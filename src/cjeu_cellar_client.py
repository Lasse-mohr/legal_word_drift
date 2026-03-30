"""
CELLAR SPARQL client for CJEU case-law data collection.

Queries the EU Publications Office SPARQL endpoint to extract decision
metadata, citation networks, subject-matter codes, procedural links,
case names, and attention metrics.

Adapted from the cjeu-py package by Stephan Meijer:
  https://github.com/step-mie/cjeu-py
  File: cjeu_py/data_collection/cellar_client.py

Local modifications:
  - Self-contained configuration (no cjeu_py.config dependency)
  - Subject-matter codes returned with extracted code suffix (subject_code)
    instead of raw URI
  - Removed save_* methods (handled by the build script)
  - Removed methods not used by our pipeline (fetch_interveners,
    fetch_referring_judgments, fetch_dossiers, fetch_summaries,
    fetch_misc_info, fetch_admin_metadata, fetch_successors,
    fetch_incorporates, fetch_cited_metadata)
"""

import logging
import time
from typing import Dict, List

import pandas as pd
from SPARQLWrapper import JSON, POST, SPARQLWrapper

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────
CELLAR_SPARQL_ENDPOINT = "https://publications.europa.eu/webapi/rdf/sparql"
CELLAR_RATE_LIMIT = 1.0       # seconds between SPARQL queries
SPARQL_BATCH_SIZE = 10_000    # OFFSET/LIMIT batch for large queries

# ── CDM namespace ────────────────────────────────────────────────────────
CDM = "http://publications.europa.eu/ontology/cdm#"

# ── CELEX sector-6 document type codes ───────────────────────────────────
# Source: cjeu-py (cjeu_py/data_collection/cellar_client.py)
# Full inventory: https://eur-lex.europa.eu/content/tools/TableOfSectors/types_of_documents_in_eurlex.html

CELEX_DOC_TYPES = {
    # Court of Justice
    "CJ": ("Judgment", "CJ"),
    "CO": ("Order", "CJ"),
    "CC": ("AG Opinion", "CJ"),
    "CV": ("Opinion (avis)", "CJ"),
    "CP": ("View (prise de position)", "CJ"),
    "CD": ("Decision", "CJ"),
    "CX": ("Ruling", "CJ"),
    "CS": ("Seizure", "CJ"),
    "CT": ("Third party proceeding", "CJ"),
    "CN": ("Communication: new case", "CJ"),
    "CA": ("Communication: judgment", "CJ"),
    "CB": ("Communication: order", "CJ"),
    "CU": ("Communication: request for opinion", "CJ"),
    "CG": ("Communication: opinion", "CJ"),
    # General Court
    "TJ": ("Judgment", "GC"),
    "TO": ("Order", "GC"),
    "TC": ("AG Opinion", "GC"),
    "TT": ("Third party proceeding", "GC"),
    "TN": ("Communication: new case", "GC"),
    "TA": ("Communication: judgment", "GC"),
    "TB": ("Communication: order", "GC"),
    # Civil Service Tribunal
    "FJ": ("Judgment", "CST"),
    "FO": ("Order", "CST"),
    "FT": ("Third party proceeding", "CST"),
    "FN": ("Communication: new case", "CST"),
    "FA": ("Communication: judgment", "CST"),
    "FB": ("Communication: order", "CST"),
}

DOC_TYPE_JUDGMENTS = ["CJ", "TJ", "FJ"]
DOC_TYPE_ORDERS = ["CO", "TO", "FO"]
DOC_TYPE_AG_OPINIONS = ["CC", "TC"]
DOC_TYPE_OTHER_JUDICIAL = ["CV", "CP", "CD", "CX"]
DOC_TYPE_ALL_JUDICIAL = (
    DOC_TYPE_JUDGMENTS + DOC_TYPE_ORDERS + DOC_TYPE_AG_OPINIONS + DOC_TYPE_OTHER_JUDICIAL
)


class CjeuCellarClient:
    """Client for querying CJEU case law from the CELLAR SPARQL endpoint.

    Adapted from cjeu-py (cjeu_py.data_collection.cellar_client.CellarClient).
    """

    def __init__(
        self,
        endpoint: str = None,
        rate_limit: float = None,
        batch_size: int = None,
    ):
        self.endpoint = endpoint or CELLAR_SPARQL_ENDPOINT
        self.rate_limit = rate_limit or CELLAR_RATE_LIMIT
        self.batch_size = batch_size or SPARQL_BATCH_SIZE
        self.sparql = SPARQLWrapper(self.endpoint)
        self.sparql.setReturnFormat(JSON)
        self.sparql.setMethod(POST)
        self._last_request_time = 0

    @staticmethod
    def _celex_filter(doc_types: List[str] = None, var: str = "celex") -> str:
        """Build a SPARQL FILTER clause for CELEX document type codes."""
        codes = doc_types or DOC_TYPE_JUDGMENTS
        alternatives = "|".join(codes)
        return f'FILTER(REGEX(?{var}, "^6[0-9]{{4}}({alternatives})"))'

    def _throttle(self):
        """Respect rate limit between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self._last_request_time = time.time()

    def _query(self, sparql_str: str) -> List[Dict]:
        """Execute a SPARQL query and return bindings as list of dicts."""
        self._throttle()
        self.sparql.setQuery(sparql_str)
        try:
            results = self.sparql.query().convert()
            bindings = results.get("results", {}).get("bindings", [])
            return [
                {k: v["value"] for k, v in row.items()}
                for row in bindings
            ]
        except Exception as e:
            logger.error(f"SPARQL query failed: {e}")
            return []

    # ── Decision metadata ─────────────────────────────────────────────────

    def fetch_decisions(
        self,
        court: str = None,
        resource_type: str = None,
        formation: str = None,
        judge: str = None,
        advocate_general: str = None,
        date_from: str = None,
        date_to: str = None,
        max_items: int = None,
        offset: int = 0,
        doc_types: List[str] = None,
    ) -> pd.DataFrame:
        """Fetch CJEU decision metadata from CELLAR.

        Returns:
            DataFrame with columns: celex, ecli, date, court_code, resource_type,
            formation_code, judge_rapporteur, advocate_general, procedure_type,
            orig_country, proc_lang, case_year, defendant_agent, applicant_agent,
            referring_court, treaty_basis, date_lodged, procjur, published_ecr,
            authentic_lang, eea_relevant, natural_celex
        """
        all_rows = []
        batch_size = self.batch_size
        current_offset = offset

        while True:
            limit = min(batch_size, max_items - len(all_rows)) if max_items else batch_size

            filters = []
            if court:
                filters.append(f'FILTER(?court_code = "{court}")')
            if resource_type:
                filters.append(f'FILTER(?resource_type = "{resource_type}")')
            if formation:
                filters.append(f'FILTER(CONTAINS(?formation_code, "{formation}"))')
            if judge:
                filters.append(f'FILTER(CONTAINS(LCASE(?judge_rapporteur), LCASE("{judge}")))')
            if advocate_general:
                filters.append(f'FILTER(CONTAINS(LCASE(?advocate_general), LCASE("{advocate_general}")))')
            if date_from:
                filters.append(f'FILTER(?date >= "{date_from}"^^xsd:date)')
            if date_to:
                filters.append(f'FILTER(?date <= "{date_to}"^^xsd:date)')

            filter_clause = "\n    ".join(filters)

            query = f"""
PREFIX cdm: <{CDM}>
SELECT DISTINCT ?celex ?ecli ?date ?court_code ?resource_type
       ?formation_code ?judge_rapporteur ?advocate_general
       ?procedure_type ?orig_country ?proc_lang ?case_year
       ?defendant_agent ?applicant_agent ?referring_court
       ?treaty_basis ?date_lodged ?procjur ?published_ecr
       ?authentic_lang ?eea_relevant ?natural_celex
WHERE {{
    ?work cdm:resource_legal_id_celex ?celex .
    {self._celex_filter(doc_types)}
    ?work cdm:work_date_document ?date .
    OPTIONAL {{ ?work cdm:case-law_ecli ?ecli }}
    OPTIONAL {{ ?work cdm:resource_legal_type ?court_code }}
    OPTIONAL {{ ?work cdm:work_has_resource-type ?restype .
               BIND(REPLACE(STR(?restype), "^.*resource-type/", "") AS ?resource_type) }}
    OPTIONAL {{ ?work cdm:case-law_delivered_by_court-formation ?form .
               BIND(REPLACE(STR(?form), "^.*formjug/", "") AS ?formation_code) }}
    OPTIONAL {{ ?work cdm:case-law_delivered_by_judge ?judgeUri .
               ?judgeUri cdm:agent_name ?judge_rapporteur }}
    OPTIONAL {{ ?work cdm:case-law_delivered_by_advocate-general ?agUri .
               ?agUri cdm:agent_name ?advocate_general }}
    OPTIONAL {{ ?work cdm:case-law_has_type_procedure_concept_type_procedure ?procUri .
               BIND(REPLACE(STR(?procUri), "^.*fd_100/", "") AS ?procedure_type) }}
    OPTIONAL {{ ?work cdm:case-law_originates_in_country ?countryUri .
               BIND(REPLACE(STR(?countryUri), "^.*country/", "") AS ?orig_country) }}
    OPTIONAL {{ ?work cdm:case-law_uses_procedure_language ?langUri .
               BIND(REPLACE(STR(?langUri), "^.*language/", "") AS ?proc_lang) }}
    OPTIONAL {{ ?work cdm:resource_legal_year ?case_year }}
    OPTIONAL {{ ?work cdm:case-law_defended_by_agent ?defUri .
               BIND(REPLACE(STR(?defUri), "^.*corporate-body/", "") AS ?defendant_agent) }}
    OPTIONAL {{ ?work cdm:case-law_requested_by_agent ?reqUri .
               BIND(REPLACE(STR(?reqUri), "^.*corporate-body/", "") AS ?applicant_agent) }}
    OPTIONAL {{ ?work cdm:case-law_delivered_by_court_national ?natCourtUri .
               BIND(REPLACE(STR(?natCourtUri), "^.*/", "") AS ?referring_court) }}
    OPTIONAL {{ ?work cdm:resource_legal_based_on_concept_treaty ?treatyUri .
               BIND(REPLACE(STR(?treatyUri), "^.*treaty/", "") AS ?treaty_basis) }}
    OPTIONAL {{ ?work cdm:resource_legal_date_request_opinion ?date_lodged }}
    OPTIONAL {{ ?work cdm:case-law_has_procjur ?procjurUri .
               BIND(REPLACE(STR(?procjurUri), "^.*procjur/", "") AS ?procjur) }}
    OPTIONAL {{ ?work cdm:case-law_published_in_erecueil ?published_ecr }}
    OPTIONAL {{ ?work cdm:resource_legal_uses_originally_language ?authLangUri .
               BIND(REPLACE(STR(?authLangUri), "^.*language/", "") AS ?authentic_lang) }}
    OPTIONAL {{ ?work cdm:resource_legal_eea ?eea_relevant }}
    OPTIONAL {{ ?work cdm:resource_legal_number_natural_celex ?natural_celex }}
    {filter_clause}
}}
ORDER BY ?date
OFFSET {current_offset}
LIMIT {limit}
"""
            logger.info(f"Fetching decisions: offset={current_offset}, limit={limit}")
            rows = self._query(query)

            if not rows:
                break

            all_rows.extend(rows)
            current_offset += len(rows)

            logger.info(f"  -> Got {len(rows)} rows (total: {len(all_rows)})")

            if max_items and len(all_rows) >= max_items:
                break
            if len(rows) < limit:
                break

        if not all_rows:
            return pd.DataFrame()

        df = pd.DataFrame(all_rows)
        if "celex" in df.columns:
            df = df.drop_duplicates(subset=["celex"], keep="first")

        logger.info(f"Fetched {len(df)} unique decisions from CELLAR")
        return df

    # ── Case names ─────────────────────────────────────────────────────────

    def fetch_case_names(
        self,
        celex_list: List[str] = None,
        max_items: int = None,
        doc_types: List[str] = None,
    ) -> pd.DataFrame:
        """Fetch case names from expression-level metadata.

        Returns:
            DataFrame with columns: celex, case_name, case_id
        """
        celex_f = self._celex_filter(doc_types)
        all_rows = []
        batch_size = self.batch_size
        offset = 0

        while True:
            limit = min(batch_size, max_items - len(all_rows)) if max_items else batch_size

            celex_filter = ""
            if celex_list:
                values = " ".join(f'"{c}"' for c in celex_list)
                celex_filter = f"VALUES ?celex {{ {values} }}"

            query = f"""
PREFIX cdm: <{CDM}>
SELECT DISTINCT ?celex ?parties ?titleAlt ?caseId
WHERE {{
    {celex_filter}
    ?work cdm:resource_legal_id_celex ?celex .
    {celex_f}
    ?expr cdm:expression_belongs_to_work ?work .
    ?expr cdm:expression_uses_language
          <http://publications.europa.eu/resource/authority/language/ENG> .
    OPTIONAL {{ ?expr cdm:expression_case-law_parties ?parties . }}
    OPTIONAL {{ ?expr cdm:expression_title_alternative ?titleAlt . }}
    OPTIONAL {{ ?expr cdm:expression_case-law_identifier_case ?caseId . }}
}}
OFFSET {offset}
LIMIT {limit}
"""
            logger.info(f"Fetching case names: offset={offset}, limit={limit}")
            rows = self._query(query)

            if not rows:
                break

            all_rows.extend(rows)
            offset += len(rows)

            if max_items and len(all_rows) >= max_items:
                break
            if len(rows) < limit:
                break

        df = pd.DataFrame(all_rows) if all_rows else pd.DataFrame(
            columns=["celex", "parties", "titleAlt", "caseId"])

        if not df.empty:
            df["case_name"] = df["titleAlt"].fillna(df["parties"])
            df["case_id"] = df.get("caseId", pd.Series(dtype=str))
            df = df[["celex", "case_name", "case_id"]].drop_duplicates(
                subset=["celex"], keep="first")

        logger.info(f"Fetched case names for {len(df)} cases")
        return df

    # ── Citations ─────────────────────────────────────────────────────────

    def fetch_citations(
        self,
        celex_list: List[str] = None,
        max_items: int = None,
        doc_types: List[str] = None,
    ) -> pd.DataFrame:
        """Fetch the citation network: which cases cite which other cases.

        Returns:
            DataFrame with columns: citing_celex, cited_celex
        """
        celex_f = self._celex_filter(doc_types, var="citing_celex")
        all_rows = []
        batch_size = self.batch_size
        offset = 0

        while True:
            limit = min(batch_size, max_items - len(all_rows)) if max_items else batch_size

            celex_filter = ""
            if celex_list:
                values = " ".join(f'"{c}"' for c in celex_list)
                celex_filter = f"VALUES ?citing_celex {{ {values} }}"

            query = f"""
PREFIX cdm: <{CDM}>
SELECT DISTINCT ?citing_celex ?cited_celex
WHERE {{
    {celex_filter}
    ?citing_work cdm:resource_legal_id_celex ?citing_celex .
    {celex_f}
    ?citing_work cdm:work_cites_work ?cited_work .
    ?cited_work cdm:resource_legal_id_celex ?cited_celex .
}}
OFFSET {offset}
LIMIT {limit}
"""
            logger.info(f"Fetching citations: offset={offset}, limit={limit}")
            rows = self._query(query)

            if not rows:
                break

            all_rows.extend(rows)
            offset += len(rows)

            if max_items and len(all_rows) >= max_items:
                break
            if len(rows) < limit:
                break

        df = pd.DataFrame(all_rows) if all_rows else pd.DataFrame(
            columns=["citing_celex", "cited_celex"])
        logger.info(f"Fetched {len(df)} citation pairs from CELLAR")
        return df

    # ── Subject matter ────────────────────────────────────────────────────

    def fetch_subject_matter(
        self,
        celex_list: List[str] = None,
        max_items: int = None,
    ) -> pd.DataFrame:
        """Fetch subject-matter codes from four CELLAR taxonomies.

        Returns:
            DataFrame with columns: celex, subject_code, subject_label, subject_source

        Note: subject_code is the URI suffix (code portion) extracted from the
        full CELLAR URI.  The original cjeu-py returns the raw URI as 'subject';
        we extract the code here for downstream compatibility.
        """
        all_rows = []

        celex_filter = ""
        if celex_list:
            values = " ".join(f'"{c}"' for c in celex_list)
            celex_filter = f"VALUES ?celex {{ {values} }}"

        sources = [
            ("eurovoc",
             "resource_legal_is_about_subject-matter",
             [1, 3, 6]),
            ("case_law_subject",
             "case-law_is-about_case-law-subject-matter",
             [6]),
            ("case_law_directory",
             "case-law_is_about_concept_new_case-law",
             [6]),
            ("case_law_directory_old",
             "case-law_is_about_concept_case-law",
             [6]),
        ]

        for source_name, cdm_prop, sectors in sources:
            offset = 0
            batch_size = self.batch_size
            sector_values = " ".join(f'"{s}"^^xsd:string' for s in sectors)

            while True:
                limit = (min(batch_size, max_items - len(all_rows))
                         if max_items else batch_size)

                query = f"""
PREFIX cdm: <{CDM}>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
SELECT DISTINCT ?celex ?subject ?subject_label
WHERE {{
    {celex_filter}
    ?work cdm:resource_legal_id_celex ?celex .
    VALUES ?sector {{ {sector_values} }}
    ?work cdm:resource_legal_id_sector ?sector .
    ?work cdm:{cdm_prop} ?subject .
    OPTIONAL {{ ?subject skos:prefLabel ?subject_label .
                FILTER(LANG(?subject_label) = "en") }}
}}
OFFSET {offset}
LIMIT {limit}
"""
                logger.info(f"Fetching {source_name} subjects: offset={offset}")
                rows = self._query(query)
                if not rows:
                    break
                for r in rows:
                    r["subject_source"] = source_name
                    # Extract code suffix from URI
                    # e.g. "http://.../fd_578/4.14.01" -> "4.14.01"
                    r["subject_code"] = r.pop("subject", "").rsplit("/", 1)[-1]
                all_rows.extend(rows)
                offset += len(rows)
                logger.info(f"  -> Got {len(rows)} rows (total {source_name}: {offset})")

                if max_items and len(all_rows) >= max_items:
                    break
                if len(rows) < limit:
                    break

        df = pd.DataFrame(all_rows) if all_rows else pd.DataFrame(
            columns=["celex", "subject_code", "subject_label", "subject_source"]
        )
        counts = df['subject_source'].value_counts().to_dict() if len(df) else {}
        logger.info(f"Fetched {len(df)} subject entries from CELLAR ({counts})")
        return df

    # ── Subject taxonomy ──────────────────────────────────────────────────

    def fetch_subject_taxonomy(self) -> pd.DataFrame:
        """Fetch the CELLAR subject-matter taxonomy (codes + labels only).

        Returns:
            DataFrame with columns: code, label, source
        """
        all_rows = []
        sources = [
            ("eurovoc",
             """
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
SELECT DISTINCT ?code ?label WHERE {{
    ?concept a skos:Concept .
    FILTER(STRSTARTS(STR(?concept), "http://eurovoc.europa.eu/"))
    ?concept skos:prefLabel ?label .
    FILTER(LANG(?label) = "en")
    BIND(REPLACE(STR(?concept), "http://eurovoc.europa.eu/", "") AS ?code)
}}
ORDER BY ?code
OFFSET {offset} LIMIT {limit}
"""),
            ("case_law_directory",
             """
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
SELECT DISTINCT ?code ?label WHERE {{
    ?concept a skos:Concept .
    FILTER(STRSTARTS(STR(?concept),
           "http://publications.europa.eu/resource/authority/fd_578/"))
    ?concept skos:prefLabel ?label .
    FILTER(LANG(?label) = "en")
    BIND(REPLACE(STR(?concept),
         "http://publications.europa.eu/resource/authority/fd_578/", "")
         AS ?code)
}}
ORDER BY ?code
OFFSET {offset} LIMIT {limit}
"""),
            ("case_law_directory_old",
             """
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
SELECT DISTINCT ?code ?label WHERE {{
    ?concept a skos:Concept .
    FILTER(STRSTARTS(STR(?concept),
           "http://publications.europa.eu/resource/authority/fd_577/"))
    ?concept skos:prefLabel ?label .
    FILTER(LANG(?label) = "en")
    BIND(REPLACE(STR(?concept),
         "http://publications.europa.eu/resource/authority/fd_577/", "")
         AS ?code)
}}
ORDER BY ?code
OFFSET {offset} LIMIT {limit}
"""),
        ]

        for source_name, query_tpl in sources:
            offset = 0
            batch_size = self.batch_size
            while True:
                query = query_tpl.format(offset=offset, limit=batch_size)
                logger.info(f"Fetching {source_name} taxonomy: offset={offset}")
                rows = self._query(query)
                if not rows:
                    break
                for r in rows:
                    r["source"] = source_name
                all_rows.extend(rows)
                offset += len(rows)
                logger.info(f"  -> Got {len(rows)} concepts (total {source_name}: {offset})")
                if len(rows) < batch_size:
                    break

        df = pd.DataFrame(all_rows) if all_rows else pd.DataFrame(
            columns=["code", "label", "source"]
        )
        df = df.drop_duplicates(subset=["code", "source"], keep="first")
        logger.info(f"Fetched {len(df)} taxonomy entries "
                     f"({df['source'].value_counts().to_dict() if len(df) else {{}}})")
        return df

    # ── Procedural links (generic pair fetcher) ───────────────────────────

    def _fetch_pairs(
        self,
        cdm_property: str,
        target_col: str,
        celex_list: List[str] = None,
        max_items: int = None,
        extract_celex: bool = False,
        doc_types: List[str] = None,
    ) -> pd.DataFrame:
        """Generic fetcher for 1-to-many CELLAR relationships."""
        celex_f = self._celex_filter(doc_types)
        all_rows = []
        batch_size = self.batch_size
        offset = 0

        while True:
            limit = min(batch_size, max_items - len(all_rows)) if max_items else batch_size

            celex_filter = ""
            if celex_list:
                values = " ".join(f'"{c}"' for c in celex_list)
                celex_filter = f"VALUES ?celex {{ {values} }}"

            if extract_celex:
                select = f"?celex ?{target_col}"
                body = f"""
    {celex_filter}
    ?work cdm:resource_legal_id_celex ?celex .
    {celex_f}
    ?work cdm:{cdm_property} ?targetWork .
    ?targetWork cdm:resource_legal_id_celex ?{target_col} ."""
            else:
                select = f"?celex ?{target_col}"
                body = f"""
    {celex_filter}
    ?work cdm:resource_legal_id_celex ?celex .
    {celex_f}
    ?work cdm:{cdm_property} ?targetUri .
    BIND(REPLACE(STR(?targetUri), "^.*/", "") AS ?{target_col})"""

            query = f"""
PREFIX cdm: <{CDM}>
SELECT DISTINCT {select}
WHERE {{{body}
}}
OFFSET {offset}
LIMIT {limit}
"""
            rows = self._query(query)
            if not rows:
                break
            all_rows.extend(rows)
            offset += len(rows)
            if max_items and len(all_rows) >= max_items:
                break
            if len(rows) < limit:
                break

        cols = ["celex", target_col]
        return pd.DataFrame(all_rows) if all_rows else pd.DataFrame(columns=cols)

    def fetch_joined_cases(
        self, celex_list: List[str] = None, max_items: int = None,
        doc_types: List[str] = None,
    ) -> pd.DataFrame:
        """Fetch joined-case links. Returns celex -> joined_celex pairs."""
        df = self._fetch_pairs("case-law_joins_case_court", "joined_celex",
                               celex_list, max_items, doc_types=doc_types)
        logger.info(f"Fetched {len(df)} joined-case links from CELLAR")
        return df

    def fetch_appeals(
        self, celex_list: List[str] = None, max_items: int = None,
        doc_types: List[str] = None,
    ) -> pd.DataFrame:
        """Fetch appeal links. Returns celex -> appeal_celex pairs."""
        df = self._fetch_pairs("case-law_subject_to_appeal_in_case_court", "appeal_celex",
                               celex_list, max_items, doc_types=doc_types)
        logger.info(f"Fetched {len(df)} appeal links from CELLAR")
        return df

    def fetch_annulled_acts(
        self, celex_list: List[str] = None, max_items: int = None,
        doc_types: List[str] = None,
    ) -> pd.DataFrame:
        """Fetch acts declared void. Returns celex -> annulled_celex pairs."""
        df = self._fetch_pairs("case-law_declares_void_resource_legal", "annulled_celex",
                               celex_list, max_items, extract_celex=True, doc_types=doc_types)
        logger.info(f"Fetched {len(df)} annulled-act links from CELLAR")
        return df

    # ── AG opinion links ───────────────────────────────────────────────────

    def fetch_ag_opinions(
        self, celex_list: List[str] = None, max_items: int = None,
        doc_types: List[str] = None,
    ) -> pd.DataFrame:
        """Fetch judgment -> AG opinion links. Returns celex -> ag_opinion_celex pairs."""
        celex_f = self._celex_filter(doc_types)
        all_rows = []
        batch_size = self.batch_size
        offset = 0

        while True:
            limit = min(batch_size, max_items - len(all_rows)) if max_items else batch_size

            celex_filter = ""
            if celex_list:
                values = " ".join(f'"{c}"' for c in celex_list)
                celex_filter = f"VALUES ?celex {{ {values} }}"

            query = f"""
PREFIX cdm: <{CDM}>
SELECT DISTINCT ?celex ?ag_opinion_celex
WHERE {{
    {celex_filter}
    ?work cdm:resource_legal_id_celex ?celex .
    {celex_f}
    ?work cdm:case-law_has_conclusions_opinion_advocate-general ?agWork .
    ?agWork cdm:resource_legal_id_celex ?ag_opinion_celex .
}}
OFFSET {offset}
LIMIT {limit}
"""
            rows = self._query(query)
            if not rows:
                break
            all_rows.extend(rows)
            offset += len(rows)
            if max_items and len(all_rows) >= max_items:
                break
            if len(rows) < limit:
                break

        df = pd.DataFrame(all_rows) if all_rows else pd.DataFrame(
            columns=["celex", "ag_opinion_celex"])
        logger.info(f"Fetched {len(df)} AG opinion links from CELLAR")
        return df

    # ── Legislation links ──────────────────────────────────────────────────

    LEGISLATION_LINK_TYPES_HIGH = [
        ("interprets", "case-law_interpretes_resource_legal"),
        ("confirms", "case-law_confirms_resource_legal"),
        ("requests_interpretation", "case-law_requests_interpretation_resource_legal"),
        ("requests_annulment", "case-law_requests_annulment_of_resource_legal"),
        ("states_failure", "case-law_states_failure_concerning_resource_legal"),
        ("amends", "case-law_amends_resource_legal"),
        ("declares_valid", "case-law_declares_valid_resource_legal"),
        ("declares_void_preliminary", "case-law_declares_void_by_preliminary_ruling_resource_legal"),
        ("declares_incidentally_valid", "case-law_declares_incidentally_valid_resource_legal"),
    ]
    LEGISLATION_LINK_TYPES_LOW = [
        ("suspends", "case-law_suspends_application_of_resource_legal"),
        ("corrects_judgment", "case-law_corrects_judgement_resource_legal"),
        ("incidentally_declares_void", "case-law_incidentally_declares_void_resource_legal"),
        ("interprets_judgment", "case-law_interpretes_judgement_resource_legal"),
        ("partially_annuls", "case-law_partially_annuls_resource_legal"),
        ("immediately_enforces", "case-law_immediately_enforces_resource_legal"),
        ("reviews_judgment", "case-law_reviews_judgement_resource_legal"),
        ("reexamined_by", "case-law_reexamined_by_case_court"),
    ]

    def fetch_legislation_links(
        self,
        celex_list: List[str] = None,
        max_items: int = None,
        include_low: bool = False,
        doc_types: List[str] = None,
    ) -> pd.DataFrame:
        """Fetch case-to-legislation links (interprets, confirms, amends, etc.).

        Returns:
            DataFrame with columns: celex, legislation_celex, link_type
        """
        celex_f = self._celex_filter(doc_types)
        link_types = list(self.LEGISLATION_LINK_TYPES_HIGH)
        if include_low:
            link_types.extend(self.LEGISLATION_LINK_TYPES_LOW)

        all_rows = []
        for link_type, cdm_prop in link_types:
            offset = 0
            batch_size = self.batch_size

            while True:
                limit = (min(batch_size, max_items - len(all_rows))
                         if max_items else batch_size)

                celex_filter = ""
                if celex_list:
                    values = " ".join(f'"{c}"' for c in celex_list)
                    celex_filter = f"VALUES ?celex {{ {values} }}"

                query = f"""
PREFIX cdm: <{CDM}>
SELECT DISTINCT ?celex ?legislation_celex
WHERE {{
    {celex_filter}
    ?work cdm:resource_legal_id_celex ?celex .
    {celex_f}
    ?work cdm:{cdm_prop} ?legWork .
    ?legWork cdm:resource_legal_id_celex ?legislation_celex .
}}
OFFSET {offset}
LIMIT {limit}
"""
                rows = self._query(query)
                if not rows:
                    break
                for r in rows:
                    r["link_type"] = link_type
                all_rows.extend(rows)
                offset += len(rows)
                if max_items and len(all_rows) >= max_items:
                    break
                if len(rows) < limit:
                    break

            if all_rows:
                count = sum(1 for r in all_rows if r["link_type"] == link_type)
                if count:
                    logger.info(f"  {link_type}: {count} links")

        df = pd.DataFrame(all_rows) if all_rows else pd.DataFrame(
            columns=["celex", "legislation_celex", "link_type"])
        logger.info(f"Fetched {len(df)} legislation links from CELLAR")
        return df

    # ── Academic citations ─────────────────────────────────────────────────

    def fetch_academic_citations(
        self,
        celex_list: List[str] = None,
        max_items: int = None,
    ) -> pd.DataFrame:
        """Fetch academic journal citations related to each case.

        Returns:
            DataFrame with columns: celex, citation_text
        """
        all_rows = []
        batch_size = self.batch_size
        offset = 0

        while True:
            limit = min(batch_size, max_items - len(all_rows)) if max_items else batch_size

            celex_filter = ""
            if celex_list:
                values = " ".join(f'"{c}"' for c in celex_list)
                celex_filter = f"VALUES ?celex {{ {values} }}"

            query = f"""
PREFIX cdm: <{CDM}>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
SELECT DISTINCT ?celex ?citation_text
WHERE {{
    {celex_filter}
    ?work cdm:resource_legal_id_celex ?celex .
    ?work cdm:resource_legal_id_sector "6"^^xsd:string .
    ?work cdm:case-law_article_journal_related ?citation_text .
}}
OFFSET {offset}
LIMIT {limit}
"""
            rows = self._query(query)
            if not rows:
                break
            all_rows.extend(rows)
            offset += len(rows)
            if max_items and len(all_rows) >= max_items:
                break
            if len(rows) < limit:
                break

        df = pd.DataFrame(all_rows) if all_rows else pd.DataFrame(
            columns=["celex", "citation_text"])
        logger.info(f"Fetched {len(df)} academic citation entries from CELLAR")
        return df
