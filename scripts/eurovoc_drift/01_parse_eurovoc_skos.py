"""Parse the EuroVoc SKOS-XL RDF/XML dump into flat tables.

Input:  data/raw/eurovoc/eurovoc-skos-ap-eu.rdf  (~456 MB)
Output: data/processed/eurovoc/concepts.jsonl
        data/processed/eurovoc/labels_en.csv         (concept_id, label, label_type, notation)
        data/processed/eurovoc/schemes.csv           (scheme_id, scheme_label, domain_code, domain_name, microthesaurus_name)
        data/processed/eurovoc/concepts_enriched.csv (concept + microthesaurus + domain attached)

The dump uses SKOS-XL: each <skos:Concept> / <skos:ConceptScheme> points to
xl:Label resources (one per language); the Label resource carries
xl:literalForm with xml:lang. We keep only English. We stream the file
with iterparse and clear elements as we go to keep memory bounded.

EuroVoc structure:
- One master scheme `100141` "EuroVoc" — every concept is in this.
- 127 microthesauri whose English labels start with a 4-digit code, e.g.
  "1216 criminal law". The first two digits identify one of 21 *domains*
  (e.g. 12 = Law). Concepts carry the microthesaurus via skos:inScheme.
- A separate `domains` scheme that lists the 21 domain roots.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from xml.etree.ElementTree import iterparse

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "data/raw/eurovoc/eurovoc-skos-ap-eu.rdf"
OUT_DIR = REPO / "data/processed/eurovoc"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NS = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "skos": "http://www.w3.org/2004/02/skos/core#",
    "xl": "http://www.w3.org/2008/05/skos-xl#",
    "dct": "http://purl.org/dc/terms/",
}
RDF_ABOUT = f"{{{NS['rdf']}}}about"
RDF_RESOURCE = f"{{{NS['rdf']}}}resource"
RDF_TYPE = f"{{{NS['rdf']}}}type"
XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"

CONCEPT_TYPE = NS["skos"] + "Concept"
SCHEME_TYPE = NS["skos"] + "ConceptScheme"
LABEL_TYPE = NS["xl"] + "Label"

MASTER_SCHEME_ID = "100141"  # "EuroVoc" — every concept belongs to this

# 21 EuroVoc domains. The 2-digit prefix on a microthesaurus label
# (e.g. "1216 criminal law" -> "12") maps to one of these.
DOMAINS = {
    "04": "Politics",
    "08": "International relations",
    "10": "European Union",
    "12": "Law",
    "16": "Economics",
    "20": "Trade",
    "24": "Finance",
    "28": "Social questions",
    "32": "Education and communications",
    "36": "Science",
    "40": "Business and competition",
    "44": "Employment and working conditions",
    "48": "Transport",
    "52": "Environment",
    "56": "Agriculture, forestry and fisheries",
    "60": "Agri-foodstuffs",
    "64": "Production, technology and research",
    "66": "Energy",
    "68": "Industry",
    "72": "Geography",
    "76": "International organisations",
}

MICROTHES_PREFIX_RE = re.compile(r"^(\d{2})(\d{2})\s+(.*)$")


def short_id(uri: str) -> str:
    return uri.rsplit("/", 1)[-1]


def parse() -> tuple[dict, dict, dict]:
    """One streaming pass. Returns (concepts, schemes, labels_en)."""
    concepts: dict[str, dict] = {}
    schemes: dict[str, dict] = {}  # scheme_id -> {pref_label_ids: [...]}
    labels_en: dict[str, str] = {}

    n_concepts = n_schemes = n_labels_en = n_labels_skipped = 0
    DESCRIPTION = f"{{{NS['rdf']}}}Description"

    for _, elem in iterparse(str(SRC), events=("end",)):
        if elem.tag != DESCRIPTION:
            continue
        about = elem.get(RDF_ABOUT)
        if about is None:
            elem.clear()
            continue

        type_uri = None
        for t in elem.findall(RDF_TYPE):
            type_uri = t.get(RDF_RESOURCE)
            if type_uri:
                break

        if type_uri == LABEL_TYPE:
            for lf in elem.findall(f"{{{NS['xl']}}}literalForm"):
                if lf.get(XML_LANG) == "en":
                    labels_en[short_id(about)] = (lf.text or "").strip()
                    n_labels_en += 1
                    break
            else:
                n_labels_skipped += 1

        elif type_uri == CONCEPT_TYPE:
            cid = short_id(about)
            rec: dict = {
                "id": cid,
                "uri": about,
                "pref_label_ids": [],
                "alt_label_ids": [],
                "broader": [],
                "narrower": [],
                "related": [],
                "in_scheme": [],
                "top_concept_of": [],
                "notation": None,
            }
            for child in elem:
                tag = child.tag
                res = child.get(RDF_RESOURCE)
                if tag == f"{{{NS['xl']}}}prefLabel" and res:
                    rec["pref_label_ids"].append(short_id(res))
                elif tag == f"{{{NS['xl']}}}altLabel" and res:
                    rec["alt_label_ids"].append(short_id(res))
                elif tag == f"{{{NS['skos']}}}broader" and res:
                    rec["broader"].append(short_id(res))
                elif tag == f"{{{NS['skos']}}}narrower" and res:
                    rec["narrower"].append(short_id(res))
                elif tag == f"{{{NS['skos']}}}related" and res:
                    rec["related"].append(short_id(res))
                elif tag == f"{{{NS['skos']}}}inScheme" and res:
                    rec["in_scheme"].append(short_id(res))
                elif tag == f"{{{NS['skos']}}}topConceptOf" and res:
                    rec["top_concept_of"].append(short_id(res))
                elif tag == f"{{{NS['skos']}}}notation":
                    rec["notation"] = (child.text or "").strip() or None
            concepts[cid] = rec
            n_concepts += 1

        elif type_uri == SCHEME_TYPE:
            sid = short_id(about)
            pref_ids = [
                short_id(res)
                for c in elem.findall(f"{{{NS['xl']}}}prefLabel")
                if (res := c.get(RDF_RESOURCE))
            ]
            schemes[sid] = {"pref_label_ids": pref_ids}
            n_schemes += 1

        elem.clear()

    print(f"concepts:        {n_concepts:,}")
    print(f"schemes:         {n_schemes:,}")
    print(f"english labels:  {n_labels_en:,}")
    print(f"non-en labels:   {n_labels_skipped:,}")
    return concepts, schemes, labels_en


def build_scheme_index(schemes: dict, labels_en: dict) -> dict:
    """scheme_id -> {label, domain_code, domain_name, microthesaurus_name}."""
    out = {}
    for sid, rec in schemes.items():
        en = next((labels_en[p] for p in rec["pref_label_ids"] if p in labels_en), None)
        domain_code = domain_name = micro_name = None
        if en:
            m = MICROTHES_PREFIX_RE.match(en)
            if m:
                domain_code = m.group(1)
                domain_name = DOMAINS.get(domain_code)
                micro_name = m.group(3)
        out[sid] = {
            "label": en,
            "domain_code": domain_code,
            "domain_name": domain_name,
            "microthesaurus_name": micro_name,
        }
    return out


def resolve_and_write(concepts: dict, schemes: dict, labels_en: dict) -> None:
    out_jsonl = OUT_DIR / "concepts.jsonl"
    out_labels = OUT_DIR / "labels_en.csv"
    out_schemes = OUT_DIR / "schemes.csv"
    out_enriched = OUT_DIR / "concepts_enriched.csv"

    scheme_idx = build_scheme_index(schemes, labels_en)

    # schemes.csv
    with out_schemes.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scheme_id", "scheme_label", "domain_code", "domain_name", "microthesaurus_name"])
        for sid, info in sorted(scheme_idx.items()):
            w.writerow([
                sid,
                info["label"] or "",
                info["domain_code"] or "",
                info["domain_name"] or "",
                info["microthesaurus_name"] or "",
            ])

    n_with_pref = n_no_pref = 0
    n_with_micro = 0
    with (
        out_jsonl.open("w") as fj,
        out_labels.open("w", newline="") as fl,
        out_enriched.open("w", newline="") as fe,
    ):
        wl = csv.writer(fl)
        wl.writerow(["concept_id", "label", "label_type", "notation"])
        we = csv.writer(fe)
        we.writerow([
            "concept_id",
            "pref_label_en",
            "n_alt_labels",
            "is_leaf",
            "depth_min",  # min broader-chain depth (0 = top concept)
            "microthesaurus_id",
            "microthesaurus_name",
            "domain_code",
            "domain_name",
            "notation",
        ])

        # Pre-compute a depth via memoised broader-walk.
        depth_cache: dict[str, int] = {}

        def depth(cid: str, seen: frozenset = frozenset()) -> int:
            if cid in depth_cache:
                return depth_cache[cid]
            if cid in seen:
                return 0  # cycle guard
            rec = concepts.get(cid)
            if not rec or not rec["broader"]:
                depth_cache[cid] = 0
                return 0
            d = 1 + min(depth(b, seen | {cid}) for b in rec["broader"])
            depth_cache[cid] = d
            return d

        for cid, rec in concepts.items():
            pref_en = [labels_en[lid] for lid in rec["pref_label_ids"] if lid in labels_en]
            alt_en = [labels_en[lid] for lid in rec["alt_label_ids"] if lid in labels_en]
            if pref_en:
                n_with_pref += 1
            else:
                n_no_pref += 1

            # Pick the microthesaurus = first non-master scheme this concept is in.
            micro_id = next((s for s in rec["in_scheme"] if s != MASTER_SCHEME_ID), None)
            micro_info = scheme_idx.get(micro_id, {}) if micro_id else {}
            if micro_id:
                n_with_micro += 1

            fj.write(json.dumps({
                "id": cid,
                "pref_label_en": pref_en[0] if pref_en else None,
                "alt_labels_en": alt_en,
                "broader": rec["broader"],
                "narrower": rec["narrower"],
                "related": rec["related"],
                "in_scheme": rec["in_scheme"],
                "top_concept_of": rec["top_concept_of"],
                "notation": rec["notation"],
                "microthesaurus_id": micro_id,
                "microthesaurus_name": micro_info.get("microthesaurus_name"),
                "domain_code": micro_info.get("domain_code"),
                "domain_name": micro_info.get("domain_name"),
            }, ensure_ascii=False) + "\n")

            for lab in pref_en:
                wl.writerow([cid, lab, "pref", rec["notation"] or ""])
            for lab in alt_en:
                wl.writerow([cid, lab, "alt", rec["notation"] or ""])

            we.writerow([
                cid,
                pref_en[0] if pref_en else "",
                len(alt_en),
                int(not rec["narrower"]),
                depth(cid),
                micro_id or "",
                micro_info.get("microthesaurus_name") or "",
                micro_info.get("domain_code") or "",
                micro_info.get("domain_name") or "",
                rec["notation"] or "",
            ])

    print(f"concepts with EN pref:    {n_with_pref:,}")
    print(f"concepts w/o EN pref:     {n_no_pref:,}")
    print(f"concepts with micro tag:  {n_with_micro:,}")
    print(f"wrote {out_jsonl}")
    print(f"wrote {out_labels}")
    print(f"wrote {out_schemes}")
    print(f"wrote {out_enriched}")


if __name__ == "__main__":
    concepts, schemes, labels_en = parse()
    resolve_and_write(concepts, schemes, labels_en)
