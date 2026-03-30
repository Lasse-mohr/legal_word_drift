# Legal Word Drift: Semantic Change in CJEU Judgments (1990--2025)

This project trains diachronic word embeddings on the full text of Court of Justice of the European Union (CJEU) judgments to measure how legal concepts shift, narrow, broaden, and cluster over time.

The pipeline fetches judgments from the EU's CELLAR database, preprocesses them with domain-specific tokenization, trains Word2Vec models per time slice, aligns them via Orthogonal Procrustes, and computes drift metrics. The result is a set of per-word time series showing how each term's semantic neighborhood evolves.

---

## Quick start

```bash
conda activate legal_word_drift

# 1. Fetch metadata (SPARQL)
python -m src.pipeline.01_fetch_metadata --start-year 2000 --end-year 2005

# 2. Fetch full texts (XHTML from CELLAR REST)
python -m src.pipeline.02_fetch_texts --concurrency 5

# 3. Tokenize paragraphs into sentence files
python -m src.pipeline.03_preprocess

# 4. Train phrase detector and re-tokenize
python -m src.pipeline.04_detect_phrases

# 5. Train Word2Vec models (sliding windows + single years)
python -m src.pipeline.05_train_embeddings

# 6. Build vocabulary tiers and align embeddings
python -m src.pipeline.06_align_embeddings

# 7. Compute drift metrics
python -m src.pipeline.07_compute_metrics
```

For a full 1990--2025 run, change `--start-year 1990 --end-year 2025` in step 1. Steps 2--7 automatically operate on whatever metadata has been fetched.

---

## Pipeline decisions

Every parameter and design choice in the pipeline falls into one of three categories:

| Label | Meaning |
|---|---|
| **Standard** | Well-established in the literature; unlikely to be controversial |
| **Reasoned** | A judgment call with clear rationale, but defensible alternatives exist |
| **Uncertain** | Genuinely unclear which option is better; worth experimenting with |

---

### 1. Data scope

| Decision | Choice | Status | Rationale | Alternatives |
|---|---|---|---|---|
| Language | English only | **Reasoned** | English has the best NLP tooling and is the working language of EU legal scholarship. | French is the CJEU's internal deliberation language and might be more "authentic." Multilingual analysis is possible but dramatically harder. |
| Document types | Judgments only (CELEX pattern `6*CJ`, `6*TJ`, `6*FJ`) | **Reasoned** | Judgments are the court's final word. AG Opinions have a different rhetorical register and might confound the signal. Orders are too short and formulaic. | AG Opinions could form a parallel analysis. Including them would increase corpus size but mix two distinct genres. |
| Temporal scope | 1990--2025 (pilot: 2000--2005) | **Reasoned** | Pre-1990 English-language coverage in CELLAR is sparse and inconsistent. | Could extend to 1954 (founding of the court) if willing to accept noisier early embeddings. |

**Key file:** `src/pipeline/01_fetch_metadata.py`

---

### 2. Text preprocessing

**Key files:** `src/preprocessing/legal_tokenizer.py`, `src/preprocessing/corpus_builder.py`

#### Tokenization basics

| Decision | Choice | Status | Rationale | Alternatives |
|---|---|---|---|---|
| Lowercasing | All text lowercased | **Standard** | Universal in word2vec pipelines. Case distinctions add noise without benefit for distributional semantics. | Preserving case for named entities is occasionally done but word2vec doesn't benefit from it. |
| Tokenization regex | `[a-z_][a-z0-9_]*` | **Reasoned** | Simple and effective. The regex preserves underscore-joined phrases from earlier pipeline steps. | spaCy tokenizer would handle edge cases better but is slower and adds dependency complexity. |
| Minimum token length | 2 characters | **Standard** | Filters single letters (section markers, list items). | Threshold of 3 would be more aggressive but would lose "ec", "eu", "ip". |
| Sentence unit | One paragraph = one "sentence" for word2vec | **Reasoned** | Legal paragraphs are typically 1--5 sentences. Using whole paragraphs lets the context window span sentence boundaries, which captures cross-sentence co-occurrence. | Splitting on actual sentence boundaries would need a sentence segmenter tuned for legal text (abbreviations like "Art.", "No.", "para." cause false splits). |

#### Citation handling

Legal text is dense with citations that need special treatment:

| Decision | Choice | Status | Rationale | Alternatives |
|---|---|---|---|---|
| Case references | Replace with `__CASEREF__` placeholder | **Reasoned** | Case numbers (e.g., "Case C-176/03") don't carry semantic weight as *words*, but their presence provides context. The placeholder preserves positional information without polluting the vocabulary. | Remove entirely (saves vocab space). Preserve case identity (e.g., `case_c_176_03`) to study citation drift -- interesting but a different research question. |
| Article references | Normalize to `article_NNN_treaty` (e.g., `article_234_ec`) | **Reasoned** | Treaty renumbering means the same legal provision appears under different numbers over time (Article 177 EEC = 234 EC = 267 TFEU). We preserve the article *as written in that era's text*, so the same concept appears under different tokens -- itself a signal of drift. | Map all articles to current TFEU numbering. This would collapse the renumbering signal but let you track the *concept* rather than the *term*. The choice depends on whether you want to study legal language or legal ideas. |
| Standalone numbers | Replace with `__NUM__` | **Reasoned** | Numbers like dates, amounts, and paragraph references add vocabulary noise. | Remove entirely (slightly smaller vocab). Keep as-is (numbers like "100%" carry meaning in context). |

#### Domain-specific preprocessing

| Decision | Choice | Status | Rationale | Alternatives |
|---|---|---|---|---|
| Latin phrases | Curated list of ~60 phrases, joined with underscores before tokenization | **Reasoned** | Legal Latin is full of multi-word terms (*nemo iudex in causa sua*, *acte clair*, *inter alia*). Joining them prevents word2vec from seeing disconnected fragments. The curated list is conservative: it may miss rare phrases but won't produce false positives. | Let gensim `Phrases` detect them automatically (would miss rare ones). Use spaCy NER or multi-word expression detection. |
| Stopword list | Standard English stopwords + 18 legal-specific words ("hereby", "whereas", "thereof", "herein", etc.) | **Reasoned** | The legal additions are genuinely formulaic -- they appear in almost every judgment regardless of topic. | Use NLTK/spaCy defaults only. Use no stopwords and let word2vec's `sample` parameter handle frequency downweighting. Add more legal terms like "applicant", "defendant", "order" -- but these might carry meaning depending on context. |
| Section filtering | Currently processes **all** paragraphs (grounds + operative part) | **Uncertain** | The operative part ("the Court hereby rules...") is highly formulaic and adds noise. Filtering to grounds-only would be cleaner. However, the section metadata from the XHTML parser isn't currently carried through to the sentence files. | Filter to grounds paragraphs only. This would require propagating the `section` field from the parser through the JSONL to the corpus builder. Worth doing. |
| Lemmatization | None | **Uncertain** | Raw word forms mean "judge", "judges", "judging", "judged" are separate vocabulary entries. This preserves morphological signal but fragments the data. For legal text, the distinction between "proportional" and "proportionality" may genuinely matter. | Lemmatize with spaCy. Would collapse word forms, giving more data per concept but losing morphological variation. **Worth testing both ways.** |
| POS filtering | None | **Uncertain** | Some drift studies restrict to nouns and adjectives (the primary meaning-bearing categories). We include all parts of speech. | Filter to nouns + adjectives only. Cleaner signal at the cost of losing verbal semantic shifts (e.g., "harmonize" changing meaning). |

#### Multi-word term detection

**Key file:** `src/preprocessing/phrase_detector.py`

| Decision | Choice | Status | Rationale | Alternatives |
|---|---|---|---|---|
| Method | Gensim `Phrases` (bigram pass, then trigram pass on bigrammed text) | **Standard** | This is the standard approach from Mikolov et al. (2013). Two passes let you detect trigrams like `free_movement_of_goods` by first finding `free_movement`, then finding `free_movement_of_goods`. | spaCy noun chunks. Domain-specific dictionary lookup. The automated approach is more scalable. |
| `min_count` | 30 | **Reasoned** | Minimum co-occurrence for a phrase to be detected. Lower values produce more phrases (risk of noise); higher values miss real collocations. 30 is conservative for a corpus of this size. | 10 (more phrases, more noise), 50 (fewer phrases, more conservative). |
| `threshold` | 10.0 | **Reasoned** | Scoring threshold -- higher means fewer phrases. The gensim default is 10.0. Controls the trade-off between precision and recall of phrase detection. | Lower (5.0) for more phrases, higher (20.0) for fewer. |
| Connector words | `ENGLISH_CONNECTOR_WORDS` (gensim default) | **Standard** | Allows words like "of", "the", "and" in the middle of phrases. Essential for legal collocations like "freedom_of_establishment", "principle_of_proportionality". | None. |

---

### 3. Embeddings

**Key file:** `src/embeddings/trainer.py`

#### Algorithm choice

| Decision | Choice | Status | Rationale | Alternatives |
|---|---|---|---|---|
| Model | Word2Vec | **Standard** | The standard choice for diachronic studies. Simple, well-understood, and the alignment methods (Procrustes) are designed for it. | GloVe (similar properties, different training). FastText (handles OOV via subwords, adds complexity). BERT contextual embeddings (captures polysemy but much harder to align across time -- active research area). |
| Architecture | Skip-gram (`sg=1`) | **Standard** | Skip-gram outperforms CBOW on smaller corpora and with rare words (Mikolov et al., 2013). Legal vocabulary has a long tail of rare technical terms, making Skip-gram the clear choice. | CBOW. Faster to train but worse with rare words. |

#### Hyperparameters

| Parameter | Value | Status | Rationale | Alternatives |
|---|---|---|---|---|
| `vector_size` | 100 | **Reasoned** | Hamilton et al. (2016) used 300 on Google Books (billions of words). Our yearly corpora are 150K--1M words -- much smaller. Lower dimensions reduce overfitting. 100 is a common choice for corpora of this size. | 50 (more stable, less expressive), 200 (more expressive, noisier on small data), 300 (standard for large corpora). |
| `window` | 5 | **Standard** | The conventional default. Smaller windows (2--3) capture syntactic relations; larger windows (8--10) capture topical/semantic relations. 5 is the standard middle ground. | For legal text where we care about semantic meaning, 8--10 might be better. **Worth experimenting with.** |
| `min_count` | 50 | **Reasoned** | Words appearing fewer than 50 times in a time slice are excluded. This is aggressive (gensim default is 5). Rare words have unstable embeddings that add noise to alignment. Hamilton et al. (2016) also used aggressive filtering. | 10--20 (includes more vocabulary at cost of stability), 100 (very conservative). The right value depends on corpus size per time slice. |
| `negative` | 10 | **Reasoned** | Negative sampling count. Gensim default is 5. Higher values help with smaller corpora (Mikolov et al., 2013 suggests 5--20). | 5 (default), 15--20 (more for very small corpora). |
| `sample` | 1e-4 | **Standard** | Subsampling threshold. Aggressively downsamples ultra-frequent words like "court", "article", "case" so they don't dominate training. Gensim default is 1e-3; we use 1e-4 for stronger downsampling because legal text is repetitive. | 1e-3 (default, less aggressive), 1e-5 (very aggressive). |
| `epochs` | 10 | **Reasoned** | Training epochs. Gensim default is 5. More epochs compensate for smaller corpus size. Hamilton et al. (2016) used 5 on a much larger corpus. | 5 (faster, may underfit on small data), 20 (more, risk of overfitting). |
| `workers` | 1 | **Standard** | Single-threaded for reproducibility. Multi-threaded word2vec in gensim is non-deterministic even with a fixed seed. Essential for scientific reproducibility and for bootstrap replicates to be meaningful. Makes training 2--5x slower. | Higher values for speed at the cost of exact reproducibility. |
| `seed` | 42 | **Standard** | Arbitrary but fixed. Combined with `workers=1`, ensures identical results on re-run. | Any fixed integer. |

**Interaction note:** `min_count`, `vector_size`, and corpus size per time slice are tightly coupled. If you change the windowing (which changes corpus size), you may need to adjust `min_count` and `vector_size` accordingly. Smaller corpora need lower `min_count` (to retain enough vocabulary) and lower `vector_size` (to avoid overfitting).

---

### 4. Temporal windowing

**Key file:** `src/embeddings/trainer.py`

| Decision | Choice | Status | Rationale | Alternatives |
|---|---|---|---|---|
| Primary unit | 5-year sliding windows, 1-year step | **Standard** | Hamilton et al. (2016) used decade-scale bins on Google Books. For our corpus (~300--500 judgments/year, 150K--1M words/year), 5-year windows give enough text per window (~3--10M words) for stable embeddings with fine enough resolution to detect gradual drift. | 3-year windows (finer resolution, noisier embeddings), 10-year windows (more stable, coarser resolution). |
| Step size | 1 year | **Standard** | Maximum temporal resolution. Each window overlaps with the next by 4 years. | Step=2 or step=5 would be faster to compute but lose resolution. |
| Secondary unit | Single-year models | **Reasoned** | A validation check: do patterns from the windowed models hold at yearly resolution? Years with small corpora (< 200K words) will produce unreliable models, but they can confirm trends seen in the windowed analysis. | Not training yearly models at all (faster, but no cross-check). |

---

### 5. Alignment

**Key file:** `src/embeddings/alignment.py`

#### Method

| Decision | Choice | Status | Rationale | Alternatives |
|---|---|---|---|---|
| Algorithm | Orthogonal Procrustes | **Standard** | The standard method from Hamilton et al. (2016). Finds the rotation matrix **R** that best maps one embedding space onto another, minimizing `‖XR - Y‖` subject to R being orthogonal. Preserves distances and angles within each space. | **Incremental training** (initialize year N+1 from year N): simpler but leaks information across periods and makes permutation tests impossible. **TWEC / Compass alignment** (Di Carlo et al., 2019): trains all periods jointly with a shared compass. More principled but less transparent. |

The key advantage of Procrustes is that each model is trained independently. This means: (1) no information leaks between time periods, (2) permutation tests are possible, (3) the internal geometry of each model is preserved exactly.

#### Anchor words

| Decision | Choice | Status | Rationale | Alternatives |
|---|---|---|---|---|
| Selection criterion | Top N words by harmonic mean frequency across time slices | **Reasoned** | Harmonic mean favors words that are *consistently* common rather than spiking in one period. The assumption is that high-frequency, temporally-stable words are semantically stable and thus good alignment anchors. | Use all shared vocabulary (noisier alignment). Hand-pick "obviously stable" words (subjective). Iterative refinement: align, find anchors that moved most, remove them, re-align. |
| Anchor count | 500 (300 for small pilot runs) | **Uncertain** | More anchors = more constraints on the rotation = more stable alignment, but also more risk of including a drifting word that distorts the rotation. Fewer anchors = less constraint = noisier alignment. | 200 (fewer constraints), 1000 (more constraints). **Worth testing sensitivity.** |

**Important caveat:** The assumption that high-frequency words are semantically stable is strong and potentially circular. A word could be frequent *and* drifting. If a core anchor word is actually shifting meaning, it would silently distort the alignment and make other words appear to shift less (or more) than they actually do.

#### Chain topology

| Decision | Choice | Status | Rationale | Alternatives |
|---|---|---|---|---|
| Alignment topology | Chain: align outward from a reference model | **Reasoned** | Each model is aligned to its temporal neighbor, forming a chain. This maximizes vocabulary overlap between each aligned pair. | **Star topology:** align every model directly to the reference. Avoids chain error propagation but each pair may have less vocabulary overlap (especially for distant periods). |
| Reference model | Most recent window | **Reasoned** | The most recent period has the largest corpus and thus the most stable embeddings. | Use the largest-corpus period (might differ from most recent). Align to the centroid of all models. |

**Chain propagation warning:** If the alignment between two adjacent models is poor (e.g., due to small vocabulary overlap), that error propagates to all models further down the chain. Star topology avoids this but has its own drawbacks.

---

### 6. Vocabulary tiers

**Key file:** `src/embeddings/vocabulary.py`

The pipeline maintains three nested vocabulary sets:

| Tier | Criterion | Purpose | Status |
|---|---|---|---|
| **V_global** | Word appears in 3+ time slices | Common vocabulary across the corpus. Used for building aligned embedding matrices. | **Reasoned** -- the threshold of 3 is somewhat arbitrary. |
| **V_analysis** | Word appears in 10+ time slices, excludes placeholder tokens (`__CASEREF__`, `__NUM__`, etc.) | Words with enough temporal coverage for meaningful trend analysis. With ~32 windows, 10+ means present in roughly 1/3 of the time span. | **Reasoned** -- threshold depends on total number of windows. For the 2000--2005 pilot (4 windows), this was lowered to 3. |
| **V_anchor** | Top N from V_global by harmonic mean frequency, present in 3+ slices | Alignment anchors. These are the words assumed to be semantically stable. | **Reasoned** -- see anchor word discussion above. |

---

### 7. Drift metrics

**Key files:** `src/metrics/shift.py`, `src/metrics/dispersion.py`

#### Core shift metrics

| Metric | Formula | Status | What it captures |
|---|---|---|---|
| **Cosine shift** | `1 - cos(v_t1, v_t2)` | **Standard** | Geometric distance between a word's vectors in two aligned spaces. The primary metric from Hamilton et al. (2016). A word can shift a lot in cosine distance even if its neighbors stay the same (rotating within the same neighborhood). |
| **Jaccard neighborhood shift** | `1 - |NN_t1 ∩ NN_t2| / |NN_t1 ∪ NN_t2|`, k=25 | **Standard** | Whether a word's nearest neighbors changed, regardless of geometric movement. Captures "change in company" even when cosine shift is small. Also from Hamilton et al. (2016). |
| **Combined shift** | `0.5 * cosine + 0.5 * jaccard` | **Uncertain** | Simple average of the two metrics. The equal weighting is arbitrary -- the two metrics capture different phenomena. |

The choice of **k=25** for nearest neighbors is a judgment call. Larger k is more stable but less sensitive to local changes; smaller k is more sensitive but noisier.

#### Dispersion metrics (semantic narrowing/broadening)

| Metric | Formula | Status | What it captures |
|---|---|---|---|
| **k-NN dispersion** | Mean cosine distance to k=25 nearest neighbors | **Reasoned** | Whether a word's neighborhood is getting tighter (narrowing) or more spread out (broadening) over time. Decreasing dispersion = the word is becoming more specialized. |
| **Neighborhood density** | Mean pairwise cosine similarity among the k=25 neighbors | **Reasoned** | Whether the neighbors themselves form a tight cluster, independent of the target word. High density = the word lives in a coherent semantic region. |
| **Effective neighborhood size** | Count of words with cosine similarity > 0.5 | **Uncertain** | A threshold-based measure of how many words are "close." The 0.5 threshold is arbitrary and should ideally be calibrated per-model (e.g., 95th percentile of random pairwise similarities). |

---

### 8. Clustering

**Key file:** `src/metrics/clustering.py`

| Decision | Choice | Status | Rationale | Alternatives |
|---|---|---|---|---|
| Community detection | Louvain algorithm on a k-NN graph (k=10) | **Standard** | Louvain is the standard algorithm for community detection in networks. k=10 for the graph is a judgment call -- too small and clusters fragment, too large and they merge. | Leiden algorithm (improved Louvain with better guarantees). HDBSCAN (density-based, doesn't require a graph). Spectral clustering. |
| Domain coherence | Mean pairwise cosine similarity within hand-curated word groups | **Reasoned** | Simple and interpretable. Measures whether words that "should" be related (based on legal domain knowledge) actually cluster together, and how that clustering evolves over time. | Derive groups from the data via unsupervised clustering rather than imposing them. This would be more objective but harder to interpret. |
| Domain word lists | 10 domains, ~10 words each (competition, free movement, data protection, state aid, fundamental rights, environment, IP, consumer protection, preliminary reference, proportionality) | **Uncertain** | Hand-curated based on EU law domain knowledge. May miss important domains (taxation, transport, agriculture). Individual word choices are debatable -- is "balance" really a proportionality word? Is "national" really a preliminary reference word? | Expand to more domains. Validate word lists with legal experts. Use automated methods to discover domains. |

---

### 9. Visualization

**Key file:** `src/visualization/embedding_plots.py`

| Decision | Choice | Status | Rationale | Alternatives |
|---|---|---|---|---|
| Stacking approach | Stack all (word, time_slice) vectors into one matrix, reduce jointly | **Reasoned** | Ensures all time points share the same 2D coordinate system, making trajectories comparable. | Reduce each time slice separately (loses cross-time comparability). Aligned UMAP / parametric UMAP (fits a projection function). |
| PCA | Standard PCA, 2 components | **Standard** | Linear, deterministic, preserves global structure. Good for a first look. Downside: may not reveal non-linear clusters. | t-SNE (non-linear, non-deterministic, focuses on local structure). |
| UMAP | n_neighbors=15, min_dist=0.1, cosine metric | **Reasoned** | UMAP preserves both local and global structure. Cosine metric matches how we measure similarity elsewhere. n_neighbors=15 and min_dist=0.1 are the defaults. **Different UMAP parameters produce very different plots** -- this is important to acknowledge. Larger n_neighbors emphasizes global structure; smaller emphasizes local clusters. | n_neighbors=5 (more local), n_neighbors=50 (more global). min_dist=0.01 (tighter clusters), min_dist=0.5 (more spread). |

---

### 10. Statistical controls

**Key file:** `src/stats/bootstrap.py`

| Decision | Choice | Status | Rationale | Alternatives |
|---|---|---|---|---|
| Method | Bootstrap confidence intervals (10 replicates per metric) | **Standard** | Train M independent word2vec models per time slice with different random seeds, compute metrics on each, report mean and 95% CI. Tests whether observed drift exceeds the noise floor of the embedding algorithm itself. | Permutation tests (shuffle time labels, re-align, compare). Frequency controls (regress out frequency effects). Displacement baselines (compare to random words with similar frequency). These were considered but dropped to keep the pipeline focused. |
| Replicate count | 10 | **Reasoned** | Enough to estimate a CI, not so many that training takes forever (each replicate trains 2 full word2vec models). 10 replicates with `workers=1` is already slow. | 5 (faster, wider CIs), 25--50 (more precise CIs, much slower). |

---

### Interactions between decisions

Several decisions interact in ways that aren't obvious from looking at them individually:

1. **Corpus size x vector_size x min_count:** Smaller corpora need smaller dimensions (to avoid overfitting) and lower min_count (to retain enough vocabulary). If you change the windowing from 5-year to 3-year windows, you halve the corpus per window and may need to reduce vector_size from 100 to 50 and min_count from 50 to 30.

2. **min_count x anchor words:** Aggressive min_count filtering reduces the shared vocabulary between time slices, which directly affects how many anchor words are available for Procrustes alignment. If too few anchors survive, the alignment degrades.

3. **Stopwords x subsampling:** Both serve to reduce the influence of ultra-frequent words. Stopwords remove them entirely; subsampling downweights them during training. With strong subsampling (1e-4), the stopword list matters less -- but stopwords still affect vocabulary size and phrase detection.

4. **Section filtering x corpus size:** Removing operative paragraphs would reduce corpus size by roughly 5--10% (they're short). This is a small reduction but could matter for years with fewer judgments.

5. **Latin phrase list x phrase detector:** The curated Latin list handles multi-word Latin terms *before* gensim Phrases runs. If a Latin phrase also appears frequently enough, Phrases would detect it independently -- but rare Latin phrases (< 30 occurrences) would be missed without the curated list.

---

## Project structure

```
legal_word_drift/
  src/
    pipeline/                  # Numbered pipeline scripts (01-07)
    preprocessing/             # Tokenizer, phrase detector, corpus builder
    embeddings/                # Word2Vec trainer, Procrustes alignment, vocabulary
    metrics/                   # Shift, dispersion, clustering
    stats/                     # Bootstrap confidence intervals
    visualization/             # PCA/UMAP trajectory plots
    cjeu_cellar_client.py      # SPARQL + REST client for EU CELLAR
    xhtml_parser.py            # Judgment XHTML parser
    text_fetcher.py            # Async text downloader
    models.py                  # Data classes (ParagraphRecord)
    utils/
      config.py                # Path constants
      io.py                    # JSONL read/write helpers
  data/
    raw/                       # Downloaded metadata and texts
    processed/                 # Sentence files, phrase models
    models/                    # Word2Vec models, aligned embeddings
    metrics/                   # Computed drift metrics
    figures/                   # Generated plots
```

---

## Key references

- Hamilton, W. L., Leskovec, J., & Jurafsky, D. (2016). Diachronic word embeddings reveal statistical laws of semantic change. *Proceedings of ACL*.
- Mikolov, T., Sutskever, I., Chen, K., Corrado, G., & Dean, J. (2013). Distributed representations of words and phrases and their compositionality. *NeurIPS*.
- Di Carlo, V., Bianchi, F., & Palmonari, M. (2019). Training temporal word embeddings with a compass. *Proceedings of AAAI*.
