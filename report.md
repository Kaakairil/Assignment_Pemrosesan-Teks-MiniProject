# Report — Spoiler Detection on GTA 6 Replay Comments (Windah Basudara)

Mini-project for Text Processing (Pemrosesan Teks), semester 5.
Report date: 2026-11-27. Full narrative in Indonesian: `laporan.md`.

## 1. Project summary

Binary classification of spoiler comments on the YouTube archive (replay)
videos of Windah Basudara's "GTA 6 Gameplay" series, episodes #1–#11,
comparing two methods:

- **Method A (baseline):** TF-IDF + Logistic Regression
- **Method B (comparison):** IndoBERT (`indobenchmark/indobert-base-p1`) fine-tuned
- **Labels:** `1` = spoiler, `0` = not spoiler, `-1` = ambiguous (excluded)

Research questions:

1. How large is the difference in spoiler-class F1 and recall between
   IndoBERT and TF-IDF + Logistic Regression?
2. On which kinds of comments do both methods fail most, and are those
   failures model errors or label errors?

**Hypothesis:** IndoBERT yields higher spoiler F1, especially on
mixed-language and sarcastic comments. **Result: not detected** — the
difference is smaller than seed variance (see §6).

## 2. Data — full counts per pipeline stage

### 2.1 Collection (YouTube Data API, `commentThreads`)

| Stage | Count | Notes |
|---|---|---|
| Raw API response pages | 41 JSON files | `data/raw/`, not committed |
| Raw comments collected | **4,007** | top-level threads across 11 videos |
| After dedup + preprocessing | **3,992** | `data/interim/komentar_bersih.csv`, 0 duplicate ids |
| Labeled sheets distributed (A and B) | 3,607 each | `lembar_label_a.csv`, `lembar_label_b.csv` |
| Agreement subset | 100 | random, seed 42 |
| Final valid labels (after dropping ambiguous) | **3,562** | `komentar_berlabel.csv` |
| Ambiguous (-1) dropped | 45 | 1.2% |

Raw comments per video (300 top-level threads per episode by design,
episode #11 capped lower by API pagination):

| Ep | Video ID | Role | Raw comments | Train/Val/Test rows |
|---|---|---|---|---|
| #1 | `45r3OyaxvFE` | train | 300 | train |
| #2 | `QSPPyA1736U` | train | 300 | train |
| #3 | `B2Bon2X1jPQ` | train | 300 | train |
| #4 | `03qhcshfeMY` | train | 300 | train |
| #5 | `lF_QYSm2T-Q` | train | 299 | train |
| #6 | `u_Lb6sh9N0M` | train | 299 | train |
| #7 | `MWQtXTTAgeM` | train | 300 | train |
| #8 | `z6sBvvg_004` | train | 300 | train |
| #9 | `zYTPbx83wCQ` | val | 299 | val |
| #10 | `wJVeWiqMJ-M` | val | 300 | val |
| #11 | `5m-Xi8lgvWY` | test | 995 | test (897 after filtering) |

Collection dates: episodes #1–#10 on **2026-11-25**, episode #11 on
**2026-11-26**. View counts recorded 2026-11-26 in
`hasil/pemilihan_video.md` and change over time.

### 2.2 Labeling

- Guideline v2.1 (`panduan/panduan_label.md`): rules S1–S6, G1, T1, R1.
  Spoiler = revealing story content or story consequences unknown to a
  viewer who has not finished the game (plot, endings, character fates,
  twists, key events, and mechanics that determine any of these — S5/S6).
- Two human labelers. Cohen's kappa on the 100-comment subset: initial
  **0.580** (below 0.6 gate) → 4 disagreements discussed, 3 label revisions
  → re-run **0.740 (PASS)** → full labeling opened.
- Final distribution of 3,562 valid labels:
  - not spoiler (0): **3,470 (97.4%)**
  - spoiler (1): **92 (2.6%)**
  - ambiguous (-1): 45 dropped before splitting
- All labels are human labels. No automatic label was ever counted as a
  human label. Old heuristic-labeled data (2,795 comments, from 10 earlier
  videos) was quarantined to `data/legacy/` and excluded from the pipeline;
  earlier evaluation results (F1 A 0.667; B 0.665 ± 0.049) were declared
  invalid.

### 2.3 Split — by video, not random

Split by episode order to prevent leakage (later-episode comments often
discuss earlier episodes):

| Split | Episodes | Rows | Spoilers | Proportion |
|---|---|---|---|---|
| Train | #1–#8 | **2,126** | 50 | 2.4% |
| Validation | #9–#10 | **539** | 20 | 3.7% |
| Test | #11 only (`5m-Xi8lgvWY`) | **897** | 22 | 2.5% |

The test set is evaluated **exactly once**, after hyperparameters and
models are locked on validation.

## 3. Methodology

### 3.1 Preprocessing

- Manual, conservative alay-normalization dictionary
  (`kamus/normalisasi_alay.csv`); `wkwk*` normalized; letter runs >2 trimmed.
- Two text versions for a fair comparison:
  - `teks_klasik` (Sastrawi stemming, no stopwords) → Method A
  - `teks_bert` (light cleaning, emoji kept) → Method B
- `MAX_LEN_BERT = 128`.

### 3.2 Method A — TF-IDF + Logistic Regression

- `TfidfVectorizer(min_df=2, sublinear_tf=True)` +
  `LogisticRegression(class_weight="balanced", max_iter=1000)`.
- Grid: ngram (1,1)/(1,2) × max_features None/10,000 × C 0.1/1/10 —
  **12 combinations scored on validation** (no random StratifiedKFold —
  same-video comments would leak across folds).
- Selected: **ngram (1,2), max_features None, C = 0.1**
  (validation F1 0.393). Refit on train, evaluated once on test.

### 3.3 Method B — IndoBERT fine-tuning

- Locked hyperparameters: lr 2e-5, 4 epochs, effective batch 8
  (train batch 4 × grad-accum 2 — RTX 3050 4GB VRAM), weight decay 0.01;
  best checkpoint per seed by validation F1 per epoch
  (`load_best_model_at_end`).
- Class weights from train data only, applied to cross-entropy.
- **Three seeds (42, 43, 44)**, full runs; predictions saved per seed.
  **No consensus seed**: no seed-merged prediction, no seed chosen on test.
- Trained locally on RTX 3050 Laptop 4GB (torch 2.11.0+cu128) after
  repeated Google Colab failures (details in README Keputusan).
- Best checkpoints: seed 42 → checkpoint-798 (val F1 0.514), seed 43 →
  checkpoint-1064 (val F1 0.500), seed 44 → checkpoint-798 (val F1 0.541).

### 3.4 Evaluation

- Primary: spoiler-class precision/recall/F1; macro F1 and accuracy secondary.
- Method B reported as **mean ± std across the three seeds**.
- **95% bootstrap CI** (N=1,000 row resamples of test) for spoiler F1;
  for B the CI is over the per-seed mean F1.
- **McNemar test** A vs B on the same test predictions, run per seed.

## 4. Results (test set: 897 comments, 22 spoilers)

### 4.1 Metrics

| Metric | A (TF-IDF + LogReg) | B (IndoBERT, mean ± std, 3 seeds) |
|---|---|---|
| Precision spoiler | 0.266 | 0.605 |
| Recall spoiler | 0.773 | 0.364 ± 0.164 |
| **F1 spoiler** | **0.395** | **0.435 ± 0.120** |
| Macro F1 | 0.682 | 0.712 |
| Accuracy | 0.942 | 0.978 |

Per-seed B: seed 42 F1 0.345 (P 0.714 / R 0.227); seed 43 F1 0.571
(P 0.600 / R 0.545); seed 44 F1 0.389 (P 0.500 / R 0.318).

### 4.2 Bootstrap CI 95% (spoiler F1)

| Method | F1 | CI 95% |
|---|---|---|
| A | 0.395 | [0.256, 0.520] |
| B | 0.435 | [0.260, 0.597] |

The two intervals **overlap**.

### 4.3 McNemar (A vs B, per seed)

| Seed | Statistic | p-value |
|---|---|---|
| 42 | 17.96 | 0.000023 |
| 43 | 21.78 | 0.000003 |
| 44 | 15.57 | 0.000079 |

All p < 0.001: the two methods make **different** errors (not that one is
better).

### 4.4 Confusion matrices

- A: [[828, 47], [5, 17]] — 47 false positives, 5 false negatives.
- B varies strongly by seed: seed 42 finds 5 of 22 spoilers (precision
  0.714); seed 43 finds 12 of 22 (precision 0.600).

Comparison figure: `hasil/perbandingan_metode.png` (F1 + CI, per-seed
points, precision/recall trade-off).

## 5. Error analysis (64 error rows, reviewed one by one by the owner)

`hasil/analisis_kesalahan.csv`: union of A and B (seed 42) errors by `id` —
A: 5 FN, 47 FP; B: 17 FN, 2 FP.

### 5.1 Label errors vs model errors (manual review)

- **5 label errors** (label contradicts the guideline; not counted as model
  failures):
  - 3 rows labeled 1 that should be 0 (world mystery = G1; live-schedule
    joke; pure mechanical tip = G1)
  - 2 rows labeled 0 that should be 1 (ending-determining decisions = S5/S6)
  - Net effect: recorded metrics slightly **understate both methods
    similarly** — the comparison conclusion does not change.
- **59 model errors.**

### 5.2 Error patterns (owner's manual categorization)

| Pattern | Count | Model error | Label error |
|---|---|---|---|
| `teks_pendek` (too short for context) | 19 | 19 | 0 |
| `sindiran` (sarcasm/irony) | 16 | 15 | 1 |
| `menyebut_jalan_cerita_dan_cara` (reveals story path + how to get it, S5/S6) | 16 | 14 | 2 |
| `tips_mekanik` (game-mechanics tips) | 13 | 11 | 2 |

Key semantic trap: most comments containing the word "ending" refer to the
**end of the stream** (e.g. selling the Bugatti, closing remarks), not the
game's ending — so they are NOT spoilers. Only comments that reveal the
story path and how to reach it (e.g. "raise Jason–Lucia relationship for
good ending", "keep robbery under 2jt for good ending") are spoilers.

## 6. Interpretation

1. **The F1 difference between A (0.395) and B (0.435 ± 0.120) is smaller
   than the between-seed std (0.120)**, and the 95% CIs overlap. Per the
   plan's criterion: **no performance difference was detected** on this
   test set. The hypothesis is not confirmed — but not refuted either;
   the test set is too small to discriminate.
2. Results come from **one test episode (#11)** with only 22 spoilers.
   Moving a single comment changes F1 by ~0.02–0.04. Per-seed B metrics
   (0.345–0.571) are noisy; no per-seed conclusions should be drawn.
3. Character trade-off: A is high-recall/low-precision (47 FPs); B leans
   the other way and varies widely by seed. Significant McNemar shows the
   methods err on different comments — ensemble potential exists but is
   out of scope.

## 7. Limitations

1. **One channel, one series, one test episode** — no support for
   generalization to other channels or games.
2. **Few test spoilers (22/897, 2.5%)** — wide F1 CIs (~26 pp wide for A)
   and high sensitivity to individual comments.
3. **Extreme class imbalance (2.6%)** — accuracy 0.94–0.98 reflects the
   majority class, not spoiler-detection quality.
4. **Large between-seed variance for B (std 0.120)** — comparable to or
   larger than the between-method gap; three seeds are a minimum, not a
   stable estimate.
5. **Label subjectivity** — the spoiler/non-spoiler boundary on replay
   comments (gameplay events, S5/S6) is a guideline decision, not ground
   truth; kappa 0.740 is strong but not perfect.
6. Episodes #10–#11 were scraped hours after the live stream — comment
   distribution still evolving.
7. Number of GTA 6 endings: unverifiable community claim
   (`TODO_VERIFIKASI`); never used as a labeling basis.

## 8. Ethics

1. **Anonymization** — no usernames, author IDs, or profile links in any
   shared data (`data/labeled/`, `hasil/`); identity replaced by
   `id_anonim` = first 16 hex chars of SHA-256(HASH_SALT + channel id);
   salt only in `.env` (never committed).
2. **Purpose** — comments used only for this course research; never
   republished with identity.
3. **Subjectivity** — spoiler is subjective; defined explicitly in
   guideline v2.1 and measured (kappa 0.740).
4. **Terms of service** — collection only via the official YouTube Data
   API; no Instagram scraping (prohibited by project policy).
5. **Open sourcing** — code and anonymized data shared for academic
   reproduction; `data/raw/`, `.env`, `models/` never committed.

## 9. Conclusion

1. Both methods were built end-to-end under strict governance: video-based
   split, validation-only hyperparameter selection, single test pass,
   three seeds without consensus, bootstrap CI, per-seed McNemar.
2. **No F1 difference was detected between Method A (0.395) and Method B
   (0.435 ± 0.120)** — the gap is smaller than seed variance and the CIs
   overlap. With 22 spoilers in one test episode, this experiment lacks
   the power to separate the methods.
3. McNemar is significant (all p < 0.001): A misses little (5 FN) but
   over-flags a lot (47 FPs, mostly "ending" = end of stream); B is more
   selective and seed-volatile.
4. Highest-impact next steps: more test episodes (more spoilers), more
   seeds, and explicit handling of the word "ending" (end-of-stream vs
   game-ending context).

## Appendix A — Repository outputs

| File | Content |
|---|---|
| `hasil/pemilihan_video.md` | 11-video table, criteria, collection dates |
| `hasil/kesepakatan.txt` | kappa 0.740, cross-table on 100 subset |
| `hasil/audit_label_timestamp.txt` | timestamp-based label audit |
| `hasil/params_metode_a.json` | Method A selected params + full grid log |
| `hasil/prediksi_metode_a.csv` | A test predictions (897 rows) |
| `hasil/prediksi_metode_b_seed{42,43,44}.csv` | B test predictions per seed |
| `hasil/metrik.csv` | both methods' metrics (B = mean ± std) |
| `hasil/metrik_metode_b_seeds.csv` | B per-seed metrics |
| `hasil/bootstrap_ci.csv` | 95% bootstrap CI, spoiler F1 |
| `hasil/mcnemar.txt` | McNemar per seed |
| `hasil/confusion_matrix_*.png` | confusion matrices (A; B per seed) |
| `hasil/perbandingan_metode.png` | A vs B comparison figure |
| `hasil/analisis_kesalahan.csv` | 64 error rows + pola + jenis_galat |

## Appendix B — Reproduction

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt          # or requirements.lock.txt for exact versions
# .env: YOUTUBE_API_KEY, HASH_SALT, TRAIN/VAL/TEST_VIDEO_IDS (see README)

python -m src.ambil_komentar --video-id $VIDEO_ID --maks 1000
python -m src.preprocessing
python -m src.buat_lembar_label
# manual labeling of lembar_label_a/b, then:
python -m src.kesepakatan
python -m src.split
python -m src.metode_a                   # Method A (CPU, minutes)
python -m src.metode_b                   # Method B (GPU required; 3 seeds)
python -m src.evaluasi
python -m src.analisis_kesalahan
pytest -q                                # 24 tests
```
