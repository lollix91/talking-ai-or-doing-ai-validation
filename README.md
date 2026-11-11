# AI Disclosure and Firm Value — Replication Package

Code, human/LLM annotations and aggregated results for a study of AI disclosure
in 10-K filings of S&P 500 firms (FY2015–2024) and its association with firm
valuation and performance.

## What is in this repository

| Path | Content |
|---|---|
| `src/`, `run.py`, `config.py` | Data pipeline (EODHD + SEC EDGAR), NLP scoring, panel/DiD/CAR analyses |
| `annotation_app.py`, `second_annotation.py`, `review_disagreements.py` | Annotation tools (blind mode, second annotation, adjudication of disagreements) |
| `data/processed/annotations*.json` | Validation labels: `annotations.json` (human, 148 filings), `annotations_human2.json` (second human, 80), `annotations_llm.json` (LLM, declared, 80), `annotations_adjudicated.json` (35 adjudicated disagreements) |
| `data/processed/annotation_sample.csv` | The 148 filings of the stratified validation sample, with dictionary scores and sampling stratum |
| `data/processed/annotation_subsample_llm.csv` | The 80-filing subsample labeled by the second human annotator and the LLM annotator |
| `data/processed/annotations_joined.csv` | One row per validation filing: stratum weight, dictionary scores and all labels (human, human 2, LLM, adjudicated, final) used by `src/analysis/validation.py` |
| `data/processed/edgar_10k_index.parquet`, `ai_adoption_scores.parquet`, `ai_scores_v1.parquet` | Filing index and AI scores computed from public SEC EDGAR 10-Ks |
| `outputs/tables`, `outputs/figures` | Aggregated results (summary statistics, regressions, validation metrics) |

## What is NOT in this repository

Raw data and firm-level financial/market data from EODHD (fundamentals, prices,
and panels built from them) are **not redistributed** (data-provider licence).
Only aggregated results are published.
SEC EDGAR filings are not included either; they are downloaded by the pipeline.

## Data provenance

Firm-level financial and market data were obtained from EODHD. Data
processing was completed on 20 September 2026.
10-K filings come from SEC EDGAR. The daily Fama–French five factors used
for abnormal returns come from the Kenneth R. French Data Library and are
downloaded automatically by `src/analysis/car.py` (to `data/raw/ff5_daily.csv`).

## Reproducing the results

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env     # add EODHD_API_KEY and SEC_USER_AGENT
```

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env   # add EODHD_API_KEY and SEC_USER_AGENT
```

Then, on either system:

```bash
# download + features + NLP (needs a valid EODHD key)
python run.py --sec --sample --fundamentals --prices --edgar --features --nlp
# tables and figures in outputs/
python run.py --analysis
```

`python run.py --all` runs the same steps and also downloads the 8-K
earnings-call exhibits (`--transcripts`). These are an exploratory artifact,
are not used in the paper and make the download much heavier, so the explicit
steps above are enough to replicate the paper.

## Paper tables and figures

| Paper | File in `outputs/` |
|---|---|
| Sample attrition | `tables/audit_attrition.csv` |
| Validation of the dictionary score | `tables/validation_metrics.csv`, `tables/validation_interannotator.csv` |
| Summary statistics | `tables/table1_p2.csv` |
| High versus low AI disclosure | `tables/p2_group_comparison.csv` |
| Panel fixed-effects estimates | `tables/p2_panel_fe.csv` (specs `A_baseline` to `F_fixed_dictionary`) |
| Operational and risk-factor disclosure, with Wald test | `tables/p2_panel_fe.csv` (spec `G_ops_vs_risk`, columns `beta`, `beta_2`, `wald_p_equal`) |
| Robustness and heterogeneity | `tables/p2_robustness.csv` |
| Abnormal returns at 10-K filing | `tables/p2_car_regressions.csv` |
| Difference-in-differences around ChatGPT | `tables/p2_event_study_did.csv` |
| Event-study figures (Tobin's Q, log market cap) | `figures/p2_event_study_tobin_q.png`, `figures/p2_event_study_log_market_cap.png`, coefficients in `tables/p2_event_study_*.csv` |

The event-study coefficients use FY2022 as the reference year (its `beta` is
0). The paper also reports them relative to FY2021: for each outcome, take the
`raw` rows of `tables/p2_event_study_<outcome>.csv` and subtract the FY2021
`beta` from the FY2023 and FY2024 `beta` (for example, Tobin's Q:
0.376 − 0.871 = −0.49 and 0.711 − 0.871 = −0.16). These are point estimates
only; no standard errors are computed for the rebased coefficients.

Annotation tools: `python annotation_app.py --serve [--blind] [--subsample]
--annotator NAME`, `python review_disagreements.py --serve`.
