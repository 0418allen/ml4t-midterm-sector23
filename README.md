# ML4T Midterm Proposal — Sector-ETF momentum with a volatility-state gate

One complete lap of the ML4T workflow on a fixed list of 23 ETFs (all in the textbook ETF universe), 2019–2026.
Development 2020-01..2025-12 (walk-forward, eight half-year folds 2022–2025); **holdout 2026-01-02..2026-10-30, fresh data never seen before, sealed and touched once**.

Universe: IYR, XLB, XLC, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY, QQQ, IBB, ITA, ITB, KRE, OIH, SOXX, XBI, XME, XRT, USO, GLD (BIL = cash; SPY, TLT, AGG for states and benchmarks).

| Deliverable | Path |
|---|---|
| Proposal PDF (Sections 1–8 + appendix) | `report/ML4T_Midterm_Proposal.pdf` |
| Presentation (10 slides, Chinese speaker notes) | `slides/ML4T_Midterm_Presentation.pptx` |
| Trading setup v1 | `setup.yaml` |
| Research ledger / run log | `ledger/research_ledger.md`, `run_log.csv` |

## Reproduce from the repository root
Python 3.12+ (results produced on 3.14.7). Exact versions: `requirements.lock.txt`.

```bash
uv venv && uv pip install -r requirements.lock.txt      # or: python -m venv .venv && pip install -r requirements.lock.txt
.venv/bin/python data/build_dataset.py                    # downloads data/prices_*.parquet from Yahoo (yfinance)
.venv/bin/python notebooks/07_results.py                  # headline table (results/headline_table.csv) + figures
```

Checked on a fresh clone with a fresh environment (2026-11-07):
- 07 on the committed data files reproduces `results/headline_table.csv` exactly.
- After re-downloading, two holdout cells move slightly: the holdout gross-Sharpe CI ([-1.94, 1.50] -> [-1.89, 1.51]) and the holdout Deflated Sharpe (0.46/0.43 -> 0.47/0.45).
  Cause: after our download Yahoo revised the dividend adjustment of the three monthly-distributing bond ETFs (BIL, TLT, AGG). BIL is the risk-free series for excess returns,
  so statistics on holdout excess returns shift; the strategy returns themselves do not change. To get the exact numbers, use the committed parquet files (`git checkout data/`).

Stage by stage: `01_setup` (protocol, ledger) → `02_data` (PIT universe, quality report) → `03_labels_features` (labels, feature table, five-gate triage)
→ `04_models` (ladder, paired comparison, SHAP, conformal) → `05_backtest` (costs, allocators, risk, inference, **freeze**) → `06_holdout` (single touch)
→ `06b_holdout_summary` (summary from 06's saved output) → `07_results`.
01-06 refuse to run again once the freeze / holdout marks exist (by design). `02_data` also reads the textbook's ETF files (`~/Downloads/machine-learning-for-trading`) for two cross-checks.
PDF and slides: `pip install reportlab python-pptx pillow`, then `python report/build_report.py` and `python slides/build_slides.py`.

Notebooks are percent-format `.py` files paired with unexecuted `.ipynb` copies (jupytext).

## Safeguards built into the code
- `src/lib.py: load_prices()` truncates every development stage at 2025-12-31; the holdout is readable only after `results/freeze.json` exists.
- `06_holdout.py` verifies the SHA-256 of all pre-freeze code against the freeze and writes `results/holdout_touched.json` first, so a second run is refused.
- Constraints (long-only, leverage ≤ 1, max weight) are asserted inside the simulator; no fill on a missing bar.

## What happened at the holdout step
`06_holdout.py` traded the holdout once and saved its daily returns, backtest table and IC series, then failed while writing the summary JSON:
the deployed strategy turned out to be the baseline rule itself, so its row appeared twice. `06b_holdout_summary.py` computes the summary from those saved
files only (no refit, no re-simulation). Both facts are in the ledger.

## Why numbers could differ
Seeds are fixed (LightGBM 42, bootstrap 7). Yahoo revises adjusted prices (see above), so a later download can differ slightly; download time is in `data/download_meta.json` and the committed parquet files are the ones used for every reported number.

## Disclosures
Three earlier versions of this project (v1: 734 backtests; v2: 52 variants; v3: 60 variants) are counted as trials (inclusive N = 906 for the Deflated Sharpe).
They had seen 2024 and 2025, which is why those years are development data here and the holdout is 2026. The 23-ETF list is a hand-picked survivor list chosen after those versions.
