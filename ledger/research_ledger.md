# Research ledger

Taxonomy: Strategy -> Trial family -> Trial -> Run (ML4T Ch. 6). Every adaptive choice is a trial; bug fixes are not.

## Prior work counted as trials (disclosed)
- v1 (textbook pipeline, 100 ETFs, 2006-2025): 734 registered backtests; its holdout 2024-2025 was run once.
- v2 (69 de-duplicated ETFs, 2020-2024): 52 strategy variants; its holdout 2024 was run once.
- v3 (54 de-duplicated ETFs, 2020-2025): 60 strategy variants; its holdout 2025 was run once.
All three are counted in the inclusive Deflated Sharpe trial count. Because 2024 and 2025 were already seen, they are development data here and the sealed holdout is 2026-01..2026-10, which no one on the team had looked at when this setup was committed.
The 23-ETF list was chosen by the team after those versions (hindsight risk in the universe choice, disclosed; it is not re-optimised here).

### 2026-11-03T18:51 (commit 7202343)
**Setup v1 committed before any backtest.** Mechanism: Industry and sector returns persist over one to six months because information diffuses slowly across industries and investors rebalance sector allocations gradually (industry momentum, flow predictability). The persistence crashes when market volatility spikes and leadership reverses abruptly, so the rotation signal should be traded only in calm-to-normal volatility states.
Source of edge: flow_predictability, risk_compensation. Holdout 2026-01-01..2026-10-30 sealed by the loader (src/lib.py: load_prices).
Pre-registered confirmatory test: holdout net Sharpe of deployed strategy minus equal-weight benchmark Sharpe (paired, 21-day block bootstrap); pass if 95% CI lower bound > 0. Secondary: holdout daily IC Newey-West 95% CI excludes zero.
Selection rule: highest mean validation IC; deployed strategy = selected rung + equal weight + gate + volatility target.

### 2026-11-03T18:51 (commit 7202343)
02 data: 27 symbols, 1760 sessions (dev). Duplicates 0, OHLC violations {'high_below_low': 0, 'close_outside_high_low': 0, 'open_outside_high_low': 0, 'non_positive_price': 0, 'zero_volume': 0}. PIT universe: 23-23 tradable names per decision. Survivorship check FAIL disclosed.

### 2026-11-03T18:51 (commit 7202343)
03 labels/features: label fwd_oo_21 (next-open to open+21). Nominal labelled rows 34,178 over 1486 dates; effective independent dates ~70. Searched set 15 signals + 4 states; merged by clustering: {'mom_126_skip21': 'mom_126_vs', 'sma50_ratio': 'ret_21', 'max_dd_63': 'vol_63', 'range_21': 'vol_63', 'rsi_14': 'ret_21'}. Triage: {'STOP': 13, 'REVISE': 2}. Model features (non-STOP signals + states): ['vol_63', 'beta_spy_63', 'spy_vol_pct', 'spy_trend_126', 'xs_dispersion_21', 'spy_tlt_corr_63']. Mechanism gate: momentum IC low-vol -0.013, high-vol +0.089.

### 2026-11-03T18:51 (commit 7202343)
04 models: ladder rule -> ridge (8 alphas, 1e-2..1e5) -> LightGBM (4 configs, L1 loss, early stopping on a purged inner split) -> ensemble. Chosen by validation IC: {'rule': 'rule_mom126vs', 'ridge': 'ridge_a10000', 'lgb': 'lgb_l7_m1000', 'ensemble': 'ensemble'}. Deployed rung (highest validation IC): rule. Paired deltas: ridge (ridge_a10000) - rule (rule_mom126vs): -0.0211 [-0.1484,+0.1061]; lgb (lgb_l7_m1000) - ridge (ridge_a10000): -0.0279 [-0.1231,+0.0674]; ensemble (ensemble) - lgb (lgb_l7_m1000): +0.0258 [-0.0312,+0.0829]

### 2026-11-03T18:52 (commit 7202343)
04 models: ladder rule -> ridge (8 alphas, 1e-2..1e5) -> LightGBM (4 configs, L1 loss, early stopping on a purged inner split) -> ensemble. Chosen by validation IC: {'rule': 'rule_mom126vs', 'ridge': 'ridge_a10000', 'lgb': 'lgb_l7_m1000', 'ensemble': 'ensemble'}. Deployed rung (highest validation IC): rule. Paired deltas: ridge (ridge_a10000) - rule (rule_mom126vs): -0.0211 [-0.1484,+0.1061]; lgb (lgb_l7_m1000) - ridge (ridge_a10000): -0.0279 [-0.1231,+0.0674]; ensemble (ensemble) - lgb (lgb_l7_m1000): +0.0258 [-0.0312,+0.0829]

### 2026-11-03T18:52 (commit 7202343)
Bug fix (not a trial): 04 conformal residuals used the raw deployed score; when the deployed rung is the rule its scale differs from the percentile target. Residuals now use the per-date percentile of the score. 04 re-run; model choices unchanged.

### 2026-11-03T18:52 (commit 7202343)
05 backtest (validation 2021-12-31..2025-12-31): 60 strategy variants. Deployed rule_mom126vs|ew|gate|vt: Sharpe 0.36, MDD -7.9%; rule rule_mom126vs|ew|gate|vt: 0.36; EW universe 0.44. Break-even 36 bp/leg. DSR(N=60) 0.63; DSR(N=906) 0.58.
**FROZEN** at 2026-11-03T18:52:39 (code sha256 362e5a0daf9f). Holdout not yet touched.

### 2026-11-03T18:53 (commit 9577d37)
06 traded the holdout once (2025-12-31..2026-10-30) and saved its artifacts, then failed while writing the summary JSON (deployed == baseline -> duplicate row). 06b computed the summary from the saved artifacts without re-running anything. Deployed rule_mom126vs|ew|gate|vt: excess Sharpe 0.07 [-2.01, 1.45]. Confirmatory test (minus EW): -1.82 [-3.60, +0.50] -> NOT PASSED. Holdout IC -0.037 [-0.174, +0.099].
