# %% [markdown]
# # 06b Holdout summary from the saved single-touch artifacts (no re-run)
# 06 traded the holdout once with the frozen code and saved its daily returns, backtest table and IC series, then
# failed while serialising the summary: the deployed strategy equals the baseline (both the rule), so its row appears
# twice and `.loc` returned a Series. This script computes the same summary from those saved files only; it never
# refits a model or re-runs the simulator, and it refuses to run if the single-touch mark is missing.

# %%
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from lib import (DEV_END, FREEZE, HO_END, HOLDOUT_MARK, RES, boot_sharpe, build_panel, ledger, load_prices, month_ends, nw, paired_boot,
                 pit_universe, set_rf, sharpe_se)

assert os.path.exists(HOLDOUT_MARK), "06 must have touched the holdout first"
fz = json.load(open(FREEZE))
mark = json.load(open(HOLDOUT_MARK))
assert mark["freeze_code_sha256"] == fz["code_sha256"]
res = pd.read_csv(os.path.join(RES, "holdout_backtests.csv"))
for s, g in res.groupby("strategy"):
    assert len(g.drop(columns="strategy").drop_duplicates()) == 1, f"duplicate rows for {s} differ"
res = res.drop_duplicates("strategy").reset_index(drop=True)
res.to_csv(os.path.join(RES, "holdout_backtests.csv"), index=False)
curves = pd.read_parquet(os.path.join(RES, "holdout_daily_returns.parquet"))
ic_series = pd.read_parquet(os.path.join(RES, "ic_series_holdout.parquet"))

P = load_prices(holdout=True)
set_rf(P)
_, state = build_panel(P)
idx = P["close"].index
dec = [d for d in month_ends(idx) if d >= DEV_END and d < HO_END]
U = pit_universe(P, dec)
sel = fz["model_selection"]
ic_out = {}
for n in ic_series:
    ic = ic_series[n].dropna()
    months = ic.groupby([ic.index.year, ic.index.month]).mean()
    ic_out[n] = {**nw(ic), "share_months_positive": float((months > 0).mean())}

DEP = fz["deployed_strategy"]
R = res.set_index("strategy")
dep = curves[DEP].dropna()
yrs = len(dep) / 252
dep_sr = float(R.loc[DEP, "sharpe"])
conf = paired_boot(dep, curves["bench_ew_universe"].dropna())
bil = P["close"]["BIL"].pct_change().loc[dep.index[0]:HO_END].dropna()
bil_ann = (1 + bil).prod() ** (252 / len(bil)) - 1
g = R.loc[DEP + "|gross", "cagr"]
n_iter = int(pd.Series(list(sel["lgb_best_iters"].values())).median())
dm = sel["deployed_model"]
summary = {"decisions": [str(d.date()) for d in dec], "universe_size": {str(d.date()): len(U[d]["kept"]) for d in dec},
           "gated_months": int(sum(state.loc[d, "spy_vol_pct"] >= 0.8 for d in dec)), "lgb_holdout_trees": n_iter, "ic": ic_out,
           "deployed": {"strategy": DEP, "sharpe": dep_sr, "ci_analytic": [dep_sr - 1.96 * sharpe_se(dep_sr, yrs), dep_sr + 1.96 * sharpe_se(dep_sr, yrs)],
                        "ci_bootstrap": boot_sharpe(dep), "breakeven_bps_per_leg": (g - bil_ann) / R.loc[DEP, "turnover_2way_ann"] * 1e4},
           "confirmatory_test": {**conf, "threshold": "CI lower bound > 0", "passed": conf["lo"] > 0},
           "secondary_test": {"ic": ic_out[dm]["mean"], "lo": ic_out[dm]["lo"], "hi": ic_out[dm]["hi"], "passed": bool(ic_out[dm]["lo"] > 0 or ic_out[dm]["hi"] < 0)},
           "deployed_vs_rule": {"diff": 0.0, "note": "deployed strategy is the rule itself"} if DEP == fz["baseline"] else paired_boot(dep, curves[fz["baseline"]].dropna()),
           "deployed_vs_spy": paired_boot(dep, curves["bench_spy"].dropna()), "bil_cagr": bil_ann, "summary_source": "06b from saved single-touch artifacts"}
json.dump(summary, open(os.path.join(RES, "holdout_summary.json"), "w"), indent=1, default=float)
pd.set_option("display.width", 230)
print(res[["strategy", "sharpe", "cagr", "vol", "max_dd", "turnover_2way_ann", "avg_exposure", "gated_share"]].round(3).to_string())
print(json.dumps({k: v for k, v in summary.items() if k not in ("universe_size",)}, indent=1, default=lambda x: round(float(x), 4)))
ledger(f"06 traded the holdout once ({dec[0].date()}..{HO_END.date()}) and saved its artifacts, then failed while writing the summary JSON (deployed == baseline -> duplicate row). "
       f"06b computed the summary from the saved artifacts without re-running anything. Deployed {DEP}: excess Sharpe {dep_sr:.2f} "
       f"[{summary['deployed']['ci_bootstrap'][0]:.2f}, {summary['deployed']['ci_bootstrap'][1]:.2f}]. Confirmatory test (minus EW): {conf['diff']:+.2f} [{conf['lo']:+.2f}, {conf['hi']:+.2f}] -> "
       f"{'PASSED' if conf['lo'] > 0 else 'NOT PASSED'}. Holdout IC {ic_out[dm]['mean']:+.3f} [{ic_out[dm]['lo']:+.3f}, {ic_out[dm]['hi']:+.3f}].")
