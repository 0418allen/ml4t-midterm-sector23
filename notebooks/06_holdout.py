# %% [markdown]
# # 06 Holdout (2026-01..2026-10): touched exactly once, after the freeze
# Refits the frozen configurations on every development row and trades the sealed year. Refuses to run twice.

# %%
import glob
import hashlib
import json
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from lib import (CFG, DATA, DEV_END, EV, FREEZE, HO_END, HOLDOUT_MARK, RES, ROOT, STATES, boot_sharpe, build_panel, daily_ic, ledger, load_prices, log_run,
                 month_ends, nw, paired_boot, perf, pit_universe, set_rf, sharpe_se, simulate, xs)

assert os.path.exists(FREEZE), "freeze first"
assert not os.path.exists(HOLDOUT_MARK), "the holdout has already been touched once; refusing to run again"
fz = json.load(open(FREEZE))
code = "".join(open(f).read() for f in sorted(glob.glob(os.path.join(ROOT, "notebooks", "0[1-5]_*.py")) + [os.path.join(ROOT, "src", "lib.py"), os.path.join(ROOT, "setup.yaml")]))
assert hashlib.sha256(code.encode()).hexdigest() == fz["code_sha256"], "pipeline code changed after the freeze"
json.dump({"touched_at": pd.Timestamp.now().isoformat(timespec="seconds"), "freeze_code_sha256": fz["code_sha256"]}, open(HOLDOUT_MARK, "w"), indent=1)

P = load_prices(holdout=True)
set_rf(P)
panel, state = build_panel(P)
idx = P["close"].index
dec = [d for d in month_ends(idx) if d >= DEV_END and d < HO_END]           # 2025-12-31 .. 2026-09-30
U = pit_universe(P, dec)
sel = fz["model_selection"]
SIG = [f for f in sel["features"] if f not in STATES]
panel = panel[panel.date >= EV["training_start"]].reset_index(drop=True)
panel["y"] = panel.groupby("date").fwd_oo_21.rank(pct=True) - 0.5
tr = ((panel.label_end <= DEV_END) & panel.y.notna()).values
sc = (panel.date >= DEV_END).values


def design(train_mask, rows_mask):
    X = panel.loc[rows_mask, SIG].groupby(panel.loc[rows_mask, "date"]).rank(pct=True) - 0.5
    t = panel.loc[train_mask, STATES]
    lo, hi = t.quantile(0.01), t.quantile(0.99)
    tc = t.clip(lo, hi, axis=1)
    X[STATES] = (panel.loc[rows_mask, STATES].clip(lo, hi, axis=1) - tc.mean()) / tc.std()
    return X[SIG + STATES].values


Xtr, Xsc, ytr = design(tr, tr), design(tr, sc), panel.y.values[tr]
pred = {"rule_mom126vs": pd.Series(panel["mom_126_vs"].values, index=panel.index)}
pred[sel["chosen"]["ridge"]] = pd.Series(np.nan, index=panel.index)
pred[sel["chosen"]["ridge"]][sc] = Ridge(alpha=sel["ridge_alpha"]).fit(Xtr, ytr).predict(Xsc)
n_iter = int(np.median(list(sel["lgb_best_iters"].values())))              # median early-stopping iteration across the validation folds
cfg = {**sel["lgb_cfg"], "n_estimators": n_iter}
pred[sel["chosen"]["lgb"]] = pd.Series(np.nan, index=panel.index)
pred[sel["chosen"]["lgb"]][sc] = lgb.LGBMRegressor(**cfg).fit(Xtr, ytr).predict(Xsc)
z = lambda s: s.groupby(panel.date).transform(lambda x: (x - x.mean()) / x.std())
pred["ensemble"] = (z(pred["rule_mom126vs"]) + z(pred[sel["chosen"]["ridge"]]) + z(pred[sel["chosen"]["lgb"]])) / 3

# %% holdout IC (rows whose 21-session label ends inside the holdout)
ho = (panel.date > DEV_END) & (panel.label_end <= HO_END)
ic_out, ic_series = {}, {}
for n, p in pred.items():
    ic = daily_ic(panel.loc[ho].assign(p=p[ho]), "p")
    ic_series[n] = ic
    s = nw(ic)
    months = ic.groupby([ic.index.year, ic.index.month]).mean()
    ic_out[n] = {**s, "share_months_positive": float((months > 0).mean())}
pd.DataFrame(ic_series).to_parquet(os.path.join(RES, "ic_series_holdout.parquet"))

# %% holdout backtests: deployed, baseline, comparisons, benchmarks — all under the frozen protocol
keys = [fz["deployed_strategy"], fz["baseline"], *fz["comparison_strategies"]]
scores = {n: panel.assign(p=p).pivot(index="date", columns="symbol", values="p").reindex(dec) for n, p in pred.items()}
curves, rows, infos = {}, [], {}
for key in keys:
    m, a, g, o = key.split("|")
    for mult in (0, 1):
        r, info = simulate(scores[m], P, state, U, dec, HO_END, alloc=a, gate=(g == "gate"), overlay=(o == "vt"), cost_mult=mult)
        tag = key if mult == 1 else key + "|gross"
        curves[tag], infos[tag] = r, info
        rows.append({"strategy": tag, **perf(r), **info})
ones = pd.DataFrame(1.0, index=dec, columns=P["close"].columns)
for key, kw in {"bench_ew_universe": dict(gate=False, top_k="all"), "bench_spy": dict(fixed={"SPY": 1.0}), "bench_6040": dict(fixed={"SPY": 0.6, "AGG": 0.4})}.items():
    r, info = simulate(ones, P, state, U, dec, HO_END, **kw)
    curves[key], infos[key] = r, info
    rows.append({"strategy": key, **perf(r), **info})
res = pd.DataFrame(rows)
res.to_csv(os.path.join(RES, "holdout_backtests.csv"), index=False)
pd.DataFrame(curves).to_parquet(os.path.join(RES, "holdout_daily_returns.parquet"))
for _, r in res.iterrows():
    log_run("holdout", r.strategy, "06_holdout", f"{dec[0].date()}..{HO_END.date()}", {"frozen": fz["code_sha256"][:12]}, sharpe=r.sharpe, decision="reported", gate="holdout (single touch)")

DEP = fz["deployed_strategy"]
yrs = len(curves[DEP]) / 252
dep_sr = res.set_index("strategy").loc[DEP, "sharpe"]
conf = paired_boot(curves[DEP], curves["bench_ew_universe"])
bil = P["close"]["BIL"].pct_change().loc[curves[DEP].index[0]:HO_END].dropna()
bil_ann = (1 + bil).prod() ** (252 / len(bil)) - 1
g = res.set_index("strategy").loc[DEP + "|gross", "cagr"]
summary = {"decisions": [str(d.date()) for d in dec], "universe_size": {str(d.date()): len(U[d]["kept"]) for d in dec},
           "gated_months": int(sum(state.loc[d, "spy_vol_pct"] >= 0.8 for d in dec)), "lgb_holdout_trees": n_iter, "ic": ic_out,
           "deployed": {"strategy": DEP, "sharpe": dep_sr, "ci_analytic": [dep_sr - 1.96 * sharpe_se(dep_sr, yrs), dep_sr + 1.96 * sharpe_se(dep_sr, yrs)],
                        "ci_bootstrap": boot_sharpe(curves[DEP]), "breakeven_bps_per_leg": (g - bil_ann) / infos[DEP]["turnover_2way_ann"] * 1e4},
           "confirmatory_test": {**conf, "threshold": "CI lower bound > 0", "passed": conf["lo"] > 0},
           "secondary_test": {"ic": ic_out[sel["deployed_model"]]["mean"], "lo": ic_out[sel["deployed_model"]]["lo"], "hi": ic_out[sel["deployed_model"]]["hi"],
                              "passed": ic_out[sel["deployed_model"]]["lo"] > 0 or ic_out[sel["deployed_model"]]["hi"] < 0},
           "deployed_vs_rule": paired_boot(curves[DEP], curves[fz["baseline"]]), "deployed_vs_spy": paired_boot(curves[DEP], curves["bench_spy"]), "bil_cagr": bil_ann}
json.dump(summary, open(os.path.join(RES, "holdout_summary.json"), "w"), indent=1, default=float)
pd.set_option("display.width", 230)
print(res[["strategy", "sharpe", "cagr", "vol", "max_dd", "turnover_2way_ann", "avg_exposure", "gated_share"]].round(3).to_string())
print(json.dumps({k: v for k, v in summary.items() if k not in ("universe_size",)}, indent=1, default=lambda x: round(float(x), 4)))
ledger(f"06 holdout touched once ({dec[0].date()}..{HO_END.date()}). Deployed {DEP}: excess Sharpe {dep_sr:.2f} "
       f"[{summary['deployed']['ci_bootstrap'][0]:.2f}, {summary['deployed']['ci_bootstrap'][1]:.2f}]. Confirmatory test (minus EW): {conf['diff']:+.2f} [{conf['lo']:+.2f}, {conf['hi']:+.2f}] -> "
       f"{'PASSED' if conf['lo'] > 0 else 'NOT PASSED'}. Holdout IC {ic_out[sel['deployed_model']]['mean']:+.3f} [{ic_out[sel['deployed_model']]['lo']:+.3f}, {ic_out[sel['deployed_model']]['hi']:+.3f}].")
