# %% [markdown]
# # 05 Backtest, baseline, costs, portfolio, risk — validation window 2022-2025; then FREEZE

# %%
import glob
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from lib import (ANN, CASH, CFG, DATA, DEV_END, EV, FIG, FREEZE, RES, ROOT, boot_sharpe, deflated_sharpe, figure_style, git_commit, ledger, load_prices,
                 log_run, min_track_years, paired_boot, perf, set_rf, sharpe_se, simulate, xs)

plt = figure_style()
assert not os.path.exists(FREEZE), "already frozen"
P = load_prices()
set_rf(P)
state = pd.read_parquet(os.path.join(DATA, "state_dev.parquet"))
U = {pd.Timestamp(k): v for k, v in json.load(open(os.path.join(RES, "universe_dev.json"))).items()}
pv = pd.read_parquet(os.path.join(DATA, "predictions_validation.parquet"))
sel = json.load(open(os.path.join(RES, "model_selection.json")))
idx = P["close"].index
v0, v1 = pd.Timestamp(EV["validation_folds"][0][0]), pd.Timestamp(EV["validation_folds"][-1][1])
dec = [d for d in U if (idx > d).any() and v0 <= idx[idx > d][0] <= v1]
END = DEV_END
scores = {c.replace("pred__", ""): pv.pivot(index="date", columns="symbol", values=c).reindex(dec) for c in pv if c.startswith("pred__")}

# %% every strategy variant (counts as trials for the Deflated Sharpe)
runs, curves, infos = [], {}, {}
for name, S in scores.items():
    for gate in ([True, False] if name == "rule_mom126vs" else [True]):
        for alloc in ("ew", "ivol"):
            for ov in (False, True):
                key = f"{name}|{alloc}|{'gate' if gate else 'nogate'}|{'vt' if ov else 'raw'}"
                r, info = simulate(S, P, state, U, dec, END, alloc=alloc, gate=gate, overlay=ov)
                curves[key], infos[key] = r, info
                pf = perf(r)
                runs.append({"strategy": key, "model": name, "alloc": alloc, "gate": gate, "overlay": ov, **pf, **info})
                log_run("strategy", key, "05_backtest", f"{dec[0].date()}..{END.date()}", {"model": name, "alloc": alloc, "gate": gate, "overlay": ov}, sharpe=pf["sharpe"], decision="evaluated", gate="validation")
ones = pd.DataFrame(1.0, index=dec, columns=P["close"].columns)
bench = {"bench_ew_universe": dict(gate=False, top_k="all"), "bench_spy": dict(fixed={"SPY": 1.0}), "bench_6040": dict(fixed={"SPY": 0.6, "AGG": 0.4})}
for key, kw in bench.items():
    r, info = simulate(ones, P, state, U, dec, END, **kw)
    curves[key], infos[key] = r, info
    runs.append({"strategy": key, "model": "benchmark", **perf(r), **info})
bt = pd.DataFrame(runs)
bt.to_csv(os.path.join(RES, "validation_backtests.csv"), index=False)
pd.DataFrame(curves).to_parquet(os.path.join(RES, "validation_daily_returns.parquet"))
DEP = f"{sel['deployed_model']}|ew|gate|vt"
RULE = "rule_mom126vs|ew|gate|vt"
B = bt.set_index("strategy")

# %% gross vs net, cost sweep and break-even (deployed and rule)
bil = (P["close"][CASH].pct_change().loc[curves[DEP].index[0]:END]).dropna()
bil_ann = (1 + bil).prod() ** (ANN / len(bil)) - 1
sweep = []
for key in (DEP, RULE):
    m, a, g, o = key.split("|")
    for mult in CFG["costs"]["sweep_multiples"]:
        r, info = simulate(scores[m], P, state, U, dec, END, alloc=a, gate=(g == "gate"), overlay=(o == "vt"), cost_mult=mult)
        sweep.append({"strategy": key, "cost_mult": mult, "sharpe": perf(r)["sharpe"], "cagr": perf(r)["cagr"], "cost_per_leg_bps": mult * info["avg_cost_per_leg_bps"]})
sweep = pd.DataFrame(sweep)
sweep.to_csv(os.path.join(RES, "cost_sweep.csv"), index=False)
be = {}
for key in (DEP, RULE):
    g = sweep[(sweep.strategy == key) & (sweep.cost_mult == 0)].cagr.iloc[0]
    t2 = infos[key]["turnover_2way_ann"]
    be[key] = {"gross_cagr": g, "bil_cagr": bil_ann, "turnover_2way_ann": t2, "breakeven_bps_per_leg": (g - bil_ann) / t2 * 1e4 if t2 > 0 else np.nan,
               "assumed_bps_per_leg": infos[key]["avg_cost_per_leg_bps"], "avg_holding_days": infos[key]["avg_holding_days"]}
    x = be[key]["breakeven_bps_per_leg"] / be[key]["assumed_bps_per_leg"]
    be[key]["cost_regime"] = "hard floor (<2x)" if x < 2 else ("gray zone (2-5x)" if x < 5 else "comfortable (>5x)")

# %% allocator comparison (same signal, same 63-day window)
alloc_tab = []
for a in ("ew", "ivol"):
    k = f"{sel['deployed_model']}|{a}|gate|raw"
    alloc_tab.append({"allocator": a, "sharpe": B.loc[k, "sharpe"], "max_dd": B.loc[k, "max_dd"], "turnover_2way_ann": B.loc[k, "turnover_2way_ann"], "hhi": B.loc[k, "hhi"], "eff_n": B.loc[k, "eff_n"]})
alloc_tab = pd.DataFrame(alloc_tab)
alloc_spread = float(alloc_tab.sharpe.max() - alloc_tab.sharpe.min())

# %% risk: overlay with/without, gate with/without, tail risk by regime, beta, stress with 3x spreads, kill-switch ladder
ov_tab = []
for k in [f"{sel['deployed_model']}|ew|gate|raw", DEP, "rule_mom126vs|ew|nogate|raw", "rule_mom126vs|ew|gate|raw", RULE]:
    ov_tab.append({"strategy": k, **{c: B.loc[k, c] for c in ("sharpe", "cagr", "vol", "max_dd", "dd_duration_sessions", "recovery_sessions", "var95", "cvar95", "avg_exposure", "gated_share")}})
ov_tab = pd.DataFrame(ov_tab)
dep = curves[DEP]
dstate = state["spy_vol_pct"].reindex(dep.index, method="ffill").shift(1)
reg_tail = {lab: {"var95": float(np.percentile(dep[m], 5)), "cvar95": float(dep[m][dep[m] <= np.percentile(dep[m], 5)].mean()), "days": int(m.sum())}
            for lab, m in (("vol_state_high(>=0.5)", dstate >= 0.5), ("vol_state_low(<0.5)", dstate < 0.5))}
spy = P["close"]["SPY"].pct_change().reindex(dep.index)
X = np.c_[np.ones(len(dep)), spy.values]
b = np.linalg.lstsq(X, dep.values, rcond=None)[0]
e = dep.values - X @ b
Xc = spy.values - spy.mean()
u = Xc * e
S0 = u @ u
for L in range(1, 22):
    S0 += 2 * (1 - L / 22) * (u[L:] @ u[:-L])
beta_se = np.sqrt(S0) / (Xc @ Xc)
roll_beta = dep.rolling(126).cov(spy) / spy.rolling(126).var()
stress = {}
for nm, a0, a1 in (("2022 rate shock", "2022-01-03", "2022-10-14"), ("Aug-2024 carry unwind", "2024-07-16", "2024-08-07"), ("Apr-2025 tariff shock", "2025-04-02", "2025-04-21")):
    stress[nm] = {}
    for key in (DEP, RULE, "bench_ew_universe", "bench_spy"):
        if key.startswith("bench"):
            rr = curves[key]
        else:
            m, a, g, o = key.split("|")
            rr, _ = simulate(scores[m], P, state, U, dec, END, alloc=a, gate=(g == "gate"), overlay=(o == "vt"), spread_mult=CFG["costs"]["stress_spread_multiple"])
        s = rr.loc[a0:a1]
        eq = (1 + s).cumprod()
        stress[nm][key] = {"return": float(eq.iloc[-1] - 1), "max_dd": float((eq / eq.cummax() - 1).min())}
eqd = (1 + dep).cumprod()
ddd = eqd / eqd.cummax() - 1
ladder = [(-0.05, "watch", "PM"), (-0.10, "halve exposure", "PM"), (-0.15, "halt new trades, move to BIL", "risk owner"), (-0.20, "unwind and review", "team")]
ladder_hits = {f"{t:.0%} {act}": (str(ddd[ddd <= t].index[0].date()) if (ddd <= t).any() else "not triggered") for t, act, _ in ladder}

# %% regime slice: volatility x trend, thresholds point-in-time (state at the decision date)
st_d = state.reindex(dep.index, method="ffill").shift(1)
reg = []
for key in (DEP, RULE, "bench_ew_universe"):
    rr = curves[key]
    for vlab, vm in (("low vol", st_d.spy_vol_pct < 0.5), ("high vol", st_d.spy_vol_pct >= 0.5)):
        for tlab, tm in (("up trend", st_d.spy_trend_126 > 0), ("down trend", st_d.spy_trend_126 <= 0)):
            msk = (vm & tm).reindex(rr.index).fillna(False)
            x, xe = rr[msk], xs(rr)[msk]
            if len(x) > 20:
                eq = (1 + x).cumprod()
                reg.append({"strategy": key, "regime": f"{vlab} / {tlab}", "days": len(x), "sharpe": xe.mean() / xe.std() * np.sqrt(ANN), "max_dd": float((eq / eq.cummax() - 1).min())})
reg = pd.DataFrame(reg)
reg.to_csv(os.path.join(RES, "regime_slice_validation.csv"), index=False)

# %% inference: Sharpe CI, Deflated Sharpe with the ledger's trial count, minimum track record
trials = bt[bt.model != "benchmark"]
n_var = len(trials)
n_incl = n_var + sum(CFG["prior_trials"].values())
var_d = (trials.sharpe / np.sqrt(ANN)).var()
yrs = len(dep) / ANN
inf = {}
for key in (DEP, RULE, "bench_ew_universe"):
    sr = B.loc[key, "sharpe"]
    dsr_n, sr0_n = deflated_sharpe(curves[key], n_var, var_d)
    di, sr0_i = deflated_sharpe(curves[key], n_incl, var_d)
    inf[key] = {"sharpe": sr, "ci_analytic": [sr - 1.96 * sharpe_se(sr, yrs), sr + 1.96 * sharpe_se(sr, yrs)], "ci_bootstrap": boot_sharpe(curves[key]),
                "dsr": dsr_n, "expected_max_sharpe": sr0_n, "dsr_inclusive": di, "expected_max_sharpe_inclusive": sr0_i, "min_track_record_years": min_track_years(sr)}
inf["paired_deployed_vs_ew"] = paired_boot(curves[DEP], curves["bench_ew_universe"])
inf["paired_deployed_vs_rule"] = paired_boot(curves[DEP], curves[RULE])
inf["trials"] = {"variants": n_var, "inclusive": n_incl}

# %% figures for Section 6 (validation): gross-to-net bar; cost sweep
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
for j, key in enumerate((DEP, RULE)):
    s0 = sweep[(sweep.strategy == key)]
    ax[0].bar(j - 0.2, s0[s0.cost_mult == 0].sharpe.iloc[0], 0.4, color="#88ccee", label="gross" if j == 0 else None)
    ax[0].bar(j + 0.2, s0[s0.cost_mult == 1].sharpe.iloc[0], 0.4, color="#4477aa", label="net (assumed cost)" if j == 0 else None)
    ax[0].text(j, max(s0.sharpe.max(), 0) + 0.05, f"break-even {be[key]['breakeven_bps_per_leg']:.0f} bp/leg", ha="center", fontsize=6.5)
    ax[1].plot(s0.cost_per_leg_bps, s0.sharpe, "o-", label=key.split("|")[0])
ax[0].set_xticks([0, 1], [f"Deployed ({sel['deployed_rung']})", "Rule"])
ax[0].axhline(0, c="k", lw=0.6)
ax[0].legend(fontsize=6)
ax[0].set_title("Gross vs net Sharpe (validation)")
ax[1].axhline(0, c="k", lw=0.6)
ax[1].set_xlabel("cost per leg (bps)")
ax[1].set_title("Cost sweep 0x-8x")
ax[1].legend(fontsize=6)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "b1_cost.png"))
plt.close(fig)

summary = {"deployed": DEP, "rule": RULE, "breakeven": be, "allocators": alloc_tab.to_dict("records"), "allocator_spread_sharpe": alloc_spread,
           "overlay_and_gate": ov_tab.to_dict("records"), "regime_tail": reg_tail, "beta_full": {"beta": float(b[1]), "hac_se": float(beta_se), "lo": float(b[1] - 1.96 * beta_se), "hi": float(b[1] + 1.96 * beta_se)},
           "stress_3x_spreads": stress, "kill_switch_first_trigger": ladder_hits, "inference": inf, "decisions": [str(d.date()) for d in dec]}
json.dump(summary, open(os.path.join(RES, "validation_summary.json"), "w"), indent=1, default=float)
roll_beta.to_frame("beta").to_parquet(os.path.join(RES, "rolling_beta_validation.parquet"))
pd.set_option("display.width", 230)
cols = ["strategy", "sharpe", "cagr", "vol", "max_dd", "turnover_2way_ann", "avg_exposure", "gated_share"]
print(bt[bt.strategy.isin([DEP, RULE, f"{sel['deployed_model']}|ew|gate|raw", "rule_mom126vs|ew|nogate|raw", "lgb_l7_m1000|ew|gate|vt", "ensemble|ew|gate|vt", *bench])][cols].round(3).to_string())
print(json.dumps({k: v for k, v in summary.items() if k not in ("overlay_and_gate", "decisions")}, indent=1, default=lambda x: round(float(x), 4)))

# %% FREEZE: specification, code hashes, trial counts; the holdout becomes readable exactly once after this
code = "".join(open(f).read() for f in sorted(glob.glob(os.path.join(ROOT, "notebooks", "0[1-5]_*.py")) + [os.path.join(ROOT, "src", "lib.py"), os.path.join(ROOT, "setup.yaml")]))
freeze = {"frozen_at": pd.Timestamp.now().isoformat(timespec="seconds"), "git_commit_before_freeze": git_commit(), "code_sha256": hashlib.sha256(code.encode()).hexdigest(),
          "deployed_strategy": DEP, "baseline": RULE, "comparison_strategies": [f"{sel['chosen']['lgb']}|ew|gate|vt", "ensemble|ew|gate|vt", f"{sel['deployed_model']}|ew|gate|raw", "rule_mom126vs|ew|nogate|vt"],
          "model_selection": sel, "benchmarks": list(bench), "trial_counts": inf["trials"], "sr_var_daily_trials": float(var_d),
          "confirmatory_test": CFG["confirmatory_test"], "holdout": EV["holdout"]}
json.dump(freeze, open(FREEZE, "w"), indent=1, default=str)
ledger(f"05 backtest (validation {dec[0].date()}..{END.date()}): {n_var} strategy variants. Deployed {DEP}: Sharpe {B.loc[DEP,'sharpe']:.2f}, MDD {B.loc[DEP,'max_dd']:.1%}; "
       f"rule {RULE}: {B.loc[RULE,'sharpe']:.2f}; EW universe {B.loc['bench_ew_universe','sharpe']:.2f}. Break-even {be[DEP]['breakeven_bps_per_leg']:.0f} bp/leg. "
       f"DSR(N={n_var}) {inf[DEP]['dsr']:.2f}; DSR(N={n_incl}) {inf[DEP]['dsr_inclusive']:.2f}.\n**FROZEN** at {freeze['frozen_at']} (code sha256 {freeze['code_sha256'][:12]}). Holdout not yet touched.")
log_run("freeze", "freeze_v1", "05_backtest", "n/a", {"deployed": DEP, "code_sha256": freeze["code_sha256"][:12]}, decision="frozen", gate="pre-holdout")
