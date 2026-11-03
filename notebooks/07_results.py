# %% [markdown]
# # 07 Results: headline table (Ch. 20 gates), outcome pattern, six required figures
# Reads registered results only; recomputes the validation gross curve for the portfolio row (no tuning).

# %%
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from lib import (ANN, DATA, DEV_END, EV, FIG, FREEZE, HO_END, HO_START, RES, boot_sharpe, deflated_sharpe, figure_style, load_prices, min_track_years, perf, set_rf, simulate, xs)

plt = figure_style()
fz = json.load(open(FREEZE))
vs = json.load(open(os.path.join(RES, "validation_summary.json")))
hs = json.load(open(os.path.join(RES, "holdout_summary.json")))
vic = pd.read_csv(os.path.join(RES, "validation_ic.csv")).set_index("model")
vb = pd.read_csv(os.path.join(RES, "validation_backtests.csv")).set_index("strategy")
hb = pd.read_csv(os.path.join(RES, "holdout_backtests.csv")).set_index("strategy")
vr = pd.read_parquet(os.path.join(RES, "validation_daily_returns.parquet"))
hr = pd.read_parquet(os.path.join(RES, "holdout_daily_returns.parquet"))
P = load_prices(holdout=True)
set_rf(P)
sel = fz["model_selection"]
DEP, RULE = fz["deployed_strategy"], fz["baseline"]
RAW = DEP.replace("|vt", "|raw")
m = sel["deployed_model"]

# validation gross curve of the pre-overlay portfolio (portfolio gate)
state = pd.read_parquet(os.path.join(DATA, "state_dev.parquet"))
U = {pd.Timestamp(k): v for k, v in json.load(open(os.path.join(RES, "universe_dev.json"))).items()}
pv = pd.read_parquet(os.path.join(DATA, "predictions_validation.parquet"))
dec = [pd.Timestamp(d) for d in vs["decisions"]]
S = pv.pivot(index="date", columns="symbol", values=f"pred__{m}").reindex(dec)
Pd = load_prices()
v_raw_gross, _ = simulate(S, Pd, state, U, dec, DEV_END, alloc="ew", gate=True, overlay=False, cost_mult=0)

n3, ni, var_d = fz["trial_counts"]["variants"], fz["trial_counts"]["inclusive"], fz["sr_var_daily_trials"]
NF = len(EV["validation_folds"])
RUNG = sel["deployed_rung"]
VLAB = f"Validation ({EV['validation_folds'][0][0][:4]}-{DEV_END.year})"
HLAB = f"Holdout ({HO_START:%Y-%m}..{HO_END:%Y-%m})"
f2, f3 = (lambda x: f"{x:.2f}"), (lambda x: f"{x:+.3f}")
pct = lambda x: f"{x * 100:.1f}%"
ci = lambda c: f"[{c[0]:.2f}, {c[1]:.2f}]"
icv, ich = vic.loc[m], hs["ic"][m]
vfold = [icv[f"fold{i}"] for i in range(NF)]
val_raw_g, ho_raw_g = perf(v_raw_gross), hb.loc[RAW + "|gross"]
dsr_v = deflated_sharpe(vr[DEP], n3, var_d)[0], deflated_sharpe(vr[DEP], ni, var_d)[0]
dsr_h = deflated_sharpe(hr[DEP].dropna(), n3, var_d)[0], deflated_sharpe(hr[DEP].dropna(), ni, var_d)[0]
be_v = (val_raw_g["cagr"] - vs["breakeven"][DEP]["bil_cagr"]) / vb.loc[RAW, "turnover_2way_ann"] * 1e4      # break-even of the pre-overlay portfolio
be_h = (ho_raw_g.cagr - hs["bil_cagr"]) / ho_raw_g.turnover_2way_ann * 1e4
be_v_ov, be_h_ov = vs["breakeven"][DEP]["breakeven_bps_per_leg"], hs["deployed"]["breakeven_bps_per_leg"]
head = pd.DataFrame([
    ["Signal", "Daily IC [NW 95% CI]; ICIR; share of positive folds (val) / months (holdout)",
     f"{f3(icv.ic)} [{f3(icv.lo)}, {f3(icv.hi)}]; {f2(icv.icir)}; {int(icv.folds_positive)}/{NF}",
     f"{f3(ich['mean'])} [{f3(ich['lo'])}, {f3(ich['hi'])}]; -; {pct(ich['share_months_positive'])}", "-"],
    ["Portfolio (top-5 EW + gate, before costs and overlay)", "Gross excess Sharpe [CI]; max DD; 2-way turnover/yr",
     f"{f2(val_raw_g['sharpe'])} {ci(boot_sharpe(v_raw_gross))}; {pct(val_raw_g['max_dd'])}; {vb.loc[RAW, 'turnover_2way_ann']:.1f}x",
     f"{f2(ho_raw_g.sharpe)} {ci(boot_sharpe(hr[RAW + '|gross'].dropna()))}; {pct(ho_raw_g.max_dd)}; {ho_raw_g.turnover_2way_ann:.1f}x",
     f"EW universe {f2(vb.loc['bench_ew_universe', 'sharpe'])} / {f2(hb.loc['bench_ew_universe', 'sharpe'])}; SPY {f2(vb.loc['bench_spy', 'sharpe'])} / {f2(hb.loc['bench_spy', 'sharpe'])}; 60/40 {f2(vb.loc['bench_6040', 'sharpe'])} / {f2(hb.loc['bench_6040', 'sharpe'])}"],
    ["Costs", "Net excess Sharpe at assumed cost; break-even (bp per leg)",
     f"{f2(vb.loc[RAW, 'sharpe'])}; {be_v:.0f} (with overlay {be_v_ov:.0f})", f"{f2(hb.loc[RAW, 'sharpe'])}; {be_h:.0f} (with overlay {be_h_ov:.0f})", "-"],
    ["Risk overlay (10% vol target)", "Net excess Sharpe; max DD",
     f"{f2(vb.loc[DEP, 'sharpe'])}; {pct(vb.loc[DEP, 'max_dd'])}", f"{f2(hb.loc[DEP, 'sharpe'])}; {pct(hb.loc[DEP, 'max_dd'])}", "-"],
    ["Inference", f"Deflated Sharpe (N={n3} / N={ni}); minimum track record (years)",
     f"{f2(dsr_v[0])} / {f2(dsr_v[1])}; {'inf' if vb.loc[DEP, 'sharpe'] <= 0 else f'{min_track_years(vb.loc[DEP, 'sharpe']):.0f}'}",
     f"{f2(dsr_h[0])} / {f2(dsr_h[1])}; {min_track_years(hb.loc[DEP, 'sharpe']):.0f}", "-"],
], columns=["Stage", "Metric", VLAB, HLAB, "Benchmark (val / holdout)"])
head.to_csv(os.path.join(RES, "headline_table.csv"), index=False)
vci, hci = vs["inference"][DEP]["ci_bootstrap"], hs["deployed"]["ci_bootstrap"]
if vci[0] < 0 < vci[1] and hci[0] < 0 < hci[1]:
    pattern = "never resolved: the deployed Sharpe interval spans zero in validation and in the holdout"
elif np.sign(vb.loc[DEP, "sharpe"]) != np.sign(hb.loc[DEP, "sharpe"]):
    pattern = "decay or sign flip"
else:
    pattern = "hold" if hb.loc[DEP, "sharpe"] <= vb.loc[DEP, "sharpe"] else "strengthen"
json.dump({"pattern": pattern, "dsr_validation": dsr_v, "dsr_holdout": dsr_h, "breakeven_raw": [be_v, be_h], "breakeven_overlay": [be_v_ov, be_h_ov]}, open(os.path.join(RES, "results_summary.json"), "w"), indent=1)
pd.set_option("display.width", 250)
print(head.to_string(index=False))
print("pattern:", pattern)

# %% Figure 1: equity curves with the validation/holdout boundary
lab = {DEP: f"Deployed: {RUNG} + gate + vol target", RULE: "Non-ML baseline: momentum rule + gate + vol target", "bench_ew_universe": "Equal-weight universe", "bench_spy": "SPY"}
fig, ax = plt.subplots(figsize=(7, 3))
for k, l in lab.items():
    s = pd.concat([vr[k].dropna(), hr[k].dropna()])
    ax.plot((1 + s).cumprod(), label=l, lw=1.1 if k.startswith("bench") is False else 0.9, ls="-" if not k.startswith("bench") else "--")
ax.axvline(HO_START, c="k", lw=0.8)
ax.text(HO_START + pd.Timedelta(days=10), ax.get_ylim()[1] * 0.97, "holdout", fontsize=7, va="top")
ax.set_title("Fig 1. Q: Does the strategy beat the baseline and the passive benchmark after costs?")
ax.legend(fontsize=6)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "r1_equity.png"))
plt.close(fig)

# Figure 2: rolling 3-month IC
vi = pd.read_parquet(os.path.join(RES, "ic_series_rungs_validation.parquet"))
hi = pd.read_parquet(os.path.join(RES, "ic_series_holdout.parquet"))
fig, ax = plt.subplots(figsize=(7, 2.8))
for col, rung, l in ((m, RUNG, f"Deployed {RUNG}"), ("rule_mom126vs", "rule", "Momentum rule")):
    s = pd.concat([vi[rung], hi[col]])
    ax.plot(s.rolling(63, min_periods=40).mean(), label=l, lw=1)
ax.axhline(0, c="k", lw=0.6)
ax.axvline(HO_START, c="k", lw=0.8, ls="--")
ax.set_title("Fig 2. Q: Is the ranking skill stable? Rolling 63-day mean daily IC")
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "r2_rolling_ic.png"))
plt.close(fig)

# Figure 3: gross-to-net Sharpe bars with break-even
fig, ax = plt.subplots(figsize=(6.4, 2.8))
labels, gross, net = ["Val raw", "Val +overlay", "Holdout raw", "Holdout +overlay"], [], []
sw = pd.read_csv(os.path.join(RES, "cost_sweep.csv"))
gross += [val_raw_g["sharpe"], sw[(sw.strategy == DEP) & (sw.cost_mult == 0)].sharpe.iloc[0], ho_raw_g.sharpe, hb.loc[DEP + "|gross", "sharpe"]]
net += [vb.loc[RAW, "sharpe"], vb.loc[DEP, "sharpe"], hb.loc[RAW, "sharpe"], hb.loc[DEP, "sharpe"]]
x = np.arange(4)
ax.bar(x - 0.18, gross, 0.36, label="gross", color="#88ccee")
ax.bar(x + 0.18, net, 0.36, label="net", color="#4477aa")
ax.axhline(0, c="k", lw=0.6)
ax.set_xticks(x, labels)
ax.text(1, max(gross) * 1.05, f"break-even: val {be_v:.0f} bp/leg, holdout {be_h:.0f} bp/leg (assumed ~{vs['breakeven'][DEP]['assumed_bps_per_leg']:.1f})", ha="center", fontsize=6.5)
ax.set_title("Fig 3. Q: How much do costs take? Gross vs net excess Sharpe, deployed strategy")
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "r3_gross_net.png"))
plt.close(fig)

# Figure 4: regime slice (volatility x trend, point-in-time state at the decision date)
st = pd.read_parquet(os.path.join(DATA, "state_dev.parquet"))
from lib import build_panel
_, st_full = build_panel(P)
dep_all = pd.concat([vr[DEP].dropna(), hr[DEP].dropna()])
s_d = st_full.reindex(dep_all.index, method="ffill").shift(1)
rows = []
for vlab, vm in (("low vol", s_d.spy_vol_pct < 0.5), ("high vol", s_d.spy_vol_pct >= 0.5)):
    for tlab, tm in (("up", s_d.spy_trend_126 > 0), ("down", s_d.spy_trend_126 <= 0)):
        for per, rng in (("validation", dep_all.index <= DEV_END), ("holdout", dep_all.index > DEV_END)):
            msk = (vm & tm).values & rng
            x_ = dep_all[msk]
            if len(x_) > 15:
                xe = xs(x_)
                eq = (1 + x_).cumprod()
                rows.append({"regime": f"{vlab}/{tlab}", "period": per, "days": len(x_), "sharpe": xe.mean() / xe.std() * np.sqrt(ANN) if xe.std() > 0 else np.nan, "max_dd": (eq / eq.cummax() - 1).min()})
rg = pd.DataFrame(rows)
rg.to_csv(os.path.join(RES, "regime_slice.csv"), index=False)
fig, ax = plt.subplots(1, 2, figsize=(7, 2.7))
for j, met in enumerate(("sharpe", "max_dd")):
    pvt = rg.pivot(index="regime", columns="period", values=met)
    pvt.plot.bar(ax=ax[j], rot=0, color=["#cc6677", "#4477aa"])
    ax[j].axhline(0, c="k", lw=0.6)
    ax[j].set_title(met)
    ax[j].tick_params(axis="x", labelsize=6.5)
fig.suptitle("Fig 4. Q: Where does the strategy work? Vol x trend regime, deployed strategy", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "r4_regime.png"))
plt.close(fig)

# Figure 6: drawdown with kill-switch thresholds (Figure 5 is the SHAP beeswarm from 04: figures/m2_shap_beeswarm.png)
eq = (1 + dep_all).cumprod()
dd = eq / eq.cummax() - 1
fig, ax = plt.subplots(figsize=(7, 2.6))
ax.fill_between(dd.index, dd, 0, color="#4477aa", alpha=0.5)
for t, l in ((-0.05, "watch"), (-0.10, "halve"), (-0.15, "halt"), (-0.20, "unwind")):
    ax.axhline(t, c="#cc6677", lw=0.7, ls="--")
    ax.text(dd.index[0], t + 0.004, f"{t:.0%} {l}", fontsize=6.5, color="#cc6677")
ax.axvline(HO_START, c="k", lw=0.8)
ax.set_ylim(-0.22, 0.01)
ax.set_title("Fig 6. Q: Would the kill-switch ladder have triggered? Drawdown of the deployed strategy")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "r6_drawdown.png"))
plt.close(fig)
print(rg.round(3).to_string())
print("max dd val/holdout", round(dd[dd.index <= DEV_END].min(), 3), round(dd[dd.index > DEV_END].min(), 3))
