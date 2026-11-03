# %% [markdown]
# # 03 Labels, features, triage (development sample only)
# Label anchored to the next-open fill. Every feature is a row with a mechanism; five gates decide PROCEED / REVISE / STOP.

# %%
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from lib import CFG, DATA, DEV_END, EV, FIG, RES, SIGNALS, STATES, bh, build_panel, daily_ic, figure_style, half_spread, ledger, load_prices, log_run, nw

plt = figure_style()
P = load_prices()
panel, state = build_panel(P)
U = {pd.Timestamp(k): v for k, v in json.load(open(os.path.join(RES, "universe_dev.json"))).items()}
panel.to_parquet(os.path.join(DATA, "panel_dev.parquet"))
state.to_parquet(os.path.join(DATA, "state_dev.parquet"))
dev = panel[(panel.label_end <= DEV_END) & (panel.date >= EV["training_start"])].copy()          # labels fully inside development
print("panel", panel.date.min().date(), panel.date.max().date(), len(panel), "| labelled dev rows", len(dev))

# %% feature table (Table 8.1 style) — mechanism, role, lookback, representation, failure mode
FT = pd.DataFrame([
    ["mom_126_vs", "Signal", "Persistence / under-reaction (flows)", "126d, monthly", "126d return / 63d vol, cross-sectional rank", "Momentum crashes at volatility spikes and trend reversals"],
    ["mom_126_skip21", "Signal", "Persistence without 1-month reversal", "126d skip 21d", "Return, rank", "Same as momentum; crowding"],
    ["mom_63", "Signal", "Medium-term persistence", "63d", "Return, rank", "Choppy markets"],
    ["ret_21", "Signal", "Short-term reversal (liquidity provision)", "21d", "Return, rank", "Trending markets"],
    ["ret_5", "Signal", "Weekly reversal", "5d", "Return, rank", "Noise at a monthly hold"],
    ["dist_63d_high", "Signal", "Anchoring to recent high", "63d", "Distance to max, rank", "Breakouts in crises"],
    ["sma50_ratio", "Signal", "Trend-following flows", "50d", "Price / SMA, rank", "Whipsaw"],
    ["vol_63", "Signal", "Risk compensation vs low-vol anomaly", "63d", "Annualised vol, rank", "Sign depends on regime"],
    ["idio_vol_63", "Signal", "Idiosyncratic risk premium", "63d", "Residual vol vs SPY, rank", "Unstable beta estimate"],
    ["beta_spy_63", "Signal", "Equity-risk compensation / betting-against-beta", "63d", "Beta to SPY, rank", "Flips sign in sell-offs"],
    ["max_dd_63", "Signal", "Loss aversion / distress", "126d (63 within 63)", "Worst drawdown, rank", "Rebounds after crashes"],
    ["volume_chg_21_63", "Signal", "Attention and flow pressure", "63d", "Log volume ratio, rank", "Event-driven volume spikes"],
    ["range_21", "Signal", "Risk compensation (trading range)", "21d", "Mean (H-L)/C, rank", "Single shock dominates"],
    ["rsi_14", "Signal", "Overbought/oversold reversal", "14d (EWM)", "Oscillator, rank", "Strong trends stay overbought"],
    ["dollar_vol_126", "Signal", "Liquidity premium", "126d", "Log median $ volume, rank", "Size proxy, slow"],
    ["spy_vol_pct", "State", "Risk environment: momentum works in calm states", "21d vol ranked over 126d", "Percentile vs own history", "Single shock dominates"],
    ["spy_trend_126", "State", "Market trend regime", "126d", "Return", "Late at turning points"],
    ["xs_dispersion_21", "State", "Opportunity set for rotation", "21d", "Cross-sectional std", "Driven by one outlier ETF"],
    ["spy_tlt_corr_63", "State", "Hedge regime (stock-bond correlation)", "63d", "Level", "Non-synchronous trading"],
], columns=["feature", "role", "driver_hypothesis", "lookback", "representation", "known_failure_mode"])
FT.to_csv(os.path.join(RES, "feature_table.csv"), index=False)
assert set(FT.feature) == set(SIGNALS + STATES)

# %% de-duplicate candidate signals by correlation clustering (|Spearman| >= 0.80), pre-declared priority = table order
me = dev[dev.date.isin(U)]
fc = me[SIGNALS].rank(pct=True).corr(method="spearman")
keep_sig, merged = [], {}
for f in SIGNALS:
    hit = [k for k in keep_sig if abs(fc.loc[f, k]) >= 0.80]
    (merged.__setitem__(f, hit[0]) if hit else keep_sig.append(f))
print("merged by correlation clustering:", merged)

# %% the five gates
blocks = [(f"{y}H{h}", pd.Timestamp(f"{y}-{1 if h == 1 else 7:02d}-01"), pd.Timestamp(f"{y}-{6 if h == 1 else 12}-{30 if h == 1 else 31}"))
          for y in range(2020, DEV_END.year + 1) for h in (1, 2)]
med_price = P["raw_close"].iloc[-252:].median()
rt_cost_bps = float(np.median([2 * (CFG["costs"]["commission_per_share"] + half_spread(s)) / med_price[s] for s in med_price.index]) * 1e4)
rows = []
for f in keep_sig:
    ic = daily_ic(dev, f)
    s = nw(ic)
    sign = np.sign(s["mean"])
    bic = [ic[(ic.index >= a) & (ic.index <= b)].mean() for _, a, b in blocks]
    icir = np.nanmean(bic) / np.nanstd(bic)
    share = np.mean([np.sign(x) == sign for x in bic])
    decay = {k: nw(daily_ic(panel[panel.label_end.notna() & (panel.date <= DEV_END - pd.Timedelta(days=100))], f, f"fwd_oo_{k}"))["mean"] for k in (5, 21, 63)}
    lagged = dev[["date", "symbol"]].assign(x=dev.groupby("symbol")[f].shift(21), y=dev["fwd_oo_21"]).dropna()
    plac = nw(lagged.groupby("date").apply(lambda g: g.x.corr(g.y, method="spearman"), include_groups=False))["mean"]
    m = me.dropna(subset=["fwd_oo_21"]).copy()
    m["q"] = m.groupby("date")[f].transform(lambda x: pd.qcut(x.rank(method="first"), 5, labels=False))
    qret = m.groupby("q").fwd_oo_21.mean()
    mono = spearmanr(qret.index, qret.values)[0]
    top = 4 if sign >= 0 else 0
    spread_m = float((qret[top] - qret[4 - top]) * 1e4)
    mem = m[m.q == top].groupby("date").symbol.apply(set)
    turn = np.mean([1 - len(a & b) / max(len(b), 1) for a, b in zip(mem.iloc[:-1], mem.iloc[1:])])
    rows.append({"feature": f, "ic": s["mean"], "ic_lo": s["lo"], "ic_hi": s["hi"], "t_nw": s["t"], "p_nw": s["p"], "icir_blocks": icir,
                 "share_blocks_same_sign": share, "ic_h5": decay[5], "ic_h21": decay[21], "ic_h63": decay[63], "placebo_ic_lag21": plac,
                 "quintile_monotonicity": mono, "q5_q1_spread_bps_month": spread_m, "top_quintile_turnover": turn, "round_trip_cost_bps": rt_cost_bps})
tri = pd.DataFrame(rows)
tri["p_bh"] = bh(tri.p_nw)
tri["g1_correctness"] = np.where(tri.placebo_ic_lag21.abs() >= 0.75 * tri.ic.abs(), "slow-state proxy", "pass")
tri["g2_association"] = np.where((tri.ic_lo > 0) | (tri.ic_hi < 0), "pass", np.where(tri.share_blocks_same_sign >= 0.6, "weak", "fail"))
tri["g3_shape"] = np.where(tri.quintile_monotonicity.abs() >= 0.9, "monotone", "non-monotone")
tri["g4_feasibility"] = np.where(tri.q5_q1_spread_bps_month.abs() >= 2 * tri.round_trip_cost_bps, "pass", "fail")
tri["g5_bh_5pct"] = np.where(tri.p_bh < 0.05, "pass", "fail")
tri["decision"] = np.select(
    [(tri.g2_association == "pass") & (tri.g5_bh_5pct == "pass") & (tri.g3_shape == "monotone") & (tri.g4_feasibility == "pass"),
     (tri.g2_association == "fail") & (tri.t_nw.abs() < 1)],
    ["PROCEED", "STOP"], "REVISE")
for f, k in merged.items():
    tri = pd.concat([tri, pd.DataFrame([{"feature": f, "decision": "STOP", "g1_correctness": f"merged into {k} (|rho|>=0.8)"}])])
tri.to_csv(os.path.join(RES, "triage.csv"), index=False)
model_features = [f for f in tri[tri.decision != "STOP"].feature] + STATES
json.dump({"model_features": model_features, "merged": merged, "round_trip_cost_bps": rt_cost_bps, "searched_set": len(SIGNALS)}, open(os.path.join(RES, "features_selected.json"), "w"), indent=1)
pd.set_option("display.width", 250)
print(tri.round(3).to_string())

# %% mechanism checks for the top 3 features by |t|: timing placebo, shared driver (beta), regime heterogeneity
top3 = tri.dropna(subset=["t_nw"]).reindex(tri.dropna(subset=["t_nw"]).t_nw.abs().sort_values(ascending=False).index).feature.head(3).tolist()
chk = []
for f in top3:
    d = dev.copy()
    ctrl = "vol_63" if f == "beta_spy_63" else "beta_spy_63"     # shared driver: market beta (or volatility for beta itself)
    def resid(g):
        X = np.c_[np.ones(len(g)), g[ctrl].rank()]
        yv = g[f].rank().values
        b = np.linalg.lstsq(X, yv, rcond=None)[0]
        yl = g["fwd_oo_21"].rank().values
        bl = np.linalg.lstsq(X, yl, rcond=None)[0]
        return np.corrcoef(yv - X @ b, yl - X @ bl)[0, 1]
    partial = nw(d.groupby("date").apply(resid, include_groups=False))
    lo = d[d.spy_vol_pct <= 1 / 3]
    hi = d[d.spy_vol_pct >= 2 / 3]
    r = tri.set_index("feature").loc[f]
    chk.append({"feature": f, "ic": r.ic, "placebo_lag21_ic": r.placebo_ic_lag21, "shared_driver_control": ctrl, "ic_after_control": partial["mean"],
                "ic_low_vol_tercile": nw(daily_ic(lo, f))["mean"], "ic_high_vol_tercile": nw(daily_ic(hi, f))["mean"], "event_time": "N/A (not event-based)"})
chk = pd.DataFrame(chk)
chk.to_csv(os.path.join(RES, "mechanism_checks.csv"), index=False)
print(chk.round(3).to_string())
mom = dev.copy()
mech = {"mom_126_vs_ic_low_vol": nw(daily_ic(mom[mom.spy_vol_pct < 0.8], "mom_126_vs"))["mean"],
        "mom_126_vs_ic_high_vol": nw(daily_ic(mom[mom.spy_vol_pct >= 0.8], "mom_126_vs"))["mean"]}
json.dump({**mech, "top3": top3}, open(os.path.join(RES, "mechanism_gate.json"), "w"), indent=1)
print(mech)

# %% DAG figure for the top features
fig, ax = plt.subplots(figsize=(6.4, 2.6))
ax.axis("off")
nodes = {"Feature (top-3)": (0.12, 0.5), "fwd 21d return": (0.88, 0.5), "Market beta /\nvol regime": (0.5, 0.88), "Flows / rebalancing": (0.5, 0.5), "Fund AUM growth\n(collider)": (0.5, 0.1)}
for n, (x, y) in nodes.items():
    ax.text(x, y, n, ha="center", va="center", fontsize=7.5, bbox=dict(boxstyle="round", fc="#eef2f7", ec="#555"))
arr = lambda a, b, c="k": ax.annotate("", xy=nodes[b], xytext=nodes[a], arrowprops=dict(arrowstyle="->", color=c, shrinkA=22, shrinkB=22))
arr("Market beta /\nvol regime", "Feature (top-3)", "#cc6677"); arr("Market beta /\nvol regime", "fwd 21d return", "#cc6677")
arr("Feature (top-3)", "Flows / rebalancing"); arr("Flows / rebalancing", "fwd 21d return")
arr("Feature (top-3)", "Fund AUM growth\n(collider)", "#999"); arr("fwd 21d return", "Fund AUM growth\n(collider)", "#999")
ax.set_title("DAG: confounder (red, control) - mediator (do not control) - collider (grey, never control)", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "d3_dag.png"))
plt.close(fig)

# %% figure: triage IC with CI
t = tri.dropna(subset=["ic"]).sort_values("ic")
fig, ax = plt.subplots(figsize=(6.4, 3.2))
col = t.decision.map({"PROCEED": "#228833", "REVISE": "#ccbb44", "STOP": "#cc6677"})
ax.errorbar(t.ic, range(len(t)), xerr=[t.ic - t.ic_lo, t.ic_hi - t.ic], fmt="none", ecolor="#bbb")
ax.scatter(t.ic, range(len(t)), c=col, zorder=3)
ax.set_yticks(range(len(t)), t.feature, fontsize=7)
ax.axvline(0, c="k", lw=0.6)
ax.set_title("Q: Which features carry signal? Daily IC, NW 95% CI (green PROCEED, yellow REVISE, red STOP)")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "d4_triage.png"))
plt.close(fig)

n_eff = int(dev.date.nunique() / 21)
ledger(f"03 labels/features: label fwd_oo_21 (next-open to open+21). Nominal labelled rows {len(dev):,} over {dev.date.nunique()} dates; effective independent dates ~{n_eff}. "
       f"Searched set {len(SIGNALS)} signals + {len(STATES)} states; merged by clustering: {merged}. Triage: {tri.decision.value_counts().to_dict()}. "
       f"Model features (non-STOP signals + states): {model_features}. Mechanism gate: momentum IC low-vol {mech['mom_126_vs_ic_low_vol']:+.3f}, high-vol {mech['mom_126_vs_ic_high_vol']:+.3f}.")
for _, r in tri.iterrows():
    log_run("feature_triage", r.feature, "03_labels_features", f"{dev.date.min().date()}..{dev.date.max().date()} (labels end <= {DEV_END.date()})", {"label": "fwd_oo_21"},
            ic=r.get("ic") if pd.notna(r.get("ic")) else None, decision=r.decision, gate="five gates")
