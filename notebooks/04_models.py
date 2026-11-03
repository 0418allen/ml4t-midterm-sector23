# %% [markdown]
# # 04 Models and validation: baseline ladder on the six walk-forward folds (development only)

# %%
import json
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from lib import CFG, DATA, DEV_END, EV, FIG, RES, STATES, daily_ic, figure_style, ledger, load_prices, log_run, month_ends, nw

plt = figure_style()
assert not os.path.exists(os.path.join(RES, "freeze.json")), "models are tuned before the freeze only"
P = load_prices()
panel = pd.read_parquet(os.path.join(DATA, "panel_dev.parquet"))
FEATS = json.load(open(os.path.join(RES, "features_selected.json")))["model_features"]
SIG = [f for f in FEATS if f not in STATES]
panel = panel[panel.date >= EV["training_start"]].reset_index(drop=True)
panel["y"] = panel.groupby("date").fwd_oo_21.rank(pct=True) - 0.5
idx = P["close"].index
folds = [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in EV["validation_folds"]]
NF = len(folds)
fold_lab = [f"{str(a.year)[2:]}H{1 if a.month < 7 else 2}" for a, _ in folds]
decisions = month_ends(idx)
exec_date = {d: idx[idx > d][0] for d in decisions if (idx > d).any()}


def design(train_mask, rows_mask):
    """Signals -> per-date percentile rank (no cross-date fit). States -> winsorise 1/99 and standardise with training-fold statistics only."""
    X = panel.loc[rows_mask, SIG].groupby(panel.loc[rows_mask, "date"]).rank(pct=True) - 0.5
    tr = panel.loc[train_mask, STATES]
    lo, hi = tr.quantile(0.01), tr.quantile(0.99)
    st = panel.loc[rows_mask, STATES].clip(lo, hi, axis=1)
    trc = tr.clip(lo, hi, axis=1)
    X[STATES] = (st - trc.mean()) / trc.std()
    return X[SIG + STATES].values


RIDGE = {f"ridge_a{a:g}": a for a in CFG["models"]["ladder"]["ridge"]["alpha"]}
L = CFG["models"]["ladder"]["lightgbm"]
LGB = {f"lgb_l{l}_m{m}": dict(num_leaves=l, min_child_samples=m) for l in L["num_leaves"] for m in L["min_child_samples"]}
BASE = dict(objective=L["objective"], learning_rate=L["learning_rate"], n_estimators=L["max_trees"], max_depth=L["max_depth"], subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, random_state=CFG["seeds"]["lightgbm"], verbose=-1, n_jobs=4)
pred = {n: pd.Series(np.nan, index=panel.index) for n in ["rule_mom126vs", *RIDGE, *LGB]}
pred["rule_mom126vs"][:] = panel["mom_126_vs"].values
fold_info, shap_by_fold, best_iters = [], {}, {}
for k, (vs, ve) in enumerate(folds):
    tr = ((panel.label_end < vs) & panel.y.notna()).values                          # purge: training labels end before validation
    ex = [d for d, e in exec_date.items() if vs <= e <= ve]
    sc = (((panel.date >= vs) & (panel.date <= ve)) | panel.date.isin(ex)).values
    Xtr, Xsc, ytr = design(tr, tr), design(tr, sc), panel.y.values[tr]
    for n, a in RIDGE.items():
        pred[n][sc] = Ridge(alpha=a).fit(Xtr, ytr).predict(Xsc)
    tr_dates = np.sort(panel.loc[tr, "date"].unique())
    iv_start = pd.Timestamp(tr_dates[-126])
    itr = tr & (panel.label_end < iv_start).values                                   # inner purge for early stopping
    iva = tr & (panel.date >= iv_start).values
    for n, cfg in LGB.items():
        m = lgb.LGBMRegressor(**BASE, **cfg)
        m.fit(design(itr, itr), panel.y.values[itr], eval_set=[(design(itr, iva), panel.y.values[iva])], callbacks=[lgb.early_stopping(L["early_stopping"], verbose=False)])
        best_iters[(n, k)] = int(m.best_iteration_ or L["max_trees"])
        m2 = lgb.LGBMRegressor(**{**BASE, "n_estimators": best_iters[(n, k)]}, **cfg).fit(Xtr, ytr)
        pred[n][sc] = m2.predict(Xsc)
        if True:  # keep TreeSHAP for every LightGBM config (secondary table)
            va = (((panel.date >= vs) & (panel.date <= ve)) & (panel.label_end <= DEV_END)).values
            sv = shap.TreeExplainer(m2).shap_values(design(tr, va))
            shap_by_fold.setdefault(n, {})[k] = (sv, design(tr, va))
    fold_info.append({"fold": k, "validation": f"{vs.date()}..{ve.date()}", "train_first": str(panel.loc[tr, "date"].min().date()),
                      "train_last_label_end": str(panel.loc[tr, "label_end"].max().date()), "train_rows": int(tr.sum()), "scored_rows": int(sc.sum())})
pd.DataFrame(fold_info).to_csv(os.path.join(RES, "folds.csv"), index=False)

# %% validation IC per configuration (yardstick)
ev = (panel.label_end <= DEV_END) & (panel.date >= folds[0][0]) & (panel.date <= folds[-1][1])
ic_series, rows = {}, []
for n, p in pred.items():
    ic = daily_ic(panel.loc[ev].assign(p=p[ev]), "p")
    ic_series[n] = ic
    s = nw(ic)
    pf = [ic[(ic.index >= a) & (ic.index <= b)] for a, b in folds]
    rows.append({"model": n, "family": n.split("_")[0], "ic": s["mean"], "lo": s["lo"], "hi": s["hi"], "t": s["t"], "icir": float(np.mean([x.mean() for x in pf]) / np.std([x.mean() for x in pf])),
                 "folds_positive": int(sum(x.mean() > 0 for x in pf)), **{f"fold{i}": x.mean() for i, x in enumerate(pf)}})
    for i, x in enumerate(pf):
        log_run("model:" + n.split("_")[0], n, f"fold{i}", f"{folds[i][0].date()}..{folds[i][1].date()}", {"model": n}, ic=x.mean(), decision="evaluated", gate="validation IC")
ics = pd.DataFrame(rows)
chosen = {fam: ics[ics.family == fam].sort_values("ic").model.iloc[-1] for fam in ("rule", "ridge", "lgb")}
z = lambda s: s.groupby(panel.date).transform(lambda x: (x - x.mean()) / x.std())
pred["ensemble"] = (z(pred[chosen["rule"]]) + z(pred[chosen["ridge"]]) + z(pred[chosen["lgb"]])) / 3
ic = daily_ic(panel.loc[ev].assign(p=pred["ensemble"][ev]), "p")
ic_series["ensemble"] = ic
s = nw(ic)
pf = [ic[(ic.index >= a) & (ic.index <= b)] for a, b in folds]
ics = pd.concat([ics, pd.DataFrame([{"model": "ensemble", "family": "ensemble", "ic": s["mean"], "lo": s["lo"], "hi": s["hi"], "t": s["t"],
                                     "icir": float(np.mean([x.mean() for x in pf]) / np.std([x.mean() for x in pf])), "folds_positive": int(sum(x.mean() > 0 for x in pf)),
                                     **{f"fold{i}": x.mean() for i, x in enumerate(pf)}}])], ignore_index=True)
chosen["ensemble"] = "ensemble"
ics.to_csv(os.path.join(RES, "validation_ic.csv"), index=False)
rung_order = ["rule", "ridge", "lgb", "ensemble"]
rung_ic = {r: ics.set_index("model").loc[chosen[r], "ic"] for r in rung_order}
deployed = max(rung_ic, key=rung_ic.get)

# paired comparison: each rung vs the previous one (daily IC difference, NW CI)
pair = []
for a, b in zip(rung_order[:-1], rung_order[1:]):
    d = (ic_series[chosen[b]] - ic_series[chosen[a]]).dropna()
    s = nw(d)
    pair.append({"comparison": f"{b} ({chosen[b]}) - {a} ({chosen[a]})", "delta_ic": s["mean"], "lo": s["lo"], "hi": s["hi"], "t": s["t"]})
pair = pd.DataFrame(pair)
pair.to_csv(os.path.join(RES, "paired_comparison.csv"), index=False)
for n in pred:
    panel[f"pred__{n}"] = pred[n]
panel[["date", "symbol", "fwd_oo_21", "label_end"] + [c for c in panel if c.startswith("pred__")]].to_parquet(os.path.join(DATA, "predictions_validation.parquet"))
pd.Series(ic_series[chosen[deployed]]).to_frame("ic").to_parquet(os.path.join(RES, "ic_series_deployed_validation.parquet"))
pd.DataFrame({r: ic_series[chosen[r]] for r in rung_order}).to_parquet(os.path.join(RES, "ic_series_rungs_validation.parquet"))

# %% per-fold IC chart with HAC band (Figure 11.8 style)
fig, ax = plt.subplots(figsize=(6.6, 2.8))
w = 0.2
for j, r in enumerate(rung_order):
    pf = [ic_series[chosen[r]][(ic_series[chosen[r]].index >= a) & (ic_series[chosen[r]].index <= b)] for a, b in folds]
    st = [nw(x, lag=21) for x in pf]
    ax.bar(np.arange(NF) + (j - 1.5) * w, [s["mean"] for s in st], w, yerr=[1.96 * s["se"] for s in st], capsize=2, label=f"{r}: {chosen[r]}")
ax.axhline(0, c="k", lw=0.6)
ax.set_xticks(range(NF), fold_lab)
ax.set_title("Q: Is the signal stable across folds? Per-fold daily IC with Newey-West 95% band")
ax.legend(fontsize=6, ncol=2)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "m1_fold_ic.png"))
plt.close(fig)

# %% SHAP for the deployed rung (linear SHAP if Ridge, TreeSHAP if LightGBM); stability and sign consistency across folds
if deployed == "ridge":
    for k, (vs, ve) in enumerate(folds):
        tr = ((panel.label_end < vs) & panel.y.notna()).values
        va = (((panel.date >= vs) & (panel.date <= ve)) & (panel.label_end <= DEV_END)).values
        Xtr, Xva = design(tr, tr), design(tr, va)
        m = Ridge(alpha=RIDGE[chosen["ridge"]]).fit(Xtr, panel.y.values[tr])
        shap_by_fold.setdefault("deployed", {})[k] = (m.coef_ * (Xva - Xtr.mean(0)), Xva)
    lg = "deployed"
    shap_label = chosen["ridge"]
else:
    lg = chosen["lgb"]
    shap_label = lg
sv_last, X_last = shap_by_fold[lg][NF - 1]
plt.figure()
shap.summary_plot(sv_last, X_last, feature_names=SIG + STATES, show=False, max_display=10, plot_size=(6.4, 3.4))
plt.title(f"Q: Does the model use features as the mechanism predicts? SHAP, {shap_label}, last fold {fold_lab[-1]}", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "m2_shap_beeswarm.png"))
plt.close("all")
imp = pd.DataFrame({k: np.abs(v[0]).mean(0) for k, v in shap_by_fold[lg].items()}, index=SIG + STATES)
sign = pd.DataFrame({k: [np.corrcoef(v[1][:, j], v[0][:, j])[0, 1] for j in range(len(SIG + STATES))] for k, v in shap_by_fold[lg].items()}, index=SIG + STATES)
stab = imp.rank().corr(method="spearman").values[np.triu_indices(NF, 1)].mean()
shap_tab = pd.DataFrame({"mean_abs_shap": imp.mean(1), "sign_corr_mean": sign.mean(1), "folds_same_sign": (np.sign(sign).eq(np.sign(sign.mean(1)), axis=0)).sum(1)}).sort_values("mean_abs_shap", ascending=False)
shap_tab.to_csv(os.path.join(RES, "shap_summary.csv"))

# %% conformal intervals (deployed rung, rank scale): calibrate on the previous fold, cover the next
score_rank = pred[chosen[deployed]].groupby(panel.date).rank(pct=True) - 0.5     # same scale as the target (per-date percentile - 0.5)
resid = (panel.y - score_rank).where(panel.label_end <= DEV_END)
conf = []
for k in range(1, NF):
    cal = resid[(panel.date >= folds[k - 1][0]) & (panel.date <= folds[k - 1][1])].dropna().abs()
    tst = resid[(panel.date >= folds[k][0]) & (panel.date <= folds[k][1])].dropna().abs()
    row = {"fold": k}
    for lvl in (0.8, 0.9):
        q = cal.quantile(lvl)
        row[f"halfwidth_{int(lvl * 100)}"] = q
        row[f"coverage_{int(lvl * 100)}"] = (tst <= q).mean()
    row["flag_below_80"] = row["coverage_80"] < 0.8
    conf.append(row)
conf = pd.DataFrame(conf)
conf.to_csv(os.path.join(RES, "conformal.csv"), index=False)

json.dump({"chosen": chosen, "deployed_rung": deployed, "deployed_model": chosen[deployed], "features": SIG + STATES, "ridge_alpha": RIDGE.get(chosen["ridge"]),
           "lgb_cfg": {**BASE, **LGB[chosen["lgb"]]}, "lgb_best_iters": {f"{n}|{k}": v for (n, k), v in best_iters.items() if n == chosen["lgb"]},
           "shap_rank_stability": float(stab)}, open(os.path.join(RES, "model_selection.json"), "w"), indent=1, default=float)
pd.set_option("display.width", 220)
print(pd.DataFrame(fold_info).to_string())
print(ics.round(4).to_string())
print(pair.round(4).to_string())
print("chosen", chosen, "deployed", deployed)
print(shap_tab.round(4).to_string())
print("shap rank stability", round(stab, 3))
print(conf.round(3).to_string())
ledger(f"04 models: ladder rule -> ridge ({len(RIDGE)} alphas, 1e-2..1e5) -> LightGBM ({len(LGB)} configs, L1 loss, early stopping on a purged inner split) -> ensemble. "
       f"Chosen by validation IC: {chosen}. Deployed rung (highest validation IC): {deployed}. Paired deltas: "
       + "; ".join(f"{r.comparison}: {r.delta_ic:+.4f} [{r.lo:+.4f},{r.hi:+.4f}]" for r in pair.itertuples()))
