"""Shared library: configuration, sealed data access, point-in-time universe, features, metrics, simulator, run log."""
import csv
import hashlib
import json
import os
import subprocess
from datetime import datetime

import numpy as np
import pandas as pd
import yaml
from scipy.stats import norm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = yaml.safe_load(open(os.path.join(ROOT, "setup.yaml")))
DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
for d in (RES, FIG):
    os.makedirs(d, exist_ok=True)
EV = CFG["evaluation"]
DEV_END = pd.Timestamp(EV["development_end"])
HO_START, HO_END = (pd.Timestamp(x) for x in EV["holdout"])
H = CFG["label"]["horizon_sessions"]
CASH = CFG["universe"]["cash_proxy"]
TRADABLE = CFG["universe"]["tradable"]
FREEZE = os.path.join(RES, "freeze.json")
HOLDOUT_MARK = os.path.join(RES, "holdout_touched.json")
ANN = 252


# ---------------------------------------------------------------- provenance and run log
def git_commit():
    try:
        return subprocess.check_output(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "uncommitted"


def data_snapshot():
    h = hashlib.sha256()
    for f in ("prices_adjusted.parquet", "prices_raw.parquet"):
        h.update(open(os.path.join(DATA, f), "rb").read())
    return h.hexdigest()[:12]


RUN_FIELDS = ["timestamp", "git_commit", "data_snapshot", "setup_version", "strategy", "trial_family", "trial", "run",
              "eval_window", "config", "metric_ic", "metric_sharpe", "decision", "gate"]


def log_run(trial_family, trial, run, window, config, ic=None, sharpe=None, decision="", gate=""):
    path = os.path.join(ROOT, "run_log.csv")
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RUN_FIELDS)
        if new:
            w.writeheader()
        w.writerow({"timestamp": datetime.now().isoformat(timespec="seconds"), "git_commit": git_commit(), "data_snapshot": data_snapshot(),
                    "setup_version": CFG["setup_version"], "strategy": CFG["strategy_id"], "trial_family": trial_family, "trial": trial,
                    "run": run, "eval_window": window, "config": json.dumps(config, default=str),
                    "metric_ic": "" if ic is None else round(float(ic), 5), "metric_sharpe": "" if sharpe is None else round(float(sharpe), 4),
                    "decision": decision, "gate": gate})


def ledger(text):
    with open(os.path.join(ROOT, "ledger", "research_ledger.md"), "a") as f:
        f.write(f"\n### {datetime.now().isoformat(timespec='minutes')} (commit {git_commit()})\n{text.strip()}\n")


# ---------------------------------------------------------------- sealed data access
def load_prices(holdout=False):
    """Development stages see data up to DEV_END only. The holdout is readable only after the freeze,
    and only by notebooks/06_holdout.py, which writes HOLDOUT_MARK first so a second run is refused."""
    a = pd.read_parquet(os.path.join(DATA, "prices_adjusted.parquet"))
    r = pd.read_parquet(os.path.join(DATA, "prices_raw.parquet"))
    if holdout:
        assert os.path.exists(FREEZE), "holdout is sealed until results/freeze.json exists"
        a, r = a[a.timestamp <= HO_END], r[r.timestamp <= HO_END]
    else:
        a, r = a[a.timestamp <= DEV_END], r[r.timestamp <= DEV_END]
    piv = lambda d, c: d.pivot(index="timestamp", columns="symbol", values=c).sort_index()
    return {"open": piv(a, "open"), "high": piv(a, "high"), "low": piv(a, "low"), "close": piv(a, "close"),
            "volume": piv(a, "volume"), "raw_close": piv(r, "close")}


def month_ends(idx):
    s = pd.Series(idx, index=idx)
    return pd.DatetimeIndex(s.groupby([s.index.year, s.index.month]).max().values)


# ---------------------------------------------------------------- point-in-time universe
def pit_universe(P, dates, lookback=126, adv_min=10e6, corr_max=None):
    """For each decision date: eligibility (and optional de-duplication) using only the trailing `lookback` sessions.
    Only the fixed tradable list is ever eligible; setup v1 applies no de-duplication (corr_max=None)."""
    close, raw, vol = P["close"], P["raw_close"], P["volume"]
    ret = close.pct_change(fill_method=None)
    dv = (raw * vol)
    rows = {}
    for d in dates:
        win_dv = dv.loc[:d].iloc[-lookback:]
        win_r = ret.loc[:d].iloc[-lookback:]
        hist = close.loc[:d].iloc[-lookback:].notna().sum()
        med = win_dv.median()
        elig = med[(med >= adv_min) & (hist >= lookback)].reindex(TRADABLE).dropna().sort_values(ascending=False)
        c = win_r[elig.index].corr()
        kept, dropped = [], []
        for s in elig.index:
            if corr_max is not None and any(c.loc[s, k] >= corr_max for k in kept):
                dropped.append(s)
            else:
                kept.append(s)
        rows[d] = {"eligible": list(elig.index), "kept": kept, "dropped": dropped}
    return rows


# ---------------------------------------------------------------- labels and features
SIGNALS = ["mom_126_vs", "mom_126_skip21", "mom_63", "ret_21", "ret_5", "dist_63d_high", "sma50_ratio", "vol_63", "idio_vol_63",
           "beta_spy_63", "max_dd_63", "volume_chg_21_63", "range_21", "rsi_14", "dollar_vol_126"]
STATES = ["spy_vol_pct", "spy_trend_126", "xs_dispersion_21", "spy_tlt_corr_63"]


def build_panel(P):
    close, opn, high, low, vol, raw = P["close"], P["open"], P["high"], P["low"], P["volume"], P["raw_close"]
    r1 = close.pct_change(fill_method=None)
    F = {}
    v63 = r1.rolling(63).std() * np.sqrt(ANN)
    F["mom_126_vs"] = (close / close.shift(126) - 1) / v63
    F["mom_126_skip21"] = close.shift(21) / close.shift(126) - 1
    F["mom_63"] = close / close.shift(63) - 1
    F["ret_21"] = close / close.shift(21) - 1
    F["ret_5"] = close / close.shift(5) - 1
    F["dist_63d_high"] = close / close.rolling(63).max() - 1
    F["sma50_ratio"] = close / close.rolling(50).mean()
    F["vol_63"] = v63
    spy = r1["SPY"]
    beta = r1.rolling(63).cov(spy).div(spy.rolling(63).var(), axis=0)
    F["beta_spy_63"] = beta
    F["idio_vol_63"] = r1.sub(beta.mul(spy, axis=0)).rolling(63).std() * np.sqrt(ANN)
    F["max_dd_63"] = (close / close.rolling(63).max() - 1).rolling(63).min()
    F["volume_chg_21_63"] = np.log(vol.rolling(21).mean() / vol.rolling(63).mean())
    F["range_21"] = ((high - low) / close).rolling(21).mean()
    d = close.diff()
    up, dn = d.clip(lower=0).ewm(alpha=1 / 14).mean(), (-d.clip(upper=0)).ewm(alpha=1 / 14).mean()
    F["rsi_14"] = 100 - 100 / (1 + up / dn)
    F["dollar_vol_126"] = np.log((raw * vol).rolling(126).median())
    spy_v21 = spy.rolling(21).std() * np.sqrt(ANN)
    state = pd.DataFrame({"spy_vol_pct": spy_v21.rolling(126).apply(lambda x: (x[:-1] < x[-1]).mean(), raw=True),
                          "spy_trend_126": close["SPY"] / close["SPY"].shift(126) - 1,
                          "xs_dispersion_21": F["ret_21"][TRADABLE].std(axis=1),
                          "spy_tlt_corr_63": spy.rolling(63).corr(r1["TLT"])})
    long = pd.concat({k: v.stack(future_stack=True) for k, v in F.items()}, axis=1)
    long.index.names = ["date", "symbol"]
    for k in (5, 21, 63):
        lab = opn.shift(-(k + 1)) / opn.shift(-1) - 1
        long[f"fwd_oo_{k}"] = lab.stack(future_stack=True)
    end = pd.Series(close.index, index=close.index).shift(-(H + 1))
    long = long.reset_index()
    long["label_end"] = long.date.map(end)
    long = long.merge(state, left_on="date", right_index=True, how="left")
    long = long[long.symbol.isin(TRADABLE)]
    return long.dropna(subset=SIGNALS + STATES), state


# ---------------------------------------------------------------- statistics
def nw(x, lag=H):
    x = np.asarray(pd.Series(x).dropna(), float)
    n = len(x)
    if n < 10:
        return dict(mean=np.nan, se=np.nan, t=np.nan, lo=np.nan, hi=np.nan, p=np.nan, n=n)
    m = x.mean()
    e = x - m
    s = e @ e / n
    for L in range(1, min(lag, n - 1) + 1):
        s += 2 * (1 - L / (lag + 1)) * (e[L:] @ e[:-L]) / n
    se = np.sqrt(s / n)
    return dict(mean=m, se=se, t=m / se, lo=m - 1.96 * se, hi=m + 1.96 * se, p=2 * (1 - norm.cdf(abs(m / se))), n=n)


def daily_ic(df, col, label="fwd_oo_21"):
    return df.groupby("date").apply(lambda g: g[col].corr(g[label], method="spearman"), include_groups=False)


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    adj = np.empty(n)
    adj[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.clip(adj, 0, 1)


RF = None  # daily risk-free series (BIL open-to-open); set by set_rf(P)


def set_rf(P):
    """Sharpe ratios are computed on returns in excess of the cash proxy (BIL, open-to-open, same marking as the simulator)."""
    global RF
    RF = P["open"][CASH].pct_change(fill_method=None)


def xs(r):
    r = pd.Series(r).dropna()
    return r - RF.reindex(r.index).fillna(0) if RF is not None else r


def perf(r):
    r = pd.Series(r).dropna()
    eq = (1 + r).cumprod()
    dd = eq / eq.cummax() - 1
    ex = xs(r)
    mu, sd = ex.mean() * ANN, r.std() * np.sqrt(ANN)
    var = np.percentile(r, 5)
    # drawdown duration and recovery (sessions)
    trough = dd.idxmin()
    peak = eq.loc[:trough].idxmax()
    rec = eq.loc[trough:][eq.loc[trough:] >= eq.loc[peak]]
    return {"sharpe": mu / sd if sd > 0 else np.nan, "cagr": eq.iloc[-1] ** (ANN / len(r)) - 1, "vol": sd, "max_dd": dd.min(),
            "dd_duration_sessions": int(len(eq.loc[peak:trough])), "recovery_sessions": int(len(eq.loc[trough:rec.index[0]])) if len(rec) else None,
            "var95": var, "cvar95": r[r <= var].mean(), "n_days": len(r)}


def sharpe_se(sr, years):
    return np.sqrt((1 + sr ** 2 / 2) / years)


def boot_sharpe(r, n=2000, block=21, seed=CFG["seeds"]["bootstrap"]):
    x = np.asarray(xs(r))
    T = len(x)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        st = rng.integers(0, T - block + 1, int(np.ceil(T / block)))
        s = np.concatenate([x[a:a + block] for a in st])[:T]
        out.append(s.mean() / s.std() * np.sqrt(ANN))
    return list(np.percentile(out, [2.5, 97.5]))


def paired_boot(a, b, n=2000, block=21, seed=CFG["seeds"]["bootstrap"]):
    j = pd.concat([xs(a), xs(b)], axis=1).dropna().values
    T = len(j)
    rng = np.random.default_rng(seed)
    f = lambda x: x.mean() / x.std() * np.sqrt(ANN)
    out = []
    for _ in range(n):
        st = rng.integers(0, T - block + 1, int(np.ceil(T / block)))
        s = np.concatenate([j[q:q + block] for q in st])[:T]
        out.append(f(s[:, 0]) - f(s[:, 1]))
    out = np.array(out)
    return {"diff": f(j[:, 0]) - f(j[:, 1]), "lo": float(np.percentile(out, 2.5)), "hi": float(np.percentile(out, 97.5)),
            "p": float(2 * min((out <= 0).mean(), (out >= 0).mean()))}


def deflated_sharpe(r, n_trials, sr_var_daily):
    r = xs(r)
    s = r.mean() / r.std()
    emc = 0.5772156649
    sr0 = np.sqrt(sr_var_daily) * ((1 - emc) * norm.ppf(1 - 1 / n_trials) + emc * norm.ppf(1 - 1 / (n_trials * np.e)))
    sk, ku = r.skew(), r.kurt() + 3
    return float(norm.cdf((s - sr0) * np.sqrt(len(r) - 1) / np.sqrt(1 - sk * s + (ku - 1) / 4 * s ** 2))), float(sr0 * np.sqrt(ANN))


def min_track_years(sr, a=0.05, b=0.80):
    return ((norm.ppf(1 - a) + norm.ppf(b)) / sr) ** 2 if sr and sr > 0 else np.inf


# ---------------------------------------------------------------- simulator
def half_spread(sym):
    hs = CFG["costs"]["half_spread_usd"]
    if sym in hs:
        return hs[sym]
    return hs["sector_XL"] if sym.startswith("XL") else hs["default"]


def simulate(scores, P, state, univ, decisions, end_mark, alloc="ew", gate=True, overlay=False, cost_mult=1.0, spread_mult=1.0, fixed=None, top_k=None):
    """Monthly event loop. Weights are decided from information at the close of d and traded at the next open.
    The gate, overlay and constraints are applied inside the loop, so they change the next action."""
    top_k = top_k or CFG["score_to_trade"]["top_k"]
    opn, close, raw = P["open"], P["close"], P["raw_close"]
    r1 = close.pct_change(fill_method=None)
    idx = opn.index
    end_mark = pd.Timestamp(end_mark)
    comm = CFG["costs"]["commission_per_share"]
    held = pd.Series(dtype=float)
    daily, turns, hhis, expo, gated, leg_costs = [], [], [], [], [], []
    for i, d in enumerate(decisions):
        if fixed is not None:
            w = pd.Series(fixed(d) if callable(fixed) else fixed, dtype=float)
        else:
            s = scores.loc[d].reindex(univ[d]["kept"]).dropna()
            if gate and state.loc[d, "spy_vol_pct"] >= 0.80:
                w = pd.Series({CASH: 1.0})
                gated.append(1)
            else:
                gated.append(0)
                sel = s.index if top_k == "all" else s.nlargest(top_k).index
                if alloc == "ivol":
                    v = r1.loc[:d].iloc[-63:][sel].std()
                    w = (1 / v) / (1 / v).sum()
                else:
                    w = pd.Series(1.0 / len(sel), index=sel)
                w = w.clip(upper=CFG["constraints"]["max_weight"] if top_k != "all" else 1.0)
                w = w / w.sum()
                if overlay:
                    pr = r1.loc[:d].iloc[-63:][w.index].fillna(0) @ w
                    scale = min(1.0, CFG["risk_overlay"]["target_annual_vol"] / (pr.std() * np.sqrt(ANN)))
                    w = w * scale
                    w[CASH] = w.get(CASH, 0) + (1 - w.sum())
        assert (w >= -1e-12).all() and w.sum() <= 1 + 1e-9, "long-only / no-leverage constraint violated"
        expo.append(float(w.drop(CASH, errors="ignore").sum()))
        hhis.append(float((w ** 2).sum()))
        after = idx[idx > d]
        if len(after) == 0 or after[0] > end_mark:
            break
        t0 = after[0]
        if i + 1 < len(decisions):
            nxt = idx[idx > decisions[i + 1]]
            t1 = min(nxt[0] if len(nxt) else idx[-1], end_mark)
        else:
            t1 = end_mark
        t1 = idx[idx <= t1][-1]
        allsym = w.index.union(held.index)
        trade = (w.reindex(allsym).fillna(0) - held.reindex(allsym).fillna(0)).abs()
        per_share = pd.Series({s: comm + spread_mult * half_spread(s) for s in allsym})
        cost_rate = per_share / raw.loc[d, allsym]
        cost = cost_mult * float((trade * cost_rate).sum())
        if trade.sum() > 0:
            leg_costs.append(float((trade * cost_rate).sum() / trade.sum()))
        turns.append(float(trade.sum()))
        seg = opn.loc[t0:t1, w.index]
        assert not seg.iloc[0].isna().any(), f"no fill on a missing bar at {t0}"
        rel = seg / seg.iloc[0]
        val = (rel * w).sum(axis=1)
        r = val.pct_change().iloc[1:]
        if len(r):
            r.iloc[0] -= cost
        else:
            r = pd.Series([-cost], index=[t1])
        daily.append(r)
        e = rel.iloc[-1] * w
        held = e / e.sum()
    out = pd.concat(daily)
    out = out[~out.index.duplicated()]
    yrs = len(out) / ANN
    t2 = float(np.sum(turns) / yrs)
    return out, {"turnover_2way_ann": t2, "avg_holding_days": (2 / t2 * ANN) if t2 > 0 else np.inf, "hhi": float(np.mean(hhis)),
                 "eff_n": float(np.mean([1 / h for h in hhis])), "avg_exposure": float(np.mean(expo)),
                 "gated_share": float(np.mean(gated)) if gated else 0.0, "avg_cost_per_leg_bps": float(np.mean(leg_costs) * 1e4) if leg_costs else 0.0}


def figure_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "figure.dpi": 150, "figure.constrained_layout.use": False, "figure.autolayout": False,
                         "axes.spines.top": False, "axes.spines.right": False})
    return plt
