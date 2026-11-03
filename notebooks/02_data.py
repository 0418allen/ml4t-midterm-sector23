# %% [markdown]
# # 02 Data: point-in-time universe and data-quality report (development sample only)

# %%
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from lib import CASH, DATA, DEV_END, EV, FIG, RES, TRADABLE, figure_style, ledger, load_prices, log_run, month_ends, pit_universe

plt = figure_style()
P = load_prices()                       # sealed: ends at DEV_END (2025-12-31)
a = pd.read_parquet(os.path.join(DATA, "prices_adjusted.parquet"))
r = pd.read_parquet(os.path.join(DATA, "prices_raw.parquet"))
a, r = a[a.timestamp <= DEV_END], r[r.timestamp <= DEV_END]

# %% quality checks
q = {
    "rows_adjusted": int(len(a)), "symbols": int(a.symbol.nunique()), "sessions": int(a.timestamp.nunique()),
    "duplicate_symbol_dates": int(a.duplicated(["symbol", "timestamp"]).sum()),
    "missing_values": {c: int(a[c].isna().sum()) for c in ["open", "high", "low", "close", "volume"]},
    "ohlc_violations": {
        "high_below_low": int((a.high < a.low).sum()),
        "close_outside_high_low": int(((a.close > a.high * 1.0001) | (a.close < a.low * 0.9999)).sum()),
        "open_outside_high_low": int(((a.open > a.high * 1.0001) | (a.open < a.low * 0.9999)).sum()),
        "non_positive_price": int((a[["open", "high", "low", "close"]] <= 0).sum().sum()),
        "zero_volume": int((a.volume == 0).sum())},
    "symbols_leaving_sample": int((a.groupby("symbol").timestamp.max() < a.timestamp.max()).sum()),
    "symbols_entering_after_start": int((a.groupby("symbol").timestamp.min() > a.timestamp.min()).sum()),
    "sessions_per_symbol_min_max": [int(a.groupby("symbol").size().min()), int(a.groupby("symbol").size().max())],
    "missing_handling": "no forward-fill; a missing bar would be excluded and the simulator refuses to fill on it",
}
ret = P["close"].pct_change(fill_method=None)
big = ret.stack()[ret.stack().abs() > 0.15]
q["daily_moves_gt_15pct"] = [[str(d.date()), s, round(float(v), 3)] for (d, s), v in big.items()]

# %% point-in-time universe on every development decision date
dec = month_ends(P["close"].index)
dec = dec[dec >= "2019-12-31"]
U = pit_universe(P, dec)
cov = pd.DataFrame({"date": list(U), "eligible": [len(v["eligible"]) for v in U.values()], "kept_after_dedup": [len(v["kept"]) for v in U.values()]})
ret_m = P["close"][TRADABLE].pct_change(fill_method=None).loc["2020-01-01":]
pair_corr = ret_m.corr().where(lambda c: ~np.eye(len(c), dtype=bool)).stack().sort_values(ascending=False)
pair_corr = pair_corr[pair_corr.index.get_level_values(0) < pair_corr.index.get_level_values(1)]
cov.to_csv(os.path.join(RES, "universe_coverage.csv"), index=False)
json.dump({str(k.date()): v for k, v in U.items()}, open(os.path.join(RES, "universe_dev.json"), "w"))
dropped_count = pd.Series([s for v in U.values() for s in v["dropped"]]).value_counts()
q["universe"] = {"decisions": len(U), "eligible_min": int(cov.eligible.min()), "eligible_max": int(cov.eligible.max()),
                 "kept_min": int(cov.kept_after_dedup.min()), "kept_max": int(cov.kept_after_dedup.max()),
                 "never_eligible": sorted(set(TRADABLE) - {s for v in U.values() for s in v["eligible"]}),
                 "highest_pair_correlations_2020_2025": {f"{a}/{b}": round(float(v), 3) for (a, b), v in pair_corr.head(6).items()}}
json.dump(q, open(os.path.join(RES, "data_quality.json"), "w"), indent=1)

# %% figures: coverage over time; corporate-action check (USO 1:8 reverse split, 2020-04-29)
fig, ax = plt.subplots(figsize=(6.5, 2.6))
ax.plot(cov.date, cov.eligible, label="eligible of 23 (ADV >= $10M, 126-session history)")
ax.set_ylabel("ETFs")
ax.set_ylim(0, 25)
ax.set_title("Q: How many names can the strategy trade at each decision date?")
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "d1_coverage.png"))
plt.close(fig)
TB = os.path.expanduser("~/Downloads/machine-learning-for-trading/data/etfs/market")
tb_raw = pd.read_parquet(os.path.join(TB, "etf_universe_unadjusted.parquet"))
tb_raw["timestamp"] = pd.to_datetime(tb_raw["timestamp"])
ur = tb_raw[tb_raw.symbol == "USO"].set_index("timestamp").close.loc["2020-02-01":"2020-07-31"]     # truly unadjusted (textbook file)
u = a[a.symbol == "USO"].set_index("timestamp").close.loc["2020-02-01":"2020-07-31"]
fig, ax = plt.subplots(figsize=(6.5, 2.6))
ax.plot(ur, label="unadjusted close (textbook file, as traded)")
ax.plot(u, label="adjusted close used here (yfinance)")
ax.axvline(pd.Timestamp("2020-04-29"), c="k", lw=0.7, ls="--")
ax.text(pd.Timestamp("2020-05-02"), ur.max() * 0.9, "1:8 reverse split", fontsize=7)
ax.set_yscale("log")
ax.set_title("Q: Are corporate actions handled? USO unadjusted vs adjusted close")
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "d2_corporate_action.png"))
plt.close(fig)
# source consistency: our yfinance download vs the textbook's Yahoo bars, daily adjusted returns 2020-2025
tb = pd.read_parquet(os.path.join(TB, "etf_universe.parquet"), columns=["timestamp", "symbol", "close"])
tb["timestamp"] = pd.to_datetime(tb["timestamp"])
tbc = tb[tb.symbol.isin(P["close"].columns)].pivot(index="timestamp", columns="symbol", values="close").loc["2020-01-02":str(DEV_END.date())]
mine = P["close"].loc["2020-01-02":].reindex(columns=tbc.columns)
rc = pd.DataFrame({"textbook": tbc.pct_change(fill_method=None).stack(), "ours": mine.pct_change(fill_method=None).stack()}).dropna()
q["source_check_vs_textbook"] = {"overlap_rows": int(len(rc)), "return_correlation": float(rc.corr().iloc[0, 1]),
                                 "share_abs_diff_gt_10bp": float(((rc.textbook - rc.ours).abs() > 0.001).mean())}
jump_raw = float((ur.pct_change().abs()).max())
jump_adj = float((u.pct_change().abs()).max())
q["corporate_action_check"] = {"symbol": "USO", "event": "1:8 reverse split 2020-04-29", "max_abs_daily_change_unadjusted": jump_raw, "max_abs_daily_change_adjusted": jump_adj,
                              "note": "prices_raw.parquet is split-adjusted by Yahoo; dollar volume (close x volume) is invariant to the split"}
json.dump(q, open(os.path.join(RES, "data_quality.json"), "w"), indent=1)

pit = pd.DataFrame([
    ["Values", "Every feature uses bars dated <= t (close of t); label starts at the t+1 open", "PASS"],
    ["Values", "Non-price data", "N/A (prices and volume only)"],
    ["Universe", "Eligibility (ADV, history) computed from the trailing 126 sessions at each t", "PASS"],
    ["Universe", "Delisted / removed ETFs included", "FAIL (disclosed): fixed list of 23 survivors chosen in 2026 by the team; no delisted sector ETFs"],
    ["Identifiers", "Ticker changes / reuse handled with effective dates", "PASS (no ticker changes in 2019-2025 for these symbols; symbols are stable keys)"],
], columns=["Level", "Check", "Result"])
pit.to_csv(os.path.join(RES, "pit_checklist.csv"), index=False)
print(json.dumps({k: v for k, v in q.items() if k != "daily_moves_gt_15pct"}, indent=1))
print(len(q["daily_moves_gt_15pct"]), "moves > 15%")
ledger(f"02 data: {q['symbols']} symbols, {q['sessions']} sessions (dev). Duplicates {q['duplicate_symbol_dates']}, OHLC violations {q['ohlc_violations']}. "
       f"PIT universe: {q['universe']['kept_min']}-{q['universe']['kept_max']} tradable names per decision. Survivorship check FAIL disclosed.")
log_run("data", "pit_universe", "02_data", f"2019-12-31..{DEV_END.date()}", {"adv_min": 10e6, "lookback": 126, "dedup": None}, decision="dataset accepted", gate="quality report")
