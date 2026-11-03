"""Download the dataset: daily bars 2019-01-02..2026-10-30 for the 23 tradable ETFs plus SPY, TLT, AGG and BIL.

Source: Yahoo Finance via yfinance (the textbook ETF case study's own source, re-downloaded so the sealed
2026 holdout exists). Two files, same layout as the textbook's ETF data:
  prices_adjusted.parquet  split/dividend-adjusted OHLCV (returns, features, fills)
  prices_raw.parquet       split-adjusted, dividend-unadjusted close + volume (dollar volume, per-share costs)
The download date is written to data/download_meta.json; Yahoo revises adjusted history, so a later download can differ slightly.
Usage (from the repo root): uv run --project <textbook-repo> python data/build_dataset.py
"""
import hashlib
import json
import os

import pandas as pd
import yaml
import yfinance as yf

out = os.path.dirname(os.path.abspath(__file__))
cfg = yaml.safe_load(open(os.path.join(out, "..", "setup.yaml")))
U = cfg["universe"]
syms = U["tradable"] + U["auxiliary_not_traded"] + [U["cash_proxy"]]
end = (pd.Timestamp(cfg["data"]["end"]) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
raw = yf.download(syms, start=cfg["data"]["start"], end=end, auto_adjust=False, actions=False, progress=False, group_by="column")
adj = yf.download(syms, start=cfg["data"]["start"], end=end, auto_adjust=True, actions=False, progress=False, group_by="column")


def long(df, cols):
    d = pd.concat({c.lower(): df[c] for c in cols}, axis=1).stack(level=1, future_stack=True).reset_index()
    d.columns = ["timestamp", "symbol", *[c.lower() for c in cols]]
    d["timestamp"] = pd.to_datetime(d["timestamp"]).dt.tz_localize(None)
    return d.dropna(subset=["close"]).sort_values(["symbol", "timestamp"]).reset_index(drop=True)


meta = {"downloaded_at": pd.Timestamp.now().isoformat(timespec="seconds"), "yfinance": yf.__version__, "symbols": syms}
for d, name in ((long(adj, ["Open", "High", "Low", "Close", "Volume"]), "prices_adjusted.parquet"), (long(raw, ["Close", "Volume"]), "prices_raw.parquet")):
    d.to_parquet(os.path.join(out, name), index=False)
    meta[name] = {"rows": len(d), "symbols": int(d.symbol.nunique()), "first": str(d.timestamp.min().date()), "last": str(d.timestamp.max().date()),
                  "sha256_12": hashlib.sha256(open(os.path.join(out, name), "rb").read()).hexdigest()[:12]}
    print(name, meta[name])
json.dump(meta, open(os.path.join(out, "download_meta.json"), "w"), indent=1)
