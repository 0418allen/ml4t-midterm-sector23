# %% [markdown]
# # 01 Setup: protocol commitment, sealed holdout, research ledger
# Runs before any data analysis or backtest. Writes the protocol table and split dates, and opens the ledger.

# %%
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from lib import CFG, EV, RES, ROOT, ledger, log_run

assert not os.path.exists(os.path.join(RES, "freeze.json")), "setup must precede the freeze"
ev = EV
protocol = pd.DataFrame([
    ["Training window", f"expanding from {ev['training_start']}"],
    ["Step / retraining cadence", f"{ev['step']} / {ev['retrain_cadence']}"],
    ["Label horizon", f"{CFG['label']['horizon_sessions']} sessions, {CFG['label']['definition']}"],
    ["Longest feature lookback", f"{CFG['admissible_information']['longest_feature_lookback_sessions']} sessions"],
    ["Validation folds", "; ".join(f"{a}..{b}" for a, b in ev["validation_folds"])],
    ["Label buffer (purge) / feature buffer (embargo)", f"{ev['label_buffer_sessions']} / {ev['feature_buffer_sessions']} trading sessions"],
    ["Hyperparameter tuning", ev["hyperparameter_tuning"]],
    ["Holdout (sealed)", f"{ev['holdout'][0]}..{ev['holdout'][1]}"],
    ["Yardstick", ev["yardstick"]],
    ["Confirmatory test", f"{CFG['confirmatory_test']['metric']}; pass if {CFG['confirmatory_test']['threshold']}"],
], columns=["Commitment", "Value"])
protocol.to_csv(os.path.join(RES, "protocol.csv"), index=False)
print(protocol.to_string(index=False))

# %%
os.makedirs(os.path.join(ROOT, "ledger"), exist_ok=True)
path = os.path.join(ROOT, "ledger", "research_ledger.md")
if not os.path.exists(path):
    open(path, "w").write("# Research ledger\n\nTaxonomy: Strategy -> Trial family -> Trial -> Run (ML4T Ch. 6). Every adaptive choice is a trial; bug fixes are not.\n"
                          "\n## Prior work counted as trials (disclosed)\n"
                          "- v1 (textbook pipeline, 100 ETFs, 2006-2025): 734 registered backtests; its holdout 2024-2025 was run once.\n"
                          "- v2 (69 de-duplicated ETFs, 2020-2024): 52 strategy variants; its holdout 2024 was run once.\n"
                          "- v3 (54 de-duplicated ETFs, 2020-2025): 60 strategy variants; its holdout 2025 was run once.\n"
                          "All three are counted in the inclusive Deflated Sharpe trial count. Because 2024 and 2025 were already seen, they are development data here "
                          "and the sealed holdout is 2026-01..2026-10, which no one on the team had looked at when this setup was committed.\n"
                          "The 23-ETF list was chosen by the team after those versions (hindsight risk in the universe choice, disclosed; it is not re-optimised here).\n")
ledger(f"""**Setup v1 committed before any backtest.** Mechanism: {CFG['hypothesis']['mechanism'].strip()}
Source of edge: {', '.join(CFG['hypothesis']['source_of_edge'])}. Holdout {ev['holdout'][0]}..{ev['holdout'][1]} sealed by the loader (src/lib.py: load_prices).
Pre-registered confirmatory test: {CFG['confirmatory_test']['metric']}; pass if {CFG['confirmatory_test']['threshold']}. Secondary: {CFG['confirmatory_test']['secondary']}.
Selection rule: {CFG['models']['selection']}.""")
log_run("setup", "setup_v1", "01_setup", "n/a", {"setup_version": CFG["setup_version"]}, decision="committed", gate="protocol fixed")
