"""Build the proposal PDF (Sections 1-8 + appendix) from registered results. Run from the repo root after 07."""
import json
import os

import pandas as pd
from PIL import Image as PI
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, FIG = f"{R}/results", f"{R}/figures"
J = lambda f: json.load(open(f"{RES}/{f}"))
C = lambda f: pd.read_csv(f"{RES}/{f}")
fz, vs, hs, dq, rs = J("freeze.json"), J("validation_summary.json"), J("holdout_summary.json"), J("data_quality.json"), J("results_summary.json")
mech, fsel = J("mechanism_gate.json"), J("features_selected.json")
tri, ft, pit, cov, folds = C("triage.csv"), C("feature_table.csv"), C("pit_checklist.csv"), C("universe_coverage.csv"), C("folds.csv")
vic, pair, conf, shp = C("validation_ic.csv").set_index("model"), C("paired_comparison.csv"), C("conformal.csv"), C("shap_summary.csv").set_index("Unnamed: 0")
vb, hb, head, mchk = C("validation_backtests.csv").set_index("strategy"), C("holdout_backtests.csv").set_index("strategy"), C("headline_table.csv"), C("mechanism_checks.csv")
sel = fz["model_selection"]
DEP, RULE = fz["deployed_strategy"], fz["baseline"]
RAW = DEP.replace("|vt", "|raw")
M = sel["deployed_model"]
pct = lambda x: f"{x * 100:.1f}%"
f2 = lambda x: f"{x:.2f}"
f3 = lambda x: f"{x:+.3f}"

ss = getSampleStyleSheet()
B = ParagraphStyle("B", parent=ss["BodyText"], fontName="Helvetica", fontSize=9.6, leading=12.8, spaceAfter=5)
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=13, spaceBefore=4, spaceAfter=5, textColor=colors.HexColor("#1f2d55"))
H2 = ParagraphStyle("H2", parent=B, fontName="Helvetica-Bold", fontSize=10, spaceBefore=4, spaceAfter=2)
BOX = ParagraphStyle("BOX", parent=B, backColor=colors.HexColor("#f4f1e8"), borderPadding=5, spaceBefore=4, spaceAfter=8)
CAP = ParagraphStyle("CAP", parent=B, fontSize=7.8, leading=9.6, textColor=colors.HexColor("#444444"))
S = []
p = lambda t, st=B: S.append(Paragraph(t, st))


def tbl(rows, widths, fs=7.6, hdr="#1f2d55"):
    c = ParagraphStyle("c", parent=B, fontSize=fs, leading=fs + 2.2, spaceAfter=0)
    ch = ParagraphStyle("ch", parent=c, fontName="Helvetica-Bold", textColor=colors.white)
    t = Table([[Paragraph(str(x), ch if i == 0 else c) for x in r] for i, r in enumerate(rows)], colWidths=[w * cm for w in widths], repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(hdr)), ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#999999")),
                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5)]))
    S.append(t)
    S.append(Spacer(1, 5))


def fig(name, w, cap):
    iw, ih = PI.open(f"{FIG}/{name}").size
    S.append(KeepTogether([Image(f"{FIG}/{name}", width=w * cm, height=w * cm * ih / iw), Paragraph(cap, CAP)]))


def wcb(t):
    p("<b>What could break this?</b> " + t, BOX)


def sec(t, brk=True):
    if brk and S:
        S.append(PageBreak())
    p(t, H1)


vh = hs["ic"][M]
EVC = __import__("yaml").safe_load(open(f"{R}/setup.yaml"))["evaluation"]
NF = len(EVC["validation_folds"])
VL, HL = head.columns[2], head.columns[3]
NDEC = len(hs["decisions"])
_pd = pd.read_parquet(f"{R}/data/panel_dev.parquet")
_pd = _pd[(_pd.label_end <= EVC["development_end"]) & (_pd.date >= EVC["training_start"])]
NROWS, NDATES = len(_pd), _pd.date.nunique()
TRI = tri.decision.value_counts().to_dict()
og = {r["strategy"]: r for r in vs["overlay_and_gate"]}
NOG_VT, NOG_RAW, G_RAW = "rule_mom126vs|ew|nogate|vt", "rule_mom126vs|ew|nogate|raw", "rule_mom126vs|ew|gate|raw"
LGB, RID = sel["chosen"]["lgb"], sel["chosen"]["ridge"]
ct = hs["confirmatory_test"]
TICK = "IYR, XLB, XLC, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY, QQQ, IBB, ITA, ITB, KRE, OIH, SOXX, XBI, XME, XRT, USO, GLD"
# ===================================================================== title page content + Section 1
p("<font size=16><b>Sector-ETF Momentum with a Volatility-State Gate: an ML4T Midterm Proposal</b></font>", ss["Title"])
p(f"Setup v1 &nbsp;|&nbsp; 23 sector/industry ETFs + QQQ, USO, GLD, 2019-2026 (Yahoo) &nbsp;|&nbsp; Development 2020-01..2025-12, sealed holdout 2026-01..2026-10 (fresh data, touched once) &nbsp;|&nbsp; Repository: github.com/0418allen/ml4t-midterm-sector23 (setup commit {fz['git_commit_before_freeze']})", CAP)
p(f"<b>Abstract.</b> We test whether sector and industry ETF trends, traded only outside high-volatility states, predict next-month returns, and whether Ridge or LightGBM improves on a momentum rule. "
  f"Under a pre-registered walk-forward protocol ({NF} half-year folds, 2022-2025) no model beat the rule, so the rule was deployed (validation IC {f3(vic.loc[M,'ic'])} [{f3(vic.loc[M,'lo'])}, {f3(vic.loc[M,'hi'])}], net excess Sharpe {f2(vb.loc[DEP,'sharpe'])}); "
  f"momentum had no IC in calm states and the gate made the rule worse, falsifying our mechanism on validation. On the fresh 2026 holdout the deployed strategy earned an excess Sharpe of {f2(hb.loc[DEP,'sharpe'])} "
  f"[{hs['deployed']['ci_bootstrap'][0]:.2f}, {hs['deployed']['ci_bootstrap'][1]:.2f}] against {f2(hb.loc['bench_ew_universe','sharpe'])} for the equal-weight universe; the pre-registered confirmatory test failed "
  f"({f3(ct['diff'])}, CI {f3(ct['lo'])}..{f3(ct['hi'])}). Outcome pattern: <b>never resolved</b>.", BOX)
sec("1. Hypothesis and Sources of Edge", brk=False)
p("<b>Mechanism (two sentences).</b> Industry and sector returns persist over one to six months because information diffuses slowly across industries and investors rebalance sector allocations gradually. "
  "The persistence crashes when market volatility spikes and leadership reverses abruptly, so the rotation signal should be traded only in calm-to-normal volatility states.")
p("<b>Source of edge (Table 6.1).</b> Flow predictability (gradual sector re-allocation by benchmarked and thematic investors) and risk compensation (holding leading risk sectors earns a premium that is lost in volatility shocks). "
  "<b>Counterparty:</b> investors who rebalance on calendar schedules or chase sector leadership late.")
p("<b>Falsification criteria (fixed in setup.yaml before data analysis).</b> (1) The momentum rule's validation IC is not positive in the low-volatility state. (2) The gated rule does not beat the ungated rule on validation net Sharpe. "
  "(3) The pre-registered confirmatory test fails on the holdout: deployed holdout net excess Sharpe minus the equal-weight benchmark, 21-day block bootstrap, pass only if the 95% CI lower bound is above zero.")
p(f"<b>Status.</b> (1) <b>triggered</b> (low-vol IC {f3(mech['mom_126_vs_ic_low_vol'])}, high-vol {f3(mech['mom_126_vs_ic_high_vol'])} - the opposite of the mechanism); (2) <b>triggered</b> (gated rule {f2(vb.loc[RULE,'sharpe'])} vs ungated {f2(vb.loc[NOG_VT,'sharpe'])}); "
  f"(3) <b>triggered</b> (Section 7). The hypothesis is not supported; the report documents where it failed.")
wcb("Six years cover few regimes (post-COVID recovery, 2022 inflation shock, 2023-24 AI rally, 2025 tariff shock). With 23 overlapping ETFs the cross-section is thin, and the 80th-percentile gate threshold was chosen a priori, not estimated.")

# ===================================================================== Section 2
sec("2. Trading Setup v1 and Evaluation Protocol")
tbl([["Setup field", "Specification (setup.yaml, committed before any backtest)"],
     ["Universe and eligibility", f"Fixed list of 23 ETFs chosen by the team: {TICK}. Point-in-time eligibility: trailing 126-session median dollar volume >= $10M and 126 sessions of history at t (all 23 qualify at every decision). No de-duplication: overlaps such as QQQ/XLK (corr 0.98) are kept and disclosed. BIL is cash; SPY, TLT, AGG are used for states and benchmarks only."],
     ["Decision schedule", "Snapshot at the month-end close; execute at the next session's open; hold to the next rebalance (about 21 sessions)."],
     ["Admissible information", "Prices and volume to the close of t; no fundamentals; longest lookback 147 sessions."],
     ["Score-to-trade mapping", "Rank, top-5, equal weight (inverse-volatility as the comparison allocator), long-only. No-trade state: SPY 21-day vol percentile vs its trailing 126 sessions >= 0.80 -> 100% BIL."],
     ["Constraints and risk", "Leverage <= 1, no shorting, max weight 20%, checked inside the simulation. Overlay: 10% annual volatility target from 63-session returns, remainder in BIL."],
     ["Cost-model class", "$0.0035/share commission + tiered half-spread ($0.005 SPY/QQQ, $0.01 sector XL*, $0.02 other); sweep 0-8x; stress 3x spreads."],
     ["Version", "setup_version: v1"]], [3.6, 13.4])
tbl([["Protocol commitment", "Value"]] + C("protocol.csv").values.tolist(), [5.0, 12.0])
p(f"<b>Why the holdout is 2026.</b> Earlier versions of this project had already looked at 2024 and 2025 outcomes, so those years cannot be an honest final test; they are development data here. The holdout is 2026-01-02..2026-10-30, "
  "downloaded from Yahoo with the textbook's own source; no one on the team had looked at it when setup.yaml was committed, and the loader refuses to read it before the freeze. "
  "<b>Engine.</b> A custom monthly event loop (src/lib.py: simulate): the gate, the volatility overlay and the constraints use information at the close of t and change the next action. Fills are at the next open; a fill on a missing bar raises an error. "
  f"<b>Search accounting.</b> The research ledger (Appendix A) was opened by 01_setup before any backtest; run_log.csv records git commit, data snapshot, configuration, window, metric and decision for every run. "
  f"This proposal tried {fz['trial_counts']['variants']} strategy variants; including the project's three earlier versions the inclusive count is {fz['trial_counts']['inclusive']}.")
wcb("The 23-ETF list was chosen by the team after earlier versions had shown 2020-2025 results, a hindsight risk in the universe that the protocol cannot remove. The holdout has only ten monthly decisions.")

# ===================================================================== Section 3
sec("3. Data: Sources, Point-in-Time Pipeline, Quality Report")
p("<b>Route A (textbook ETF universe, extended).</b> All 27 symbols are in the textbook's ETF dataset; we re-downloaded them from Yahoo (yfinance) for 2019-01-02..2026-10-30 so that a never-seen 2026 holdout exists. "
  "<b>Timestamp semantics:</b> bar date = NYSE session; close known at 16:00 New York, open at 09:30. <b>Corporate actions:</b> split- and dividend-adjusted OHLCV for returns and fills; split-adjusted, dividend-unadjusted close for dollar volume and per-share costs. "
  "<b>Identifiers:</b> ticker = key; no ticker changes in the period. <b>Storage:</b> data/prices_adjusted.parquet and data/prices_raw.parquet, rebuilt by data/build_dataset.py (download time in data/download_meta.json).")
sc = dq["source_check_vs_textbook"]
tbl([["Quality check (development 2019-2025)", "Result"],
     ["Symbols x sessions; duplicate symbol-dates", f"{dq['symbols']} x {dq['sessions']}; {dq['duplicate_symbol_dates']}"],
     ["Missing values (open/high/low/close/volume)", " / ".join(str(v) for v in dq["missing_values"].values()) + " (no forward-fill; the simulator refuses to fill on a missing bar)"],
     ["OHLC invariant violations", ", ".join(f"{k} {v}" for k, v in dq["ohlc_violations"].items())],
     ["Daily moves > 15%", f"{len(dq['daily_moves_gt_15pct'])}, all in March-November 2020 except SOXX +18.6% on 2025-04-09 (tariff pause); verified as real and kept"],
     ["Source consistency", f"Daily adjusted returns vs the textbook's Yahoo bars, {sc['overlap_rows']:,} overlapping rows: correlation {sc['return_correlation']:.8f}, share differing by > 10 bp {sc['share_abs_diff_gt_10bp']:.1%}"],
     ["Corporate-action check", f"USO 1:8 reverse split 2020-04-29: largest unadjusted daily change {dq['corporate_action_check']['max_abs_daily_change_unadjusted']:.0%}, adjusted {dq['corporate_action_check']['max_abs_daily_change_adjusted']:.0%} (Figure D2)"],
     ["Survivorship", "All 23 names trade through 2026: a hand-picked survivor list (bias disclosed)"],
     ["Coverage", f"{cov.eligible.min()}-{cov.eligible.max()} of 23 eligible at every decision; highest pairwise return correlations: " + ", ".join(f"{k} {v}" for k, v in list(dq['universe']['highest_pair_correlations_2020_2025'].items())[:4])]], [5.5, 11.5])
tbl([["PIT level", "Check", "Result"]] + pit.values.tolist(), [2.2, 7.6, 7.2])
fig("d2_corporate_action.png", 12.5, "Figure D2. Q: Are corporate actions handled? The unadjusted series shows the 1:8 reverse split as an 8x jump; the adjusted series used here is continuous.")
wcb("Yahoo revises adjusted history, so a later download can differ slightly (download time recorded). Because the dividend-unadjusted close is split-adjusted, USO's per-share cost before its 2020 split is understated; no validation trade falls in that period.")

# ===================================================================== Section 4
sec("4. Labels, Features, and Triage")
p(f"<b>Label.</b> Fixed-horizon continuous return fwd_oo_21 = open[t+22]/open[t+1] - 1, anchored to the next-open fill (a close-to-close label would credit the overnight gap the strategy cannot earn). "
  f"Models learn its cross-sectional percentile. <b>Overlap:</b> {NROWS:,} nominal labelled rows over {NDATES:,} sessions, but with a 21-session horizon only about {NDATES // 21} non-overlapping dates exist, each with just 23 names; every IC interval uses Newey-West with lag 21.")
tbl([["Feature", "Role", "Driver hypothesis", "Lookback", "Representation", "Known failure mode"]] + ft.values.tolist(), [2.6, 1.1, 4.2, 2.2, 3.4, 3.5], fs=6.6)
p(f"<b>Searched set and de-duplication.</b> {fsel['searched_set']} signals + 4 states. Correlation clustering (|Spearman| >= 0.80, pre-declared priority = table order) merged: "
  + ", ".join(f"{k} -> {v}" for k, v in fsel["merged"].items()) + ".")
t = tri.dropna(subset=["ic"]).copy()
tbl([["Feature", "IC [NW CI]", "ICIR", "Same sign", "IC h5/h21/h63", "Placebo", "Monot.", "Q5-Q1 bp/mo", "BH p", "Gates G1/G2/G3/G4/G5", "Decision"]] +
    [[r.feature, f"{f3(r.ic)} [{f3(r.ic_lo)},{f3(r.ic_hi)}]", f2(r.icir_blocks), f"{r.share_blocks_same_sign:.0%}", f"{r.ic_h5:+.2f}/{r.ic_h21:+.2f}/{r.ic_h63:+.2f}", f"{r.placebo_ic_lag21:+.3f}",
      f"{r.quintile_monotonicity:+.1f}", f"{r.q5_q1_spread_bps_month:.0f}", f"{r.p_bh:.2f}", f"{r.g1_correctness[:4]}/{r.g2_association}/{r.g3_shape[:4]}/{r.g4_feasibility}/{r.g5_bh_5pct}", r.decision]
     for r in t.itertuples()], [2.4, 2.6, 0.9, 1.1, 2.0, 1.1, 0.9, 1.3, 0.8, 2.9, 1.2], fs=6.2)
p(f"Gates: G1 correctness (observable at t by construction; timing placebo = IC of the feature lagged 21 sessions; 'slow' flags a slow state proxy whose IC survives the shift); G2 IC with NW CI, ICIR over twelve half-year blocks, share of blocks with the pooled sign, decay over 5/21/63-day horizons; "
  f"G3 quintile monotonicity (|rho| >= 0.9); G4 sign-aligned Q5-Q1 spread must exceed 2x the median round trip ({fsel['round_trip_cost_bps']:.1f} bp); G5 Benjamini-Hochberg at 5% over the searched set. "
  f"Rule: PROCEED if G2, G3, G4, G5 pass; STOP if G2 fails with |t| < 1 or merged; otherwise REVISE. Result: {TRI.get('PROCEED', 0)} PROCEED, {TRI.get('REVISE', 0)} REVISE, {TRI.get('STOP', 0)} STOP. "
  f"No feature survives BH; the two REVISE signals ({', '.join(x for x in fsel['model_features'] if x in t.feature.values)}) plus the four states enter the models. The momentum rule is kept as the pre-registered baseline regardless of its triage outcome.")
tbl([["Top-3 feature", "IC", "Timing placebo", "Shared driver controlled", "IC after control", "IC low-vol tercile", "IC high-vol tercile", "Event time"]] +
    [[r.feature, f3(r.ic), f3(r.placebo_lag21_ic), r.shared_driver_control, f3(r.ic_after_control), f3(r.ic_low_vol_tercile), f3(r.ic_high_vol_tercile), "N/A"] for r in mchk.itertuples()],
    [2.8, 1.3, 1.8, 2.6, 1.8, 2.0, 2.0, 1.4], fs=6.8)
fig("d3_dag.png", 11.5, "Figure D3. DAG used for the top features: market beta / volatility regime is a confounder (controlled in the shared-driver check); flows are the mediator (not controlled); fund AUM growth is a collider (never controlled).")
mc = mchk.set_index("feature")
p(f"<b>Reading.</b> The strongest feature is market beta (IC {f3(mc.loc['beta_spy_63','ic'])}), a risk-compensation variable, not trend: its IC survives the 21-day timing placebo (a slow state proxy), survives a volatility control ({f3(mc.loc['beta_spy_63','ic_after_control'])}) and is larger in high-volatility terciles. "
  f"The 21-day return's IC disappears once beta is controlled ({f3(mc.loc['ret_21','ic_after_control'])}). Momentum's IC is negative in calm states and positive only in volatile ones (low-vol {f3(mech['mom_126_vs_ic_low_vol'])}, high-vol {f3(mech['mom_126_vs_ic_high_vol'])}) - the reverse of what the gate assumes.")
wcb("Triage used the whole development period, including the validation folds, so feature screening is not out-of-sample (only STOP removes features, which limits the leak). Beta's positive IC is a risk-on effect of 2020-2025 and can reverse in a bear market.")

# ===================================================================== Section 5
sec("5. Models, Validation, Interpretation, Uncertainty")
p("<b>Ladder (same features, folds, purge).</b> (1) Non-ML rule: vol-scaled 126-day momentum. (2) Ridge, alpha 1e-2..1e5 (eight values); signals enter as per-date percentile ranks, states are winsorised (1/99) and standardised with training-fold statistics only. "
  f"(3) LightGBM, L1 loss, depth <= 4, 7/15 leaves x 300/1000 minimum leaf size, early stopping (50 rounds) on a purged inner split (last 126 training sessions). (4) Ensemble: mean of the three rungs' per-date z-scores. "
  f"Selection by the yardstick (mean daily Spearman IC, NW 95% CI); hyperparameters never touched the holdout.")
tbl([["Fold", "Validation", "Training from", "Last training label ends", "Training rows", "Scored rows"]] + folds.values.tolist(), [1.0, 4.0, 2.6, 3.6, 2.6, 2.6])
rows = [["Model", "Validation IC [NW 95% CI]", "t", "ICIR", "Folds > 0"]]
for k in [sel["chosen"]["rule"], *[x for x in vic.index if x.startswith("ridge")][::3], RID, *[x for x in vic.index if x.startswith("lgb")], "ensemble"]:
    r = vic.loc[k]
    rows.append([k + (" (chosen)" if k in sel["chosen"].values() else ""), f"{f3(r.ic)} [{f3(r.lo)}, {f3(r.hi)}]", f2(r.t), f2(r.icir), f"{int(r.folds_positive)}/{NF}"])
tbl(rows, [5.2, 5.0, 1.6, 1.6, 1.8], fs=7)
tbl([["Paired comparison (daily IC difference)", "Delta IC", "NW 95% CI", "t"]] + [[r.comparison, f3(r.delta_ic), f"[{f3(r.lo)}, {f3(r.hi)}]", f2(r.t)] for r in pair.itertuples()], [8.4, 2.0, 4.0, 1.6])
it = list(sel["lgb_best_iters"].values())
p(f"No rung's IC interval excludes zero, so by the course yardstick <b>no signal is established</b>. The pre-registered rule deploys the highest-IC rung, which is the <b>non-ML momentum rule</b>: Ridge ({RID}, {f3(vic.loc[RID,'ic'])}) and LightGBM ({LGB}, {f3(vic.loc[LGB,'ic'])}) are both lower, "
  f"and every LightGBM configuration has a negative IC (two significantly so). Early stopping chose {min(it)}-{max(it)} trees across folds, a symptom of very low signal-to-noise. ML adds nothing on this universe.")
fig("m1_fold_ic.png", 13, f"Figure M1. Q: Is the signal stable across folds? Per-fold IC with Newey-West bands; no rung is positive in more than {int(vic.folds_positive.max())} of {NF} folds and every band spans zero in most folds.")
fig("m2_shap_beeswarm.png", 11.5, f"Figure 5 (required). Q: Does the ML candidate use features as the mechanism predicts? TreeSHAP for LightGBM ({LGB}), last fold. The deployed rule uses one feature (vol-scaled momentum), so its attribution is trivial.")
st_ = shp.sort_values("mean_abs_shap", ascending=False)
p(f"<b>SHAP reading (four layers).</b> Magnitude: {st_.index[0]} and {st_.index[1]} carry the largest attribution; sign consistency: {st_.index[1]} has the same sign in {int(st_.loc[st_.index[1],'folds_same_sign'])}/{NF} folds (high volatility lowers the score) while beta keeps its sign in only {int(st_.loc['beta_spy_63','folds_same_sign'])}/{NF}; "
  f"stability: mean rank correlation of attributions across folds {sel['shap_rank_stability']:.2f}. <b>Surprising attribution:</b> the trees split on market-wide state variables that are identical for all ETFs on a date, so they cannot rank ETFs by themselves - LightGBM is fitting time variation, not the cross-section, which explains its negative IC. SHAP is descriptive, not causal (Ch. 15).")
tbl([["Fold", "80% half-width (rank)", "Coverage 80%", "90% half-width", "Coverage 90%", "Flag < 80%"]] + [[int(r.fold), f"{r.halfwidth_80:.3f}", pct(r.coverage_80), f"{r.halfwidth_90:.3f}", pct(r.coverage_90), "yes" if r.flag_below_80 else "no"] for r in conf.itertuples()], [1.2, 3.4, 2.6, 3.0, 2.6, 2.2])
p(f"Conformal intervals on the deployed score's cross-sectional rank (calibrated on the previous fold) are roughly calibrated ({int(conf.flag_below_80.sum())} of {len(conf)} folds flagged below 80%) but uninformative: a half-width of about {conf.halfwidth_80.median():.2f} on a -0.5..+0.5 scale covers most of the range, i.e. the score barely narrows an ETF's expected rank.")
wcb("With IC intervals spanning zero in validation and holdout, the evidence cannot distinguish prediction-quality drift from noise. A cross-section of 23 correlated ETFs gives few independent bets per date.")

# ===================================================================== Section 6
sec("6. Backtest Protocol, Baseline, Portfolio, Costs, Risk")
tbl([["Component", "Specification"], ["Signal timing", "Month-end close signal, next-open fill (no same-bar fills)"], ["Rebalance and holding", "Monthly; holding = 21-session label horizon"],
     ["Position sizing", "Equal weight top-5 (20% cap); inverse-vol comparison"], ["Fill assumptions", "Adjusted open; no fills on missing bars (assertion)"],
     ["Constraints", "Long-only, leverage <= 1, cash in BIL; asserted inside the loop"], ["Costs", "Per-share commission + tiered half-spread per instrument; 0-8x sweep; 3x spread stress"]], [4.0, 13.0])
rows = [[f"{VL} (net, excess of BIL)", "Sharpe", "CAGR", "Vol", "Max DD", "Turnover 2-way/yr", "Avg exposure"]]
for k, l in ((DEP, "Deployed = rule + gate + vol target"), (G_RAW, "Rule + gate, no overlay"), (NOG_VT, "Rule, no gate (+vol target)"), (NOG_RAW, "Rule, no gate, no overlay"),
             (f"{RID}|ew|gate|vt", f"Ridge ({RID}) + gate + vol target"), (f"{LGB}|ew|gate|vt", f"LightGBM ({LGB}) + gate + vol target"), ("ensemble|ew|gate|vt", "Ensemble + gate + vol target"),
             ("bench_ew_universe", "Benchmark: equal-weight 23 ETFs"), ("bench_6040", "Benchmark: 60/40 SPY/AGG"), ("bench_spy", "Benchmark: SPY")):
    r = vb.loc[k]
    rows.append([l, f2(r.sharpe), pct(r.cagr), pct(r.vol), pct(r.max_dd), f"{r.turnover_2way_ann:.1f}x", pct(r.avg_exposure)])
tbl(rows, [5.8, 1.5, 1.5, 1.5, 1.7, 2.6, 2.4])
be = vs["breakeven"][DEP]
x = rs['breakeven_raw'][0] / be['assumed_bps_per_leg']
p(f"<b>Costs.</b> Assumed cost {be['assumed_bps_per_leg']:.1f} bp per leg; average holding {be['avg_holding_days']:.0f} days. Break-even c* = gross excess return / two-way turnover: {rs['breakeven_raw'][0]:.0f} bp per leg before the overlay and {be['breakeven_bps_per_leg']:.0f} bp with it "
  f"({x:.0f}x the assumed cost: {'comfortable zone' if x >= 5 else ('gray zone' if x >= 2 else 'hard floor')}). Costs are not what limits this strategy on validation; the signal is.")
fig("b1_cost.png", 13, "Figure B1. Q: How sensitive is the result to costs? Gross vs net and a 0-8x sweep (validation); the deployed strategy and the rule coincide.")
al = vs["allocators"]
p(f"<b>Allocator (same signal, same 63-day window).</b> Equal weight: Sharpe {f2(al[0]['sharpe'])}, MDD {pct(al[0]['max_dd'])}, turnover {al[0]['turnover_2way_ann']:.1f}x; inverse-vol: {f2(al[1]['sharpe'])}, {pct(al[1]['max_dd'])}, {al[1]['turnover_2way_ann']:.1f}x. "
  f"Best-worst spread {vs['allocator_spread_sharpe']:.3f} Sharpe - far below the textbook's median 0.49; with five similar-volatility sector ETFs the allocator does not matter.")
tbl([["Risk control", "Proxy", "Control", "Frequency"], ["Market", "Beta to SPY (rolling 126d; full-period HAC CI)", "Vol-state gate + 10% vol target", "Monthly decision, daily monitor"],
     ["Factor / overlap", "Count of tech-heavy names (QQQ, XLK, SOXX) held together", "20% cap per name; disclosed overlap", "Monthly"], ["Leverage", "Gross exposure", "<= 1.0 asserted in the simulation", "Every rebalance"],
     ["Concentration", "HHI / effective N", "Max weight 20%", "Monthly"], ["Liquidity", "Trailing median $ volume", ">= $10M eligibility; ETF trade sizes << ADV", "Monthly"],
     ["Model", "Rolling 63-day IC", "Pause if 6-month IC < 0 (Section 8)", "Monthly"], ["Operational", "Data checks; code hash vs freeze", "Halt on unexplained nulls or hash mismatch", "Daily"]], [2.4, 5.0, 5.6, 4.0])
d = og[DEP]
p(f"<b>Tail and path risk.</b> Daily VaR95 / CVaR95 {pct(d['var95'])} / {pct(d['cvar95'])}; by state: high-vol {pct(vs['regime_tail']['vol_state_high(>=0.5)']['var95'])} / {pct(vs['regime_tail']['vol_state_high(>=0.5)']['cvar95'])}, low-vol {pct(vs['regime_tail']['vol_state_low(<0.5)']['var95'])} / {pct(vs['regime_tail']['vol_state_low(<0.5)']['cvar95'])}. "
  f"Max drawdown {pct(d['max_dd'])}, {d['dd_duration_sessions']} sessions peak-to-trough, recovered after {d['recovery_sessions']} sessions. Beta to SPY {vs['beta_full']['beta']:.2f} [{vs['beta_full']['lo']:.2f}, {vs['beta_full']['hi']:.2f}] (HAC): the strategy is mostly cash and a little equity.")
st = vs["stress_3x_spreads"]
tbl([["Stress, spreads 3x: return / max DD"] + list(st)] + [[l] + [f"{pct(st[w][k]['return'])} / {pct(st[w][k]['max_dd'])}" for w in st] for k, l in ((DEP, "Deployed (= rule)"), ("bench_ew_universe", "EW universe"), ("bench_spy", "SPY"))], [4.4, 4.2, 4.2, 4.2])
p(f"<b>Overlay and gate, with and without.</b> Overlay: {f2(og[G_RAW]['sharpe'])} -> {f2(og[DEP]['sharpe'])} Sharpe, MDD {pct(og[G_RAW]['max_dd'])} -> {pct(og[DEP]['max_dd'])}, at an average exposure of only {pct(og[DEP]['avg_exposure'])}. "
  f"Gate on the rule (no overlay): {f2(og[NOG_RAW]['sharpe'])} -> {f2(og[G_RAW]['sharpe'])} - <b>the gate lowers Sharpe</b> while it cuts drawdown ({pct(og[NOG_RAW]['max_dd'])} -> {pct(og[G_RAW]['max_dd'])}); its value is defensive, as the stress table shows, not a better signal. "
  f"<b>Kill-switch ladder:</b> -5% watch (PM), -10% halve (PM), -15% halt to BIL (risk owner), -20% unwind (team); on validation only the -5% level triggered (first {vs['kill_switch_first_trigger']['-5% watch']}).")
inf = vs["inference"][DEP]
p(f"<b>Statistical inference.</b> Deployed validation Sharpe {f2(inf['sharpe'])}, analytic CI [{inf['ci_analytic'][0]:.2f}, {inf['ci_analytic'][1]:.2f}], bootstrap [{inf['ci_bootstrap'][0]:.2f}, {inf['ci_bootstrap'][1]:.2f}]; "
  f"Deflated Sharpe {inf['dsr']:.2f} with N={fz['trial_counts']['variants']} (expected max Sharpe of zero-skill trials {inf['expected_max_sharpe']:.2f}) and {inf['dsr_inclusive']:.2f} with N={fz['trial_counts']['inclusive']}; "
  f"minimum track record to confirm this Sharpe: {inf['min_track_record_years']:.0f} years. Versus the equal-weight universe: {f3(vs['inference']['paired_deployed_vs_ew']['diff'])} [{f3(vs['inference']['paired_deployed_vs_ew']['lo'])}, {f3(vs['inference']['paired_deployed_vs_ew']['hi'])}].")
wcb("The gate and stops act at month-end only, so an intra-month crash is not cut; the 3x-spread stress keeps linear costs; BIL as cash ignores its own small price risk.")

# ===================================================================== Section 7
sec("7. Results, Diagnosis, and Next Iteration")
tbl([list(head.columns)] + head.values.tolist(), [2.6, 4.0, 3.6, 3.6, 3.2], fs=6.8)
p(f"<b>Outcome pattern (Ch. 20): {rs['pattern']}.</b> Confirmatory test: deployed minus equal-weight holdout Sharpe {f3(ct['diff'])} [{f3(ct['lo'])}, {f3(ct['hi'])}], p = {ct['p']:.2f} -> <b>not passed</b>. "
  f"Secondary: holdout IC {f3(vh['mean'])} [{f3(vh['lo'])}, {f3(vh['hi'])}] -> not passed. Versus SPY {f3(hs['deployed_vs_spy']['diff'])} [{f3(hs['deployed_vs_spy']['lo'])}, {f3(hs['deployed_vs_spy']['hi'])}]. "
  f"The gate was active in {hs['gated_months']} of {NDEC} holdout decisions. ML comparisons on the holdout (frozen, reported, not selected): LightGBM {f2(hb.loc[f'{LGB}|ew|gate|vt','sharpe'])}, ensemble {f2(hb.loc['ensemble|ew|gate|vt','sharpe'])}, rule without gate {f2(hb.loc[NOG_VT,'sharpe'])}.")
fig("r1_equity.png", 14, "Figure 1. Q: Does the strategy beat the baseline and the passive benchmark after costs? Answer: no - the deployed strategy is the baseline, and it trails the equal-weight universe and SPY in validation and in the 2026 holdout; flat stretches are gated months in BIL.")
fig("r2_rolling_ic.png", 14, "Figure 2. Q: Is the ranking skill stable? Answer: no - the rolling IC swings between about -0.27 and +0.35 and turns negative within weeks of the holdout start.")
fig("r3_gross_net.png", 12, "Figure 3. Q: How much do costs take? Answer: little - gross and net Sharpe are close; the weak signal, not costs, limits the strategy.")
fig("r4_regime.png", 13, "Figure 4. Q: Where does the strategy work? Answer: only in low-volatility up-trend months on validation (point-in-time thresholds); it loses in down-trends and is flat-to-negative in every holdout regime.")
fig("r6_drawdown.png", 13, "Figure 6. Q: Would the kill-switch ladder have triggered? Answer: only the -5% watch level, several times in validation and once in mid-2026; the overlay keeps drawdowns small because exposure averages under 50%.")
p(f"<b>Diagnosis.</b> The strategy lost value at the <b>signal gate</b>: no rung established IC, momentum's IC is near zero overall and negative in calm states, and the holdout IC is {f3(vh['mean'])}. "
  f"2026 rewarded holding all sectors at once (equal-weight Sharpe {f2(hb.loc['bench_ew_universe','sharpe'])}, CAGR {pct(hb.loc['bench_ew_universe','cagr'])}); a top-5 rotation that is about half in cash cannot keep up in such a year. "
  f"Drift diagnosis: prediction quality fell from {f3(vic.loc[M,'ic'])} to {f3(vh['mean'])} but both intervals span zero, and portfolio translation is stable (gross {f2(hb.loc[DEP+'|gross','sharpe'])} vs net {f2(hb.loc[DEP,'sharpe'])}), so the gap is noise and regime, not a pipeline defect. "
  f"Holdout break-even {rs['breakeven_raw'][1]:.0f} bp per leg vs ~{be['assumed_bps_per_leg']:.1f} assumed: costs became material only because gross return collapsed. "
  "<b>Binding constraint:</b> label richness - 23 correlated names, monthly decisions and about 70 independent dates cannot resolve an IC of a few hundredths.")
p("<b>One next step.</b> Keep the 23-ETF universe but lengthen the evidence before any new idea: extend the walk-forward backwards with the textbook's 2006-2019 history for these ETFs (about 25 more half-year folds) and re-test the momentum rule without the gate, "
  "since the gate was the part the data contradicted most clearly. A new holdout (2027 data) would be required. None of these first-pass strategies goes to capital.")
wcb(f"Ten holdout decisions; a single month moves the holdout Sharpe by several tenths. The holdout summary was computed by 06b from the saved single-touch artifacts after 06 failed at its final write step (Appendix A).")

# ===================================================================== Section 8
sec("8. Monitoring and Deployment Plan (design, not built)")
p("<b>Parity.</b> The same src/lib.py code path would run live in shadow mode for three months; compare stage by stage (raw bars, features, scores, target weights, orders, fills) against a backtest replay of the same days, and alert on any mismatch beyond rounding. "
  "<b>Monitoring.</b> Rolling 30/60/90-day Sharpe, drawdown, hit rate and daily IC; realisation ratio = live / backtest return over the same window (investigate below 0.5). "
  "<b>Drift.</b> PSI on the score input (vol-scaled 126-day momentum) and on the two risk features (beta, 63-day vol): watch 0.10-0.25, retrain above 0.25; monthly check of the top-5 holdings' overlap with the tech cluster (QQQ/XLK/SOXX). "
  "<b>Lifecycle.</b> Re-check every six months on the expanding window with the frozen protocol; pause if the 6-month IC is negative or the -15% kill-switch level is hit; retire if the pause lasts two consecutive reviews or the confirmatory test fails on a second fresh year. "
  "<b>Bugs vs decay vs costs (Ch. 26).</b> A parity mismatch or data-check failure is a bug (fix, rerun); falling IC with clean parity is statistical decay (retrain or retire); stable IC with falling net returns and rising fill slippage is cost inflation (slow the cadence or tighten the universe).")
wcb("Shadow mode cannot reveal capacity or market-impact problems; PSI thresholds are textbook defaults, not calibrated on this strategy.")

# ===================================================================== Appendix
sec("Appendix A. Research Ledger, Run Log, Repository Map, Checklist")
p("<b>A.1 Research ledger</b> (ledger/research_ledger.md, verbatim excerpt).", H2)
for line in open(f"{R}/ledger/research_ledger.md").read().split("\n"):
    if line.strip():
        p(line.replace("**", "").replace("###", "&bull;").replace("<", "&lt;"), CAP)
rl = pd.read_csv(f"{R}/run_log.csv")
p("<b>A.2 Run log</b> (run_log.csv): " + ", ".join(f"{k}: {v} rows" for k, v in rl.trial_family.value_counts().items()) +
  f". Each row stores timestamp, git commit, data snapshot hash ({rl.data_snapshot.iloc[0]}), setup version, trial family, trial, run, evaluation window, configuration, metric and decision. Rows repeated by the 04 bug-fix re-execution are identified in the ledger.", B)
p("<b>A.3 Repository map</b>", H2)
tbl([["Path", "Content"], ["README.md", "How to reproduce from the root; environment; why numbers could differ"], ["setup.yaml", "Trading setup v1 and protocol (Section 2)"],
     ["data/build_dataset.py", "Downloads data/prices_*.parquet from Yahoo (yfinance)"], ["notebooks/01_setup ... 07_results (.py and .ipynb)", "One stage each; 06 refuses to run twice; 06b summarises its saved output"],
     ["src/lib.py", "Sealed loader, PIT universe, features, statistics, simulator, run log"], ["results/, figures/", "Every number and figure in this document"],
     ["run_log.csv, ledger/research_ledger.md", "Run log and research ledger"], ["requirements.txt, requirements.lock.txt", "Environment (textbook uv env, Python 3.14)"],
     ["report/, slides/", "This PDF and the presentation, built by report/build_report.py and slides/build_slides.py"]], [6.5, 10.5])
p("<b>A.4 Pre-submission checklist (ten questions)</b>", H2)
tbl([["#", "Question", "Answer"],
     ["1", "Time each feature became knowable?", "Yes: close of t for all features (prices/volume only)"],
     ["2", "Universe formed at t, leavers included?", "Eligibility at t: yes. Fixed survivor list: disclosed FAIL"],
     ["3", "Label price convention matches fills?", "Yes: next open to open+21"],
     ["4", "Scalers/winsorisation fitted on training folds only?", "Yes (states); signals use per-date ranks"],
     ["5", "Buffers in trading days, every fold?", "Yes: 21-session purge; embargo not needed (expanding, forward)"],
     ["6", "Holdout touched once, after the freeze?", f"Yes: freeze {fz['frozen_at']}, single touch recorded in results/holdout_touched.json (2026 data never seen before)"],
     ["7", "Ledger counts every adaptive choice; DSR uses it?", f"Yes: N={fz['trial_counts']['variants']} (this setup) and {fz['trial_counts']['inclusive']} (inclusive)"],
     ["8", "Per-instrument costs with sweep and break-even?", "Yes: commission + tiered spread; 0-8x; break-even reported"],
     ["9", "Constraints and stops inside the simulation, t-1 info?", "Yes: asserted in the loop; gate/overlay use the close of t, trade at t+1 open"],
     ["10", "Dataset and headline table regenerate from the root?", "Yes: see README; LightGBM seed 42, bootstrap seed 7; Yahoo revisions disclosed"]], [0.8, 7.6, 8.6])

doc = SimpleDocTemplate(f"{R}/report/ML4T_Midterm_Proposal.pdf", pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.5 * cm, bottomMargin=1.5 * cm, title="ML4T Midterm Proposal")
doc.build(S, onFirstPage=lambda c, d: None, onLaterPages=lambda c, d: (c.setFont("Helvetica", 7), c.drawRightString(A4[0] - 1.8 * cm, 0.9 * cm, f"{d.page}")))
print("built")
