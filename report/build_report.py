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
_pd = pd.read_parquet(f"{R}/data/panel_dev.parquet")
_pd = _pd[_pd.label_end <= "2024-12-31"]
NROWS, NDATES = len(_pd), _pd.date.nunique()
cv = cov[pd.to_datetime(cov.date) >= "2021-12-31"]
# ===================================================================== title page content + Section 1
p("<font size=16><b>ETF Momentum with a Volatility-State Gate: an ML4T Midterm Proposal</b></font>", ss["Title"])
p(f"Setup v1 &nbsp;|&nbsp; Textbook ETF dataset, 2020-2025 &nbsp;|&nbsp; Development 2020-07..2024-12, sealed holdout 2025 (touched once) &nbsp;|&nbsp; Repository: ml4t-midterm-v3 (git {fz['git_commit_before_freeze']} at freeze)", CAP)
p(f"<b>Abstract.</b> We test whether cross-asset ETF trends, traded only outside high-volatility states, predict next-month returns, and whether Ridge or LightGBM improves on a momentum rule. "
  f"Under a pre-registered walk-forward protocol the selected model (Ridge, alpha 1e5) had validation IC {f3(vic.loc[M,'ic'])} [{f3(vic.loc[M,'lo'])}, {f3(vic.loc[M,'hi'])}] and a net excess Sharpe of "
  f"{f2(vb.loc[DEP,'sharpe'])}; the momentum rule's IC was zero and the gate made the rule worse, falsifying our mechanism on validation. On the 2025 holdout the deployed strategy returned an excess Sharpe of "
  f"{f2(hb.loc[DEP,'sharpe'])} {hs['deployed']['ci_bootstrap'][0]:.2f}..{hs['deployed']['ci_bootstrap'][1]:.2f}, but the pre-registered confirmatory test against the equal-weight universe failed "
  f"({f3(hs['confirmatory_test']['diff'])}, CI {f3(hs['confirmatory_test']['lo'])}..{f3(hs['confirmatory_test']['hi'])}). Outcome pattern: <b>never resolved</b>.", BOX)
sec("1. Hypothesis and Sources of Edge", brk=False)
p("<b>Mechanism (two sentences).</b> Cross-asset ETF trends persist over one to six months because investors and slow-moving institutional capital under-react to macro news and rebalance gradually. "
  "The same persistence crashes when volatility spikes and trends reverse abruptly, so the signal should be traded only in calm-to-normal volatility states.")
p("<b>Source of edge (Table 6.1).</b> Flow predictability (gradual rebalancing by benchmarked and risk-parity investors) and risk compensation (holding trending risk assets earns a premium that is lost in volatility shocks). "
  "<b>Counterparty:</b> investors who rebalance on calendar schedules or de-risk late.")
p("<b>Falsification criteria (fixed in setup.yaml before data analysis).</b> (1) The momentum rule's validation IC is not positive in the low-volatility state. (2) The gated rule does not beat the ungated rule on validation net Sharpe. "
  "(3) The pre-registered confirmatory test fails on the holdout: deployed holdout net excess Sharpe minus the equal-weight benchmark, 21-day block bootstrap, pass only if the 95% CI lower bound is above zero.")
p(f"<b>Status.</b> (1) not triggered (low-vol IC {f3(mech['mom_126_vs_ic_low_vol'])}, positive but not significant); (2) <b>triggered</b> (gated rule {f2(vb.loc[RULE,'sharpe'])} vs ungated {f2(vb.loc['rule_mom126vs|ew|nogate|vt','sharpe'])}); "
  f"(3) <b>triggered</b> (Section 7). The mechanism-first hypothesis is therefore not supported; the report documents where it failed.")
wcb("Five and a half years cover few regimes (post-COVID recovery, 2022 inflation shock, 2023-24 AI rally, 2025 tariff shock). A trend premium can be arbitraged away or crowded, and the gate threshold (80th percentile) was chosen a priori, not estimated.")

# ===================================================================== Section 2
sec("2. Trading Setup v1 and Evaluation Protocol")
tbl([["Setup field", "Specification (setup.yaml, committed before any backtest)"],
     ["Universe and eligibility", "100 textbook ETFs; point-in-time: trailing 126-session median dollar volume >= $10M and 126 sessions of history at t; then de-duplicate: walk by trailing $ volume, drop a name whose trailing 126-session return correlation with a kept name is >= 0.95. BIL is the cash proxy and is never ranked."],
     ["Decision schedule", "Snapshot at the month-end close; execute at the next session's open; hold to the next rebalance (about 21 sessions)."],
     ["Admissible information", "Prices and volume to the close of t; no fundamentals; longest lookback 147 sessions."],
     ["Score-to-trade mapping", "Rank, top-5, equal weight (inverse-volatility as the comparison allocator), long-only. No-trade state: SPY 21-day vol percentile vs its trailing 126 sessions >= 0.80 -> 100% BIL."],
     ["Constraints and risk", "Leverage <= 1, no shorting, max weight 20%, checked inside the simulation. Overlay: 10% annual volatility target from 63-session returns, remainder in BIL."],
     ["Cost-model class", "$0.0035/share commission + tiered half-spread ($0.005 SPY/QQQ/IWM/EFA/EEM, $0.01 sector XL*, $0.02 other); sweep 0-8x; stress 3x spreads."],
     ["Version", "setup_version: v1"]], [3.6, 13.4])
tbl([["Protocol commitment", "Value"]] + C("protocol.csv").values.tolist(), [5.0, 12.0])
p(f"<b>Engine.</b> A custom monthly event loop (src/lib.py: simulate): the gate, the volatility overlay and the constraints use information at the close of t and change the next action, so a vectorised engine would not suffice. "
  "Fills are at the next open; a fill on a missing bar raises an error. <b>Search accounting.</b> The research ledger (Appendix A) was opened by 01_setup before any backtest; run_log.csv records git commit, data snapshot, configuration, window, metric and decision for every run. "
  f"This proposal tried {fz['trial_counts']['v3_strategy_variants']} strategy variants; including earlier versions of the project (v1: 734 backtests, v2: 52 variants) the inclusive count is {fz['trial_counts']['inclusive_with_v1_v2']}.")
wcb("2025 market behaviour was seen once inside v1's holdout (different universe, model and protocol). No choice here used it, but the holdout is sealed procedurally rather than informationally. Six half-year folds give little power for model selection.")

# ===================================================================== Section 3
sec("3. Data: Sources, Point-in-Time Pipeline, Quality Report")
p("<b>Route A (textbook case study, etfs).</b> Starting setup: the textbook's case_studies/etfs (Yahoo daily OHLCV); changes: universe restricted to 2020-2025, point-in-time de-duplication added, label anchored to the next open. "
  "<b>Timestamp semantics:</b> bar date = NYSE session; close known at 16:00 New York, open at 09:30. <b>Corporate actions:</b> adjusted OHLCV for returns and fills, raw close for dollar volume and share-based costs. "
  "<b>Identifiers:</b> ticker = key; no ticker changes in the period. <b>Revisions:</b> none for exchange prices. <b>Storage:</b> data/prices_adjusted.parquet and data/prices_raw.parquet, regenerated exactly by data/build_dataset.py.")
tbl([["Quality check (development 2020-2024)", "Result"],
     ["Symbols x sessions; duplicate symbol-dates", f"{dq['symbols']} x {dq['sessions']}; {dq['duplicate_symbol_dates']}"],
     ["Missing values (open/high/low/close/volume)", " / ".join(str(v) for v in dq["missing_values"].values()) + " (no forward-fill; the simulator refuses to fill on a missing bar)"],
     ["OHLC invariant violations", ", ".join(f"{k} {v}" for k, v in dq["ohlc_violations"].items())],
     ["Daily moves > 15%", f"{len(dq['daily_moves_gt_15pct'])}, concentrated in March-April 2020 (e.g. USO -25% on 2020-04-21); verified as real and kept"],
     ["Corporate-action check", f"USO 1:8 reverse split 2020-04-29: largest raw daily change {dq['corporate_action_check']['max_abs_daily_change_raw']:.0%}, adjusted {dq['corporate_action_check']['max_abs_daily_change_adjusted']:.0%} (Figure D2)"],
     ["Survivorship", f"{dq['symbols_leaving_sample']} names leave the sample: the textbook list was chosen backward-looking (bias towards survivors, disclosed)"],
     ["Coverage (validation decisions)", f"{cv.eligible.min()}-{cv.eligible.max()} eligible, {cv.kept_after_dedup.min()}-{cv.kept_after_dedup.max()} after de-duplication; never eligible: {', '.join(dq['universe']['never_eligible'])}"]], [5.5, 11.5])
tbl([["PIT level", "Check", "Result"]] + pit.values.tolist(), [2.2, 7.6, 7.2])
fig("d1_coverage.png", 12.5, "Figure D1. Q: How many names can be traded? Eligible and de-duplicated ETFs per decision date (zero at 2020-06-30 because 126 sessions of history are required).")
fig("d2_corporate_action.png", 12.5, "Figure D2. Q: Are corporate actions handled? The raw series shows the 1:8 reverse split as an 8x jump; the adjusted series is continuous.")
wcb("The list of 100 ETFs is survivorship-biased; any ETF that closed is absent, which flatters the equal-weight benchmark and the strategies alike. Yahoo-derived adjusted prices can be silently revised.")

# ===================================================================== Section 4
sec("4. Labels, Features, and Triage")
p(f"<b>Label.</b> Fixed-horizon continuous return fwd_oo_21 = open[t+22]/open[t+1] - 1, anchored to the next-open fill (a close-to-close label would credit the overnight gap the strategy cannot earn). "
  f"Models learn its cross-sectional percentile. <b>Overlap:</b> {NROWS:,} nominal labelled rows over {NDATES:,} sessions, but with a 21-session horizon only about {NDATES // 21} non-overlapping dates exist (effective sample); every IC interval uses Newey-West with lag 21.")
tbl([["Feature", "Role", "Driver hypothesis", "Lookback", "Representation", "Known failure mode"]] + ft.values.tolist(), [2.6, 1.1, 4.2, 2.2, 3.4, 3.5], fs=6.6)
p(f"<b>Searched set and de-duplication.</b> {fsel['searched_set']} signals + 4 states. Correlation clustering (|Spearman| >= 0.80, pre-declared priority = table order) merged: "
  + ", ".join(f"{k} -> {v}" for k, v in fsel["merged"].items()) + ".")
t = tri.dropna(subset=["ic"]).copy()
tbl([["Feature", "IC [NW CI]", "ICIR", "Same sign", "IC h5/h21/h63", "Placebo", "Monot.", "Q5-Q1 bp/mo", "BH p", "Gates G1/G2/G3/G4/G5", "Decision"]] +
    [[r.feature, f"{f3(r.ic)} [{f3(r.ic_lo)},{f3(r.ic_hi)}]", f2(r.icir_blocks), f"{r.share_blocks_same_sign:.0%}", f"{r.ic_h5:+.2f}/{r.ic_h21:+.2f}/{r.ic_h63:+.2f}", f"{r.placebo_ic_lag21:+.3f}",
      f"{r.quintile_monotonicity:+.1f}", f"{r.q5_q1_spread_bps_month:.0f}", f"{r.p_bh:.2f}", f"{r.g1_correctness[:4]}/{r.g2_association}/{r.g3_shape[:4]}/{r.g4_feasibility}/{r.g5_bh_5pct}", r.decision]
     for r in t.itertuples()], [2.4, 2.6, 0.9, 1.1, 2.0, 1.1, 0.9, 1.3, 0.8, 2.9, 1.2], fs=6.2)
p(f"Gates: G1 correctness (observable at t by construction; timing placebo = IC of the feature lagged 21 sessions; 'slow' flags a slow state proxy whose IC survives the shift); G2 fold-level IC with NW CI, ICIR over nine half-year blocks, share of blocks with the pooled sign, decay over 5/21/63-day horizons; "
  f"G3 quintile monotonicity (|rho| >= 0.9); G4 sign-aligned Q5-Q1 spread must exceed 2x the median round trip ({fsel['round_trip_cost_bps']:.1f} bp); G5 Benjamini-Hochberg at 5% over the searched set. "
  f"Rule: PROCEED if G2, G3, G4, G5 pass; STOP if G2 fails with |t| < 1 or merged; otherwise REVISE. Result: {tri.decision.value_counts().to_dict()}. Only volume_chg_21_63 survives BH; non-STOP signals plus the four states enter the models.")
tbl([["Top-3 feature", "IC", "Timing placebo", "Shared driver controlled", "IC after control", "IC low-vol tercile", "IC high-vol tercile", "Event time"]] +
    [[r.feature, f3(r.ic), f3(r.placebo_lag21_ic), r.shared_driver_control, f3(r.ic_after_control), f3(r.ic_low_vol_tercile), f3(r.ic_high_vol_tercile), "N/A"] for r in mchk.itertuples()],
    [2.8, 1.3, 1.8, 2.6, 1.8, 2.0, 2.0, 1.4], fs=6.8)
fig("d3_dag.png", 11.5, "Figure D3. DAG used for the top features: market beta / volatility regime is a confounder (controlled in the shared-driver check); flows are the mediator (not controlled); fund AUM growth is a collider (never controlled).")
p("<b>Reading.</b> The two strongest features are risk-compensation variables (beta, volatility), not trend: their ICs survive the 21-day timing placebo (slow state proxies) and are larger in high-volatility terciles - the opposite of what a momentum-crash mechanism needs. "
  f"Volatility's IC disappears once beta is controlled ({f3(mchk.set_index('feature').loc['vol_63','ic_after_control'])}), so it is largely a beta proxy. Momentum's IC is small in both states (low-vol {f3(mech['mom_126_vs_ic_low_vol'])}, high-vol {f3(mech['mom_126_vs_ic_high_vol'])}).")
wcb("Triage used the whole development period, including the validation folds, so feature screening is not out-of-sample (only STOP removes features, which limits the leak). Beta's positive IC is a 2020-2024 risk-on effect and can reverse in a bear market.")

# ===================================================================== Section 5
sec("5. Models, Validation, Interpretation, Uncertainty")
p("<b>Ladder (same features, folds, purge).</b> (1) Non-ML rule: vol-scaled 126-day momentum. (2) Ridge, alpha 1e-2..1e5 (eight values, seven orders of magnitude); signals enter as per-date percentile ranks, states are winsorised (1/99) and standardised with training-fold statistics only. "
  f"(3) LightGBM, L1 loss, depth <= 4, 7/15 leaves x 300/1000 minimum leaf size, early stopping (50 rounds) on a purged inner split (last 126 training sessions). (4) Ensemble: mean of the three rungs' per-date z-scores. "
  f"Selection by the yardstick (mean daily Spearman IC, NW 95% CI); hyperparameters never touched the holdout.")
tbl([["Fold", "Validation", "Training from", "Last training label ends", "Training rows", "Scored rows"]] + folds.values.tolist(), [1.0, 4.0, 2.6, 3.6, 2.6, 2.6])
rows = [["Model", "Validation IC [NW 95% CI]", "t", "ICIR", "Folds > 0"]]
for k in [sel["chosen"]["rule"], *[x for x in vic.index if x.startswith("ridge")][::3], sel["chosen"]["ridge"], *[x for x in vic.index if x.startswith("lgb")], "ensemble"]:
    r = vic.loc[k]
    rows.append([k + (" (chosen)" if k in sel["chosen"].values() else ""), f"{f3(r.ic)} [{f3(r.lo)}, {f3(r.hi)}]", f2(r.t), f2(r.icir), f"{int(r.folds_positive)}/6"])
tbl(rows, [5.2, 5.0, 1.6, 1.6, 1.8], fs=7)
tbl([["Paired comparison (daily IC difference)", "Delta IC", "NW 95% CI", "t"]] + [[r.comparison, f3(r.delta_ic), f"[{f3(r.lo)}, {f3(r.hi)}]", f2(r.t)] for r in pair.itertuples()], [8.4, 2.0, 4.0, 1.6])
p(f"No rung's IC interval excludes zero, so by the course yardstick <b>no signal is established</b>. The pre-registered rule deploys the highest-IC rung, Ridge with alpha 1e5 - consistent with the textbook's observation that the ICIR peak appears at very large alpha. "
  "LightGBM does not beat Ridge (negative paired delta) and its early stopping chose few trees, a symptom of low signal-to-noise; the ensemble's dispersion across folds is lower (ICIR 0.99 vs 0.62) but its mean IC is not higher.")
fig("m1_fold_ic.png", 13, "Figure M1. Q: Is the signal stable across folds? Per-fold IC with Newey-West bands; Ridge, LightGBM and the ensemble are all negative in 2022H1 (the rate shock).")
fig("m2_shap_beeswarm.png", 11.5, "Figure 5 (required). Q: Does the deployed Ridge use features as the mechanism predicts? Linear SHAP, fold 2024H2.")
p(f"<b>SHAP reading (four layers).</b> Sign consistency: beta, volatility and momentum push scores up and volume change down in all six folds; magnitude: beta has twice momentum's attribution; stability: mean rank correlation of attributions across folds {sel['shap_rank_stability']:.2f}. "
  "<b>Surprising attribution:</b> the top driver is market beta, i.e. the model mostly ranks ETFs by equity risk, not by trend - a risk-compensation reading that contradicts the momentum mechanism. State variables have zero attribution because they are identical across ETFs on a date. SHAP is descriptive, not causal (Ch. 15).")
tbl([["Fold", "80% half-width (rank)", "Coverage 80%", "90% half-width", "Coverage 90%", "Flag < 80%"]] + [[int(r.fold), f"{r.halfwidth_80:.3f}", pct(r.coverage_80), f"{r.halfwidth_90:.3f}", pct(r.coverage_90), "yes" if r.flag_below_80 else "no"] for r in conf.itertuples()], [1.2, 3.4, 2.6, 3.0, 2.6, 2.2])
p("Conformal intervals on the predicted cross-sectional rank (calibrated on the previous fold) are well calibrated (fold 2 at 79.9% is flagged) but wide: +/-0.40 on a -0.5..+0.5 scale, i.e. the model barely narrows an ETF's expected rank.")
wcb("Validation and holdout disagree in level (Section 7); with IC intervals spanning zero in both, the evidence cannot distinguish prediction-quality drift from noise. Ridge at alpha 1e5 is close to a fixed linear blend of beta and volatility.")

# ===================================================================== Section 6
sec("6. Backtest Protocol, Baseline, Portfolio, Costs, Risk")
tbl([["Component", "Specification"], ["Signal timing", "Month-end close signal, next-open fill (no same-bar fills)"], ["Rebalance and holding", "Monthly; holding = 21-session label horizon"],
     ["Position sizing", "Equal weight top-5 (20% cap); inverse-vol comparison"], ["Fill assumptions", "Adjusted open; no fills on missing bars (assertion)"],
     ["Constraints", "Long-only, leverage <= 1, cash in BIL; asserted inside the loop"], ["Costs", "Per-share commission + tiered half-spread per instrument; 0-8x sweep; 3x spread stress"]], [4.0, 13.0])
rows = [["Validation 2022-2024 (net, excess of BIL)", "Sharpe", "CAGR", "Vol", "Max DD", "Turnover 2-way/yr", "Avg exposure"]]
for k, l in ((DEP, "Deployed: Ridge + gate + vol target"), (RAW, "Ridge + gate, no overlay"), (RULE, "Non-ML rule + gate + vol target"), ("rule_mom126vs|ew|nogate|vt", "Rule, no gate (+vol target)"),
             ("bench_ew_universe", "Benchmark: equal-weight eligible universe"), ("bench_6040", "Benchmark: 60/40 SPY/AGG"), ("bench_spy", "Benchmark: SPY")):
    r = vb.loc[k]
    rows.append([l, f2(r.sharpe), pct(r.cagr), pct(r.vol), pct(r.max_dd), f"{r.turnover_2way_ann:.1f}x", pct(r.avg_exposure)])
tbl(rows, [5.8, 1.5, 1.5, 1.5, 1.7, 2.6, 2.4])
be = vs["breakeven"][DEP]
p(f"<b>Costs.</b> Assumed cost {be['assumed_bps_per_leg']:.1f} bp per leg; average holding {be['avg_holding_days']:.0f} days. Break-even c* = gross excess return / two-way turnover: {rs['breakeven_raw'][0]:.0f} bp per leg before the overlay "
  f"({'hard floor' if rs['breakeven_raw'][0] / be['assumed_bps_per_leg'] < 2 else ('gray zone' if rs['breakeven_raw'][0] / be['assumed_bps_per_leg'] < 5 else 'comfortable zone')}, {rs['breakeven_raw'][0] / be['assumed_bps_per_leg']:.1f}x the assumed cost) and negative with the overlay, because the overlay's gross return ({pct(be['gross_cagr'])}) is below BIL ({pct(be['bil_cagr'])}): "
  "the strategy is in the hard-floor regime once risk control is added.")
fig("b1_cost.png", 13, "Figure B1. Q: How sensitive is the result to costs? Gross vs net and a 0-8x sweep (validation).")
al = vs["allocators"]
p(f"<b>Allocator (same signal, same 63-day window).</b> Equal weight: Sharpe {f2(al[0]['sharpe'])}, MDD {pct(al[0]['max_dd'])}, turnover {al[0]['turnover_2way_ann']:.1f}x, effective N {al[0]['eff_n']:.1f}; inverse-vol: {f2(al[1]['sharpe'])}, {pct(al[1]['max_dd'])}, {al[1]['turnover_2way_ann']:.1f}x, {al[1]['eff_n']:.1f}. "
  f"Best-worst spread {vs['allocator_spread_sharpe']:.2f} Sharpe - far below the textbook's median 0.49 and too small to matter; effective N averages below 5 because gated months hold only BIL.")
tbl([["Risk control", "Proxy", "Control", "Frequency"], ["Market", "Beta to SPY (rolling 126d; full-period HAC CI)", "Vol-state gate + 10% vol target", "Monthly decision, daily monitor"],
     ["Factor", "Asset-class concentration of the 5 holdings", "De-duplication (corr < 0.95), 20% cap", "Monthly"], ["Leverage", "Gross exposure", "<= 1.0 asserted in the simulation", "Every rebalance"],
     ["Concentration", "HHI / effective N", "Max weight 20%", "Monthly"], ["Liquidity", "Trailing median $ volume", ">= $10M eligibility; ETF trade sizes << ADV", "Monthly"],
     ["Model", "Rolling 63-day IC; SHAP rank", "Pause if 6-month IC < 0 (Section 8)", "Monthly"], ["Operational", "Data checks; code hash vs freeze", "Halt on unexplained nulls or hash mismatch", "Daily"]], [2.4, 5.0, 5.6, 4.0])
og = {r["strategy"]: r for r in vs["overlay_and_gate"]}
d = og[DEP]
p(f"<b>Tail and path risk.</b> Daily VaR95 / CVaR95 {pct(d['var95'])} / {pct(d['cvar95'])}; by state: high-vol {pct(vs['regime_tail']['vol_state_high(>=0.5)']['var95'])} / {pct(vs['regime_tail']['vol_state_high(>=0.5)']['cvar95'])}, low-vol {pct(vs['regime_tail']['vol_state_low(<0.5)']['var95'])} / {pct(vs['regime_tail']['vol_state_low(<0.5)']['cvar95'])}. "
  f"Max drawdown {pct(d['max_dd'])}, {d['dd_duration_sessions']} sessions peak-to-trough, not recovered by end of validation. Beta to SPY {vs['beta_full']['beta']:.2f} [{vs['beta_full']['lo']:.2f}, {vs['beta_full']['hi']:.2f}] (HAC).")
st = vs["stress_3x_spreads"]
tbl([["Stress, spreads 3x: return / max DD"] + list(st)] + [[l] + [f"{pct(st[w][k]['return'])} / {pct(st[w][k]['max_dd'])}" for w in st] for k, l in ((DEP, "Deployed"), (RULE, "Rule"), ("bench_ew_universe", "EW universe"), ("bench_spy", "SPY"))], [5.0, 6.0, 6.0])
p(f"<b>Overlay and gate, with and without.</b> Overlay: Ridge {f2(og[RAW]['sharpe'])} -> {f2(og[DEP]['sharpe'])} (MDD {pct(og[RAW]['max_dd'])} -> {pct(og[DEP]['max_dd'])}). Gate on the rule (no overlay): {f2(og['rule_mom126vs|ew|nogate|raw']['sharpe'])} -> {f2(og['rule_mom126vs|ew|gate|raw']['sharpe'])}. "
  "<b>The overlay halves drawdown but turns a small positive excess Sharpe negative</b> because it parks ~70% in BIL; the overlay does not redeem the signal and we say so plainly. "
  f"<b>Kill-switch ladder:</b> -5% watch (PM), -10% halve (PM), -15% halt to BIL (risk owner), -20% unwind (team); on validation only the -5% level triggered (first {vs['kill_switch_first_trigger']['-5% watch']}).")
inf = vs["inference"][DEP]
p(f"<b>Statistical inference.</b> Deployed validation Sharpe {f2(inf['sharpe'])}, analytic CI [{inf['ci_analytic'][0]:.2f}, {inf['ci_analytic'][1]:.2f}], bootstrap [{inf['ci_bootstrap'][0]:.2f}, {inf['ci_bootstrap'][1]:.2f}]; "
  f"Deflated Sharpe {inf['dsr_v3']:.2f} with N={fz['trial_counts']['v3_strategy_variants']} (expected max Sharpe of zero-skill trials {inf['expected_max_sharpe_v3']:.2f}) and {inf['dsr_inclusive']:.2f} with N={fz['trial_counts']['inclusive_with_v1_v2']}; minimum track record: infinite (non-positive Sharpe).")
wcb("Stops and the gate act at month-end only, so an intra-month crash is not cut; the 3x-spread stress keeps linear costs; BIL as cash ignores its own small price risk.")

# ===================================================================== Section 7
sec("7. Results, Diagnosis, and Next Iteration")
tbl([["Stage", "Metric", "Validation (2022-2024)", "Holdout (2025)", "Benchmark (val / holdout)"]] + head.values.tolist(), [2.6, 4.0, 3.6, 3.6, 3.2], fs=6.8)
p(f"<b>Outcome pattern (Ch. 20): {rs['pattern']}.</b> Confirmatory test: deployed minus equal-weight holdout Sharpe {f3(hs['confirmatory_test']['diff'])} [{f3(hs['confirmatory_test']['lo'])}, {f3(hs['confirmatory_test']['hi'])}], p = {hs['confirmatory_test']['p']:.2f} -> <b>not passed</b>. "
  f"Secondary: holdout IC {f3(vh['mean'])} [{f3(vh['lo'])}, {f3(vh['hi'])}] -> not passed. Versus the momentum rule {f3(hs['deployed_vs_rule']['diff'])} [{f3(hs['deployed_vs_rule']['lo'])}, {f3(hs['deployed_vs_rule']['hi'])}]; versus SPY {f3(hs['deployed_vs_spy']['diff'])}. "
  f"The gate was active in {hs['gated_months']} of 12 holdout months.")
fig("r1_equity.png", 14, "Figure 1. Q: Does the strategy beat the baseline and the passive benchmark after costs? Answer: it beats the momentum rule but not the equal-weight universe; flat stretches are gated months in BIL.")
fig("r2_rolling_ic.png", 14, "Figure 2. Q: Is the ranking skill stable? Answer: no - the Ridge IC swings between -0.4 and +0.5 within months, including a deep negative spell during the April-2025 tariff shock.")
fig("r3_gross_net.png", 12, "Figure 3. Q: How much do costs take? Answer: little (turnover is modest); the overlay, not costs, removes most of the validation edge.")
fig("r4_regime.png", 13, "Figure 4. Q: Where does the strategy work? Answer: only in low-volatility up-trend regimes (point-in-time thresholds); elsewhere it loses or is flat - the gate threshold of 0.8 is too permissive.")
fig("r6_drawdown.png", 13, "Figure 6. Q: Would the kill-switch ladder have triggered? Answer: only the -5% watch level, in 2023 and in the 2025 tariff shock (drawdown measured from the running peak across both periods).")
p("<b>Diagnosis.</b> The strategy lost value at the <b>signal gate</b>: no rung established IC, and the momentum mechanism itself failed (the gate hurt the rule, momentum IC is near zero, SHAP shows beta doing the work). "
  "The higher 2025 Sharpe is not evidence of a working mechanism: the holdout IC interval is wide, and 2025 rewarded broad risk exposure (equal-weight universe Sharpe 1.19). "
  "Drift diagnosis: prediction quality (IC) was higher, not lower, in the holdout and portfolio translation was stable (gross and net close), so the validation-holdout gap reflects regime and noise, not a pipeline defect. "
  "<b>Binding constraint:</b> regime persistence and label richness - six half-year folds and one holdout year cannot resolve an IC of 0.05.")
p("<b>One next step.</b> Re-specify the hypothesis as risk compensation (beta/volatility tilt) in setup v2, extend the walk-forward with the textbook's 2007-2019 history to gain about 25 more folds, and test it on the first fresh holdout (2026 data). "
  "None of these first-pass strategies goes to capital.")
wcb("One holdout year contains 12 decisions; a single month (April 2025) moves the holdout Sharpe by several tenths. The inclusive trial count (846) is approximate.")

# ===================================================================== Section 8
sec("8. Monitoring and Deployment Plan (design, not built)")
p("<b>Parity.</b> The same src/lib.py code path would run live in shadow mode for three months; compare stage by stage (raw bars, features, predictions, target weights, orders, fills) against a backtest replay of the same days, and alert on any mismatch beyond rounding. "
  "<b>Monitoring.</b> Rolling 30/60/90-day Sharpe, drawdown, hit rate and daily IC; realisation ratio = live / backtest return over the same window (investigate below 0.5). "
  "<b>Drift.</b> PSI on the top-5 SHAP features (beta, vol, momentum, volume change, 21-day return): watch 0.10-0.25, retrain above 0.25; monthly SHAP-rank check against the validation ranking (Spearman below 0.5 = review). "
  "<b>Lifecycle.</b> Retrain every six months on the expanding window with the frozen protocol; pause if the 6-month IC is negative or the -15% kill-switch level is hit; retire if the pause lasts two consecutive reviews or the confirmatory test fails on a second fresh year. "
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
  f". Each row stores timestamp, git commit, data snapshot hash ({rl.data_snapshot.iloc[0]}), setup version, trial family, trial, run, evaluation window, configuration, metric and decision. Rows repeated by the two bug-fix re-executions are identified in the ledger.", B)
p("<b>A.3 Repository map</b>", H2)
tbl([["Path", "Content"], ["README.md", "How to reproduce from the root; environment; why numbers could differ"], ["setup.yaml", "Trading setup v1 and protocol (Section 2)"],
     ["data/build_dataset.py", "Regenerates data/prices_*.parquet from the textbook dataset"], ["notebooks/01_setup ... 07_results (.py and .ipynb)", "One stage each; 06 refuses to run twice"],
     ["src/lib.py", "Sealed loader, PIT universe, features, statistics, simulator, run log"], ["results/, figures/", "Every number and figure in this document"],
     ["run_log.csv, ledger/research_ledger.md", "Run log and research ledger"], ["requirements.txt, requirements.lock.txt", "Environment (textbook uv env, Python 3.14)"],
     ["report/, slides/", "This PDF and the presentation, built by report/build_report.py and slides/build_slides.py"]], [6.5, 10.5])
p("<b>A.4 Pre-submission checklist (ten questions)</b>", H2)
tbl([["#", "Question", "Answer"],
     ["1", "Time each feature became knowable?", "Yes: close of t for all features (prices/volume only)"],
     ["2", "Universe formed at t, leavers included?", "Formed at t: yes. Leavers: no - disclosed survivorship FAIL"],
     ["3", "Label price convention matches fills?", "Yes: next open to open+21"],
     ["4", "Scalers/winsorisation fitted on training folds only?", "Yes (states); signals use per-date ranks"],
     ["5", "Buffers in trading days, every fold?", "Yes: 21-session purge; embargo not needed (expanding, forward)"],
     ["6", "Holdout touched once, after the freeze?", f"Yes: freeze {fz['frozen_at']}, single touch recorded in results/holdout_touched.json"],
     ["7", "Ledger counts every adaptive choice; DSR uses it?", f"Yes: N={fz['trial_counts']['v3_strategy_variants']} (v3) and {fz['trial_counts']['inclusive_with_v1_v2']} (inclusive)"],
     ["8", "Per-instrument costs with sweep and break-even?", "Yes: commission + tiered spread; 0-8x; break-even reported"],
     ["9", "Constraints and stops inside the simulation, t-1 info?", "Yes: asserted in the loop; gate/overlay use the close of t, trade at t+1 open"],
     ["10", "Dataset and headline table regenerate from the root?", "Yes: see README; LightGBM seed 42, bootstrap seed 7"]], [0.8, 7.6, 8.6])

doc = SimpleDocTemplate(f"{R}/report/ML4T_Midterm_Proposal.pdf", pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.5 * cm, bottomMargin=1.5 * cm, title="ML4T Midterm Proposal")
doc.build(S, onFirstPage=lambda c, d: None, onLaterPages=lambda c, d: (c.setFont("Helvetica", 7), c.drawRightString(A4[0] - 1.8 * cm, 0.9 * cm, f"{d.page}")))
print("built")
