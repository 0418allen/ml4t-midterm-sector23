"""10-minute deck (<= 12 slides): mechanism, setup, headline table, 3 required figures, diagnosis, next step."""
import json
import os

import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
J = lambda f: json.load(open(f"{R}/results/{f}"))
fz, hs, vs, rs = J("freeze.json"), J("holdout_summary.json"), J("validation_summary.json"), J("results_summary.json")
vb = pd.read_csv(f"{R}/results/validation_backtests.csv").set_index("strategy")
hb = pd.read_csv(f"{R}/results/holdout_backtests.csv").set_index("strategy")
vic = pd.read_csv(f"{R}/results/validation_ic.csv").set_index("model")
head = pd.read_csv(f"{R}/results/headline_table.csv")
TRI = pd.read_csv(f"{R}/results/triage.csv").decision.value_counts().to_dict()
DEP, RULE, M = fz["deployed_strategy"], fz["baseline"], fz["model_selection"]["deployed_model"]
NAVY = RGBColor(0x1F, 0x2D, 0x55)
prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.33), Inches(7.5)


def slide(title, bullets=(), img=None, notes="", img_w=7.4):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    t = s.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(12.3), Inches(0.8)).text_frame
    t.text = title
    r = t.paragraphs[0].runs[0]
    r.font.size, r.font.bold, r.font.color.rgb = Pt(28), True, NAVY
    if bullets:
        tf = s.shapes.add_textbox(Inches(0.6), Inches(1.4), Inches(12.2 - (img_w + 0.3 if img else 0)), Inches(5.6)).text_frame
        tf.word_wrap = True
        for i, b in enumerate(bullets):
            para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            para.text = b
            para.font.size = Pt(19)
            para.space_after = Pt(12)
    if img:
        s.shapes.add_picture(f"{R}/figures/{img}", Inches(13.33 - img_w - 0.4), Inches(1.4), width=Inches(img_w))
    s.notes_slide.notes_text_frame.text = notes
    return s


f2 = lambda x: f"{x:.2f}"
slide("ETF momentum with a volatility-state gate",
      ["Question: do calm-state ETF trends predict next-month returns, and does ML beat a momentum rule?",
       "One lap of the ML4T workflow; holdout 2025 sealed and touched once",
       f"Answer: mechanism falsified on validation; holdout positive (Sharpe {f2(hb.loc[DEP, 'sharpe'])}) but not better than equal weight",
       "Pattern: never resolved"],
      notes="開場 30 秒:一句話講結論。")
slide("1. Mechanism and falsification",
      ["Trends persist 1-6 months: slow capital under-reacts and rebalances gradually (flow predictability, risk compensation)",
       "Trends crash in volatility spikes -> trade only when SPY vol percentile < 0.80",
       "Stop if: (1) momentum IC <= 0 in calm states, (2) gate does not help the rule, (3) holdout test fails"],
      notes="組員1。強調機制先行、證偽條件在看結果前寫好。")
slide("2. Setup v1 and protocol (committed before any backtest)",
      ["Universe: 100 textbook ETFs, point-in-time ADV >= $10M, de-duplicated (corr < 0.95) each month",
       "Month-end close signal -> next-open fill; top-5 equal weight, long-only; BIL as cash",
       "Label: next open -> open +21; costs $0.0035/share + tiered half-spread",
       "Walk-forward: 6 half-year folds 2022-2024, 21-session purge; holdout 2025",
       f"Ledger + run log from day one; {fz['trial_counts']['v3_strategy_variants']} variants ({fz['trial_counts']['inclusive_with_v1_v2']} incl. earlier versions)"],
      notes="組員1。老師第一個問題:決策當下知道什麼?答:t 日收盤前的價量,t+1 開盤成交。")
slide("3. Data and features",
      ["0 missing bars, 0 duplicates, 0 OHLC violations; USO 1:8 split handled; survivorship FAIL disclosed",
       "15 signals + 4 states; 5 merged by correlation clustering",
       f"Five gates: {TRI.get('PROCEED', 0)} PROCEED, {TRI.get('REVISE', 0)} REVISE, {TRI.get('STOP', 0)} STOP; only volume change survives BH",
       "Strongest features are beta and volatility (risk compensation), not momentum"],
      img="d4_triage.png", img_w=6.4, notes="組員2、3。大多數 REVISE 是正常的。")
s = slide("4. Model ladder (yardstick: daily IC, Newey-West 95% CI)",
          [f"Rule {vic.loc['rule_mom126vs', 'ic']:+.3f}  |  Ridge {vic.loc[M, 'ic']:+.3f}  |  LightGBM {vic.loc[fz['model_selection']['chosen']['lgb'], 'ic']:+.3f}  |  Ensemble {vic.loc['ensemble', 'ic']:+.3f}",
           "No interval excludes zero -> no signal established",
           "Deployed by pre-registered rule: Ridge, alpha 1e5",
           "SHAP surprise: beta drives the ranking, not trend"],
          img="m2_shap_beeswarm.png", img_w=6.2, notes="組員4。Figure 5 SHAP:最重要是 beta,和動能機制相反。")
s = prs.slides.add_slide(prs.slide_layouts[6])
t = s.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(12.3), Inches(0.8)).text_frame
t.text = "5. Headline table (Ch. 20 gates)"
t.paragraphs[0].runs[0].font.size, t.paragraphs[0].runs[0].font.bold, t.paragraphs[0].runs[0].font.color.rgb = Pt(28), True, NAVY
rows = [list(head.columns)] + head.values.tolist()
tb = s.shapes.add_table(len(rows), 5, Inches(0.4), Inches(1.3), Inches(12.5), Inches(5.0)).table
for j, w in enumerate((1.9, 3.0, 2.6, 2.6, 2.4)):
    tb.columns[j].width = Inches(w)
for i, r in enumerate(rows):
    for j, v in enumerate(r):
        c = tb.cell(i, j)
        c.text = str(v)
        for para in c.text_frame.paragraphs:
            para.font.size = Pt(11 if i else 12)
            para.font.bold = i == 0
b = s.shapes.add_textbox(Inches(0.5), Inches(6.5), Inches(12.3), Inches(0.8)).text_frame
b.text = f"Confirmatory test (minus equal weight): {hs['confirmatory_test']['diff']:+.2f} [{hs['confirmatory_test']['lo']:+.2f}, {hs['confirmatory_test']['hi']:+.2f}] -> not passed. Pattern: never resolved."
b.paragraphs[0].font.size = Pt(16)
b.paragraphs[0].font.bold = True
s.notes_slide.notes_text_frame.text = "組員5。逐列講:訊號、投組、成本、風險覆蓋、推論。DSR 和最短紀錄長度一定要提。"
slide("6. Equity curves: validation | holdout",
      [f"Validation excess Sharpe: deployed {f2(vb.loc[DEP, 'sharpe'])}, rule {f2(vb.loc[RULE, 'sharpe'])}, EW {f2(vb.loc['bench_ew_universe', 'sharpe'])}",
       f"Holdout 2025: deployed {f2(hb.loc[DEP, 'sharpe'])}, rule {f2(hb.loc[RULE, 'sharpe'])}, EW {f2(hb.loc['bench_ew_universe', 'sharpe'])}",
       "Overlay halves drawdown but parks ~70% in BIL"],
      img="r1_equity.png", img_w=7.6, notes="組員5。Figure 1:贏動能規則,輸等權重。")
slide("7. Where it works: regime slice",
      ["Works only in low-vol / up-trend months",
       "Loses or is flat in every other regime",
       "Gate at 0.80 is too permissive; momentum crash story not supported"],
      img="r4_regime.png", img_w=7.6, notes="組員6。Figure 4:門檻用當時資訊算,不是事後挑。")
slide("8. Diagnosis",
      ["Value lost at the signal gate: IC never clears zero",
       "Gate hurt the rule on validation -> mechanism falsified",
       "2025 rewarded broad risk; holdout IC higher, not lower -> regime + noise, not a pipeline bug",
       "Binding constraint: too little history (6 folds, 1 holdout year)"],
      notes="組員6。誠實歸類比數字重要。")
slide("9. Next step and monitoring",
      ["Next step (one): setup v2 = risk-compensation tilt, add 2007-2019 history, test on fresh 2026 holdout",
       "Monitor: 30/60/90-day Sharpe, IC, drawdown; PSI on top-5 features (retrain > 0.25)",
       "Kill switch: -5% watch, -10% halve, -15% halt, -20% unwind",
       "Not deployed to capital"],
      img="r6_drawdown.png", img_w=6.4, notes="組員6。Figure 6:只觸發 -5% 警戒。然後進入 Q&A。")
prs.save(f"{R}/slides/ML4T_Midterm_Presentation.pptx")
print(len(prs.slides), "slides")
