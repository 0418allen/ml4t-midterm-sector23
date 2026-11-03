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
mech = J("mechanism_gate.json")
DEP = fz["deployed_strategy"]
NOG = "rule_mom126vs|ew|nogate|vt"
LGB, RID = fz["model_selection"]["chosen"]["lgb"], fz["model_selection"]["chosen"]["ridge"]
ct = hs["confirmatory_test"]
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
EW_H = hb.loc["bench_ew_universe", "sharpe"]
slide("Sector-ETF momentum with a volatility-state gate",
      ["Question: do sector-ETF trends predict next-month returns in calm markets, and does ML beat a momentum rule?",
       "23 ETFs: 11 sector SPDRs + QQQ, IBB, ITA, ITB, KRE, OIH, SOXX, XBI, XME, XRT, USO, GLD; 2020-2026",
       f"Answer: no. Mechanism falsified on validation; fresh 2026 holdout Sharpe {f2(hb.loc[DEP, 'sharpe'])} vs equal weight {f2(EW_H)}",
       "Pattern: never resolved"],
      notes="開場 30 秒:一句話講結論。我們的策略在全新的 2026 年資料上,沒有贏過把 23 檔等權重全部抱著。")
slide("1. Mechanism and falsification",
      ["Industry momentum: information diffuses slowly across sectors; money rotates gradually",
       "Rotation crashes when volatility spikes -> trade only when SPY vol percentile < 0.80",
       "Stop if: (1) momentum IC <= 0 in calm states, (2) gate does not help the rule, (3) holdout test fails",
       "Result: all three triggered"],
      notes="組員1。強調機制先行:三個證偽條件在看任何結果之前就寫進 setup.yaml 並 commit。")
slide("2. Setup v1 and protocol (committed before any backtest)",
      ["Fixed list of 23 ETFs, no de-duplication (QQQ kept); BIL = cash",
       "Month-end close signal -> next-open fill; top-5 equal weight, long-only",
       "Label: next open -> open +21; costs $0.0035/share + tiered half-spread",
       "Walk-forward: 8 half-year folds 2022-2025, 21-session purge",
       "Holdout 2026-01..10: data nobody had seen, read once after the freeze",
       f"{fz['trial_counts']['variants']} variants ({fz['trial_counts']['inclusive']} incl. earlier versions)"],
      notes="組員1。為什麼 holdout 是 2026:2024、2025 在之前的版本已經看過,不能再當期末考。2026 用 Yahoo 下載,凍結設定後才讀一次。")
slide("3. Data and features",
      ["0 missing, 0 duplicates, 0 OHLC violations; matches textbook bars (corr 0.99999999)",
       "15 signals + 4 states; 5 merged by correlation clustering",
       f"Five gates: {TRI.get('PROCEED', 0)} PROCEED, {TRI.get('REVISE', 0)} REVISE, {TRI.get('STOP', 0)} STOP; nothing survives BH",
       f"Momentum IC: calm {mech['mom_126_vs_ic_low_vol']:+.3f}, volatile {mech['mom_126_vs_ic_high_vol']:+.3f} - opposite of the story",
       "Strongest feature is market beta (risk compensation), not trend"],
      img="d4_triage.png", img_w=6.4, notes="組員2、3。動能在平靜時期反而是負的,第一個證偽條件被觸發。")
slide("4. Model ladder (yardstick: daily IC, Newey-West 95% CI)",
      [f"Rule {vic.loc['rule_mom126vs', 'ic']:+.3f}  |  Ridge {vic.loc[RID, 'ic']:+.3f}  |  LightGBM {vic.loc[LGB, 'ic']:+.3f}  |  Ensemble {vic.loc['ensemble', 'ic']:+.3f}",
       "No interval excludes zero -> no signal established",
       "Deployed by pre-registered rule: the non-ML momentum rule",
       "ML adds nothing: LightGBM learns market timing from state variables, not the cross-section"],
      img="m2_shap_beeswarm.png", img_w=6.2,
      notes="組員4。Figure 5 SHAP:LightGBM 最依賴 vol 和市場狀態變數,但狀態變數同一天對所有 ETF 都一樣,沒辦法拿來排序,所以 IC 是負的。")
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
b.text = f"Confirmatory test (minus equal weight): {ct['diff']:+.2f} [{ct['lo']:+.2f}, {ct['hi']:+.2f}] -> not passed. Pattern: never resolved."
b.paragraphs[0].font.size = Pt(16)
b.paragraphs[0].font.bold = True
s.notes_slide.notes_text_frame.text = "組員5。逐列講:訊號、投組、成本、風險覆蓋、推論。DSR 和最短紀錄長度一定要提:要 49 年才能確認驗證期的 Sharpe。"
slide("6. Equity curves: validation | holdout",
      [f"Validation excess Sharpe: deployed {f2(vb.loc[DEP, 'sharpe'])}, no-gate rule {f2(vb.loc[NOG, 'sharpe'])}, EW {f2(vb.loc['bench_ew_universe', 'sharpe'])}",
       f"Holdout 2026: deployed {f2(hb.loc[DEP, 'sharpe'])}, no-gate rule {f2(hb.loc[NOG, 'sharpe'])}, EW {f2(EW_H)}, SPY {f2(hb.loc['bench_spy', 'sharpe'])}",
       f"Overlay keeps max DD small ({hb.loc[DEP, 'max_dd']:.1%}) but holds about half in BIL"],
      img="r1_equity.png", img_w=7.6, notes="組員5。Figure 1:2026 年所有產業一起漲,只挑 5 檔、一半放現金的策略追不上。")
slide("7. Where it works: regime slice",
      ["Validation: works only in low-vol / up-trend months",
       "Loses in down-trends; flat or negative in every 2026 regime",
       "Gate + overlay stayed flat in 2022 and April 2025, but the gate lowered Sharpe"],
      img="r4_regime.png", img_w=7.6, notes="組員6。Figure 4:門檻用當時資訊算,不是事後挑。閘門有防守價值,但不是好訊號。")
slide("8. Diagnosis",
      [f"Value lost at the signal gate: IC never clears zero (holdout {hs['ic']['rule_mom126vs']['mean']:+.3f})",
       "Gate hurt the rule on validation -> mechanism falsified",
       "2026 rewarded owning every sector; costs were not the problem",
       "Binding constraint: 23 correlated names, about 70 independent months"],
      notes="組員6。誠實歸類比數字重要。毛報酬和淨報酬差不多,所以不是成本也不是程式錯誤。")
slide("9. Next step and monitoring",
      ["Next step (one): same 23 ETFs, add 2006-2019 history (~25 more folds), drop the gate; new 2027 holdout",
       "Monitor: 30/60/90-day Sharpe, IC, drawdown; PSI on momentum, beta, vol (retrain > 0.25)",
       "Kill switch: -5% watch, -10% halve, -15% halt, -20% unwind",
       "Not deployed to capital"],
      img="r6_drawdown.png", img_w=6.4, notes="組員6。Figure 6:只觸發 -5% 警戒。然後進入 Q&A。")
prs.save(f"{R}/slides/ML4T_Midterm_Presentation.pptx")
print(len(prs.slides), "slides")
