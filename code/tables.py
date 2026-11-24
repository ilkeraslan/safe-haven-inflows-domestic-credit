"""LaTeX tables for the manuscript."""
import json
import numpy as np
import pandas as pd

O = "../results/"
R = json.load(open(O + "lp_results.json"))
RB = json.load(open(O + "robust.json"))
w = pd.read_csv(O + "weekly.csv", parse_dates=["date"]).set_index("date")
m = pd.read_csv(O + "monthly.csv", parse_dates=["date"]).set_index("date").loc["2004-02-29":]


def cell(v):
    b, se = v[0], v[1]
    t = abs(b / se)
    s = "^{***}" if t > 2.58 else "^{**}" if t > 1.96 else "^{*}" if t > 1.645 else ""
    return f"${b:.2f}{s}$", f"$({se:.2f})$"


REG = ["R0", "R1", "R2", "R3", "R4"]
RL = {"R0": "Jan 2009--Sep 2011", "R1": "Sep 2011--Jan 2015", "R2": "Jan 2015--Jun 2022",
      "R3": "Jun 2022--Dec 2023", "R4": "Jan 2024--Nov 2026"}

# ---- Table 1: descriptive statistics
rows = []
def addrow(name, s, unit):
    s = s.dropna()
    rows.append(f"{name} & {unit} & {s.index.min():%Y-%m} & {s.index.max():%Y-%m} & {len(s)} & {s.mean():.2f} & {s.std():.2f} \\\\")
addrow("Risk-off shock (VIX innovation)", w["shock"], "s.d.")
addrow("Financial-stress shock (STLFSI)", w["shock_stl"], "s.d.")
addrow("Weekly change, EUR/CHF", w["lEURCHF"].diff(), "\\%")
addrow("Weekly change, banks' sight deposits", w["sd_GI_bn"].diff(), "CHF bn")
addrow("Monthly growth, domestic mortgage loans", m["l_mort"].diff(), "\\%")
addrow("Monthly growth, total domestic loans", m["l_loans"].diff(), "\\%")
addrow("Monthly change, SNB liabilities to banks", m["cbliab_bn"].diff(), "CHF bn")
addrow("SNB foreign currency investments", m["fxinv_bn"], "CHF bn")
t1 = "\n".join(rows)
open(O + "tab_desc.tex", "w").write(t1)

# ---- Table 2: weekly regime responses at h = 0, 4, 8
lines = []
for rg in REG:
    c1 = [cell(R["w_lEURCHF_reg"][str(h)][rg]) for h in (0, 4, 8)]
    c2 = [cell(R["w_sd_GI_bn_reg"][str(h)][rg]) for h in (0, 4, 8)]
    n = R["w_lEURCHF_reg"]["0"][rg][2]
    lines.append(f"{RL[rg]} & " + " & ".join(c[0] for c in c1) + " & " + " & ".join(c[0] for c in c2) + f" & {n} \\\\")
    lines.append(" & " + " & ".join(c[1] for c in c1) + " & " + " & ".join(c[1] for c in c2) + " & \\\\[2pt]")
open(O + "tab_weekly.tex", "w").write("\n".join(lines))

# ---- Table 3: comparison currencies, pooled
lines = []
for y, name in [("lEURCHF", "Swiss franc"), ("lEURJPY", "Japanese yen"), ("lEURUSD", "US dollar"),
                ("lEURNOK", "Norwegian krone"), ("lEURSEK", "Swedish krona")]:
    cs = [cell(R[f"w_{y}_pooled"][str(h)]) for h in (0, 1, 4, 8, 12)]
    lines.append(f"{name} & " + " & ".join(c[0] for c in cs) + " \\\\")
    lines.append(" & " + " & ".join(c[1] for c in cs) + " \\\\[2pt]")
open(O + "tab_currencies.tex", "w").write("\n".join(lines))

# ---- Table 4: monthly responses
lines = []
for y, name in [("cbliab_bn", "SNB liabilities to banks (CHF bn)"), ("l_mort", "Domestic mortgage loans (\\%)"),
                ("l_loans", "Total domestic loans (\\%)"), ("l_other", "Other domestic loans (\\%)")]:
    for rg, rn in [("INT", "Intervention regimes"), ("NON", "Other months")]:
        cs = [cell(R[f"m_{y}_reg"][str(h)][rg]) for h in (0, 6, 12, 18, 24)]
        lab = name if rg == "INT" else ""
        lines.append(f"{lab} & {rn} & " + " & ".join(c[0] for c in cs) + " \\\\")
        lines.append(" & & " + " & ".join(c[1] for c in cs) + " \\\\[2pt]")
open(O + "tab_monthly.tex", "w").write("\n".join(lines))

# ---- Table 5: robustness, h=0 for EUR/CHF and h=8 for sight deposits
lines = []
specs = [("Baseline", "w_{y}_reg", R), ("Excluding regime-change weeks", "nobreak_{y}", RB),
         ("Excluding Feb--Apr 2020", "nocovid_{y}", RB), ("Eight lags", "lag8_{y}", RB),
         ("Financial-stress shock", "stl_{y}", RB)]
for name, key, src in specs:
    a = [cell(src[key.format(y="lEURCHF")]["0"][rg]) for rg in REG]
    b = [cell(src[key.format(y="sd_GI_bn")]["8"][rg]) for rg in REG]
    lines.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{name}}}}} \\\\")
    lines.append("EUR/CHF, impact & " + " & ".join(c[0] for c in a) + " \\\\")
    lines.append(" & " + " & ".join(c[1] for c in a) + " \\\\")
    lines.append("Sight deposits, 8 weeks & " + " & ".join(c[0] for c in b) + " \\\\")
    lines.append(" & " + " & ".join(c[1] for c in b) + " \\\\[3pt]")
open(O + "tab_robust.tex", "w").write("\n".join(lines))

# ---- Table 6: monthly robustness for mortgages
lines = []
for name, key, src in [("Baseline (6 lags)", "m_l_mort_reg", R), ("Twelve lags", "m12_l_mort", RB),
                       ("Controlling for lagged EUR/CHF", "mfx_l_mort", RB)]:
    for rg, rn in [("INT", "Intervention regimes"), ("NON", "Other months")]:
        cs = [cell(src[key][str(h)][rg]) for h in (0, 6, 12, 18, 24)]
        lines.append(f"{name if rg == 'INT' else ''} & {rn} & " + " & ".join(c[0] for c in cs) + " \\\\")
        lines.append(" & & " + " & ".join(c[1] for c in cs) + " \\\\[2pt]")
open(O + "tab_mrobust.tex", "w").write("\n".join(lines))

# ---- magnitudes for text
mort_stock = m["ln_AV1_H"].loc["2011-09":"2022-06"].mean() / 1000  # CHF bn
b12, se12 = R["m_l_mort_reg"]["12"]["INT"][0], R["m_l_mort_reg"]["12"]["INT"][1]
ub = b12 + 1.645 * se12
mag = {
    "mort_stock_bn": mort_stock,
    "mort12_int": b12, "mort12_int_se": se12, "mort12_int_ub_pct": ub,
    "mort12_int_ub_bn": ub / 100 * mort_stock,
    "sd8_R1": R["w_sd_GI_bn_reg"]["8"]["R1"][0], "sd4_R0": R["w_sd_GI_bn_reg"]["4"]["R0"][0],
    "shock_sd_dlvix": float(w["dlvix"].std()),
    "n_weeks": int(w["shock"].notna().sum()),
    "n_months": int(m["shock"].notna().sum()),
    "fxtr_2020": 38507 + 51510 + 10973 + 8685,
}
json.dump(mag, open(O + "magnitudes.json", "w"), indent=1)
print(json.dumps(mag, indent=1))
print(t1)
