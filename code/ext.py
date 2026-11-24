"""Extensions: bank balance-sheet trace, big banks versus other banks, power, pre-trends, shock exogeneity."""
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from lp import lp

D = "../data/"
O = "../results/"

m = pd.read_csv(O + "monthly.csv", parse_dates=["date"]).set_index("date")
b = pd.read_csv(D + "snb_bankbs.csv")
b.columns = ["date", "item", "res", "cur", "grp", "v"]
b["date"] = pd.PeriodIndex(b["date"], freq="M").to_timestamp(how="end").normalize()
b["key"] = b["item"] + "_" + b["res"] + "_" + b["cur"] + "_" + b["grp"]
bw = b.pivot_table(index="date", columns="key", values="v") / 1000.0  # CHF bn

series = {
    "dep_nonres_chf": "VKE_A_CHF_A40",   # customer deposits of non-residents, CHF
    "dep_res_chf": "VKE_I_CHF_A40",      # customer deposits of residents, CHF
    "fbank_chf": "VBA_A_CHF_A40",        # amounts due to foreign banks, CHF
    "liquid": "FMI_T_T_A40",             # liquid assets (cash and sight deposits at the SNB)
    "mort_chf": "HYP_I_CHF_A40",         # domestic mortgage loans, CHF
    "cust_chf": "FKU_I_CHF_A40",         # other amounts due from domestic customers, CHF
    "liquid_big": "FMI_T_T_G15",
    "mort_big": "HYP_I_CHF_G15",
    "cust_big": "FKU_I_CHF_G15",
    "dep_nonres_big": "VKE_A_CHF_G15",
}
for k, c in series.items():
    m[k] = bw[c]
for k in ["liquid", "mort_chf", "cust_chf", "dep_nonres_chf"]:
    m[k + "_oth"] = m[k] - m[k.replace("_chf", "") + "_big" if k != "dep_nonres_chf" else "dep_nonres_big"]
# percent-of-stock versions for group comparison
for k in ["mort_big", "mort_chf_oth", "cust_big", "cust_chf_oth"]:
    m["l_" + k] = 100 * np.log(m[k])
# loans by bank category (SNB credit statistics), to May 2023
for c, nm in [("ln_AV3_H", "mortcat_big"), ("ln_AV2_H", "mortcat_cant"), ("ln_AV10_H", "mortcat_raif"),
              ("ln_AV4_H", "mortcat_reg")]:
    m["l_" + nm] = 100 * np.log(m[c])

mm = m.loc["2004-02-29":].copy()
mm["regime"] = np.where(mm["interv"] == 1, "INT", "NON")
res = {}
H = 24
for y in ["dep_nonres_chf", "fbank_chf", "dep_res_chf", "liquid", "mort_chf", "cust_chf"]:
    res[f"bs_{y}"] = lp(mm, y, "shock", H, 6, by_regime=True, regimes=["INT", "NON"], extra=["rate"])
mb = mm.loc[:"2023-05-31"]
for y in ["liquid_big", "liquid_oth", "l_mort_big", "l_mort_chf_oth", "l_cust_big", "l_cust_chf_oth",
          "l_mortcat_big", "l_mortcat_cant", "l_mortcat_raif", "l_mortcat_reg"]:
    res[f"grp_{y}"] = lp(mb, y, "shock", H, 6, by_regime=True, regimes=["INT", "NON"], extra=["rate"])

# ---- pre-trend placebo: does today's shock "predict" past credit growth?
def lead_test(df, y, ks=(3, 6, 12)):
    out = {}
    d = df.copy()
    for k in ks:
        d["lhs"] = d[y].shift(1) - d[y].shift(1 + k)
        X = pd.DataFrame({"s": d["shock"], "c": 1.0, "rate": d["rate"].diff().shift(1 + k)})
        Z = pd.concat([d["lhs"], X], axis=1).dropna()
        r = sm.OLS(Z["lhs"], Z.drop(columns="lhs")).fit(cov_type="HAC", cov_kwds={"maxlags": k})
        out[k] = (float(r.params["s"]), float(r.bse["s"]), float(r.pvalues["s"]))
    return out

res["pre_l_mort"] = lead_test(mm, "l_mort")
res["pre_l_loans"] = lead_test(mm, "l_loans")

# ---- shock exogeneity: do lagged Swiss variables predict the weekly shock?
w = pd.read_csv(O + "weekly.csv", parse_dates=["date"]).set_index("date")
X = pd.DataFrame(index=w.index)
for i in range(1, 5):
    X[f"fx{i}"] = w["lEURCHF"].diff().shift(i)
    X[f"sd{i}"] = w["sd_GI_bn"].diff().shift(i)
X["c"] = 1.0
Z = pd.concat([w["shock"], X], axis=1).dropna()
r = sm.OLS(Z["shock"], Z.drop(columns="shock")).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
names = [c for c in X.columns if c != "c"]
ft = r.f_test(" , ".join(f"{n} = 0" for n in names))
res["exog_F"] = (float(np.squeeze(ft.fvalue)), float(ft.pvalue), int(r.nobs))

# ---- power: minimum detectable effect (80% power, 5% two-sided) for the INT mortgage response
R = json.load(open(O + "lp_results.json"))
mde = {}
stock = m["ln_AV1_H"].loc["2011-09":"2022-05"].mean() / 1000
for h in ["6", "12", "24"]:
    se = R["m_l_mort_reg"][h]["INT"][1]
    mde[h] = {"mde_pct": 2.80 * se, "mde_bn": 2.80 * se / 100 * stock}
res["power"] = mde
res["power_stock_bn"] = stock
res["cb_int_h3"] = R["m_cbliab_bn_reg"]["3"]["INT"]
json.dump(res, open(O + "ext.json", "w"), indent=1)


def f(v):
    b_, se = v[0], v[1]
    t = abs(b_ / se)
    s = "***" if t > 2.58 else "**" if t > 1.96 else "*" if t > 1.645 else ""
    return f"{b_:7.2f}({se:.2f}){s}"


for k in res:
    if k.startswith(("bs_", "grp_")):
        for rg in ["INT", "NON"]:
            print(f"{k:22s} {rg} " + " ".join(f(res[k][h][rg]) for h in [0, 3, 6, 12, 24]))
print("pre", res["pre_l_mort"], res["pre_l_loans"])
print("exog", res["exog_F"])
print("power", res["power"], res["cb_int_h3"])
