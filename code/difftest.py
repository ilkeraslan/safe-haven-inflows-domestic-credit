"""Wald test: does the credit response differ between intervention and other months?"""
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

O = "../results/"
m = pd.read_csv(O + "monthly.csv", parse_dates=["date"]).set_index("date").loc["2004-02-29":]
out = {}
for y in ["l_mort", "l_loans", "cbliab_bn", "l_eurchf"]:
    d = m.copy()
    d["dy"] = d[y].diff()
    for h in [3, 6, 12, 18, 24]:
        d["lhs"] = d[y].shift(-h) - d[y].shift(1)
        X = pd.DataFrame(index=d.index)
        X["s"] = d["shock"]
        X["sxI"] = d["shock"] * d["interv"]
        X["I"] = d["interv"]
        X["const"] = 1.0
        for i in range(1, 7):
            X[f"dy{i}"] = d["dy"].shift(i)
            X[f"sh{i}"] = d["shock"].shift(i)
            X[f"r{i}"] = d["rate"].diff().shift(i)
        Z = pd.concat([d["lhs"], X], axis=1).dropna()
        r = sm.OLS(Z["lhs"], Z.drop(columns="lhs")).fit(cov_type="HAC", cov_kwds={"maxlags": h + 1})
        out[f"{y}_{h}"] = (float(r.params["sxI"]), float(r.bse["sxI"]), float(r.pvalues["sxI"]))
json.dump(out, open(O + "difftest.json", "w"), indent=1)
for k, v in out.items():
    print(f"{k:16s} diff={v[0]:7.3f} se={v[1]:6.3f} p={v[2]:.2f}")
