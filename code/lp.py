"""Local projections: weekly safe-haven pass-through, monthly balance-sheet responses, quarterly house prices."""
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

O = "../results/"
REG = ["R0", "R1", "R2", "R3", "R4"]


def lp(df, y, shock, H, nlag, by_regime=False, regimes=REG, extra=None, level_diff=True, ctrl_shock_lags=True):
    """y_{t+h} - y_{t-1} on shock_t, lags of dy and shock. Newey-West SEs with h+1 lags.
    Returns dict h -> {regime: (b, se, n)} or (b, se, n)."""
    d = df.copy()
    d["dy"] = d[y].diff()
    out = {}
    for h in range(H + 1):
        d["lhs"] = d[y].shift(-h) - d[y].shift(1)
        X = pd.DataFrame(index=d.index)
        if by_regime:
            for r in regimes:
                ind = (d["regime"] == r).astype(float)
                X["s_" + r] = d[shock] * ind
                X["c_" + r] = ind
        else:
            X["s"] = d[shock]
            X["const"] = 1.0
        for i in range(1, nlag + 1):
            X[f"dy{i}"] = d["dy"].shift(i)
            if ctrl_shock_lags:
                X[f"sh{i}"] = d[shock].shift(i)
        if extra:
            for e in extra:
                for i in range(1, nlag + 1):
                    X[f"{e}{i}"] = d[e].diff().shift(i)
        Z = pd.concat([d["lhs"], X], axis=1).dropna()
        if by_regime:
            Z = Z.loc[:, (Z != 0).any(axis=0)]
        res = sm.OLS(Z["lhs"], Z.drop(columns="lhs")).fit(cov_type="HAC", cov_kwds={"maxlags": h + 1})
        if by_regime:
            out[h] = {}
            for r in regimes:
                k = "s_" + r
                if k in res.params:
                    out[h][r] = (float(res.params[k]), float(res.bse[k]), int((Z["c_" + r] == 1).sum()))
        else:
            out[h] = (float(res.params["s"]), float(res.bse["s"]), int(res.nobs))
    return out


def lp_iv(df, y, x, z, H, nlag, sample=None):
    """LP-IV: y_{t+h}-y_{t-1} on (x_t - x_{t-1}) instrumented by z_t; lags of dy, dx, z as controls."""
    from statsmodels.sandbox.regression.gmm import IV2SLS  # noqa: F401
    d = df.copy()
    if sample is not None:
        d = d[sample]
    d["dy"] = d[y].diff()
    d["dx"] = d[x].diff()
    out = {}
    for h in range(H + 1):
        d["lhs"] = d[y].shift(-h) - d[y].shift(1)
        C = pd.DataFrame(index=d.index)
        C["const"] = 1.0
        for i in range(1, nlag + 1):
            C[f"dy{i}"] = d["dy"].shift(i)
            C[f"dx{i}"] = d["dx"].shift(i)
            C[f"z{i}"] = d[z].shift(i)
        Z = pd.concat([d[["lhs", "dx", z]], C], axis=1).dropna()
        Cm = Z[C.columns]
        # first stage
        fs = sm.OLS(Z["dx"], pd.concat([Z[[z]], Cm], axis=1)).fit(cov_type="HAC", cov_kwds={"maxlags": h + 1})
        xhat = fs.fittedvalues
        ss = sm.OLS(Z["lhs"], pd.concat([xhat.rename("dx"), Cm], axis=1)).fit()
        # correct residuals with actual dx, HAC
        u = Z["lhs"] - ss.predict(pd.concat([Z["dx"], Cm], axis=1))
        Xh = pd.concat([xhat.rename("dx"), Cm], axis=1).values
        n, k = Xh.shape
        XtXi = np.linalg.inv(Xh.T @ Xh)
        L = h + 1
        S = np.zeros((k, k))
        g = Xh * u.values[:, None]
        S += g.T @ g
        for l in range(1, L + 1):
            wgt = 1 - l / (L + 1)
            G = g[l:].T @ g[:-l]
            S += wgt * (G + G.T)
        V = XtXi @ S @ XtXi
        out[h] = (float(ss.params["dx"]), float(np.sqrt(V[0, 0])), int(n), float(fs.tvalues[z] ** 2))
    return out


results = {}

# ---------------- weekly ----------------
w = pd.read_csv(O + "weekly.csv", parse_dates=["date"]).set_index("date")
Hw = 12
for y in ["lEURCHF", "lEURNOK", "lEURSEK", "lEURJPY", "lEURUSD", "sd_GI_bn", "sd_TG_bn"]:
    results[f"w_{y}_pooled"] = lp(w, y, "shock", Hw, 4)
    results[f"w_{y}_reg"] = lp(w, y, "shock", Hw, 4, by_regime=True)
    results[f"w_{y}_stl_reg"] = lp(w, y, "shock_stl", Hw, 4, by_regime=True)

# ---------------- monthly ----------------
m = pd.read_csv(O + "monthly.csv", parse_dates=["date"]).set_index("date")
m = m.loc["2004-02-29":]
m["regime2"] = np.where(m["interv"] == 1, "INT", "NON")
m2 = m.copy()
m2["regime"] = m2["regime2"]
Hm = 24
for y in ["fxinv_bn", "sight_bn", "cbliab_bn", "l_mort", "l_loans", "l_other", "l_eurchf"]:
    results[f"m_{y}_pooled"] = lp(m, y, "shock", Hm, 6, extra=["rate"])
    results[f"m_{y}_reg"] = lp(m2, y, "shock", Hm, 6, by_regime=True, regimes=["INT", "NON"], extra=["rate"])
    results[f"m_{y}_stl_reg"] = lp(m2, y, "shock_stl", Hm, 6, by_regime=True, regimes=["INT", "NON"], extra=["rate"])

# LP-IV: mortgage growth per CHF bn of central-bank reserves created, instrumented by risk-off shocks
results["iv_mort_sight_int"] = lp_iv(m, "l_mort", "sight_bn", "shock", Hm, 6, sample=(m["interv"] == 1))
results["iv_mort_sight_all"] = lp_iv(m, "l_mort", "sight_bn", "shock", Hm, 6)
results["iv_loans_sight_int"] = lp_iv(m, "l_loans", "sight_bn", "shock", Hm, 6, sample=(m["interv"] == 1))

# ---------------- quarterly house prices ----------------
q = pd.read_csv(O + "quarterly.csv", parse_dates=["date"]).set_index("date")
q2 = q.copy()
q2["regime"] = np.where(q2["interv"] == 1, "INT", "NON")
for y in ["lhp_EW", "lhp_EH"]:
    results[f"q_{y}_pooled"] = lp(q, y, "shock", 8, 2, extra=["rate"])
    results[f"q_{y}_reg"] = lp(q2, y, "shock", 8, 2, by_regime=True, regimes=["INT", "NON"], extra=["rate"])

json.dump(results, open(O + "lp_results.json", "w"), indent=1)


def show(key, hs, reg=None):
    r = results[key]
    row = []
    for h in hs:
        v = r[h] if reg is None else r[h].get(reg)
        if v is None:
            row.append("   na   ")
            continue
        b, se = v[0], v[1]
        star = "***" if abs(b / se) > 2.58 else "**" if abs(b / se) > 1.96 else "*" if abs(b / se) > 1.645 else ""
        row.append(f"{b:8.3f}({se:.3f}){star}")
    print(f"{key:28s} {reg or '':4s} " + " ".join(row))


print("WEEKLY h=0,1,4,8,12")
for y in ["lEURCHF", "lEURNOK", "lEURSEK", "lEURJPY", "sd_GI_bn"]:
    show(f"w_{y}_pooled", [0, 1, 4, 8, 12])
    for rg in REG:
        show(f"w_{y}_reg", [0, 1, 4, 8, 12], rg)
print("MONTHLY h=0,3,6,12,18,24")
for y in ["fxinv_bn", "sight_bn", "cbliab_bn", "l_mort", "l_loans", "l_eurchf"]:
    show(f"m_{y}_pooled", [0, 3, 6, 12, 18, 24])
    for rg in ["INT", "NON"]:
        show(f"m_{y}_reg", [0, 3, 6, 12, 18, 24], rg)
print("LP-IV")
for k in ["iv_mort_sight_int", "iv_mort_sight_all", "iv_loans_sight_int"]:
    r = results[k]
    print(k, [(h, round(r[h][0], 3), round(r[h][1], 3), round(r[h][3], 1)) for h in [0, 3, 6, 12, 18, 24]])
print("QUARTERLY h=0,2,4,6,8")
for y in ["lhp_EW", "lhp_EH"]:
    show(f"q_{y}_pooled", [0, 2, 4, 6, 8])
    for rg in ["INT", "NON"]:
        show(f"q_{y}_reg", [0, 2, 4, 6, 8], rg)
