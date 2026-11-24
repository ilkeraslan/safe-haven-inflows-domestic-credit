"""Moving-block bootstrap for the credit-to-reserves pass-through ratio, Switzerland and Denmark."""
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

O = "../results/"
rng = np.random.default_rng(20261124)
B = 1999
BLOCK = 12


def design(df, y, imp, h, p, extra=None, inter=None):
    """Rows (lhs, X) for y_{t+h}-y_{t-1} on impulse; optional interaction subsample indicator."""
    d = df.copy()
    d["dy"] = d[y].diff()
    lhs = d[y].shift(-h) - d[y].shift(1)
    X = pd.DataFrame({"imp": d[imp], "c": 1.0}, index=d.index)
    for i in range(1, p + 1):
        X[f"dy{i}"] = d["dy"].shift(i)
        X[f"imp{i}"] = d[imp].shift(i)
        for e in extra or []:
            X[f"{e}{i}"] = d[e].diff().shift(i)
    Z = pd.concat([lhs.rename("lhs"), X], axis=1)
    if inter is not None:
        Z = Z[inter.reindex(Z.index).fillna(False).astype(bool)]
    return Z.dropna()


def design_int(df, y, imp, h, p, extra, ind):
    """Regime-interaction design matching lp(..., by_regime=True): first column is imp x INT."""
    d = df.copy()
    d["dy"] = d[y].diff()
    lhs = d[y].shift(-h) - d[y].shift(1)
    I = ind.astype(float)
    X = pd.DataFrame({"impI": d[imp] * I, "impN": d[imp] * (1 - I), "cI": I, "cN": 1 - I}, index=d.index)
    for i in range(1, p + 1):
        X[f"dy{i}"] = d["dy"].shift(i)
        X[f"imp{i}"] = d[imp].shift(i)
        for e in extra:
            X[f"{e}{i}"] = d[e].diff().shift(i)
    return pd.concat([lhs.rename("lhs"), X], axis=1).dropna()


def coef(Z):
    y = Z["lhs"].values
    X = Z.drop(columns="lhs").values
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    return b[0]


def block_idx(n):
    k = int(np.ceil(n / BLOCK))
    starts = rng.integers(0, n - BLOCK + 1, size=k)
    return np.concatenate([np.arange(s, s + BLOCK) for s in starts])[:n]


def boot_ratio(Znum, Zden, scale_num):
    """Bootstrap numerator and denominator jointly on their common dates."""
    common = Znum.index.intersection(Zden.index)
    Zn, Zd = Znum.loc[common], Zden.loc[common]
    pt = scale_num * coef(Zn) / coef(Zd)
    draws, nums = [], []
    for _ in range(B):
        ix = block_idx(len(common))
        a = scale_num * coef(Zn.iloc[ix])
        dn = coef(Zd.iloc[ix])
        nums.append(a)
        draws.append(a / dn if abs(dn) > 1e-6 else np.nan)
    draws = np.array(draws)
    nums = np.array(nums)
    return {"point": float(pt), "ci90": [float(np.nanpercentile(draws, 5)), float(np.nanpercentile(draws, 95))],
            "se": float(np.nanstd(draws)), "num_point": float(scale_num * coef(Zn)),
            "num_ci90": [float(np.percentile(nums, 5)), float(np.percentile(nums, 95))], "n": int(len(common))}


out = {}
# ---- Switzerland: intervention months, mortgages (CHF bn) at 12 months per CHF of SNB liabilities at 3 months
m = pd.read_csv(O + "monthly.csv", parse_dates=["date"]).set_index("date").loc["2004-02-29":]
intv = m["interv"] == 1
stock = m["ln_AV1_H"].loc["2011-09":"2022-05"].mean() / 1000
for hnum in [6, 12, 24]:
    Zn = design_int(m, "l_mort", "shock", hnum, 6, ["rate"], intv)
    Zd = design_int(m, "cbliab_bn", "shock", 3, 6, ["rate"], intv)
    out[f"ch_mort_h{hnum}"] = boot_ratio(Zn, Zd, stock / 100)
    Zn = design_int(m, "l_loans", "shock", hnum, 6, ["rate"], intv)
    out[f"ch_loans_h{hnum}"] = boot_ratio(Zn, Zd, m["ln_AV1_T1"].loc["2011-09":"2022-05"].mean() / 1000 / 100)

# ---- Denmark: DKK of household mortgages / private bank loans per DKK of net position, both at the same horizon
dk = pd.read_csv(O + "dk_monthly.csv", parse_dates=["d"]).set_index("d")
dk["iv"] = dk["interv"] / 10.0
st = json.load(open(O + "dk.json"))["dk_stock"]
for h in [3, 6, 12]:
    Zd = design(dk, "netpos", "iv", h, 6, extra=["rate"])
    Zn = design(dk, "l_mort_hh", "iv", h, 6, extra=["rate"])
    out[f"dk_mort_h{h}"] = boot_ratio(Zn, Zd, st["mort_hh"] / 100)
    Zn = design(dk, "l_bank_priv", "iv", h, 6, extra=["rate"])
    out[f"dk_priv_h{h}"] = boot_ratio(Zn, Zd, st["bank_priv"] / 100)

json.dump(out, open(O + "boot.json", "w"), indent=1)
for k, v in out.items():
    print(k, {kk: (np.round(vv, 3) if not isinstance(vv, list) else [round(x, 3) for x in vv]) for kk, vv in v.items()})

# ---- LaTeX table
rows = []
for key, ctry, item, h in [("ch_mort_h6", "Switzerland", "Mortgages", 6), ("ch_mort_h12", "", "Mortgages", 12),
                           ("ch_mort_h24", "", "Mortgages", 24), ("ch_loans_h12", "", "Total domestic loans", 12),
                           ("dk_mort_h3", "Denmark", "Household mortgages", 3), ("dk_mort_h6", "", "Household mortgages", 6),
                           ("dk_mort_h12", "", "Household mortgages", 12), ("dk_priv_h6", "", "Bank loans to households and firms", 6)]:
    v = out[key]
    rows.append(f"{ctry} & {item} & {h} & ${v['point']:.2f}$ & $[{v['ci90'][0]:.2f},\\ {v['ci90'][1]:.2f}]$ \\\\")
open(O + "tab_boot.tex", "w").write("\n".join(rows))
