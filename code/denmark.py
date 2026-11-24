"""Denmark: safe-haven pressure, Danmarks Nationalbank intervention, bank liquidity and lending."""
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from lp import lp

D = "../data/"
O = "../results/"


def dt(s):
    return pd.to_datetime(s.str.replace("M", "-").str.replace("D", "-"))


vi = pd.read_csv(D + "dk_dnvali.csv", sep=";")
vi["d"] = dt(vi["TID"])
vi = vi.set_index("d")["INDHOLD"].astype(float)              # net FX purchases, DKK bn, days with intervention
pr = pd.read_csv(D + "dk_dnprnd.csv", sep=";")
pr["d"] = dt(pr["TID"])
pr["k"] = pr["BALANC"] + "_" + pr["INSTRUMENT"]
pr = pr.pivot_table(index="d", columns="k", values="INDHOLD")  # DKK bn
fx = pd.read_csv(D + "dk_dnvald.csv", sep=";")
fx["d"] = dt(fx["TID"])
fx = fx.set_index("d")["INDHOLD"].astype(float).sort_index()
rt = pd.read_csv(D + "dk_dnrentd.csv", sep=";")
rt["d"] = dt(rt["TID"])
rt = rt.pivot_table(index="d", columns="INSTRUMENT", values="INDHOLD")

# ---------------- weekly ----------------
wk = pd.read_csv(O + "weekly.csv", parse_dates=["date"]).set_index("date")
fred = pd.read_csv(D + "fred_daily.csv", parse_dates=["date"]).set_index("date")
# rebuild the shock on the full 2004- sample (as in build.py)
wf = fred[["VIXCLS"]].resample("W-FRI").last()
v = 100 * np.log(wf["VIXCLS"]).diff().dropna()
X = pd.concat([v.shift(i) for i in range(1, 5)], axis=1)
Z = pd.concat([v, X], axis=1).dropna()
r = sm.OLS(Z.iloc[:, 0], sm.add_constant(Z.iloc[:, 1:])).fit()
shock = (r.resid / r.resid.std()).rename("shock")

w = pd.DataFrame(index=shock.index)
w["shock"] = shock
w["lEURDKK"] = 100 * np.log(fx.resample("W-FRI").last())
w["interv_cum"] = vi.resample("W-FRI").sum().reindex(w.index).fillna(0).loc["2005":"2025-06-27"].cumsum()
w["netpos"] = pr["N_NS"].resample("W-FRI").last()
w["curacc"] = pr["P_IR"].resample("W-FRI").last()
w["cd"] = pr["P_IB"].resample("W-FRI").last()
w["cdrate"] = rt["OIBNAA"].resample("W-FRI").last()
w = w.loc["2005-01-07":"2025-06-27"]


def reg(d):
    if d < pd.Timestamp("2012-01-01"):
        return "P0"
    if d < pd.Timestamp("2016-01-01"):
        return "P1"
    return "P2"


w["regime"] = [reg(d) for d in w.index]
w.to_csv(O + "dk_weekly.csv")
res = {}
for y in ["lEURDKK", "interv_cum", "netpos", "cdrate"]:
    res[f"dkw_{y}_pooled"] = lp(w, y, "shock", 12, 4)
    res[f"dkw_{y}_reg"] = lp(w, y, "shock", 12, 4, by_regime=True, regimes=["P0", "P1", "P2"])

# ---------------- monthly lending and the 2015 intervention episode ----------------
ri = pd.read_csv(D + "dk_dnruuri.csv", sep=";")
ri["d"] = pd.PeriodIndex(ri["TID"].str.replace("M", "-"), freq="M").to_timestamp(how="end").normalize()
ri = ri.pivot_table(index="d", columns="INDSEK", values="INDHOLD")
ui = pd.read_csv(D + "dk_dnruupi.csv", sep=";")
ui["d"] = pd.PeriodIndex(ui["TID"].str.replace("M", "-"), freq="M").to_timestamp(how="end").normalize()
ui = ui.pivot_table(index="d", columns="INDSEK", values="INDHOLD")
m = pd.DataFrame(index=ri.index)
m["l_mort_hh"] = 100 * np.log(ri[1400])
m["l_mort_all"] = 100 * np.log(ri[1000])
m["l_bank_all"] = 100 * np.log(ui[1000])
m["l_bank_nfc"] = 100 * np.log(ui[1100])
m["l_bank_hh"] = 100 * np.log(ui[1400])
m["l_bank_priv"] = 100 * np.log(ui[1100] + ui[1400])
m["bank_fin_bn"] = (ui[1000] - ui[1100] - ui[1400]) / 1000.0
m["interv"] = vi.resample("ME").sum().reindex(m.index).fillna(0)
m["netpos"] = pr["N_NS"].resample("ME").mean().reindex(m.index)
m["dnetpos"] = m["netpos"].diff()
m["rate"] = rt["OIBNAA"].resample("ME").mean().reindex(m.index)
m["shock"] = shock.resample("ME").sum().reindex(m.index)
m = m.loc[:"2025-06-30"]
m.to_csv(O + "dk_monthly.csv")

# LP with actual intervention (DKK 10 bn net purchases) as impulse; controls: lags of dy, interventions, rate
def lp_interv(df, y, H=24, p=6, scale=10.0):
    d = df.copy()
    d["dy"] = d[y].diff()
    d["iv"] = d["interv"] / scale
    out = {}
    for h in range(H + 1):
        d["lhs"] = d[y].shift(-h) - d[y].shift(1)
        X = pd.DataFrame({"iv": d["iv"], "c": 1.0}, index=d.index)
        for i in range(1, p + 1):
            X[f"dy{i}"] = d["dy"].shift(i)
            X[f"iv{i}"] = d["iv"].shift(i)
            X[f"r{i}"] = d["rate"].diff().shift(i)
        Z = pd.concat([d["lhs"], X], axis=1).dropna()
        rr = sm.OLS(Z["lhs"], Z.drop(columns="lhs")).fit(cov_type="HAC", cov_kwds={"maxlags": h + 1})
        out[h] = (float(rr.params["iv"]), float(rr.bse["iv"]), int(rr.nobs))
    return out


for y in ["netpos", "l_mort_hh", "l_mort_all", "l_bank_all", "l_bank_nfc", "l_bank_hh", "l_bank_priv", "bank_fin_bn"]:
    res[f"dkm_{y}_iv"] = lp_interv(m, y)

# episode: cumulative interventions and lending growth around Jan-Feb 2015
ep = m.loc["2014-06-30":"2016-06-30", ["interv", "netpos", "l_mort_hh", "l_bank_priv", "bank_fin_bn"]].copy()
ep["interv_cum"] = ep["interv"].cumsum()
ep.to_csv(O + "dk_episode.csv")
json.dump(res, open(O + "dk.json", "w"), indent=1)


def f(v_):
    b, se = v_[0], v_[1]
    t = abs(b / se)
    s = "***" if t > 2.58 else "**" if t > 1.96 else "*" if t > 1.645 else ""
    return f"{b:7.3f}({se:.3f}){s}"


for k in res:
    if k.endswith("_reg"):
        for rg in ["P0", "P1", "P2"]:
            print(f"{k:22s} {rg} " + " ".join(f(res[k][h][rg]) for h in [0, 1, 4, 8, 12]))
    elif k.startswith("dkw"):
        print(f"{k:22s}    " + " ".join(f(res[k][h]) for h in [0, 1, 4, 8, 12]))
    else:
        print(f"{k:22s}    " + " ".join(f(res[k][h]) for h in [0, 3, 6, 12, 24]))
print(ep.round(2).to_string())
print(m.describe().T[["count", "mean", "std"]])

# ---------------- power, tables, figure ----------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

stock_mort = float(np.exp(m["l_mort_hh"].loc["2015-01-31"] / 100) / 1000)   # DKK bn
stock_priv = float(np.exp(m["l_bank_priv"].loc["2015-01-31"] / 100) / 1000)
mde = {}
for k, stock in [("l_mort_hh", stock_mort), ("l_bank_priv", stock_priv)]:
    se = res[f"dkm_{k}_iv"][6][1]
    mde[k] = {"mde_pct": 2.8 * se, "mde_bn": 2.8 * se / 100 * stock,
              "per_unit_liquidity": 2.8 * se / 100 * stock / res["dkm_netpos_iv"][6][0]}
res["dk_power"] = mde
res["dk_stock"] = {"mort_hh": stock_mort, "bank_priv": stock_priv}
json.dump(res, open(O + "dk.json", "w"), indent=1)
print(json.dumps(mde, indent=1), stock_mort, stock_priv)


def cell(v_):
    b, se = v_[0], v_[1]
    t = abs(b / se)
    s = "^{***}" if t > 2.58 else "^{**}" if t > 1.96 else "^{*}" if t > 1.645 else ""
    return f"${b:.2f}{s}$", f"$({se:.2f})$"


lines = []
for key, name in [("dkw_lEURDKK_pooled", "EUR/DKK, \\%"), ("dkw_interv_cum_pooled", "Cumulative FX purchases, DKK bn"),
                  ("dkw_netpos_pooled", "Banks' net position at DN, DKK bn")]:
    cs = [cell(res[key][h]) for h in (0, 1, 4, 8, 12)]
    lines.append(f"{name} & " + " & ".join(c[0] for c in cs) + " \\\\")
    lines.append(" & " + " & ".join(c[1] for c in cs) + " \\\\[2pt]")
open(O + "tab_dkw.tex", "w").write("\n".join(lines))
lines = []
for key, name in [("dkm_netpos_iv", "Banks' net position at DN, DKK bn"),
                  ("dkm_l_mort_hh_iv", "Mortgage loans to households, \\%"),
                  ("dkm_l_bank_hh_iv", "Bank loans to households, \\%"),
                  ("dkm_l_bank_nfc_iv", "Bank loans to non-financial firms, \\%"),
                  ("dkm_bank_fin_bn_iv", "Bank loans to financial corporations, DKK bn")]:
    cs = [cell(res[key][h]) for h in (0, 3, 6, 12, 24)]
    lines.append(f"{name} & " + " & ".join(c[0] for c in cs) + " \\\\")
    lines.append(" & " + " & ".join(c[1] for c in cs) + " \\\\[2pt]")
open(O + "tab_dkm.tex", "w").write("\n".join(lines))

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
e = m.loc["2014-01-31":"2016-12-31"]
fig, axes = plt.subplots(1, 3, figsize=(10, 2.6))
axes[0].bar(e.index, e["interv"], width=20, color="0.3")
axes[0].set_title("Net FX purchases, DKK bn", fontsize=8.5)
axes[1].plot(e.index, e["netpos"], color="0.15")
axes[1].set_title("Banks' net position at DN, DKK bn", fontsize=8.5)
base = e.loc["2014-12-31"]
for c, lab, ls in [("l_mort_hh", "Mortgages to households", "-"), ("l_bank_priv", "Bank loans to households and firms", "--")]:
    axes[2].plot(e.index, e[c] - base[c], color="0.15", ls=ls, label=lab)
axes[2].axhline(0, color="0.5", lw=0.6)
axes[2].set_title("Lending, % change since Dec 2014", fontsize=8.5)
axes[2].legend(fontsize=7, frameon=False, loc="upper left")
for ax in axes:
    ax.axvspan(pd.Timestamp("2015-01-01"), pd.Timestamp("2015-02-28"), color="0.9", zorder=0)
    ax.tick_params(axis="x", labelrotation=45, labelsize=7)
fig.tight_layout()
fig.savefig(O + "fig_denmark_2015.pdf")
