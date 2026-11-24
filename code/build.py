"""Build weekly, monthly and quarterly analysis datasets from SNB and FRED files."""
import numpy as np
import pandas as pd
import statsmodels.api as sm

D = "../data/"
O = "../results/"

# ---------- daily market data (FRED) ----------
f = pd.read_csv(D + "fred_daily.csv", parse_dates=["date"]).set_index("date").sort_index()
f["EURCHF"] = f["DEXSZUS"] * f["DEXUSEU"]
f["EURNOK"] = f["DEXNOUS"] * f["DEXUSEU"]
f["EURSEK"] = f["DEXSDUS"] * f["DEXUSEU"]
f["EURJPY"] = f["DEXJPUS"] * f["DEXUSEU"]
f["EURUSD"] = f["DEXUSEU"]
cols = ["VIXCLS", "STLFSI4", "DGS10", "EURCHF", "EURNOK", "EURSEK", "EURJPY", "EURUSD"]
# week ending Friday, last available observation (SNB sight deposits are reported for Fridays)
wk = f[cols].resample("W-FRI").last()

# ---------- weekly sight deposits (SNB) ----------
sd = pd.read_csv(D + "snb_weekly.csv", parse_dates=["date"]).set_index("date")
sd = sd.resample("W-FRI").last()
w = wk.join(sd, how="inner")
w = w.loc["2009-01-09":]

# global risk-off shock: AR(4) innovation in weekly change of log VIX
def ar_innov(x, p=4):
    df = pd.DataFrame({"x": x})
    for i in range(1, p + 1):
        df[f"l{i}"] = df["x"].shift(i)
    df = df.dropna()
    r = sm.OLS(df["x"], sm.add_constant(df.drop(columns="x"))).fit()
    e = r.resid
    return e / e.std()

# shocks are built on the full 2004- daily sample, then aligned
wk_full = f[cols].resample("W-FRI").last()
wk_full["dlvix"] = 100 * np.log(wk_full["VIXCLS"]).diff()
wk_full["dstl"] = wk_full["STLFSI4"].diff()
shock_vix = ar_innov(wk_full["dlvix"].dropna())
shock_stl = ar_innov(wk_full["dstl"].dropna())
wk_full["shock"] = shock_vix
wk_full["shock_stl"] = shock_stl

w = w.join(wk_full[["shock", "shock_stl", "dlvix"]])

def regime(d):
    if d < pd.Timestamp("2011-09-06"):
        return "R0"
    if d < pd.Timestamp("2015-01-15"):
        return "R1"
    if d < pd.Timestamp("2022-06-16"):
        return "R2"
    if d < pd.Timestamp("2024-01-01"):
        return "R3"
    return "R4"

w["regime"] = [regime(d) for d in w.index]
for c in ["EURCHF", "EURNOK", "EURSEK", "EURJPY", "EURUSD"]:
    w["l" + c] = 100 * np.log(w[c])
w["sd_GI_bn"] = w["sd_GI"] / 1000.0
w["sd_TG_bn"] = w["sd_TG"] / 1000.0
w.to_csv(O + "weekly.csv")

# ---------- monthly ----------
m = pd.read_csv(D + "snb_monthly.csv")
m["date"] = pd.PeriodIndex(m["date"], freq="M").to_timestamp(how="end").normalize()
m = m.set_index("date")
ms = wk_full[["shock", "shock_stl"]].resample("ME").sum()
mm = wk_full[["VIXCLS", "EURCHF"]].resample("ME").mean()
m = m.join(ms).join(mm, rsuffix="_mkt")
m["l_mort"] = 100 * np.log(m["ln_AV1_H"])
m["l_loans"] = 100 * np.log(m["ln_AV1_T1"])
m["l_other"] = 100 * np.log(m["ln_AV1_T2"])
# chain-link the September 2006 reporting break in the bank lending statistics:
# the break month's growth is replaced by the average growth of the adjacent six months
for c in ["l_mort", "l_loans", "l_other"]:
    d = m[c].diff()
    b = pd.Timestamp("2006-09-30")
    nb = d.loc[b - pd.DateOffset(months=3):b + pd.DateOffset(months=3)].drop(b)
    d.loc[b] = nb.mean()
    first = m[c].first_valid_index()
    m[c] = m[c].loc[first] + d.fillna(0).cumsum().where(m[c].notna())
m["fxinv_bn"] = m["snb_D"] / 1000.0
m["sight_bn"] = m["snb_GB"] / 1000.0
# central-bank liabilities to banks: sight deposits plus SNB bills plus CHF repo liabilities
m["cbliab_bn"] = (m["snb_GB"] + m["snb_ES"].fillna(0) + m["snb_VRGSF"].fillna(0)) / 1000.0
m["l_eurchf"] = 100 * np.log(m["fx_EUR1"])
m["infl"] = 100 * np.log(m["cpi"]).diff(12)
mmr = pd.read_csv(D + "snb_mmrates.csv")
mmr["date"] = pd.PeriodIndex(mmr["date"], freq="M").to_timestamp(how="end").normalize()
mmr = mmr.pivot_table(index="date", columns="code", values="value")
# short rate: 3-month yield on Confederation money-market debt register claims (continuous 2003-2026)
m = m.join(mmr[["EG3M"]])
m["rate"] = m["EG3M"]
m["regime"] = [regime(d) for d in m.index]
m["interv"] = m["regime"].isin(["R1", "R2"]).astype(int)
m.to_csv(O + "monthly.csv")

# ---------- quarterly house prices ----------
h = pd.read_csv(D + "snb_houseq.csv")
h = h[(h["provider"] == "TP") & (h["type"].isin(["EW", "EH"]))]
h["date"] = pd.PeriodIndex(h["date"].astype(str).str.replace("-Q", "Q"), freq="Q").to_timestamp(how="end").normalize()
hp = h.pivot_table(index="date", columns="type", values="value")
hp.columns = ["hp_" + c for c in hp.columns]
qs = wk_full[["shock", "shock_stl"]].resample("QE").sum()
q = hp.join(qs, how="inner")
mq = m[["l_mort", "sight_bn", "rate", "infl"]].resample("QE").last()
q = q.join(mq)
for c in ["hp_EW", "hp_EH"]:
    q["l" + c] = 100 * np.log(q[c])
q["regime"] = [regime(d) for d in q.index]
q["interv"] = q["regime"].isin(["R1", "R2"]).astype(int)
q.to_csv(O + "quarterly.csv")

print(w.shape, m.dropna(subset=["shock"]).shape, q.shape)
print(w[["shock", "shock_stl"]].corr())
print(w.groupby("regime").size())
