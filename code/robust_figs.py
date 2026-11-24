"""Robustness checks and figures."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lp import lp, results, REG  # noqa: E402  (re-runs baseline, stores in results)

O = "../results/"
w = pd.read_csv(O + "weekly.csv", parse_dates=["date"]).set_index("date")
m = pd.read_csv(O + "monthly.csv", parse_dates=["date"]).set_index("date").loc["2004-02-29":]
m["regime"] = np.where(m["interv"] == 1, "INT", "NON")

rob = {}
# (a) drop weeks within +/- 2 weeks of the three SNB regime announcements
breaks = pd.to_datetime(["2011-09-06", "2015-01-15", "2022-06-16"])
mask = pd.Series(True, index=w.index)
for b in breaks:
    mask &= ~((w.index >= b - pd.Timedelta(days=14)) & (w.index <= b + pd.Timedelta(days=14)))
w_nb = w.copy()
w_nb.loc[~mask, "shock"] = np.nan
for y in ["lEURCHF", "sd_GI_bn", "lEURNOK", "lEURJPY"]:
    rob[f"nobreak_{y}"] = lp(w_nb, y, "shock", 12, 4, by_regime=True)
    rob[f"stl_{y}"] = results[f"w_{y}_stl_reg"]
# (b) drop March 2020
w_nc = w.copy()
w_nc.loc["2020-02-21":"2020-04-30", "shock"] = np.nan
for y in ["lEURCHF", "sd_GI_bn"]:
    rob[f"nocovid_{y}"] = lp(w_nc, y, "shock", 12, 4, by_regime=True)
# (c) eight lags
for y in ["lEURCHF", "sd_GI_bn"]:
    rob[f"lag8_{y}"] = lp(w, y, "shock", 12, 8, by_regime=True)
# (d) monthly with 12 lags and with ECB-period control (EURCHF lags)
for y in ["l_mort", "l_loans", "cbliab_bn"]:
    rob[f"m12_{y}"] = lp(m, y, "shock", 24, 12, by_regime=True, regimes=["INT", "NON"], extra=["rate"])
    rob[f"mfx_{y}"] = lp(m, y, "shock", 24, 6, by_regime=True, regimes=["INT", "NON"], extra=["rate", "l_eurchf"])
json.dump(rob, open(O + "robust.json", "w"), indent=1)


def fmt(v):
    b, se = v[0], v[1]
    t = abs(b / se)
    s = "***" if t > 2.58 else "**" if t > 1.96 else "*" if t > 1.645 else ""
    return f"{b:.3f}({se:.3f}){s}"


for k in rob:
    for rg in (REG if k.split("_")[0] in ["nobreak", "stl", "nocovid", "lag8"] else ["INT", "NON"]):
        hs = [0, 1, 4, 8, 12] if not k.startswith("m") else [0, 6, 12, 18, 24]
        print(f"{k:22s} {rg:4s} " + " ".join(fmt(rob[k][h][rg]) if rg in rob[k][h] else "na" for h in hs))

# ---------------- figures ----------------
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
labels = {"R0": "2009–2011 (discretionary)", "R1": "2011–2015 (floor)", "R2": "2015–2022 (negative rate)",
          "R3": "2022–2023 (FX sales)", "R4": "2024– (limited intervention)"}


def panel(ax, key, reg, title, H=12, color="0.15"):
    r = results[key]
    hs = list(range(H + 1))
    b = np.array([r[str(h)][reg][0] if isinstance(list(r.keys())[0], str) else r[h][reg][0] for h in hs])
    se = np.array([r[str(h)][reg][1] if isinstance(list(r.keys())[0], str) else r[h][reg][1] for h in hs])
    ax.fill_between(hs, b - 1.645 * se, b + 1.645 * se, color="0.85")
    ax.plot(hs, b, color=color, lw=1.4)
    ax.axhline(0, color="0.4", lw=0.6)
    ax.set_title(title, fontsize=8.5)


fig, axes = plt.subplots(2, 5, figsize=(10, 4.2), sharex=True)
for j, rg in enumerate(REG):
    panel(axes[0, j], "w_lEURCHF_reg", rg, labels[rg])
    panel(axes[1, j], "w_sd_GI_bn_reg", rg, "")
axes[0, 0].set_ylabel("EUR/CHF, % (fall = CHF\nappreciation)")
axes[1, 0].set_ylabel("Sight deposits of\ndomestic banks, CHF bn")
for ax in axes[1]:
    ax.set_xlabel("Weeks after shock")
fig.tight_layout()
fig.savefig(O + "fig_weekly_regimes.pdf")

fig, axes = plt.subplots(1, 4, figsize=(10, 2.5), sharex=True)
for ax, (y, t) in zip(axes, [("lEURCHF", "Swiss franc"), ("lEURJPY", "Japanese yen"),
                             ("lEURNOK", "Norwegian krone"), ("lEURSEK", "Swedish krona")]):
    panel(ax, f"w_{y}_pooled", None, t) if False else None
    r = results[f"w_{y}_pooled"]
    hs = list(range(13))
    b = np.array([r[h][0] for h in hs]); se = np.array([r[h][1] for h in hs])
    ax.fill_between(hs, b - 1.645 * se, b + 1.645 * se, color="0.85")
    ax.plot(hs, b, color="0.15", lw=1.4); ax.axhline(0, color="0.4", lw=0.6)
    ax.set_title(t, fontsize=8.5); ax.set_xlabel("Weeks after shock")
axes[0].set_ylabel("Price of 1 EUR, %")
fig.tight_layout()
fig.savefig(O + "fig_weekly_currencies.pdf")

fig, axes = plt.subplots(2, 3, figsize=(9, 4.4), sharex=True)
for i, rg in enumerate(["INT", "NON"]):
    for j, (y, t) in enumerate([("cbliab_bn", "SNB liabilities to banks, CHF bn"),
                                 ("l_mort", "Domestic mortgage loans, %"),
                                 ("l_loans", "Total domestic loans, %")]):
        r = results[f"m_{y}_reg"]
        hs = list(range(25))
        b = np.array([r[h][rg][0] for h in hs]); se = np.array([r[h][rg][1] for h in hs])
        ax = axes[i, j]
        ax.fill_between(hs, b - 1.645 * se, b + 1.645 * se, color="0.85")
        ax.plot(hs, b, color="0.15", lw=1.4); ax.axhline(0, color="0.4", lw=0.6)
        ax.set_title(t if i == 0 else "", fontsize=8.5)
    axes[i, 0].set_ylabel("Intervention regimes\n2011–2022" if rg == "INT" else "Other months\n2004–11, 2022–26")
for ax in axes[1]:
    ax.set_xlabel("Months after shock")
fig.tight_layout()
fig.savefig(O + "fig_monthly.pdf")

# shock time series
fig, ax = plt.subplots(figsize=(9, 2.2))
ax.plot(w.index, w["shock"], color="0.3", lw=0.6)
top = w["shock"].nlargest(8)
for d, v in top.items():
    ax.annotate(d.strftime("%b %Y"), (d, v), fontsize=6.5, ha="center", va="bottom")
ax.set_ylabel("Std. dev.")
fig.tight_layout()
fig.savefig(O + "fig_shock.pdf")
print(top)
