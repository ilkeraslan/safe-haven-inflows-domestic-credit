"""Does the countercyclical capital buffer mask a reserve channel? Mortgage responses within the intervention regime, with and without the buffer in force."""
import json
import pandas as pd
from lp import lp

O = "../results/"
m = pd.read_csv(O + "monthly.csv", parse_dates=["date"]).set_index("date").loc["2004-02-29":]
m["interv"] = ((m.index >= "2011-09-01") & (m.index <= "2022-05-31")).astype(int)
# sectoral countercyclical capital buffer on residential mortgages: in force 30 Sep 2013 to 27 Mar 2020
buf = (m.index >= "2013-10-01") & (m.index <= "2020-03-31")
m["regime"] = "OTH"
m.loc[(m.interv == 1) & buf, "regime"] = "IB"
m.loc[(m.interv == 1) & ~buf, "regime"] = "IN"
res = {}
for y in ["l_mort", "l_loans", "cbliab_bn"]:
    r = lp(m, y, "shock", 24, 6, by_regime=True, regimes=["IB", "IN", "OTH"], extra=["rate"])
    res[y] = {str(h): {k: list(v) for k, v in r[h].items()} for h in (3, 6, 12, 24)}
json.dump(res, open(O + "ccyb.json", "w"), indent=1)
for y in res:
    print(y, {h: {k: (round(v[0], 3), round(v[1], 3), v[2]) for k, v in res[y][h].items()} for h in res[y]})
