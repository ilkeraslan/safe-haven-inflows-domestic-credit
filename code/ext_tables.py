"""LaTeX tables for the extension analyses."""
import json
O = "../results/"
E_ = json.load(open(O + "ext.json"))

def cell(v):
    b, se = v[0], v[1]
    t = abs(b / se)
    s = "^{***}" if t > 2.58 else "^{**}" if t > 1.96 else "^{*}" if t > 1.645 else ""
    return f"${b:.2f}{s}$", f"$({se:.2f})$"

def block(rows, hs=(0, 3, 6, 12, 24)):
    out = []
    for key, name in rows:
        for rg, rn in [("INT", "Intervention"), ("NON", "Other")]:
            cs = [cell(E_[key][str(h)][rg]) for h in hs]
            out.append(f"{name if rg == 'INT' else ''} & {rn} & " + " & ".join(c[0] for c in cs) + " \\\\")
            out.append(" & & " + " & ".join(c[1] for c in cs) + " \\\\[1pt]")
    return "\n".join(out)

open(O + "tab_bs.tex", "w").write(block([
    ("bs_liquid", "Liquid assets"),
    ("bs_dep_nonres_chf", "Non-resident CHF deposits"),
    ("bs_dep_res_chf", "Resident CHF deposits"),
    ("bs_fbank_chf", "CHF due to foreign banks"),
    ("bs_mort_chf", "Domestic CHF mortgages"),
    ("bs_cust_chf", "Other domestic CHF loans"),
]))
open(O + "tab_grp.tex", "w").write(block([
    ("grp_liquid_big", "Liquid assets, big banks (CHF bn)"),
    ("grp_liquid_oth", "Liquid assets, other banks (CHF bn)"),
    ("grp_l_mortcat_big", "Mortgages, big banks (\\%)"),
    ("grp_l_mortcat_cant", "Mortgages, cantonal banks (\\%)"),
    ("grp_l_mort_chf_oth", "Mortgages, all other banks (\\%)"),
]))
p = E_["power"]
pre = E_
lines = []
for k, lab in [("pre_l_mort", "Domestic mortgage loans"), ("pre_l_loans", "Total domestic loans")]:
    cs = [cell(pre[k][str(h)]) for h in (3, 6, 12)]
    lines.append(f"{lab} & " + " & ".join(c[0] for c in cs) + " \\\\")
    lines.append(" & " + " & ".join(c[1] for c in cs) + " \\\\[2pt]")
open(O + "tab_pre.tex", "w").write("\n".join(lines))
print(json.dumps({k: E_[k] for k in ["power", "exog_F", "power_stock_bn", "cb_int_h3"]}, indent=1))
