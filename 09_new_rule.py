"""
The 2023 rule, and the corrected reading of level compression (run 06_histograms.py and 07 first).

  N1 sensitivity of the panel decomposition to the old formula's prior weight C (35 / fitted / 65). (T44)
  N2 level compression across the change, archive films with histograms on both sides: per star of pre-change raw
     level, displayed change = members (raw) + formula (gap); by time between readings and pooled with a time
     interaction. Corrects the timing-based reading in 02_jump.py (T9). Panel version by level band. (T45, T45b)
  N3 model-free test of "a function of the public histogram": first-order stochastic dominance reversals at
     near-equal counts. (T46, fig9)
  N4 persistence of the hidden adjustment (residual of displayed - raw after flexible histogram controls) from the
     first post-change archive reading to Oct 2026, in-sample and 5-fold out-of-sample; and the within-film
     pass-through of raw-mean changes. (T47)
  N5 interior whole-star / half-star contrast H within films, last pre-change reading (2022 on) vs 2026. (T48)
  N6 half-star share and displayed - raw by audience size. (T49)
  fig6: members, shrinkage decay and formula switch by 2021 raw level.

  .venv/bin/python 09_new_rule.py
"""
import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import lbx

lbx.style()
STARS = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
HC = [f"h{int(s * 10):02d}" for s in STARS]
rng = np.random.default_rng(2026)
t29 = pd.read_csv(lbx.TAB / "T29_old_formula_fit.csv", index_col=0)
C0 = t29.loc["fitted C (prior weight, in ratings)", "readings"]
M0 = t29.loc["fitted m (prior mean, stars)", "readings"]
FLEX = "bs(lvl, df=5) + bs(ln, df=5) + lvl:ln + sd + s5 + slo"


def feats(df):
    """Histogram features per row; index = slug."""
    m = df[HC].fillna(0).to_numpy(float)
    n = m.sum(axis=1)
    P = m / np.where(n > 0, n, np.nan)[:, None]
    raw = P @ STARS
    disp = df["shown"].to_numpy() if "shown" in df else df["displayed"].to_numpy()
    return pd.DataFrame({"n": n, "raw": raw, "sd": np.sqrt((P * (STARS - raw[:, None]) ** 2).sum(axis=1)),
                         "s5": P[:, 9], "slo": P[:, :2].sum(axis=1), "half": P[:, [0, 2, 4, 6, 8]].sum(axis=1), "disp": disp,
                         "lvl": raw - 3.5, "ln": np.log10(np.where(n > 0, n, np.nan))}, index=df.slug.to_numpy())


def clustered(formula, df):
    return smf.ols(formula, df).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(df.cluster)[0]})


p = lbx.load_panel()
clean = p[p.clean].copy()
clean["cluster"] = clean.director1.fillna("film:" + clean.slug)
CL = clean.set_index("slug")[["cluster", "director1", "lang", "cohort", "genre_list", "english"]]
live = pd.read_sql("SELECT * FROM hist WHERE status = 'ok'", sqlite3.connect(
    f"file:{lbx.DATA / 'histograms.sqlite'}?mode=ro", uri=True))
ar = pd.read_sql("SELECT * FROM readings WHERE status = 'ok'", sqlite3.connect(
    f"file:{lbx.DATA / 'archive_hist_pilot.sqlite'}?mode=ro", uri=True))
ar["t"] = pd.to_datetime(ar.served.str[:8])
Z = feats(live)
dec = pd.read_csv(lbx.DATA / "derived" / "panel_decomposition.csv")
dec = dec[dec.shown.notna()]

# ----------------------------------------------------------------------------- N1 sensitivity to C
rows = []
bands = [("all scored", lambda x: x.rating_count_2021 > 0), ("n2021 30-199", lambda x: x.rating_count_2021 < 200),
         ("n2021 >= 200", lambda x: x.rating_count_2021 >= 200),
         ("n2021 200-999", lambda x: (x.rating_count_2021 >= 200) & (x.rating_count_2021 < 1000)),
         ("n2021 >= 1,000", lambda x: x.rating_count_2021 >= 1000)]
for prior_w in (35.0, C0, 65.0):          # not "C": that name is patsy's categorical marker in formulas
    raw21 = (dec.rating_2021 * (dec.rating_count_2021 + prior_w) - prior_w * M0) / dec.rating_count_2021
    old26 = (dec.n_hist * dec.raw + prior_w * M0) / (dec.n_hist + prior_w)
    for lab, sel in bands:
        s = sel(dec)
        rows.append({"C": round(prior_w, 1), "group": lab, "films": int(s.sum()), "members": (dec.raw - raw21)[s].mean(),
                     "decay": ((old26 - dec.raw) - (dec.rating_2021 - raw21))[s].mean(),
                     "switch": (dec.shown - old26)[s].mean(),
                     "old-formula displayed change": (old26 - dec.rating_2021)[s].mean()})
t44 = pd.DataFrame(rows).set_index(["group", "C"]).sort_index()
lbx.write_table(t44, "T44_decomposition_sensitivity_to_C", floatfmt=".4f")

# ----------------------------------------------------------------------------- N2 level compression across the change
b = pd.read_csv(lbx.DATA / "derived" / "pilot_split.csv")
b["lvl"], b["lc"], b["yrs"] = b.raw_b - 3.5, np.log10(b.shown_n_b), b.days / 365.25
rows = []
for lab, s in [("all", b), ("readings <= 180 days apart", b[b.days <= 180]),
               ("181-450 days", b[(b.days > 180) & (b.days <= 450)]), ("> 450 days", b[b.days > 450])]:
    r = {"window": lab, "films": len(s), "median days": s.days.median()}
    for y, nm in [("d_shown", "displayed"), ("d_raw", "members"), ("d_gap", "formula")]:
        f = smf.ols(f"{y} ~ lvl + lc" + (" + yrs" if lab == "all" else ""), s).fit(cov_type="HC3")
        lo, hi = f.conf_int().loc["lvl"]
        r.update({f"{nm} per star": f.params.lvl, f"{nm} ci_lo": lo, f"{nm} ci_hi": hi})
    rows.append(r)
f = smf.ols("d_raw ~ lvl * yrs + lc", b).fit(cov_type="HC3")
for term, lab in [("lvl", "members, pooled: level gradient at zero time between readings"),
                  ("lvl:yrs", "members, pooled: added level gradient per year between readings")]:
    rows.append({"window": lab, "films": len(b), "members per star": f.params[term],
                 "members ci_lo": f.conf_int().loc[term, 0], "members ci_hi": f.conf_int().loc[term, 1]})
t45 = pd.DataFrame(rows).set_index("window")
lbx.write_table(t45, "T45_level_gradient_members_vs_formula", floatfmt=".4f")
dec["lvl_band"] = pd.cut(dec.raw21, [0, 2.5, 3, 3.5, 4, 5], labels=["<=2.5", "2.5-3", "3-3.5", "3.5-4", ">4"])
dec["size"] = pd.cut(dec.rating_count_2021, [29, 199, 999, 1e9], labels=["30-199", "200-999", ">=1,000"])
t45b = dec.groupby(["size", "lvl_band"], observed=True).agg(
    films=("slug", "size"), displayed=("d_shown", "mean"), members=("d_raw", "mean"), decay=("d_decay", "mean"),
    switch=("d_switch", "mean"))
lbx.write_table(t45b, "T45b_decomposition_by_level_and_size", floatfmt=".3f")

# fig6: members / decay / switch by 2021 raw level, for >= 200-rating films in two size bands
fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2), sharey=True)
for ax, size, part in zip(axes, ["200-999", ">=1,000"], ["(a)", "(b)"]):
    g = t45b.loc[size]
    x = np.arange(len(g))
    for k, (c, lab, col) in enumerate([("members", "raw-mean change", lbx.S1),
                                        ("decay", "change in old-rule adjustment", lbx.S3),
                                        ("switch", "2026 rule contrast", lbx.S2)]):
        ax.bar(x + (k - 1) * 0.26, g[c], 0.24, color=col, label=lab)
    ax.scatter(x, g.displayed, marker="D", s=26, color=lbx.INK, zorder=3, label="displayed change")
    ax.axhline(0, color=lbx.INK2, lw=0.8)
    ax.set_xticks(x, [f"{i}\n(n={int(v):,})" for i, v in zip(g.index, g.films)])
    ax.set_title(part, loc="left")
    ax.set_xlabel("2021 raw mean (stars)")
axes[0].set_ylabel("mean change 2021-2026, stars")
axes[1].legend(loc="upper right", fontsize=7)
lbx.save(fig, "fig6_level_members_vs_formula")

# ----------------------------------------------------------------------------- N3 dominance reversals
L = live[live.slug.isin(clean.slug) & live.shown.notna()].merge(
    clean[["slug", "rating", "rating_count", "lang"]], on="slug")
M = L[HC].fillna(0).to_numpy(float)
L["n"] = M.sum(axis=1)
keep = ((L.n >= 1000) & (L.n / L.rating_count).between(0.99, 1.02) & ((L.shown - L.rating).abs() <= 0.0101)).to_numpy()
L, M = L[keep].reset_index(drop=True), M[keep]
o = np.argsort(L.n.to_numpy(), kind="stable")
L, M = L.iloc[o].reset_index(drop=True), M[o]
F = np.cumsum(M / M.sum(1, keepdims=True), 1)[:, :9]
nn, disp, lang = L.n.to_numpy(), L.shown.to_numpy(), L.lang.fillna("?").to_numpy()
cats = ["near-equal counts (ratio <= 1.10)", "counts within 2%", "exactly equal counts", "both >= 10k", "both >= 100k",
        "same 2021 language", "different 2021 language"]
tot = {c: [0, 0] for c in cats}
films_in_reversal, examples = set(), []
for i in range(len(L)):
    j = np.arange(i + 1, np.searchsorted(nn, nn[i] * 1.10, side="right"))
    if not len(j):
        continue
    i_dom = (F[i] <= F[j] + 1e-12).all(1) & (F[i] < F[j] - 1e-12).any(1)
    j_dom = (F[j] <= F[i] + 1e-12).all(1) & (F[j] < F[i] - 1e-12).any(1)
    dom = i_dom | j_dom
    rev = (i_dom & (disp[i] <= disp[j] - 0.02)) | (j_dom & (disp[j] <= disp[i] - 0.02))
    masks = {cats[0]: np.ones(len(j), bool), cats[1]: nn[j] <= nn[i] * 1.02, cats[2]: nn[j] == nn[i],
             cats[3]: np.minimum(nn[i], nn[j]) >= 10_000, cats[4]: np.minimum(nn[i], nn[j]) >= 100_000,
             cats[5]: lang[j] == lang[i], cats[6]: lang[j] != lang[i]}
    for c, mk in masks.items():
        tot[c][0] += (dom & mk).sum()
        tot[c][1] += (rev & mk).sum()
    for k in j[rev]:
        films_in_reversal.update((L.slug[i], L.slug[k]))
        if nn[k] == nn[i]:
            sep = np.abs(F[i] - F[k]).min()
            examples.append((abs(disp[i] - disp[k]) * sep, i, k))
t46 = pd.DataFrame({c: {"dominance-ordered pairs": v[0], "reversed by >= 0.02 stars": v[1],
                        "reversal rate": v[1] / v[0] if v[0] else np.nan} for c, v in tot.items()}).T
t46.loc["films (>= 1,000 ratings, counts and score consistent with the crawl)", "dominance-ordered pairs"] = len(L)
t46.loc["films involved in at least one reversal", "dominance-ordered pairs"] = len(films_in_reversal)
lbx.write_table(t46, "T46_dominance_reversals", floatfmt=".4f")

# fig9: the exact-count reversal with the clearest separation x score gap
_, i, k = max(examples)
lo_i, hi_i = (i, k) if disp[i] < disp[k] else (k, i)          # lo_i = dominant film with the lower score
fig, ax = plt.subplots(figsize=(6.4, 3.0))
w = 0.2
for off, idx, col in [(-w / 2, lo_i, lbx.S1), (w / 2, hi_i, lbx.S2)]:
    sh = M[idx] / M[idx].sum()
    title = clean.set_index("slug").title.get(L.slug[idx], L.slug[idx])
    ax.bar(STARS + off, sh * 100, w * 0.95, color=col,
           label=f"{title}: raw mean {sh @ STARS:.2f}, displayed {disp[idx]:.2f}")
ax.set_xticks(STARS)
ax.set_xlabel("star rating")
ax.set_ylabel("share of ratings, %")

ax.set_ylim(0, 100 * max((M[lo_i] / M[lo_i].sum()).max(), (M[hi_i] / M[hi_i].sum()).max()) * 1.35)
ax.legend(loc="upper right", fontsize=7)
lbx.save(fig, "fig9_dominance_reversal_example")

# ----------------------------------------------------------------------------- N4 persistence and pass-through
post = ar[(ar.t >= "2023-07-01") & (ar.t <= "2025-12-31")].sort_values("t").groupby("slug").head(1)
A = feats(post)
w2 = A.join(Z, lsuffix="_a", rsuffix="_z", how="inner")
w2 = w2[w2.index.isin(clean.slug) & (w2.n_a >= 1000) & (w2.n_z > w2.n_a) & w2.disp_a.notna() & w2.disp_z.notna()]
w2 = w2.join(CL)
for sfx in ("_a", "_z"):
    w2[f"wedge{sfx}"] = w2[f"disp{sfx}"] - w2[f"raw{sfx}"]
fa = FLEX.replace("lvl", "lvl_a").replace("ln", "ln_a").replace("sd", "sd_a").replace("s5", "s5_a").replace("slo", "slo_a")
fz = FLEX.replace("lvl", "lvl_z").replace("ln", "ln_z").replace("sd", "sd_z").replace("s5", "s5_z").replace("slo", "slo_z")
ra = smf.ols(f"wedge_a ~ {fa}", w2).fit().resid
rz = smf.ols(f"wedge_z ~ {fz}", w2).fit().resid
fold = rng.permutation(len(w2)) % 5                             # 5-fold out-of-sample residuals


def bounded(sfx):
    """FLEX with spline boundary knots fixed to the full-sample range, so held-out films can be predicted."""
    b_ = {v: (w2[f"{v}_{sfx}"].min() - 1e-9, w2[f"{v}_{sfx}"].max() + 1e-9) for v in ("lvl", "ln")}
    return (f"bs(lvl_{sfx}, df=5, lower_bound={b_['lvl'][0]}, upper_bound={b_['lvl'][1]}) + "
            f"bs(ln_{sfx}, df=5, lower_bound={b_['ln'][0]}, upper_bound={b_['ln'][1]}) + "
            f"lvl_{sfx}:ln_{sfx} + sd_{sfx} + s5_{sfx} + slo_{sfx}")


oa, oz = pd.Series(np.nan, index=w2.index), pd.Series(np.nan, index=w2.index)
for k in range(5):
    tr, te = w2[fold != k], w2[fold == k]
    oa[te.index] = te.wedge_a - smf.ols(f"wedge_a ~ {bounded('a')}", tr).fit().predict(te)
    oz[te.index] = te.wedge_z - smf.ols(f"wedge_z ~ {bounded('z')}", tr).fit().predict(te)
rows = {"films": len(w2), "median years between readings": ((pd.Timestamp("2026-10-19") - post.set_index("slug").t
                                                              .reindex(w2.index)).dt.days / 365.25).median(),
        "residual corr, in-sample": np.corrcoef(ra, rz)[0, 1],
        "residual corr, 5-fold out-of-sample": np.corrcoef(oa, oz)[0, 1],
        "slope of 2026 residual on earlier residual (out-of-sample)": smf.ols("z ~ a", pd.DataFrame({"z": oz, "a": oa})).fit().params.a,
        "share of 2026 residual MSE removed by earlier residual (out-of-sample)":
            1 - ((oz - smf.ols("z ~ a", pd.DataFrame({"z": oz, "a": oa})).fit().fittedvalues) ** 2).mean() / (oz ** 2).mean()}
w2["d_disp"], w2["d_raw"] = w2.disp_z - w2.disp_a, w2.raw_z - w2.raw_a
w2["d_sd"], w2["d_s5"], w2["d_slo"] = w2.sd_z - w2.sd_a, w2.s5_z - w2.s5_a, w2.slo_z - w2.slo_a
w2["new_share"] = (w2.n_z - w2.n_a) / w2.n_z
w2["exp_bayes"] = ((w2.n_z * w2.raw_z + 500 * 3.45) / (w2.n_z + 500)) - ((w2.n_a * w2.raw_a + 500 * 3.45) / (w2.n_a + 500))
rows["pass-through implied by a fixed Bayesian prior (C=500, m=3.45)"] = smf.ols("exp_bayes ~ d_raw", w2).fit().params.d_raw
for lab, rhs, df in [("pass-through, raw slope", "d_raw", w2),
                     ("pass-through, + level, count, new-rating share", "d_raw + lvl_a + ln_a + new_share", w2),
                     ("pass-through, + shape changes", "d_raw + lvl_a + ln_a + new_share + d_sd + d_s5 + d_slo", w2),
                     ("pass-through, 2026 count 1k-10k", "d_raw + lvl_a + ln_a + new_share", w2[w2.n_z < 10_000]),
                     ("pass-through, 10k-100k", "d_raw + lvl_a + ln_a + new_share", w2[(w2.n_z >= 10_000) & (w2.n_z < 100_000)]),
                     ("pass-through, >= 100k", "d_raw + lvl_a + ln_a + new_share", w2[w2.n_z >= 100_000])]:
    f = clustered(f"d_disp ~ {rhs}", df)
    rows[lab] = f"{f.params.d_raw:.3f} [{f.conf_int().loc['d_raw', 0]:.3f}, {f.conf_int().loc['d_raw', 1]:.3f}] (n={int(f.nobs)})"
w2["ns_t"] = pd.qcut(w2.new_share, 3, labels=["low", "mid", "high"])
for t, s in w2.groupby("ns_t", observed=True):
    f = clustered("d_disp ~ d_raw + lvl_a + ln_a", s)
    rows[f"pass-through, new-rating share {t} (median {s.new_share.median():.2f})"] = \
        f"{f.params.d_raw:.3f} [{f.conf_int().loc['d_raw', 0]:.3f}, {f.conf_int().loc['d_raw', 1]:.3f}]"
t47 = pd.Series(rows, name="value").to_frame()
lbx.write_table(t47, "T47_hidden_adjustment_persistence_and_passthrough", floatfmt=".3f")

# ----------------------------------------------------------------------------- N5 interior whole-star contrast
x = np.arange(2, 9)                                            # bins 1.5 ... 4.5 (no 0.5 or 5-star bins)
s_ = STARS[x]


def H(df, degree=3):
    X = np.column_stack([s_ ** d for d in range(degree + 1)] + [(s_ == np.round(s_)).astype(float)])
    return np.log(df[HC].fillna(0).to_numpy(float)[:, x] + 0.5) @ np.linalg.pinv(X)[-1]


pre = ar[(ar.t >= "2022-01-01") & (ar.t < lbx.CHANGE)].sort_values("t").groupby("slug").tail(1)
pre = pre[pre[HC].fillna(0).sum(axis=1) >= 1000]
c3 = pd.DataFrame({"slug": pre.slug.to_numpy(), "Hb": H(pre), "Hb4": H(pre, 4), "nb": pre[HC].fillna(0).sum(axis=1).to_numpy(),
                   "ihb": pre[HC].fillna(0).to_numpy(float)[:, [2, 4, 6, 8]].sum(axis=1) / pre[HC].fillna(0).to_numpy(float)[:, 1:9].sum(axis=1)})
zz = pd.DataFrame({"slug": live.slug.to_numpy(), "Hz": H(live), "Hz4": H(live, 4), "nz": live[HC].fillna(0).sum(axis=1).to_numpy(),
                   "ihz": live[HC].fillna(0).to_numpy(float)[:, [2, 4, 6, 8]].sum(axis=1) / live[HC].fillna(0).to_numpy(float)[:, 1:9].sum(axis=1)})
c3 = c3.merge(zz, on="slug")
c3 = c3[c3.slug.isin(clean.slug) & (c3.nz > c3.nb)].merge(clean[["slug", "cluster"]], on="slug")


def boot_mean(v, groups, reps=2000):
    g = pd.DataFrame({"v": v.to_numpy(), "g": groups.to_numpy()}).groupby("g").v.agg(["sum", "size"]).to_numpy()
    draws = [g[idx, 0].sum() / g[idx, 1].sum() for idx in (rng.integers(0, len(g), len(g)) for _ in range(reps))]
    return np.percentile(draws, [2.5, 97.5])


rows = []
for lab, s, col in [("main (cubic shape)", c3, "c"), ("quartic shape", c3, "q"), ("baseline count >= 10k", c3[c3.nb >= 10_000], "c")]:
    dH = (s.Hz - s.Hb) if col == "c" else (s.Hz4 - s.Hb4)
    lo, hi = boot_mean(dH, s.cluster)
    rows.append({"spec": lab, "films": len(s), "mean change in H": dH.mean(), "whole-star factor change %": 100 * (np.exp(dH.mean()) - 1),
                 "ci_lo %": 100 * (np.exp(lo) - 1), "ci_hi %": 100 * (np.exp(hi) - 1), "share of films with H up": (dH > 0).mean()})
di = c3.ihz - c3.ihb
lo, hi = boot_mean(di, c3.cluster)
rows.append({"spec": "interior (1-4.5 star) half-star share, change in points", "films": len(c3), "mean change in H": np.nan,
             "whole-star factor change %": 100 * di.mean(), "ci_lo %": 100 * lo, "ci_hi %": 100 * hi, "share of films with H up": (di < 0).mean()})
t48 = pd.DataFrame(rows).set_index("spec")
t48["median years between readings"] = ((pd.Timestamp("2026-10-19") - pre.set_index("slug").t.reindex(c3.slug)).dt.days / 365.25).median()
lbx.write_table(t48, "T48_interior_whole_star_shift", floatfmt=".4f")

# ----------------------------------------------------------------------------- N6 half-star share by audience size
q = Z[Z.disp.notna()].join(CL, how="inner")
q["wedge"] = q.disp - q.raw
GEN = sorted({g for gs in q.genre_list for g in gs})
for g in GEN:
    q["g_" + g.replace(" ", "_")] = q.genre_list.map(lambda gs, g=g: g in gs).astype(int)
q["eng"] = q.english.astype(int)
q["coh"] = q.cohort.astype(str)
GT = " + ".join("g_" + g.replace(" ", "_") for g in GEN)
rows = []
for lab, s in [("200-999", q[q.n < 1000]), ("1k-10k", q[(q.n >= 1000) & (q.n < 10_000)]), (">= 10k", q[q.n >= 10_000])]:
    f = clustered(f"wedge ~ half + {FLEX} + {GT} + eng + C(coh)", s)
    lo, hi = f.conf_int().loc["half"]
    rows.append({"2026 ratings": lab, "films": int(f.nobs), "per +10 pp half-star share": 0.1 * f.params.half,
                 "ci_lo": 0.1 * lo, "ci_hi": 0.1 * hi, "p": f.pvalues.half, "half-star share sd": s.half.std()})
lbx.write_table(pd.DataFrame(rows).set_index("2026 ratings"), "T49_half_star_share_and_adjustment", floatfmt=".4f")

for t in ["T44_decomposition_sensitivity_to_C", "T45_level_gradient_members_vs_formula", "T46_dominance_reversals",
          "T47_hidden_adjustment_persistence_and_passthrough", "T48_interior_whole_star_shift", "T49_half_star_share_and_adjustment"]:
    print(f"--- {t}\n" + (lbx.TAB / f"{t}.md").read_text())
