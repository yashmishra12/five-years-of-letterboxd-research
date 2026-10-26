"""
Overall 2021 -> 2026 change in displayed ratings for the matched panel.
Tables: T1 sample flow, T2 change by strata, T3 regressions, T4 regression-to-the-mean check.
Figures: fig1 distribution, fig2 change by 2021 rating level x count band, fig3 by release cohort, fig4 by growth.

  .venv/bin/python 01_panel.py
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import lbx

lbx.style()
p = lbx.load_panel()
a = lbx.analysis_panel(p)                       # clean matches with a 2026 average
s = a[a.rating_count_2021 >= lbx.MIN_2026]      # selection-free: >=200 ratings already in 2021
films26 = pd.read_csv(lbx.DATA / "films_2026.csv", usecols=["slug", "rating", "tmdb_id"])

# ----------------------------------------------------------------------------- T1 sample flow
raw_path = lbx.ROOT / "feb2021.json"
n_listed = len(json.load(open(raw_path))) if raw_path.exists() else lbx.FILMS_LISTED_2021
m = p[p.matched]
flow = [
    ("2021 scrape: films listed", n_listed),
    ("2021: films with a displayed rating (deduplicated title+year)", len(p)),
    ("Matched to a 2026 Letterboxd page", int(p.matched.sum())),
    ("  excluded: duplicate slug", int(m.dup_slug.sum())),
    ("  excluded: 2021 release (pre-release ratings)", int(m.prerelease.sum())),
    ("  excluded: suspect match (runtime x1.5, disjoint genres or count shrank)", int(m.suspect.sum())),
    ("  excluded: confirmed title swap, manual list (not caught above)",
     int((m.manual_mismatch & ~m.suspect & ~m.dup_slug & ~m.prerelease).sum())),
    ("Clean matches", int(p.clean.sum())),
    ("  of which flagged for identity review (kept)", int((p.clean & p.identity_review).sum())),
    ("Clean matches still showing a 2026 average (analysis panel)", len(a)),
    ("  of which >=200 ratings in 2021 (selection-free subsample)", len(s)),
    ("  of which >=1,000 ratings in 2021 (archive-eligible)", int((a.rating_count_2021 >= 1000).sum())),
]
t1 = pd.DataFrame(flow, columns=["step", "films"]).set_index("step")
lbx.write_table(t1, "T1_sample_flow", floatfmt=",.0f")

# ----------------------------------------------------------------------------- T2 change by strata
def by(df, col, label):
    out = df.groupby(col, observed=True).d.apply(lbx.mean_ci).unstack()
    out["median_growth"] = df.groupby(col, observed=True).growth.median()
    out.index = pd.MultiIndex.from_product([[label], out.index.astype(str)])
    return out


def strata(df):
    df = df.assign(
        r21_band=pd.cut(df.rating_2021, [0, 2, 2.5, 3, 3.5, 4, 5], labels=["<=2.0", "2.0-2.5", "2.5-3.0",
                                                                              "3.0-3.5", "3.5-4.0", ">4.0"]),
        n21_band=pd.cut(df.rating_count_2021, [29, 99, 199, 999, 9999, 1e8],
                        labels=["30-99", "100-199", "200-999", "1k-10k", ">=10k"]),
        growth_q=pd.qcut(df.growth, 5, labels=["Q1 (least)", "Q2", "Q3", "Q4", "Q5 (most)"]),
        lang_g=df.lang.where(df.lang.isin(df.lang.value_counts().index[:8]), "Other"),
        kind=np.where(df.is_tv, "TV entry", "Film"),
    )
    parts = [lbx.mean_ci(df.d).to_frame().T.assign(median_growth=df.growth.median())]
    parts[0].index = pd.MultiIndex.from_tuples([("All", "all")])
    for col, label in [("r21_band", "2021 rating"), ("n21_band", "2021 rating count"), ("cohort", "Release year"),
                       ("lang_g", "Language"), ("growth_q", "Audience growth quintile"), ("kind", "Type")]:
        parts.append(by(df, col, label))
    return pd.concat(parts)


lbx.write_table(strata(a), "T2a_change_by_strata_all")
lbx.write_table(strata(s), "T2b_change_by_strata_n200")

# ----------------------------------------------------------------------------- T3 regressions
F = "d ~ r21c + log10_n21 + log_growth + C(cohort, Treatment('2000-09')) + english + is_tv"
names = {"r21c": "2021 rating (centred at 3.2)", "log10_n21": "log10 2021 rating count",
         "log_growth": "log audience growth (n2026/n2021)", "english[T.True]": "English-language",
         "is_tv[T.True]": "TV entry", "Intercept": "Intercept"}
rows = {}
s_dir = s[s.director1.notna()]
for label, df, cl in [("(1) all", a, None), ("(2) n2021>=200", s, None),
                      ("(2b) n2021>=200, SEs clustered by director", s_dir, "director1"),
                      ("(3) n2021>=1000", a[a.rating_count_2021 >= 1000], None)]:
    df = df.assign(r21c=df.rating_2021 - 3.2)
    fit = (smf.ols(F, df).fit(cov_type="cluster", cov_kwds={"groups": df[cl].astype("category").cat.codes})
           if cl else smf.ols(F, df).fit(cov_type="HC3"))
    ci = fit.conf_int()
    col = {}
    for k in fit.params.index:
        nm = names.get(k, k.replace("C(cohort, Treatment('2000-09'))[T.", "cohort ").rstrip("]"))
        col[nm] = f"{fit.params[k]:+.3f} [{ci.loc[k, 0]:+.3f}, {ci.loc[k, 1]:+.3f}]"
    col["N"], col["R2"] = f"{int(fit.nobs):,}", f"{fit.rsquared:.3f}"
    rows[label] = col
t3 = pd.DataFrame(rows)
lbx.write_table(t3, "T3_regressions_HC3")

# ----------------------------------------------------------------------------- T4 regression to the mean
# If true ratings did not change, 2021 sampling noise alone gives slope(d ~ r2021) = -noise_var / var(r2021),
# with noise_var ~ sd_user^2 / n. sd_user = 1.0 star is conservative (typical per-film SD of 0.5-5 star ratings
# is ~0.8-1.0), and the displayed weighted average is already shrunk, so this overstates the RTM slope.
rtm = []
for lo, hi, lab in [(30, 100, "30-99"), (100, 200, "100-199"), (200, 1000, "200-999"), (1000, 3000, "1k-3k"),
                    (3000, 1e9, ">=3k")]:
    x = a[(a.rating_count_2021 >= lo) & (a.rating_count_2021 < hi)]
    fit = smf.ols("d ~ rating_2021", x).fit(cov_type="HC3")
    lo_ci, hi_ci = fit.conf_int().loc["rating_2021"]
    rtm.append({"2021 count band": lab, "N": len(x), "observed slope": fit.params["rating_2021"],
                "ci_lo": lo_ci, "ci_hi": hi_ci,
                "slope expected from RTM alone": -np.mean(1 / x.rating_count_2021) / x.rating_2021.var()})
lbx.write_table(pd.DataFrame(rtm).set_index("2021 count band"), "T4_regression_to_mean", floatfmt=".4f")

# ----------------------------------------------------------------------------- figures
# fig1: distribution
fig, ax = plt.subplots(figsize=(5.2, 3.0))
bins = np.arange(-0.805, 1.0, 0.02)
ax.hist(a.d.clip(-0.8, 0.99), bins=bins, color=lbx.S1, edgecolor="white", linewidth=0.4)
ax.axvline(0, color=lbx.INK2, lw=0.8)
ax.axvline(a.d.mean(), color=lbx.INK, lw=1.2)
ax.text(a.d.mean() + 0.02, ax.get_ylim()[1] * 0.92,
        f"mean {a.d.mean():+.3f}\nmedian {a.d.median():+.2f}\n{(a.d > 0).mean():.0%} rose, {(a.d < 0).mean():.0%} fell",
        color=lbx.INK, fontsize=8, va="top")
ax.set_xlabel("Change in displayed rating, Feb 2021 → Oct 2026 (stars)")
ax.set_ylabel("Films")
lbx.save(fig, "fig1_change_distribution")
plt.close(fig)

# fig2: change by 2021 rating, by 2021 count band
fig, ax = plt.subplots(figsize=(5.2, 3.4))
edges = np.arange(1.0, 4.76, 0.25)
bands = [(30, 200, "30–199 ratings in 2021", lbx.S1), (200, 1000, "200–999", lbx.S2),
         (1000, 1e9, "≥1,000", lbx.S3)]
for lo, hi, lab, c in bands:
    x = a[(a.rating_count_2021 >= lo) & (a.rating_count_2021 < hi)]
    g = x.groupby(pd.cut(x.rating_2021, edges), observed=True).d.apply(lbx.mean_ci).unstack()
    g = g[g.n >= 20]
    mid = [iv.mid for iv in g.index]
    ax.errorbar(mid, g["mean"], yerr=[g["mean"] - g.ci_lo, g.ci_hi - g["mean"]], color=c, marker="o",
                ms=4.5, lw=1.6, elinewidth=0.9, capsize=0, label=lab, mec="white", mew=0.8)
ax.axhline(0, color=lbx.INK2, lw=0.8)
ax.set_xlabel("Displayed rating in Feb 2021 (stars, 0.25-star bins)")
ax.set_ylabel("Mean change to Oct 2026 (stars)")
ax.legend(loc="upper right")
lbx.save(fig, "fig2_change_by_level")
plt.close(fig)

# fig3: by release cohort (selection-free subsample)
g = s.groupby("cohort", observed=True).d.apply(lbx.mean_ci).unstack()
fig, ax = plt.subplots(figsize=(5.2, 3.0))
xs = np.arange(len(g))
ax.errorbar(xs, g["mean"], yerr=[g["mean"] - g.ci_lo, g.ci_hi - g["mean"]], color=lbx.S1, marker="o", ms=5,
            lw=0, elinewidth=1.2, capsize=0, mec="white", mew=0.8)
ax.set_xticks(xs, g.index.astype(str))
ax.axhline(0, color=lbx.INK2, lw=0.8)
ax.set_ylim(0, None)
ax.set_xlabel("Release year")
ax.set_ylabel("Mean change, 2021 → 2026 (stars)")
ax.set_title(f"The newest films rose least (films with ≥200 ratings in 2021, n = {len(s):,})", loc="left")
lbx.save(fig, "fig3_change_by_cohort")
plt.close(fig)

# fig4: by audience growth decile, raw and adjusted for 2021 rating, count and cohort
x = s.assign(r21c=s.rating_2021 - 3.2)
base = smf.ols("d ~ r21c + log10_n21 + C(cohort) + english + is_tv", x).fit()
x["d_adj"] = base.resid + x.d.mean()
x["gdec"] = pd.qcut(x.growth, 10)
fig, ax = plt.subplots(figsize=(5.2, 3.2))
for col, lab, c in [("d", "raw", lbx.S1), ("d_adj", "adjusted for 2021 rating, count, cohort, language", lbx.S2)]:
    g = x.groupby("gdec", observed=True)[col].apply(lbx.mean_ci).unstack()
    gx = x.groupby("gdec", observed=True).growth.median()
    ax.errorbar(gx, g["mean"], yerr=[g["mean"] - g.ci_lo, g.ci_hi - g["mean"]], color=c, marker="o", ms=4.5,
                lw=1.6, elinewidth=0.9, capsize=0, label=lab, mec="white", mew=0.8)
ax.set_xscale("log")
ax.set_xticks([2, 3, 4, 6, 10, 20], ["2×", "3×", "4×", "6×", "10×", "20×"])
ax.minorticks_off()
ax.set_xlabel("Audience growth, ratings 2026 / ratings 2021 (decile medians, log scale)")
ax.set_ylabel("Mean change (stars)")
ax.set_title("Films whose audience grew more did not fall", loc="left")
ax.legend(loc="upper left")
lbx.save(fig, "fig4_change_vs_growth")
plt.close(fig)

print(t1.to_string())
print(f"panel mean change {a.d.mean():+.4f}; n>=200 {s.d.mean():+.4f}")

# films awaiting the author's identity review: kept in the analysis; listed with the raw data (film-level, not released)
rev = p[p.identity_review][["slug", "title_2021", "year_2021", "language_2021", "primary_language", "duration_2021",
                            "runtime", "genres_2021", "genres", "directors", "rating_count_2021", "rating_count",
                            "rating_2021", "rating", "d", "clean"]]
rev = rev.assign(note=rev.slug.map(lbx.IDENTITY_REVIEW))
(lbx.DATA / "derived").mkdir(exist_ok=True)
rev.to_csv(lbx.DATA / "derived" / "identity_review.csv", index=False)
print(f"identity review: {len(rev)} film(s) kept but flagged -> {lbx.DATA / 'derived' / 'identity_review.csv'}")
