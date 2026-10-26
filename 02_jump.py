"""
The 19 Jun 2023 weighted-rating change, separated from organic drift, on the Internet Archive sample
(films with >= 1,000 ratings in Feb 2021, random-order prefix of the 2,932-film queue).

Each film's dated readings = Feb-2021 scrape + archive snapshots + Oct-2026 crawl. Estimators of the step J:
  E1 raw bracket:      mean(r_after - r_before), first/last reading either side of the change (includes drift)
  E2 gap intercept:    OLS  r_after - r_before = J + delta * gap_years   (drift over the gap is absorbed by delta)
  E3 film drift-adj.:  each film's own pre- and post-change linear drift removed from its bracket difference
  E4 interval model:   all consecutive reading pairs, d_r = J*crosses + d_pre*years_before + d_post*years_after,
                       SEs clustered by film; placebo dates added one at a time
Also: decomposition of each film's 2021->2026 change into pre-drift + jump + post-drift, measurement checks,
heterogeneity of the jump by rating level / count / cohort / language (genre: separate script, after full crawl).

  .venv/bin/python 02_jump.py
"""
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import lbx

lbx.style()
CHG = lbx.CHANGE
YEAR = 365.25
MIN_SPAN = 180  # days; shortest span used to estimate a film's own drift rate

long, films = lbx.load_archive()
long["yr"] = (long.t - CHG).dt.days / YEAR   # years relative to the change


# ----------------------------------------------------------------------------- per-film brackets
def bracket(g):
    g = g.sort_values("t")
    pre, post = g[g.t < CHG], g[g.t >= CHG]
    if pre.empty or post.empty:
        return None
    b, a = pre.iloc[-1], post.iloc[0]
    out = {"slug": g.slug.iloc[0], "detailed": g.detailed.iloc[0], "t_before": b.t, "t_after": a.t,
           "r_before": b.rating, "r_after": a.rating, "n_before": b.rating_count, "n_after": a.rating_count,
           "src_before": b.source, "src_after": a.source,
           "r2021": g[g.source == "scrape2021"].rating.iloc[0], "r2026": g[g.source == "crawl2026"].rating.iloc[0],
           "n2021": g[g.source == "scrape2021"].rating_count.iloc[0],
           "n2026": g[g.source == "crawl2026"].rating_count.iloc[0]}
    # own drift rates: from the latest pre reading >= MIN_SPAN days before `before`, and the earliest post
    # reading >= MIN_SPAN days after `after` (usually the Feb-2021 scrape and the Oct-2026 crawl)
    pre_ref = pre[pre.t <= b.t - pd.Timedelta(days=MIN_SPAN)]
    post_ref = post[post.t >= a.t + pd.Timedelta(days=MIN_SPAN)]
    out["rate_pre"] = ((b.rating - pre_ref.iloc[-1].rating) / ((b.t - pre_ref.iloc[-1].t).days / YEAR)
                       if len(pre_ref) else np.nan)
    out["rate_post"] = ((post_ref.iloc[0].rating - a.rating) / ((post_ref.iloc[0].t - a.t).days / YEAR)
                        if len(post_ref) else np.nan)
    return out


br = pd.DataFrame([x for _, g in long.groupby("slug") if (x := bracket(g)) is not None])
br["gap_yr"] = (br.t_after - br.t_before).dt.days / YEAR
br["pre_gap_yr"] = (CHG - br.t_before).dt.days / YEAR
br["post_gap_yr"] = (br.t_after - CHG).dt.days / YEAR
br["jump_raw"] = br.r_after - br.r_before
# films without an own rate (no reading far enough away) borrow the sample median rate
rate_pre = br.rate_pre.fillna(br.rate_pre.median())
rate_post = br.rate_post.fillna(br.rate_post.median())
br["r_minus"] = br.r_before + rate_pre * br.pre_gap_yr     # value just before the change
br["r_plus"] = br.r_after - rate_post * br.post_gap_yr     # value just after the change
br["jump_adj"] = br.r_plus - br.r_minus
br["pre_drift"] = br.r_minus - br.r2021
br["post_drift"] = br.r2026 - br.r_plus
br["total"] = br.r2026 - br.r2021
br = br.merge(films[["slug", "year_2021", "cohort", "english", "is_tv", "genre_list", "lang"]], on="slug")
(lbx.DATA / "derived").mkdir(exist_ok=True)
br.to_csv(lbx.DATA / "derived" / "jump_per_film.csv", index=False)   # film-level: kept with the raw data, not released
n = len(br)


# ----------------------------------------------------------------------------- estimators
def ols(formula, df, cluster=None):
    if cluster is not None:
        return smf.ols(formula, df).fit(cov_type="cluster", cov_kwds={"groups": cluster})
    return smf.ols(formula, df).fit(cov_type="HC3")


def row(label, est, lo, hi, nobs, note=""):
    return {"estimator": label, "J (stars)": est, "ci_lo": lo, "ci_hi": hi, "N": nobs, "note": note}


est = []
m = lbx.mean_ci(br.jump_raw)
est.append(row("E1 raw bracket mean", m["mean"], m.ci_lo, m.ci_hi, n,
               f"median gap {br.gap_yr.median() * 12:.1f} months; {(br.jump_raw > 0).mean():.0%} rose"))
f2 = ols("jump_raw ~ gap_yr", br)
ci = f2.conf_int().loc["Intercept"]
est.append(row("E2 gap-regression intercept", f2.params.Intercept, ci[0], ci[1], n,
               f"drift over gap {f2.params.gap_yr:+.3f}/yr"))
for months in (6, 12):
    sub = br[br.gap_yr <= months / 12]
    if len(sub) >= 30:
        f = ols("jump_raw ~ gap_yr", sub)
        ci = f.conf_int().loc["Intercept"]
        est.append(row(f"E2 intercept, gap <= {months} months", f.params.Intercept, ci[0], ci[1], len(sub)))
m = lbx.mean_ci(br.jump_adj)
est.append(row("E3 film drift-adjusted mean", m["mean"], m.ci_lo, m.ci_hi, n,
               f"median {m['median']:+.3f}; {(br.jump_adj > 0).mean():.0%} positive"))

# E4: consecutive intervals
iv = []
for slug, g in long.sort_values("t").groupby("slug"):
    g = g.reset_index(drop=True)
    for i in range(len(g) - 1):
        s, e = g.loc[i], g.loc[i + 1]
        iv.append({"slug": slug, "t0": s.t, "t1": e.t, "dr": e.rating - s.rating})
iv = pd.DataFrame(iv)


def split(df, date):
    cross = (df.t0 < date) & (df.t1 >= date)
    yrs_pre = ((np.minimum(df.t1, date) - df.t0).dt.days.clip(lower=0)) / YEAR
    yrs_post = ((df.t1 - np.maximum(df.t0, date)).dt.days.clip(lower=0)) / YEAR
    return cross.astype(int), yrs_pre, yrs_post


iv["cross"], iv["yrs_pre"], iv["yrs_post"] = split(iv, CHG)
f4 = ols("dr ~ 0 + cross + yrs_pre + yrs_post", iv, cluster=iv.slug.astype("category").cat.codes)
ci = f4.conf_int().loc["cross"]
est.append(row("E4 interval model (clustered by film)", f4.params.cross, ci[0], ci[1], len(iv),
               f"drift {f4.params.yrs_pre:+.3f}/yr before, {f4.params.yrs_post:+.3f}/yr after"))

# E5: tight brackets - last reading before within ~2.5 months, first after within ~3.5 months (model-free)
tight = br[(br.t_before >= "2023-04-01") & (br.t_after <= "2023-09-30")]
m = lbx.mean_ci(tight.jump_raw)
est.append(row("E5 tight brackets (Apr-Jun -> Jun-Sep 2023)", m["mean"], m.ci_lo, m.ci_hi, len(tight),
               f"SD {tight.jump_raw.std():.3f}; drift over gap not removed"))

# E6: film fixed effects, readings since 2019: rating = a_i + b*t + J*post + k*(t - break)+
fe_data = long[long.t >= "2019-01-01"]
fe_data = fe_data[fe_data.groupby("slug").slug.transform("size") >= 3].copy()


def fe_fit(brk, step=True, kink=True):
    t = (fe_data.t - pd.Timestamp("2021-01-01")).dt.days / YEAR
    X = pd.DataFrame({"t": t})
    if step:
        X["step"] = (fe_data.t >= brk).astype(float)
    if kink:
        X["kink"] = ((fe_data.t - brk).dt.days / YEAR).clip(lower=0)
    Xd = X - X.groupby(fe_data.slug).transform("mean")                     # within-film demeaning
    yd = fe_data.rating - fe_data.groupby("slug").rating.transform("mean")
    k = fe_data.slug.nunique()
    import statsmodels.api as sm
    f = sm.OLS(yd.values, Xd).fit(cov_type="cluster", cov_kwds={"groups": fe_data.slug.astype("category").cat.codes})
    f.df_resid_fe = len(yd) - k - X.shape[1]
    return f


f6 = fe_fit(CHG)
lo, hi = f6.conf_int().loc["step"]
est.append(row("E6 film fixed effects: step + slope change", f6.params.step, lo, hi, len(fe_data),
               f"slope {f6.params.t:+.3f}/yr before, {f6.params.t + f6.params.kink:+.3f}/yr after "
               f"({fe_data.slug.nunique()} films)"))

# E7: bracket difference with a step and a drift-over-gap that both vary with rating count and level.
# Films snapshotted close to the change are much more popular (and higher rated) than the rest, and the
# step shrinks with both, so short-gap subsets understate the sample-average step; E7 evaluates at the means.
h = br.assign(lcc=np.log10(br.n_before) - np.log10(br.n_before).mean(), lvc=br.r_minus - br.r_minus.mean())
f7 = ols("jump_raw ~ lcc + lvc + gap_yr + gap_yr:lcc + gap_yr:lvc", h)
lo, hi = f7.conf_int().loc["Intercept"]
est.append(row("E7 gap model, step & drift vary with count and level", f7.params.Intercept, lo, hi, n,
               f"step {f7.params.lcc:+.3f} per log10 count, {f7.params.lvc:+.3f} per star"))
t_est = pd.DataFrame(est).set_index("estimator")

# break-date scan: where does a step + slope change fit best? (independent check of the announced date)
scan = []
for brk in pd.date_range("2021-09-01", "2025-06-01", freq="MS").append(pd.DatetimeIndex([CHG])).sort_values():
    f = fe_fit(brk)
    scan.append({"break": brk.date(), "SSR": f.ssr, "step": f.params.step, "slope change": f.params.kink,
                 "SSR step only": fe_fit(brk, kink=False).ssr, "SSR slope change only": fe_fit(brk, step=False).ssr})
scan = pd.DataFrame(scan).set_index("break")
scan["dSSR vs best"] = scan.SSR - scan.SSR.min()
lbx.write_table(scan, "T6b_break_date_scan", floatfmt=".4f")


# The single-break scan forces the step and the slope change onto one date. On the full sample that puts the
# "break" in early 2022 (driven by the slope change) while the step-only curve dips at mid-2023. So scan the
# two dates separately: rating = film + slope*t + step*1[t >= s] + kink*(t - k)+ over a grid of (s, k).
def within(df, cols):
    """Within-film demeaned design (dict of column -> array) and outcome, as numpy arrays."""
    X = pd.DataFrame(cols, index=df.index)
    return ((X - X.groupby(df.slug.values).transform("mean")).values,
            (df.rating - df.groupby("slug").rating.transform("mean")).values)


def scan_two(df, steps, kinks):
    t = ((df.t - pd.Timestamp("2021-01-01")).dt.days / YEAR).values
    rows = []
    for s in steps:
        for k in kinks:
            cols = {"t": t, "step": (df.t >= s).astype(float).values}
            if k is not None:
                cols["kink"] = ((df.t - k).dt.days / YEAR).clip(lower=0).values
            X, y = within(df, cols)
            b = np.linalg.lstsq(X, y, rcond=None)[0]
            rows.append({"step date": s.date(), "slope-change date": None if k is None else k.date(),
                         "SSR": ((y - X @ b) ** 2).sum(), "slope": b[0], "step": b[1],
                         "slope change": b[2] if k is not None else np.nan})
    out = pd.DataFrame(rows)
    out["dSSR/sigma2"] = (out.SSR - out.SSR.min()) / (out.SSR.min() / (len(df) - df.slug.nunique() - 3))
    return out


steps_grid = pd.date_range("2021-09-01", "2025-06-01", freq="MS")
two = scan_two(fe_data, steps_grid, list(pd.date_range("2020-06-01", "2025-01-01", freq="MS")))
two.to_csv(lbx.TAB / "T20_two_break_scan_grid.csv", index=False)
step_prof = two.groupby("step date")["dSSR/sigma2"].min()            # best slope-change date for each step date
kink_prof = two.groupby("slope-change date")["dSSR/sigma2"].min()
fe21 = fe_data[fe_data.t >= "2021-01-01"]
fe21 = fe21[fe21.groupby("slug").slug.transform("size") >= 3]
one21 = scan_two(fe21, steps_grid, [None]).set_index("step date")["dSSR/sigma2"]   # 2021+, one constant slope
lbx.write_table(pd.DataFrame({"two-break model, 2019+ (min over slope-change date)": step_prof,
                              "one slope, readings 2021+": one21}), "T20_step_date_profiles", floatfmt=".2f")


def fe_models(df, label):
    t = (df.t - pd.Timestamp("2021-01-01")).dt.days / YEAR
    step = (df.t >= CHG).astype(float)
    kink = ((df.t - CHG).dt.days / YEAR).clip(lower=0)
    out = []
    for name, cols in [("step only (one slope)", {"slope": t, "step": step}),
                       ("step + slope change at the change", {"slope": t, "step": step, "kink": kink})]:
        X = pd.DataFrame(cols)
        Xd = X - X.groupby(df.slug.values).transform("mean")
        yd = df.rating - df.groupby("slug").rating.transform("mean")
        import statsmodels.api as sm
        f = sm.OLS(yd.values, Xd).fit(cov_type="cluster", cov_kwds={"groups": df.slug.astype("category").cat.codes})
        lo, hi = f.conf_int().loc["step"]
        out.append({"readings": label, "model": name, "step": f.params.step, "ci_lo": lo, "ci_hi": hi,
                    "slope before (/yr)": f.params.slope,
                    "slope after (/yr)": f.params.slope + f.params.get("kink", 0.0),
                    "slope change p": f.pvalues.get("kink", np.nan)})
    return out


best2 = two.loc[two.SSR.idxmin()]
t21 = pd.DataFrame(fe_models(fe21, "2021 onward") + fe_models(fe_data, "2019 onward") + [{
    "readings": "2019 onward", "model": f"two-break best: step {best2['step date']}, "
                                         f"slope change {best2['slope-change date']}",
    "step": best2.step, "slope before (/yr)": best2.slope, "slope after (/yr)": best2.slope + best2["slope change"]}])
lbx.write_table(t21.set_index(["readings", "model"]), "T21_step_and_slope_models", floatfmt=".4f")
est.append(row("E8 film FE, one slope, readings 2021+ (step only)", t21.iloc[0].step, t21.iloc[0].ci_lo,
               t21.iloc[0].ci_hi, len(fe21), f"slope {t21.iloc[0]['slope before (/yr)']:+.3f}/yr throughout"))
est.append(row("E9 two-break model (step date and slope-change date free)", best2.step, np.nan, np.nan,
               len(fe_data), f"step at {best2['step date']}, slope change at {best2['slope-change date']}"))
t_est = pd.DataFrame(est).set_index("estimator")
lbx.write_table(t_est, "T5_jump_estimators", floatfmt=".4f")

# placebo dates: same interval model plus a crossing dummy for a fake date
plac = []
for d in ["2019-06-19", "2020-06-19", "2021-06-19", "2022-06-19", "2024-06-19", "2025-06-19"]:
    x = iv.copy()
    x["fake"], _, _ = split(x, pd.Timestamp(d))
    f = ols("dr ~ 0 + cross + fake + yrs_pre + yrs_post", x, cluster=x.slug.astype("category").cat.codes)
    lo, hi = f.conf_int().loc["fake"]
    plac.append({"placebo date": d, "placebo step": f.params.fake, "ci_lo": lo, "ci_hi": hi,
                 "p": f.pvalues.fake, "intervals crossing": int(x.fake.sum()), "real step (same model)": f.params.cross})
lbx.write_table(pd.DataFrame(plac).set_index("placebo date"), "T6_placebo_dates", floatfmt=".4f")

# ----------------------------------------------------------------------------- decomposition
dec = br[["total", "pre_drift", "jump_adj", "post_drift"]].apply(lbx.mean_ci).T
dec["share_of_total"] = dec["mean"] / dec.loc["total", "mean"]
dec["per_year"] = [np.nan,
                   (br.pre_drift / ((CHG - lbx.T2021).days / YEAR)).mean(), np.nan,
                   (br.post_drift / ((pd.to_datetime("2026-10-16") - CHG).days / YEAR)).mean()]
dec.index = ["Total change Feb 2021 → Oct 2026", "Organic drift before the change (Feb 2021 → Jun 2023)",
             "Step at 19 Jun 2023", "Organic drift after the change (Jun 2023 → Oct 2026)"]
lbx.write_table(dec, "T7_decomposition", floatfmt=".4f")

# ----------------------------------------------------------------------------- measurement checks
chk = []
for src, date_col in [("scrape2021", None), ("crawl2026", None)]:
    ref = long[long.source == src][["slug", "t", "rating", "rating_count"]]
    arc = long[long.source == "archive"][["slug", "t", "rating", "rating_count"]]
    j = ref.merge(arc, on="slug", suffixes=("_ref", "_arc"))
    j["days"] = (j.t_arc - j.t_ref).dt.days
    j = j[j.days.abs() <= 60].sort_values("days", key=abs).drop_duplicates("slug")
    if len(j):
        chk.append({"comparison": f"{src} vs archive snapshot within 60 days", "films": len(j),
                    "mean |diff| (stars)": (j.rating_ref - j.rating_arc).abs().mean(),
                    "max |diff|": (j.rating_ref - j.rating_arc).abs().max(),
                    "share identical": (j.rating_ref == j.rating_arc).mean(),
                    "median |days apart|": j.days.abs().median()})
lbx.write_table(pd.DataFrame(chk).set_index("comparison"), "T8_measurement_checks", floatfmt=".4f")

# ----------------------------------------------------------------------------- heterogeneity (non-genre)
het = ols("jump_adj ~ I(r_minus - 3.5) + np.log10(n_before) + C(cohort, Treatment('2000-09')) + english + is_tv",
          br)
drift = ols("I(pre_drift + post_drift) ~ I(r_minus - 3.5) + np.log10(n_before) + C(cohort, Treatment('2000-09'))"
            " + english + is_tv", br)
tab = pd.DataFrame({"jump: coef": het.params, "jump: p": het.pvalues,
                    "organic drift (pre+post): coef": drift.params, "organic drift: p": drift.pvalues})
tab.loc["N"] = [het.nobs, np.nan, drift.nobs, np.nan]
lbx.write_table(tab, "T9_jump_vs_drift_heterogeneity", floatfmt=".4f")

# written-review intensity before the change (review entries per rating at the last pre-change archive
# reading) as a predictor of each component of our decomposition
rv = lbx.archive_tables()[1].query("note == 'ok'")[["slug", "timestamp", "review_count", "rating_count"]]
rv = rv.assign(t_before=pd.to_datetime(rv.timestamp.str[:8]), rr=rv.review_count / rv.rating_count)
rr = br.merge(rv[["slug", "t_before", "rr"]], on=["slug", "t_before"]).dropna(subset=["rr"])
rows = []
for y, lab in [("pre_drift", "drift before (Feb 2021 → change)"), ("jump_adj", "step (E3)"),
               ("post_drift", "drift after (change → Oct 2026)")]:
    for ctl, cl in [("I(r_minus - 3.5) + np.log10(n_before)", "level, count"),
                    ("I(r_minus - 3.5) + np.log10(n_before) + C(cohort) + english + is_tv",
                     "level, count, cohort, language, TV")]:
        older = rr[rr.year_2021 < 2018].assign(cohort=lambda x: x.cohort.cat.remove_unused_categories())
        for samp, df in [("", rr), (", without 2018-2020 releases", older)]:
            f = ols(f"{y} ~ rr + {ctl}", df)
            lo, hi = f.conf_int().loc["rr"]
            rows.append({"component": lab, "controls": cl + samp, "per +0.10 review ratio": 0.1 * f.params.rr,
                         "ci_lo": 0.1 * lo, "ci_hi": 0.1 * hi, "p": f.pvalues.rr, "N": int(f.nobs)})
lbx.write_table(pd.DataFrame(rows).set_index(["component", "controls"]), "T19_review_intensity_vs_components",
                floatfmt=".4f")

# ----------------------------------------------------------------------------- figures
# fig5 (headline): A) mean rating relative to each film's own pre-change level, half-year bins;
#                  B) bracket difference vs gap, intercept = the step
long2 = long.merge(br[["slug", "r_minus"]], on="slug")
long2["rel"] = long2.rating - long2.r_minus
long2 = long2[long2.t >= "2019-01-01"]
long2["bin"] = (np.floor(long2.yr * 2) / 2 + 0.25)   # half-year bins centred
g = long2.groupby("bin").rel.apply(lbx.mean_ci).unstack()
g = g[g.n >= 15]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.4), gridspec_kw={"width_ratios": [1.25, 1]})
pre_g, post_g = g[g.index < 0], g[g.index > 0]
for part in (pre_g, post_g):
    ax1.errorbar(part.index, part["mean"], yerr=[part["mean"] - part.ci_lo, part.ci_hi - part["mean"]],
                 color=lbx.S1, marker="o", ms=4.5, lw=1.6, elinewidth=0.9, capsize=0, mec="white", mew=0.8)
ax1.axvline(0, color=lbx.INK2, lw=0.8)
ax1.axhline(0, color=lbx.GRID, lw=0.8)
ax1.text(0.05, ax1.get_ylim()[1] * 0.95 if ax1.get_ylim()[1] > 0 else 0, "19 Jun 2023\nformula change",
         color=lbx.INK2, fontsize=7.5, va="top")
ax1.set_xlabel("Years relative to the formula change (half-year bins)")
ax1.set_ylabel("Rating minus film's own pre-change level")
ax1.set_title("(a)", loc="left")
ax2.plot(pd.to_datetime(step_prof.index), step_prof.values, color=lbx.S1, lw=1.8, marker="o", ms=3.5,
         mec="white", mew=0.6, label="readings 2019+, slope-change date free")
ax2.plot(pd.to_datetime(one21.index), one21.values, color=lbx.S2, lw=1.4, label="readings 2021+, one slope")
ax2.axvline(CHG, color=lbx.INK2, lw=0.8)
best = pd.Timestamp(step_prof.idxmin())
ax2.annotate(f"best step date: {best:%b %Y}", xy=(best, 0), xytext=(best + pd.Timedelta(days=90), -6),
             color=lbx.INK, fontsize=7.5, va="top")
ax2.set_ylim(-14, None)
ax2.xaxis.set_major_locator(mdates.YearLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax2.set_xlabel("Candidate date of the step")
ax2.set_ylabel("Worse fit than best date (ΔSSR/σ²)")
ax2.set_title("(b)", loc="left")
ax2.legend(loc="center left", bbox_to_anchor=(0.0, 0.42))
fig.tight_layout()
lbx.save(fig, "fig5_formula_jump")
plt.close(fig)

# figS1: why the raw bracket difference overstates the step - it grows with the gap between snapshots
fig, ax = plt.subplots(figsize=(5.2, 3.2))
ax.scatter(br.gap_yr * 12, br.jump_raw, s=9, color=lbx.S1, alpha=0.35, linewidths=0)
xx = np.linspace(0, br.gap_yr.quantile(0.98), 50)
ax.plot(xx * 12, f2.params.Intercept + f2.params.gap_yr * xx, color=lbx.S2, lw=2)
ax.axhline(0, color=lbx.INK2, lw=0.8)
lo, hi = f2.conf_int().loc["Intercept"]
ax.annotate(f"intercept {f2.params.Intercept:+.3f} [{lo:+.3f}, {hi:+.3f}]", xy=(0, f2.params.Intercept),
            xytext=(br.gap_yr.quantile(0.6) * 12, br.jump_raw.quantile(0.98)), color=lbx.INK, fontsize=7.5,
            arrowprops=dict(arrowstyle="-", color=lbx.INK2, lw=0.6))
ax.set_xlim(0, br.gap_yr.quantile(0.98) * 12)
ax.set_ylim(br.jump_raw.quantile(0.005), br.jump_raw.quantile(0.995))
ax.set_xlabel("Months between last pre- and first post-change reading")
ax.set_ylabel("Rating after − before (stars)")
ax.set_title("Raw before/after differences grow with the snapshot gap", loc="left")
lbx.save(fig, "figS1_bracket_vs_gap")
plt.close(fig)

# volatility lead: SD of short (<= 1 yr) non-crossing changes, before vs after the change, by audience size
iv["gap"] = (iv.t1 - iv.t0).dt.days / YEAR
iv = iv.merge(long[["slug", "t", "rating_count"]].rename(columns={"t": "t0", "rating_count": "n0"}), on=["slug", "t0"])
short = iv[(iv.cross == 0) & (iv.gap <= 1) & (iv.t0 >= "2019-01-01")].copy()
short["period"] = np.where(short.t0 < CHG, "before change", "after change")
short["n_band"] = pd.cut(short.n0, [0, 10_000, 50_000, 1e9], labels=["<10k ratings", "10k-50k", ">=50k"])
vol = short.groupby(["n_band", "period"], observed=True).agg(
    intervals=("dr", "size"), median_gap_yr=("gap", "median"), mean_change=("dr", "mean"), sd_change=("dr", "std"),
    mean_abs_change=("dr", lambda v: v.abs().mean()))
lbx.write_table(vol, "T10_volatility_before_after", floatfmt=".4f")

# figS4: step vs drift by pre-change rating, timing-based. The level story is covered by the
# histogram measurement (09_new_rule.py: T45, fig6): the level gradient is mostly members' gradual change.
br["lvl"] = pd.cut(br.r_minus, [0, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 5])
fig, ax = plt.subplots(figsize=(5.2, 3.2))
for col, lab, c in [("jump_adj", "step at Jun 2023", lbx.S1), ("pre_drift", "drift Feb 2021 → Jun 2023", lbx.S2),
                    ("post_drift", "drift Jun 2023 → Oct 2026", lbx.S3)]:
    gg = br.groupby("lvl", observed=True)[col].apply(lbx.mean_ci).unstack()
    gg = gg[gg.n >= 15]
    xm = br.groupby("lvl", observed=True).r_minus.median().loc[gg.index]
    ax.errorbar(xm, gg["mean"], yerr=[gg["mean"] - gg.ci_lo, gg.ci_hi - gg["mean"]], color=c, marker="o", ms=4.5,
                lw=1.6, elinewidth=0.9, capsize=0, label=lab, mec="white", mew=0.8)
ax.axhline(0, color=lbx.INK2, lw=0.8)
ax.set_xlabel("Rating just before the change (bin medians)")
ax.set_ylabel("Mean change (stars)")
ax.set_title(f"Step vs organic drift by rating level (n = {n})", loc="left")
ax.legend(loc="upper right")
lbx.save(fig, "figS4_step_vs_drift_by_level_timing_based")
plt.close(fig)

print(f"films with both sides: {n} (detailed {br.detailed.sum()})")
print(t_est.round(4).to_string())
print(dec.round(4).to_string())
