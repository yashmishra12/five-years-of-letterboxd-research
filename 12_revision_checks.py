"""
Four supporting checks (run 06, 07 and 09 first). Each one decides how strongly a headline claim
can be stated; none is meant as a new finding.

  V1 old-rule validation: the pre-change fit by count band and by period, with whole films held out; the
     rule refitted separately on the 2016-20 film pages and the 2021-23 fragments, each tested on the other period;
     film bootstraps of (C, m) and the prior mean m by year; and how far each part of the decomposition moves with
     the 2021-23 parameters and under their uncertainty. (T55, T55b, T55c)
  V2 spread of the decomposition: film-level distribution of the formula switch, members' change, the
     change in the old-rule adjustment and the displayed change, by 2021 audience size. (T58, fig10)
  V3 traits given the full histogram: the "given the 2026 histogram" trait comparisons redone with
     all nine bin shares and the count, (a) linear in the shares, (b) a flexible series model cross-fitted by
     director (partialling out, five folds, repeated over 10 random splits and aggregated by the median), (c) (b)
     trimmed to common support; absolute group means alongside.
     Out-of-fold fit of the displayed score from the histogram, and persistence of the hidden adjustment under the
     flexible model. (T57, T57b)
  V4 break-date simulation: the archive readings were picked around 19 Jun 2023, so the two-break scan of
     T54 is rerun on synthetic ratings at exactly the real films' reading dates: smooth trends with no step, smooth
     phase-ins (at random dates, and centred on the announcement), and true steps at other dates. (T56)

  .venv/bin/python 12_revision_checks.py
"""
import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from scipy.optimize import least_squares
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

import lbx

lbx.style()
STARS = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
HC = [f"h{int(s * 10):02d}" for s in STARS]
PC = [f"p{int(s * 10):02d}" for s in STARS[:9]]          # nine independent shares; the 5-star share is the rest
YEAR = 365.25


def bayes(q, n, raw):
    return (n * raw + q[0] * q[1]) / (n + q[0])


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def hframe(df):
    """Histogram features per row: nine shares, log10 count, level, spread, 5-star and low shares."""
    m = df[HC].fillna(0).to_numpy(float)
    n = m.sum(axis=1)
    P = m / np.where(n > 0, n, np.nan)[:, None]
    raw = P @ STARS
    out = pd.DataFrame(P[:, :9], columns=PC, index=df.index)
    out["n"], out["raw"], out["ln"], out["lvl"] = n, raw, np.log10(np.where(n > 0, n, np.nan)), raw - 3.5
    out["sd"] = np.sqrt((P * (STARS - raw[:, None]) ** 2).sum(axis=1))
    out["share5"], out["share_lo"] = P[:, 9], P[:, 0] + P[:, 1]
    return out


# ----------------------------------------------------------------------------- V1 old-rule validation
rng = np.random.default_rng(2026)                      # own seed per section: edits elsewhere don't move it
rd = pd.read_sql("SELECT * FROM readings", sqlite3.connect(
    f"file:{lbx.DATA / 'archive_hist_pilot.sqlite'}?mode=ro", uri=True))
hf = hframe(rd)
rd["n_hist"], rd["raw"] = hf.n, hf.raw
rd["t"] = pd.to_datetime(rd.served.str[:8], errors="coerce")
pre = rd[(rd.status == "ok") & rd.shown.notna() & (rd.n_hist > 0) & (rd.t < lbx.CHANGE)].reset_index(drop=True)
EARLY, LATE = pre.t.dt.year <= 2020, pre.t.dt.year >= 2021      # 2016-20 film pages | 2021-23 fragments


def fit_old(df, C=None):
    """Least-squares fit of the old rule on readings (as in 06); with C given, only m is free."""
    if C is None:
        return least_squares(lambda q: bayes(q, df.shown_n.values, df.raw.values) - df.shown.values, x0=[50, 3.0],
                             bounds=([0, 0.5], [1e5, 5])).x
    m = least_squares(lambda q: bayes((C, q[0]), df.shown_n.values, df.raw.values) - df.shown.values, x0=[3.4],
                      bounds=([0.5], [5])).x[0]
    return np.array([C, m])


def film_boot(s, reps, **kw):
    g, f = {k: v for k, v in s.groupby("slug")}, s.slug.unique()
    return np.array([fit_old(pd.concat([g[x] for x in rng.choice(f, len(f))]), **kw) for _ in range(reps)])


def held_out(s):
    """Residuals with whole films held out: five folds, each predicted from a fit on the other films of s."""
    f = pd.Series(rng.permutation(s.slug.nunique()) % 5, index=s.slug.unique())[s.slug].to_numpy()
    out = pd.Series(np.nan, index=s.index)
    for k in range(5):
        te = f == k
        out[te] = s.shown[te] - bayes(fit_old(s[~te]), s.shown_n[te], s.raw[te])
    return out


def summarize(s, q, held=None):
    """Fit quality of the rule with parameters q on readings s."""
    pred = bayes(q, s.shown_n, s.raw)
    res = s.shown - pred
    return {"readings": len(s), "films": s.slug.nunique(), "median count": s.shown_n.median(),
            "mean old-rule pull": (pred - s.raw).abs().mean(), "RMS error": rms(res), "mean error": res.mean(),
            "share within rounding": (res.abs() <= 0.0051).mean(),
            "RMS error, films held out": rms(held) if held is not None else np.nan,
            "share within rounding, held out": (held.abs() <= 0.0051).mean() if held is not None else np.nan,
            "RMS error of inverted raw mean": rms(res * (s.shown_n + q[0]) / s.shown_n)}


q0, qE, qL = fit_old(pre), fit_old(pre[EARLY]), fit_old(pre[LATE])
pre["held_pooled"] = held_out(pre)
pre["held_period"] = pd.concat([held_out(pre[EARLY]), held_out(pre[LATE])])
pre["band"] = pd.cut(pre.shown_n, [0, 199, 999, 4999, np.inf], labels=["< 200", "200-999", "1,000-4,999", ">= 5,000"])
pre["period"] = pd.cut(pre.t.dt.year, [0, 2020, 2021, 2022, 2023], labels=["2016-2020", "2021", "2022", "2023 (to 18 Jun)"])
near21 = pre.kind == "frag_near_2021"
rows = {"pooled fit (T29): all pre-change readings": summarize(pre, q0, pre.held_pooled)}
for col in ("band", "period"):
    for b, s in pre.groupby(col, observed=True):
        rows[f"pooled fit: {'count ' if col == 'band' else ''}{b}"] = summarize(s, q0, s.held_pooled)
rows["pooled fit: fragments near Feb 2021 (the inversion check)"] = summarize(pre[near21], q0, pre.held_pooled[near21])
for lab, q, mask in [("2016-20 fit", qE, EARLY), ("2021-23 fit", qL, LATE)]:
    s = pre[mask]
    rows[f"{lab}: its own readings"] = summarize(s, q, s.held_period)
    for b, sb in s.groupby("band", observed=True):
        rows[f"{lab}: count {b}"] = summarize(sb, q, sb.held_period)
rows["2021-23 fit: fragments near Feb 2021"] = summarize(pre[near21], qL, pre.held_period[near21])
rows["2021-23 fit applied to 2016-20 readings"] = summarize(pre[EARLY], qL)
rows["2016-20 fit applied to 2021-23 readings"] = summarize(pre[LATE], qE)
t55 = pd.DataFrame(rows).T
lbx.write_table(t55, "T55_old_rule_validation", floatfmt=".4f")

# the parameters: pooled and by period (film bootstrap), and the prior mean by period with C held at the pooled fit
boot0, bootE, bootL = film_boot(pre, 500), film_boot(pre[EARLY], 300), film_boot(pre[LATE], 300)
rows = {}
for lab, s, q, b in [("pooled 2016-23 (T29)", pre, q0, boot0), ("2016-20 film pages", pre[EARLY], qE, bootE),
                     ("2021-23 fragments", pre[LATE], qL, bootL)]:
    rows[f"{lab}: C and m free"] = {"readings": len(s), "films": s.slug.nunique(), "C": q[0],
                                    "C ci_lo": np.percentile(b[:, 0], 2.5), "C ci_hi": np.percentile(b[:, 0], 97.5),
                                    "m": q[1], "m ci_lo": np.percentile(b[:, 1], 2.5), "m ci_hi": np.percentile(b[:, 1], 97.5),
                                    "corr(C, m)": np.corrcoef(b.T)[0, 1]}
for lab, s in [("2016-2020", pre[EARLY]), ("2016-2020, count >= 1,000", pre[EARLY & (pre.shown_n >= 1000)]),
               ("2021", pre[pre.t.dt.year == 2021]), ("2022", pre[pre.t.dt.year == 2022]),
               ("2023 (to 18 Jun)", pre[pre.t.dt.year == 2023]), ("2021-23, count >= 1,000", pre[LATE & (pre.shown_n >= 1000)])]:
    b = film_boot(s, 300, C=q0[0])[:, 1]
    if lab == "2021":
        m21_ci = np.percentile(b, [2.5, 97.5])
    rows[f"m by period, C fixed at {q0[0]:.1f}: {lab}"] = {"readings": len(s), "films": s.slug.nunique(), "C": q0[0],
                                                          "m": fit_old(s, C=q0[0])[1], "m ci_lo": np.percentile(b, 2.5),
                                                          "m ci_hi": np.percentile(b, 97.5)}
t55c = pd.DataFrame(rows).T
lbx.write_table(t55c, "T55c_old_rule_parameters_by_period", floatfmt=".4f")

dec = pd.read_csv(lbx.DATA / "derived" / "panel_decomposition.csv")
sc = dec[dec.shown.notna()].copy()                       # clean panel films with a 2026 score


def parts(q21, q26, s):
    """Mean decomposition with the old rule's parameters q21 for the 2021 inversion and q26 for the counterfactual."""
    raw21 = (s.rating_2021 * (s.rating_count_2021 + q21[0]) - q21[0] * q21[1]) / s.rating_count_2021
    old26 = bayes(q26, s.n_hist, s.raw)
    return np.array([(s.raw - raw21).mean(), ((old26 - s.raw) - (s.rating_2021 - raw21)).mean(),
                     (s.shown - old26).mean(), (old26 - s.rating_2021).mean()])


PARTS = ["members", "decay", "switch", "old-rule change (members + decay)"]
BANDS21 = {"30-199": (30, 199), "200-999": (200, 999), "1,000-9,999": (1000, 9999), ">= 10,000": (10000, np.inf),
           ">= 200": (200, np.inf), "all scored": (0, np.inf)}
rows = []
for lab, (lo, hi) in BANDS21.items():
    s = sc[sc.rating_count_2021.between(lo, hi)]
    a0, aL, aM = parts(q0, q0, s), parts(qL, qL, s), parts(qL, (qL[0], qL[1] + 0.05), s)
    a21 = [parts((qL[0], m_), (qL[0], m_), s) for m_ in m21_ci]
    s0 = np.array([parts(q, q, s) for q in boot0]).std(0)
    sL = np.array([parts(q, q, s) for q in bootL]).std(0)
    for i, nm in enumerate(PARTS):
        rows.append({"2021 ratings": lab, "part": nm, "films": len(s), "2021-23 fit (main tables)": aL[i],
                     "sd, 2021-23 bootstrap": sL[i],
                     "m at the 2021-only 95% limits": f"{min(a21[0][i], a21[1][i]):+.3f} to {max(a21[0][i], a21[1][i]):+.3f}",
                     "old rule's m +0.05 by 2026": aM[i], "pooled 2016-23 fit (earlier tables)": a0[i],
                     "sd, pooled bootstrap": s0[i],
                     "median rounding half-width of 2021 raw mean": (0.005 * (1 + qL[0] / s.rating_count_2021)).median()})
t55b = pd.DataFrame(rows).set_index(["2021 ratings", "part"])
lbx.write_table(t55b, "T55b_decomposition_rule_uncertainty", floatfmt=".4f")
print(t55.round(4).to_string())
print(t55c.round(4).to_string())
print(t55b.round(4).to_string())

# ----------------------------------------------------------------------------- V2 spread of the decomposition
rng = np.random.default_rng(2026)                      # own seed per section: edits elsewhere don't move it
COMPS = {"formula switch": "d_switch", "members (raw-mean change)": "d_raw",
         "change in old-rule adjustment (decay)": "d_decay", "displayed change": "d_shown"}
rows = []
for comp, col in COMPS.items():
    for lab, (lo, hi) in BANDS21.items():
        v = sc.loc[sc.rating_count_2021.between(lo, hi), col]
        q = v.quantile([0.05, 0.10, 0.25, 0.5, 0.75, 0.90, 0.95])
        rows.append({"component": comp, "2021 ratings": lab, "films": len(v), "mean": v.mean(), "sd": v.std(),
                     "p5": q[0.05], "p10": q[0.10], "p25": q[0.25], "median": q[0.5], "p75": q[0.75],
                     "p90": q[0.90], "p95": q[0.95], "share > 0": (v > 0).mean(), "share > +0.05": (v > 0.05).mean(),
                     "share < -0.05": (v < -0.05).mean(), "mean |x|": v.abs().mean()})
t58 = pd.DataFrame(rows).set_index(["component", "2021 ratings"])
lbx.write_table(t58, "T58_decomposition_spread_by_size", floatfmt=".3f")
print(t58.round(3).to_string())

fig, ax = plt.subplots(figsize=(6.4, 3.3))
bands = ["30-199", "200-999", "1,000-9,999", ">= 10,000"]
for off, comp, colr, legend in [(-0.13, "formula switch", lbx.S2, "2026 rule contrast"), (0.13, "members (raw-mean change)", lbx.S1, "raw-mean change")]:
    for i, b in enumerate(bands):
        if comp.startswith("members") and b == "30-199":
            continue                                     # depends on C for small films (T44, T55b); not shown
        r = t58.loc[(comp, b)]
        ax.plot([i + off] * 2, [r.p10, r.p90], color=colr, lw=1.5, solid_capstyle="round")
        ax.plot([i + off] * 2, [r.p25, r.p75], color=colr, lw=7, solid_capstyle="round")
        ax.plot(i + off, r["mean"], "o", ms=5, color="white", mec=lbx.INK, mew=1.2, zorder=4)
    ax.plot([], [], color=colr, lw=7, label=legend)
ax.plot([], [], "o", ms=5, color="white", mec=lbx.INK, mew=1.2, label="mean")
ax.axhline(0, color=lbx.AXIS, lw=1, zorder=1)
ax.set_xticks(range(len(bands)), [f"{b}\n({int(t58.loc[('formula switch', b), 'films']):,} films)" for b in bands])
ax.set_xlabel("ratings in February 2021")
ax.set_ylabel("change 2021 to 2026, stars")
ax.legend(loc="upper right", fontsize=7)
lbx.save(fig, "fig10_decomposition_spread_by_size")

# ----------------------------------------------------------------------------- V3 traits given the full histogram
rng = np.random.default_rng(2026)                      # own seed per section: edits elsewhere don't move it
p = lbx.load_panel()
live = pd.read_sql("SELECT * FROM hist WHERE status = 'ok'", sqlite3.connect(
    f"file:{lbx.DATA / 'histograms.sqlite'}?mode=ro", uri=True))
lv = hframe(live).assign(slug=live.slug.to_numpy()).drop(columns="raw").rename(columns={"n": "n26"})
d = p[p.clean].merge(dec[["slug", "d_raw", "d_switch", "shown", "rating_count_2021"]].rename(
    columns={"rating_count_2021": "n21"}), on="slug").merge(lv, on="slug")
d = d.merge(lbx.load_leads(p).drop_duplicates("slug"), on="slug", how="left")
rv = lbx.archive_tables()[1].query("note == 'ok'")[["slug", "timestamp", "review_count", "rating_count"]]
rv = rv[pd.to_datetime(rv.timestamp.str[:8], errors="coerce") < lbx.CHANGE].sort_values("timestamp")
rv = rv.groupby("slug").tail(1).assign(rr=lambda x: x.review_count / x.rating_count)[["slug", "rr"]]
d = d.merge(rv, on="slug", how="left")
d["cluster"] = d.director1.fillna("film:" + d.slug)
GENRES = sorted({g for gs in d.genre_list for g in gs})
COL = {g: "g_" + g.replace(" ", "_").replace("-", "_") for g in GENRES}
for g in GENRES:
    d[COL[g]] = d.genre_list.map(lambda gs, g=g: g in gs).astype(int)
d["g_none"] = (d.genre_list.str.len() == 0).astype(int)
d["english"], d["is_tv"] = d.english.astype(int), d.is_tv.astype(int)
d["rr10"] = d.rr * 10                                    # effect per +0.10 written reviews per rating
scored = d[d.shown.notna()].reset_index(drop=True)
DBASE = list(COL.values()) + ["g_none", "english", "is_tv"]
SIX = "bs(lvl, df=5) + bs(ln, df=5) + lvl:ln + sd + share5 + share_lo"


def basis(df, kind):
    """Histogram design matrix (no intercept), built on the given films only (no outcome used)."""
    if kind == "six":
        return patsy.dmatrix(SIX + " - 1", df, return_type="matrix")
    if kind == "nine":
        return np.column_stack([df[PC].to_numpy(), patsy.dmatrix("bs(ln, df=5) - 1", df, return_type="matrix")])
    S = df[PC].to_numpy()
    i, j = np.triu_indices(9, 1)
    parts_ = [patsy.dmatrix(f"cr({c}, df=4, constraints='center') - 1", df, return_type="matrix") for c in PC]
    parts_ += [patsy.dmatrix(SIX + " + bs(ln, df=8) + I(sd**2) + sd:ln - 1", df, return_type="matrix"),
               S * df.ln.to_numpy()[:, None], S[:, i] * S[:, j]]
    return np.column_stack(parts_)


REPEATS = 10        # cross-fitting repeated over random fold splits; estimates aggregated by the median


def folds_for(groups, k=5):
    g = pd.factorize(groups)[0]
    return rng.permutation(g.max() + 1)[g] % k


def crossfit(B, Y, groups=None, k=5, f=None):
    """Out-of-fold least-squares predictions of each column of Y from basis B; folds by group (director), or `f`."""
    f = folds_for(groups, k) if f is None else f
    Bs = (B - B.mean(0)) / np.where(B.std(0) > 0, B.std(0), 1)
    B1 = np.column_stack([np.ones(len(B)), Bs])
    out = np.empty(Y.shape)
    for j in range(k):
        tr, te = f != j, f == j
        out[te] = B1[te] @ np.linalg.lstsq(B1[tr], Y[tr], rcond=None)[0]
    return out


def cluster_ols(y, X, groups):
    return sm.OLS(y, sm.add_constant(X, has_constant="add")).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(groups)[0]})


def trait_models(df, dcols):
    """Coefficient of each trait column on the formula switch under four ways of holding the histogram fixed."""
    y, Dm = df.d_switch.to_numpy(float), df[dcols].to_numpy(float)
    out = {}
    for kind in ("six", "nine"):
        f = cluster_ols(y, np.column_stack([Dm, basis(df, kind)]), df.cluster)
        out[kind] = (f.params[1:1 + len(dcols)], f.bse[1:1 + len(dcols)], f.pvalues[1:1 + len(dcols)])
    B = basis(df, "flex")
    est, var, r2s, trims = [], [], [], []
    for _ in range(REPEATS):
        pred = crossfit(B, np.column_stack([y, Dm]), df.cluster)
        ry, rD = y - pred[:, 0], Dm - pred[:, 1:]
        f = cluster_ols(ry, rD, df.cluster)
        est.append(f.params[1:])
        var.append(f.bse[1:] ** 2)
        r2s.append(1 - (rD ** 2).mean(0) / Dm.var(0))
        trim = []
        for i in range(len(dcols)):
            binary = set(np.unique(Dm[:, i])) <= {0.0, 1.0}
            e = pred[:, 1 + i]
            if not (binary and 0.1 <= Dm[:, i].mean() <= 0.9):
                trim.append((np.nan, np.nan))
                continue
            keep = (e >= 0.05) & (e <= 0.95)
            trim.append((cluster_ols(ry[keep], rD[keep], df.cluster[keep]).params[1 + i], keep.mean()))
        trims.append(trim)
    est, var = np.array(est), np.array(var)
    theta = np.median(est, 0)
    se = np.sqrt(np.median(var + (est - theta) ** 2, 0))    # median aggregation over splits (Chernozhukov et al. 2018)
    out["flex"] = (theta, se, 2 * norm.sf(np.abs(theta / se)))
    out["flex range"] = (est.min(0), est.max(0))
    return out, np.median(r2s, 0), np.median(np.array(trims), 0)


def smd(x, dmask):
    a, b = x[dmask], x[~dmask]
    return (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)


rows = []
specs = [(DBASE, scored, [(g, COL[g]) for g in GENRES] + [("English-language", "english"), ("TV entry", "is_tv")]),
         (DBASE + ["lead_female"], scored.dropna(subset=["lead_female"]).reset_index(drop=True),
          [("female lead (IMDb first billed)", "lead_female")]),
         (DBASE + ["rr10"], scored.dropna(subset=["rr10"]).reset_index(drop=True),
          [("review intensity, +0.10 per rating", "rr10")])]
big21 = d[d.shown.notna() & (d.n21 >= 200)]
for dcols, df, wanted in specs:
    out, r2, trim = trait_models(df, dcols)
    for lab, c in wanted:
        i = dcols.index(c)
        binary = df[c].isin([0, 1]).all()
        m1 = df[c] == 1 if binary else df[c] > df[c].median()
        bb = big21.dropna(subset=[c])
        mb = (bb[c] == 1) if binary else (bb[c] > df[c].median())
        rows.append({
            "trait": lab, "films with trait": int(m1.sum()),
            "switch, mean: trait / others": f"{df.d_switch[m1].mean():+.3f} / {df.d_switch[~m1].mean():+.3f}",
            "members (n2021>=200), mean: trait / others": f"{bb.d_raw[mb].mean():+.3f} / {bb.d_raw[~mb].mean():+.3f}",
            "six summaries (Table 4)": out["six"][0][i], "p six": out["six"][2][i],
            "nine shares + count spline": out["nine"][0][i], "ci nine": 1.96 * out["nine"][1][i], "p nine": out["nine"][2][i],
            "flexible, cross-fitted": out["flex"][0][i], "ci flexible": 1.96 * out["flex"][1][i], "p flexible": out["flex"][2][i],
            "flexible: range over splits": f"{out['flex range'][0][i]:+.4f} to {out['flex range'][1][i]:+.4f}",
            "flexible, overlap-trimmed": trim[i][0], "share kept by trimming": trim[i][1],
            "out-of-fold R2 of trait on histogram": r2[i],
            "SMD raw mean": smd(df.lvl.to_numpy(), m1.to_numpy()), "SMD log count": smd(df.ln.to_numpy(), m1.to_numpy())})
t57 = pd.DataFrame(rows).set_index("trait")
gm = t57.index.isin(GENRES)
for col in ("six", "nine", "flexible"):
    t57.loc[gm, f"p {col} Holm"] = multipletests(t57.loc[gm, f"p {col}"], method="holm")[1]
lbx.write_table(t57, "T57_traits_full_histogram", floatfmt=".4f")
print(t57.round(4).to_string())

# out-of-fold fit of the displayed score from the 2026 histogram alone, and persistence of the residual
rows = {}
y = scored.shown.to_numpy(float)
oof = {k: y - crossfit(basis(scored, k), y[:, None], scored.slug)[:, 0] for k in ("six", "nine", "flex")}
for k, lab in [("six", "six summaries"), ("nine", "nine shares + count spline"), ("flex", "flexible series")]:
    rows[f"displayed score from histogram, out-of-fold RMS: {lab}"] = {"all": rms(oof[k])} | {
        b: rms(oof[k][scored.n26.between(lo, hi).to_numpy()])
        for b, (lo, hi) in {"200-999": (200, 999), "1k-10k": (1000, 9999), "10k-100k": (10000, 99999),
                            ">=100k": (100000, np.inf)}.items()}
ar = pd.read_sql("SELECT * FROM readings WHERE status = 'ok'", sqlite3.connect(
    f"file:{lbx.DATA / 'archive_hist_pilot.sqlite'}?mode=ro", uri=True))
ar["t"] = pd.to_datetime(ar.served.str[:8])
post = ar[(ar.t >= "2023-07-01") & (ar.t <= "2025-12-31")].sort_values("t").groupby("slug").head(1)
A = hframe(post).assign(slug=post.slug.to_numpy(), disp=post.shown.to_numpy()).set_index("slug")
Z = hframe(live).assign(slug=live.slug.to_numpy(), disp=live.shown.to_numpy()).set_index("slug")
w = A.join(Z, lsuffix="_a", rsuffix="_z", how="inner")
w = w[w.index.isin(p[p.clean].slug) & (w.n_a >= 1000) & (w.n_z > w.n_a) & w.disp_a.notna() & w.disp_z.notna()]
FEAT = PC + ["ln", "lvl", "sd", "share5", "share_lo"]
for k, lab in [("six", "six summaries"), ("flex", "flexible series")]:
    Bs = {sfx: basis(w[[c + sfx for c in FEAT]].set_axis(FEAT, axis=1), k) for sfx in ("_a", "_z")}
    ys = {sfx: (w[f"disp{sfx}"] - w[f"raw{sfx}"]).to_numpy(float) for sfx in ("_a", "_z")}
    rs, slopes = [], []
    for _ in range(2 * REPEATS):
        f = folds_for(w.index.to_numpy())                  # the same films held out at both dates
        res = {sfx: ys[sfx] - crossfit(Bs[sfx], ys[sfx][:, None], f=f)[:, 0] for sfx in ("_a", "_z")}
        rs.append(np.corrcoef(res["_a"], res["_z"])[0, 1])
        slopes.append(np.polyfit(res["_a"], res["_z"], 1)[0])
    rows[f"persistence of hidden adjustment, out-of-fold: {lab}"] = {
        "all": np.median(rs), "200-999": np.nan, "1k-10k": np.nan, "10k-100k": np.nan, ">=100k": np.nan,
        "films": len(w), "slope": np.median(slopes), "range over splits": f"r {min(rs):.3f} to {max(rs):.3f}"}
t57b = pd.DataFrame(rows).T
lbx.write_table(t57b, "T57b_full_histogram_fit_and_persistence", floatfmt=".4f")
print(t57b.round(4).to_string())

# ----------------------------------------------------------------------------- V4 break-date simulation
rng = np.random.default_rng(2026)                      # own seed per section: edits elsewhere don't move it
long, _ = lbx.load_archive()
fe = long[long.t >= "2019-01-01"]
fe = fe[fe.groupby("slug").slug.transform("size") >= 3].sort_values(["slug", "t"]).reset_index(drop=True)
codes = pd.factorize(fe.slug)[0]
cnt = np.bincount(codes)
nf = len(cnt)


def dm(x):
    return x - (np.bincount(codes, x) / cnt)[codes]


def yrs(ts):
    return (pd.Timestamp(ts) - pd.Timestamp("2021-01-01")).days / YEAR


tt = ((fe.t - pd.Timestamp("2021-01-01")).dt.days / YEAR).to_numpy()
since = tt - yrs(lbx.CHANGE)
STEPS = pd.date_range("2022-01-01", "2024-12-01", freq="MS")      # same grid as T54
KINKS = pd.date_range("2020-07-01", "2024-10-01", freq="QS")
cols = [tt] + [(tt >= yrs(s_)).astype(float) for s_ in STEPS] + [np.clip(tt - yrs(k_), 0, None) for k_ in KINKS]
cols += [(since >= 0).astype(float), np.clip(since, 0, None),                   # T17 design: step at 19 Jun 2023,
         np.clip(since / 0.5, 0, 1), np.clip(since - 0.5, 0, None)]             # or a 6-month ramp from it
Dd = np.column_stack([dm(c) for c in cols])
G = Dd.T @ Dd
iS = 1 + np.arange(len(STEPS))
iK = 1 + len(STEPS) + np.arange(len(KINKS))
iR = 1 + len(STEPS) + len(KINKS)
PAIRS = np.array([(0, a, b) for a in iS for b in iK])
NOSTEP = np.array([(0, b) for b in iK])
G_pairs = G[PAIRS[:, :, None], PAIRS[:, None, :]]
G_nostep = G[NOSTEP[:, :, None], NOSTEP[:, None, :]]
DOF = len(fe) - nf - 3


def scan(y):
    """Two-break scan (step date and slope-change date free) as in T54; plus how much a step improves on the best
    no-step fit and how much worse a 6-month ramp fits than a step at 19 Jun 2023, both in units of sigma2."""
    yd = dm(y)
    b, yy = Dd.T @ yd, yd @ yd
    bp = b[PAIRS]
    sol = np.linalg.solve(G_pairs, bp[..., None])[..., 0]
    ssr = yy - (sol * bp).sum(1)
    j = ssr.argmin()
    s2 = ssr[j] / DOF
    bn = b[NOSTEP]
    ssr_n = (yy - (np.linalg.solve(G_nostep, bn[..., None])[..., 0] * bn).sum(1)).min()

    def ssr_of(idx):
        return yy - b[idx] @ np.linalg.solve(G[np.ix_(idx, idx)], b[idx])

    return {"best step": STEPS[PAIRS[j, 1] - 1], "best kink": KINKS[PAIRS[j, 2] - iK[0]], "step": sol[j, 1],
            "slope": sol[j, 0], "kink": sol[j, 2], "ssr": ssr[j], "strength": (ssr_n - ssr[j]) / s2,
            "ramp6 minus step": (ssr_of(np.array([0, iR + 2, iR + 3])) - ssr_of(np.array([0, iR, iR + 1]))) / s2}


obs = scan(fe.rating.to_numpy(float))
J, KSTAR = obs["step"], obs["best kink"]
trend = obs["slope"] * tt + obs["kink"] * np.clip(tt - yrs(KSTAR), 0, None)
fitted = trend + J * (tt >= yrs(obs["best step"]))
mu = (np.bincount(codes, fe.rating.to_numpy() - fitted) / cnt)[codes]   # each film's own level
sigma2 = obs["ssr"] / DOF
pos = fe.groupby(codes).cumcount().to_numpy()
tbar = (np.bincount(codes, tt) / cnt)[codes]


def ou(tau):
    x = rng.standard_normal(len(fe))
    for j in range(1, pos.max() + 1):
        r = np.flatnonzero(pos == j)
        rho = np.exp(-(tt[r] - tt[r - 1]) / tau)
        x[r] = rho * x[r - 1] + np.sqrt(1 - rho ** 2) * x[r]
    return x


GEN = {"white": lambda: rng.standard_normal(len(fe)),
       "slopes": lambda: rng.standard_normal(nf)[codes] * (tt - tbar),
       "ou1": lambda: ou(1.0), "ou3": lambda: ou(3.0)}
MIX = {"white": {"white": 1.0}, "mixed": {"white": 1 / 3, "slopes": 1 / 3, "ou1": 1 / 3},
       "persistent": {"slopes": 0.5, "ou3": 0.5}}
# scale each component so it contributes its share of the real residual variance after the fixed effects and trend
Xtrue = np.column_stack([dm(tt), dm(np.clip(tt - yrs(KSTAR), 0, None)), dm((tt >= yrs(obs["best step"])).astype(float))])


def resid_var(e):
    ed = dm(e)
    return ((ed - Xtrue @ np.linalg.lstsq(Xtrue, ed, rcond=None)[0]) ** 2).sum() / DOF


UNIT = {k: np.mean([resid_var(f()) for _ in range(20)]) for k, f in GEN.items()}
target = sigma2 - 0.01 ** 2 / 12                            # leave room for rounding to two decimals


def noise(kind):
    return sum(np.sqrt(w_ * target / UNIT[c]) * GEN[c]() for c, w_ in MIX[kind].items())


def logistic(centre, months):
    """Smooth phase-in of size J whose 10-90% rise takes `months`."""
    return J / (1 + np.exp(-(tt - yrs(centre)) / (months / 12 / (2 * np.log(9)))))


SCEN = [("no step: trend only, white noise", "white", lambda c: 0 * tt, None),
        ("no step: trend only, mixed noise", "mixed", lambda c: 0 * tt, None),
        ("no step: trend only, persistent noise", "persistent", lambda c: 0 * tt, None),
        ("no step: 6-month phase-in at a random date", "mixed", lambda c: logistic(c, 6), "random"),
        ("no step: 3-month phase-in centred 19 Jun 2023", "mixed", lambda c: logistic(c, 3), lbx.CHANGE),
        ("no step: 6-month phase-in centred 19 Jun 2023", "mixed", lambda c: logistic(c, 6), lbx.CHANGE),
        ("no step: 12-month phase-in centred 19 Jun 2023", "mixed", lambda c: logistic(c, 12), lbx.CHANGE),
        ("true step 19 Jun 2023 (the fitted model)", "mixed", lambda c: J * (tt >= yrs(c)), lbx.CHANGE),
        ("true step 19 Jun 2023, persistent noise", "persistent", lambda c: J * (tt >= yrs(c)), lbx.CHANGE)] + [
       (f"true step {c:%d %b %Y}", "mixed", lambda c: J * (tt >= yrs(c)), c)
       for c in pd.to_datetime(["2022-10-15", "2023-02-15", "2023-10-15", "2024-04-15"])]
REPS = 300
rows = [{"scenario": "observed data", "reps": 1, "best step date: median": obs["best step"].date(),
         "share best step May-Jul 2023": np.nan, "share best step within 45 days of true date": np.nan,
         "step at best date: median": J, "step improvement dSSR/s2: median": obs["strength"],
         "share >= observed improvement": np.nan, "6-month ramp minus step dSSR/s2: median": obs["ramp6 minus step"],
         "share >= observed ramp penalty": np.nan}]
for lab, kind, sig, centre in SCEN:
    out = []
    for _ in range(REPS):
        c = (pd.Timestamp("2022-03-01") + pd.Timedelta(days=int(rng.integers(0, 945)))) if centre == "random" else centre
        y = np.round(mu + trend + sig(c) + noise(kind), 2)
        r = scan(y)
        r["true"] = c
        out.append(r)
    o = pd.DataFrame(out)
    bst = pd.to_datetime(o["best step"])
    near = (bst - pd.to_datetime(o.true)).abs() <= pd.Timedelta(days=45) if centre is not None else pd.Series(np.nan)
    rows.append({"scenario": lab, "reps": REPS, "best step date: median": bst.sort_values().iloc[REPS // 2].date(),
                 "share best step May-Jul 2023": bst.between("2023-05-01", "2023-07-01").mean(),
                 "share best step within 45 days of true date": near.mean(),
                 "step at best date: median": o.step.median(), "step improvement dSSR/s2: median": o.strength.median(),
                 "share >= observed improvement": (o.strength >= obs["strength"]).mean(),
                 "6-month ramp minus step dSSR/s2: median": o["ramp6 minus step"].median(),
                 "share >= observed ramp penalty": (o["ramp6 minus step"] >= obs["ramp6 minus step"]).mean()})
t56 = pd.DataFrame(rows).set_index("scenario")
t56.loc["design", "best step date: median"] = f"{nf:,} films / {len(fe):,} readings (2019+, >= 3 readings); residual sd {np.sqrt(sigma2):.4f}"
lbx.write_table(t56, "T56_break_date_simulation", floatfmt=".3f")
print(t56.to_string())
