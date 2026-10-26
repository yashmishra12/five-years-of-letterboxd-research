"""
Histogram analyses on the full panel (run 06_histograms.py first: it writes derived/panel_decomposition.csv and T29).

  H6 the 2023 formula: which rule maps a film's 2026 histogram to its displayed score? Bayesian-average fits with
     one prior weight, a weight that varies with the rating count, and one that also varies with the rating spread;
     free fits within count bands. (T33, T34)
  H7 who gained: members' change and formula switch by genre, language, TV, lead gender and pre-change review
     intensity. The switch is modelled twice: (a) with 2021 baseline controls (level, count) = who gained;
     (b) with the 2026 histogram itself (raw mean, count, spread, extreme shares) = does the formula use anything
     beyond the histogram? Members' change only for films with >= 200 ratings in 2021 (robust to the old-formula
     prior weight). Director-clustered SEs; Holm across genres. (T35, T36)
  H8 spread and polarization: within-film change in rating SD and extreme-rating shares for archive films with a
     pre-change histogram. (T37)
  H9 small films and selection: films with < 200 ratings in 2021 that crossed 200 (score shown) vs stayed below,
     by audience growth. (T38)
  Headline means with director-clustered bootstrap CIs. (T39) Figure: fig8 decomposition by audience size.

  .venv/bin/python 07_histogram_analysis.py
"""
import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.optimize import least_squares
from statsmodels.stats.multitest import multipletests

import lbx

lbx.style()
STARS = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
HCOLS = [f"h{int(s * 10):02d}" for s in STARS]
rng = np.random.default_rng(2026)


def with_raw(df):
    h = df[HCOLS].fillna(0).to_numpy(float)
    n = h.sum(1)
    nn = np.where(n > 0, n, np.nan)
    raw = (h * STARS).sum(1) / nn
    return df.assign(n_hist=n, raw=raw, sd=np.sqrt((h * (STARS - raw[:, None]) ** 2).sum(1) / nn),
                     share5=h[:, 9] / nn, share_lo=(h[:, 0] + h[:, 1]) / nn)


def bayes(C, m, n, raw):
    return (n * raw + C * m) / (n + C)


t29 = pd.read_csv(lbx.TAB / "T29_old_formula_fit.csv", index_col=0)
C0 = t29.loc["fitted C (prior weight, in ratings)", "readings"]
M0 = t29.loc["fitted m (prior mean, stars)", "readings"]

live = with_raw(pd.read_sql("SELECT * FROM hist WHERE status = 'ok'", sqlite3.connect(
    f"file:{lbx.DATA / 'histograms.sqlite'}?mode=ro", uri=True)))
lv = live[live.shown.notna() & (live.n_hist > 0)].copy()
BANDS = [199, 999, 9999, 99999, 1e9]
BLAB = ["200-999", "1k-10k", "10k-100k", ">=100k"]
lv["n_band"] = pd.cut(lv.n_hist, BANDS, labels=BLAB)

# ----------------------------------------------------------------------------- H6: the 2023 formula
n, raw, sd, y = lv.n_hist.values, lv.raw.values, lv.sd.values, lv.shown.values
models = {
    f"old formula (C={C0:.0f}, m={M0:.2f}), for reference": (lambda q: bayes(C0, M0, n, raw), None),
    "Bayesian average, one prior weight C": (lambda q: bayes(q[0], q[1], n, raw), [200, 3.3]),
    "C grows with count: C = a * n^b": (lambda q: bayes(np.exp(q[0]) * n ** q[2], q[1], n, raw), [4, 3.3, 0.3]),
    "C = a * n^b * sd^k (spread too)": (lambda q: bayes(np.exp(q[0]) * n ** q[2] * sd ** q[3], q[1], n, raw),
                                        [4, 3.3, 0.3, 0.0]),
}
rows, fits = {}, {}
for name, (f, x0) in models.items():
    q = least_squares(lambda q: f(q) - y, x0=x0).x if x0 else None
    res = y - f(q)
    fits[name] = res
    r = {"RMS": np.sqrt(np.mean(res ** 2)), "mean residual": res.mean(),
         "share within ±0.005": np.mean(np.abs(res) <= 0.0051), "share within ±0.02": np.mean(np.abs(res) <= 0.0201)}
    r.update({f"RMS {b}": np.sqrt(np.mean(res[lv.n_band.values == b] ** 2)) for b in BLAB})
    if q is not None:
        r["parameters"] = ", ".join(f"{v:.3f}" for v in q)
    rows[name] = r
# films with >= 10k ratings, where any prior is negligible: can ANY rule on the histogram reproduce the score?
# (a) each star value gets its own weight; (b) each star value is re-scored (monotone). Fitted on these films only.
P = lv[HCOLS].fillna(0).to_numpy(float) / n[:, None]
bigf = n >= 10000


def star_weights(q, P):
    w = np.exp(np.insert(q, 5, 0.0))
    return (P * w) @ STARS / (P @ w)


def rescored(q, P):
    return P @ np.cumsum(np.concatenate([[q[0]], np.exp(q[1:])]))


for name, f, x0 in [("films >= 10k: own weight per star value", star_weights, np.zeros(9)),
                    ("films >= 10k: star values re-scored (monotone)", rescored,
                     np.concatenate([[0.5], np.log(np.full(9, 0.5))]))]:
    q = least_squares(lambda q: f(q, P[bigf]) - y[bigf], x0).x
    res = y[bigf] - f(q, P[bigf])
    rows[name] = {"RMS": np.sqrt(np.mean(res ** 2)), "mean residual": res.mean(),
                  "share within ±0.005": np.mean(np.abs(res) <= 0.0051),
                  "share within ±0.02": np.mean(np.abs(res) <= 0.0201), "films fitted": int(bigf.sum())}
gap_big = (y - raw)[bigf]
rows["films >= 10k: displayed - raw (no model)"] = {"RMS": np.sqrt(np.mean(gap_big ** 2)), "mean residual": gap_big.mean(),
                                                     "share within ±0.005": np.mean(np.abs(gap_big) <= 0.0051),
                                                     "share within ±0.02": np.mean(np.abs(gap_big) <= 0.0201),
                                                     "films fitted": int(bigf.sum())}
t33 = pd.DataFrame(rows).T
t33.insert(0, "films", len(lv))
lbx.write_table(t33, "T33_new_formula_fits", floatfmt=".4f")

# free Bayesian fit within count bands: how strong is the prior at each size?
rows = {}
for b in BLAB:
    s = lv[lv.n_band == b]
    q = least_squares(lambda q: bayes(q[0], q[1], s.n_hist.values, s.raw.values) - s.shown.values,
                      x0=[200, 3.3], bounds=([0, 0.5], [1e7, 5])).x
    res = s.shown - bayes(q[0], q[1], s.n_hist, s.raw)
    rows[b] = {"films": len(s), "median count": s.n_hist.median(), "fitted C": q[0], "fitted m": q[1],
               "C / median count": q[0] / s.n_hist.median(), "RMS": np.sqrt((res ** 2).mean()),
               "residual vs sd (corr)": np.corrcoef(res, s.sd)[0, 1],
               "old C for comparison": C0}
lbx.write_table(pd.DataFrame(rows).T, "T34_new_formula_by_count_band", floatfmt=".3f")
print(t33.round(4).to_string())
print(pd.DataFrame(rows).T.round(3).to_string())

# ----------------------------------------------------------------------------- panel with decomposition
p = lbx.load_panel()
dec = pd.read_csv(lbx.DATA / "derived" / "panel_decomposition.csv",
                  usecols=["slug", "raw21", "d_raw", "d_decay", "d_switch", "d_shown", "old26"])
d = p[p.clean].merge(dec, on="slug").merge(live[["slug", "raw", "n_hist", "sd", "share5", "share_lo", "shown"]],
                                           on="slug")
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
d["english"] = d.english.astype(int)
d["is_tv"] = d.is_tv.astype(int)
d["lvl21"] = d.raw21 - 3.5
d["lvl26"] = d.raw - 3.5
d["ln26"] = np.log10(d.n_hist)
scored = d[d.shown.notna()].copy()
big = scored[scored.rating_count_2021 >= 200]
GT = " + ".join(COL.values()) + " + g_none"
BASE = "lvl21 + log10_n21 + english + is_tv"
HIST = "bs(lvl26, df=5) + bs(ln26, df=5) + lvl26:ln26 + sd + share5 + share_lo + english + is_tv"
SPECS = {"members (n2021>=200)": ("d_raw", big, BASE),
         "switch | 2021 baseline": ("d_switch", scored, BASE),
         "switch | 2026 histogram": ("d_switch", scored, HIST)}


def fit(y, df, rhs, need=()):
    df = df.dropna(subset=list(need))
    return smf.ols(f"{y} ~ {rhs}", df).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(df.cluster)[0]})


# H7a genres
out = {}
for lab, (yv, df, ctl) in SPECS.items():
    f = fit(yv, df, f"{GT} + {ctl}")
    ci = f.conf_int()
    pv = np.array([f.pvalues[COL[g]] for g in GENRES])
    out[(lab, "coef")] = [f.params[COL[g]] for g in GENRES]
    out[(lab, "ci_lo")] = [ci.loc[COL[g], 0] for g in GENRES]
    out[(lab, "ci_hi")] = [ci.loc[COL[g], 1] for g in GENRES]
    out[(lab, "p Holm")] = multipletests(pv, method="holm")[1]
    out[(lab, "N")] = [int(f.nobs)] * len(GENRES)
t35 = pd.DataFrame(out, index=GENRES)
t35.insert(0, ("", "films (scored)"), [int(scored[COL[g]].sum()) for g in GENRES])
t35.columns = [" | ".join(c).strip(" |") for c in t35.columns]
lbx.write_table(t35, "T35_genre_members_vs_switch", floatfmt=".4f")

# H7b other film traits: one model per trait (with genre controls)
rows = []
for trait, extra, scale in [("English-language", "english", 1), ("TV", "is_tv", 1),
                            ("female lead (IMDb first billed)", "lead_female", 1),
                            ("review intensity, +0.10 per rating (archive films)", "rr", 0.1)]:
    for lab, (yv, df, ctl) in SPECS.items():
        rhs = f"{GT} + {ctl}" + ("" if extra in ctl else f" + {extra}")
        f = fit(yv, df, rhs, need=[extra])
        lo, hi = f.conf_int().loc[extra]
        rows.append({"trait": trait, "outcome": lab, "effect": scale * f.params[extra], "ci_lo": scale * lo,
                     "ci_hi": scale * hi, "p": f.pvalues[extra], "N": int(f.nobs)})
t36 = pd.DataFrame(rows).set_index(["trait", "outcome"])
lbx.write_table(t36, "T36_traits_members_vs_switch", floatfmt=".4f")
print(t35.round(3).to_string())
print(t36.round(4).to_string())

# ----------------------------------------------------------------------------- H8: spread and polarization
ar = with_raw(pd.read_sql("SELECT * FROM readings", sqlite3.connect(
    f"file:{lbx.DATA / 'archive_hist_pilot.sqlite'}?mode=ro", uri=True)))
ar["t"] = pd.to_datetime(ar.served.str[:8], errors="coerce")
pre = ar[(ar.status == "ok") & (ar.n_hist > 0) & (ar.t < lbx.CHANGE)].sort_values("t").groupby("slug").tail(1)
w = pre[["slug", "t", "n_hist", "raw", "sd", "share5", "share_lo"]].merge(
    d[["slug", "n_hist", "raw", "sd", "share5", "share_lo", "d_switch"]], on="slug", suffixes=("_pre", "_26"))
w["years"] = (pd.Timestamp("2026-10-19") - w.t).dt.days / 365.25
rows = {}
for v, lab in [("raw", "raw mean"), ("sd", "rating SD"), ("share5", "share of 5-star ratings"),
               ("share_lo", "share of 0.5-1 star ratings")]:
    ch = w[f"{v}_26"] - w[f"{v}_pre"]
    rows[lab] = {**lbx.mean_ci(ch)[["n", "mean", "ci_lo", "ci_hi", "share_up"]].to_dict(),
                 "pre-change level": w[f"{v}_pre"].mean(), "2026 level": w[f"{v}_26"].mean()}
t37 = pd.DataFrame(rows).T
t37["median years between readings"] = w.years.median()
w["dsd"] = w.sd_26 - w.sd_pre
f = smf.ols("d_switch ~ dsd + I(raw_26 - 3.5) + np.log10(n_hist_26) + sd_26", w.dropna()).fit(cov_type="HC3")
t37.loc["switch per +0.1 SD change (level, count, 2026 SD held)", ["n", "mean", "ci_lo", "ci_hi"]] = [
    f.nobs, 0.1 * f.params.dsd, 0.1 * f.conf_int().loc["dsd", 0], 0.1 * f.conf_int().loc["dsd", 1]]
lbx.write_table(t37, "T37_spread_and_polarization_change", floatfmt=".4f")
print(t37.round(4).to_string())

# ----------------------------------------------------------------------------- H9: small films and selection
small = d[d.rating_count_2021 < 200].copy()
small["crossed"] = small.shown.notna()
small["growth_q"] = pd.qcut(small.n_hist / small.rating_count_2021, 5,
                            labels=["lowest fifth", "2nd", "3rd", "4th", "highest fifth"])
g = small.groupby("growth_q", observed=True)
t38 = pd.DataFrame({"films": g.size(), "median growth": g.apply(lambda x: (x.n_hist / x.rating_count_2021).median()),
                    "share crossed 200 (score shown)": g.crossed.mean(), "members (raw change)": g.d_raw.mean(),
                    "raw mean 2021 (inverted)": g.raw21.mean()})
for lab, s in [("stayed below 200 (no score)", small[~small.crossed]), ("crossed 200", small[small.crossed])]:
    t38.loc[lab] = [len(s), (s.n_hist / s.rating_count_2021).median(), s.crossed.mean(), s.d_raw.mean(),
                    s.raw21.mean()]
lbx.write_table(t38, "T38_small_films_growth_selection", floatfmt=".3f")
print(t38.round(3).to_string())

# ----------------------------------------------------------------------------- headline means, cluster bootstrap
def boot(df, cols, reps=2000):
    gsum = df.groupby("cluster")[cols].sum()
    gcnt = df.groupby("cluster").size()
    k = len(gcnt)
    S, N = gsum.to_numpy(), gcnt.to_numpy()
    draws = np.empty((reps, len(cols)))
    for i in range(reps):
        idx = rng.integers(0, k, k)
        draws[i] = S[idx].sum(0) / N[idx].sum()
    return np.percentile(draws, [2.5, 97.5], axis=0)


rows = {}
n21 = scored.rating_count_2021
for lab, s in [("all scored films", scored), ("n2021 >= 200", scored[n21 >= 200]),
               ("n2021 30-199", scored[n21 < 200]), ("n2021 >= 1,000", scored[n21 >= 1000])]:
    cols = ["d_shown", "d_raw", "d_decay", "d_switch"]
    lohi = boot(s, cols)
    for j, c in enumerate(cols):
        rows[(lab, c)] = {"films": len(s), "directors (clusters)": s.cluster.nunique(), "mean": s[c].mean(),
                          "boot ci_lo": lohi[0, j], "boot ci_hi": lohi[1, j]}
t39 = pd.DataFrame(rows).T
lbx.write_table(t39, "T39_headline_cluster_bootstrap", floatfmt=".4f")
print(t39.round(4).to_string())

# ----------------------------------------------------------------------------- ranking: new vs old formula
# Same 2026 histograms, two rules: how far does the switch reorder films? (panel films only, not the full catalogue)
from scipy.stats import spearmanr

rows = {}
for lab, s in [("all scored panel films", scored), ("films with >= 10k ratings in 2026", scored[scored.n_hist >= 10000]),
               ("films with >= 100k ratings in 2026", scored[scored.n_hist >= 100000])]:
    mv = (s.shown.rank(pct=True) - s.old26.rank(pct=True)).abs()
    k = min(100, len(s) // 10)
    rows[lab] = {"films": len(s), "Spearman (new vs old formula)": spearmanr(s.shown, s.old26)[0],
                 "median |percentile move|": 100 * mv.median(), "share moving > 10 percentile points": (mv > 0.10).mean(),
                 "top-k": k, "top-k overlap": len(set(s.nlargest(k, "shown").slug) & set(s.nlargest(k, "old26").slug)),
                 "mean |switch|": s.d_switch.abs().mean()}
lbx.write_table(pd.DataFrame(rows).T, "T40_ranking_new_vs_old_formula", floatfmt=".3f")

# ----------------------------------------------------------------------------- fig8
bands = [("30-199", scored[n21 < 200]), ("200-999", scored[(n21 >= 200) & (n21 < 1000)]),
         ("1k-10k", scored[(n21 >= 1000) & (n21 < 10000)]), ("10k+", scored[n21 >= 10000])]
fig, ax = plt.subplots(figsize=(6.4, 3.3))
x = np.arange(len(bands))
wd = 0.24
for k, (c, lab, col) in enumerate([("d_raw", "raw-mean change", lbx.S1),
                                    ("d_decay", "change in old-rule adjustment", lbx.S3),
                                    ("d_switch", "2026 rule contrast (new score minus old rule)", lbx.S2)]):
    vals = [s[c].mean() for _, s in bands]
    ax.bar(x + (k - 1) * wd, vals, wd * 0.92, color=col, label=lab)
ax.scatter(x, [s.d_shown.mean() for _, s in bands], marker="D", s=30, color=lbx.INK, zorder=3,
           label="displayed change 2021-2026")
ax.axhline(0, color=lbx.INK2, lw=0.8)
ax.set_xticks(x, [f"{b}\n(n={len(s):,})" for b, s in bands])
ax.set_xlabel("ratings in Feb 2021")
ax.set_ylabel("mean change, stars")
ax.legend(loc="lower right", fontsize=7)
lbx.save(fig, "fig8_decomposition_by_size")
