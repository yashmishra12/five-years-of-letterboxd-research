"""
Genre analysis. Which genres changed more or less, at the June-2023 formula change and in the gradual drift,
and do genre gaps survive controls for rating level and audience size? (The step compresses scores by level
and count, so a raw genre gap can simply be compression; "Behind the Stars" reports documentary/music penalties.)

Genres are the film's 2021 genres (2026 tags are post-treatment). Films carry several genres, so all genre
indicators enter together: each coefficient compares films with that genre to otherwise similar films without it.
Holm and Benjamini-Hochberg corrections are applied within each outcome across the 19 genres.

Tables: T23 genre counts; T24 archive step / drift by genre (E3 per-film decomposition); T25 step by genre in a
film fixed-effects model (robustness); T26 panel 2021->2026 change by genre; T27 review-intensity result
with genre controls. Figure: fig7 genre effects.

  .venv/bin/python 05_genre.py      (run 02_jump.py first)
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

import lbx

lbx.style()
YEAR = 365.25
p = lbx.load_panel()
a = lbx.analysis_panel(p)
GENRES = sorted({g for gs in a.genre_list for g in gs})
COL = {g: "g_" + g.replace(" ", "_") for g in GENRES}


def add_genres(df):
    df = df.copy()
    for g in GENRES:
        df[COL[g]] = df.genre_list.map(lambda gs, g=g: g in gs).astype(int)
    df["g_none"] = (df.genre_list.str.len() == 0).astype(int)
    return df


GTERMS = " + ".join(COL[g] for g in GENRES) + " + g_none"
a = add_genres(a)
s = a[a.rating_count_2021 >= lbx.MIN_2026]

# per-film decomposition from 02_jump.py, with 2021 genres attached
br = pd.read_csv(lbx.DATA / "derived" / "jump_per_film.csv").drop(columns=["genre_list"])
br = add_genres(br.merge(a[["slug", "genre_list"]], on="slug"))
br["lvl"] = br.r_minus - 3.5
br["lc"] = np.log10(br.n_before)

# ----------------------------------------------------------------------------- T23 counts
t23 = pd.DataFrame({g: {"panel": int(a[COL[g]].sum()), "n2021>=200": int(s[COL[g]].sum()),
                        "archive (>=1,000)": int(br[COL[g]].sum())} for g in GENRES}).T
t23 = t23.sort_values("panel", ascending=False)
lbx.write_table(t23, "T23_genre_counts", floatfmt=".0f")


def genre_table(df, y, controls, cov="HC3", groups=None):
    f = smf.ols(f"{y} ~ {GTERMS}" + (f" + {controls}" if controls else ""), df)
    f = f.fit(cov_type="cluster", cov_kwds={"groups": groups}) if cov == "cluster" else f.fit(cov_type="HC3")
    out = pd.DataFrame({"coef": [f.params[COL[g]] for g in GENRES],
                        "ci_lo": [f.conf_int().loc[COL[g], 0] for g in GENRES],
                        "ci_hi": [f.conf_int().loc[COL[g], 1] for g in GENRES],
                        "p": [f.pvalues[COL[g]] for g in GENRES]}, index=GENRES)
    out["p_holm"] = multipletests(out.p, method="holm")[1]
    out["q_bh"] = multipletests(out.p, method="fdr_bh")[1]
    return out


CTRL_ARCH = "lvl + lc + C(cohort) + english + is_tv"
CTRL_PANEL = "I(rating_2021 - 3.2) + log10_n21 + C(cohort) + english + is_tv"

# ----------------------------------------------------------------------------- T24 archive: step and drift
blocks = {}
for y, lab in [("jump_adj", "step"), ("post_drift", "drift after"), ("pre_drift", "drift before")]:
    raw = genre_table(br, y, "")
    adj = genre_table(br, y, CTRL_ARCH)
    blocks[lab] = adj
    blocks[lab + " (raw)"] = raw
t24 = pd.concat({k: v for k, v in blocks.items()}, axis=1)
t24.insert(0, "n films", [int(br[COL[g]].sum()) for g in GENRES])
t24 = t24.sort_values(("step", "coef"))
lbx.write_table(t24, "T24_genre_step_drift_archive", floatfmt=".4f")

# ----------------------------------------------------------------------------- T25 robustness: FE model
# rating = film + slope*t + step*post + kink*(t-C)+, with step, slope and kink each allowed to differ by genre,
# and the step also by 2021 rating level and log count (the compression), so genre steps are net of compression.
long, films = lbx.load_archive()
x = long[long.t >= "2019-01-01"]
x = x[x.groupby("slug").slug.transform("size") >= 3].copy()
x = add_genres(x.merge(a[["slug", "genre_list", "rating_2021", "log10_n21"]], on="slug"))
t_ = (x.t - pd.Timestamp("2021-01-01")).dt.days / YEAR
post = (x.t >= lbx.CHANGE).astype(float)
kink = ((x.t - lbx.CHANGE).dt.days / YEAR).clip(lower=0)
cols = {"t": t_, "post": post, "kink": kink,
        "post_lvl": post * (x.rating_2021 - 3.5), "post_lc": post * (x.log10_n21 - x.log10_n21.mean())}
for g in GENRES:
    gi = x[COL[g]]
    cols[f"post_{COL[g]}"], cols[f"t_{COL[g]}"], cols[f"kink_{COL[g]}"] = post * gi, t_ * gi, kink * gi
X = pd.DataFrame(cols)
Xd = X - X.groupby(x.slug.values).transform("mean")
yd = x.rating - x.groupby("slug").rating.transform("mean")
fe = sm.OLS(yd.values, Xd).fit(cov_type="cluster", cov_kwds={"groups": x.slug.astype("category").cat.codes})
t25 = pd.DataFrame({"step coef": [fe.params[f"post_{COL[g]}"] for g in GENRES],
                    "ci_lo": [fe.conf_int().loc[f"post_{COL[g]}", 0] for g in GENRES],
                    "ci_hi": [fe.conf_int().loc[f"post_{COL[g]}", 1] for g in GENRES],
                    "p": [fe.pvalues[f"post_{COL[g]}"] for g in GENRES]}, index=GENRES)
t25["p_holm"] = multipletests(t25.p, method="holm")[1]
t25["q_bh"] = multipletests(t25.p, method="fdr_bh")[1]
t25.loc["(step x rating level, per star)"] = [fe.params.post_lvl, *fe.conf_int().loc["post_lvl"], fe.pvalues.post_lvl,
                                               np.nan, np.nan]
t25.loc["(step x log10 count)"] = [fe.params.post_lc, *fe.conf_int().loc["post_lc"], fe.pvalues.post_lc, np.nan, np.nan]
lbx.write_table(t25, "T25_genre_step_fixed_effects", floatfmt=".4f")

# ----------------------------------------------------------------------------- T26 panel 2021->2026 change
pan = {}
for lab, df, ctl in [("all, raw", a, ""), ("all, adjusted", a, CTRL_PANEL),
                     ("n2021>=200, raw", s, ""), ("n2021>=200, adjusted", s, CTRL_PANEL)]:
    pan[lab] = genre_table(df, "d", ctl)
t26 = pd.concat(pan, axis=1)
t26.insert(0, "n (all)", [int(a[COL[g]].sum()) for g in GENRES])
t26.insert(1, "mean change (all)", [a.loc[a[COL[g]] == 1, "d"].mean() for g in GENRES])
t26 = t26.sort_values(("n2021>=200, adjusted", "coef"))
lbx.write_table(t26, "T26_genre_total_change_panel", floatfmt=".4f")

# ----------------------------------------------------------------------------- T27 review intensity with genre controls
q, r = lbx.archive_tables()
r = r[r.note == "ok"].assign(t_before=lambda d: pd.to_datetime(d.timestamp.str[:8]),
                             rr=lambda d: d.review_count / d.rating_count)
br["t_before"] = pd.to_datetime(br.t_before)
rb = br.merge(r[["slug", "t_before", "rr"]], on=["slug", "t_before"]).dropna(subset=["rr"])
rows = []
for y, lab in [("jump_adj", "step"), ("post_drift", "drift after"), ("pre_drift", "drift before")]:
    for ctl, cl in [(CTRL_ARCH, "level, count, cohort, language, TV"),
                    (CTRL_ARCH + " + I(lvl**2) + " + GTERMS, "... + level squared + 2021 genres")]:
        f = smf.ols(f"{y} ~ rr + {ctl}", rb).fit(cov_type="HC3")
        rows.append({"component": lab, "controls": cl, "per +0.10 review ratio": 0.1 * f.params.rr,
                     "ci_lo": 0.1 * f.conf_int().loc["rr", 0], "ci_hi": 0.1 * f.conf_int().loc["rr", 1],
                     "p": f.pvalues.rr, "N": int(f.nobs)})
lbx.write_table(pd.DataFrame(rows).set_index(["component", "controls"]), "T27_review_intensity_genre_controls",
                floatfmt=".4f")

# ----------------------------------------------------------------------------- fig7
order = t24.index.tolist()           # genres sorted by adjusted step
ypos = np.arange(len(order))
fig, axes = plt.subplots(1, 3, figsize=(10.5, 5.4), sharey=True)
panels = [(axes[0], t24["step (raw)"], t24["step"], "(a)"), 
          (axes[1], t24["drift after (raw)"], t24["drift after"], "(b)"),
          (axes[2], t26["n2021>=200, raw"], t26["n2021>=200, adjusted"],
          "(c)")]
for ax, raw, adj, title in panels:
    raw, adj = raw.loc[order], adj.loc[order]
    ax.errorbar(raw.coef, ypos + 0.17, xerr=[raw.coef - raw.ci_lo, raw.ci_hi - raw.coef], fmt="o", color=lbx.S2,
                ms=4, elinewidth=0.9, capsize=0, mec="white", mew=0.6, label="raw gap")
    ax.errorbar(adj.coef, ypos - 0.17, xerr=[adj.coef - adj.ci_lo, adj.ci_hi - adj.coef], fmt="o", color=lbx.S1,
                ms=4, elinewidth=0.9, capsize=0, mec="white", mew=0.6,
                label="adjusted for rating level, audience size, cohort, language")
    sig = adj.q_bh < 0.05
    ax.scatter(adj.coef[sig], ypos[sig.values] - 0.17, s=46, facecolors="none", edgecolors=lbx.INK, linewidths=0.8,
               zorder=5, label="adjusted, BH q < 0.05")
    ax.axvline(0, color=lbx.INK2, lw=0.8)
    ax.set_title(title, loc="left", fontsize=9)
    ax.set_xlabel("Difference vs films without the genre (stars)")
    ax.grid(axis="y", visible=False)
axes[0].set_yticks(ypos, order)
axes[0].invert_yaxis()
handles, labels = axes[0].get_legend_handles_labels()
idx = [labels.index(l) for l in ("raw gap", "adjusted for rating level, audience size, cohort, language", "adjusted, BH q < 0.05")]
fig.legend([handles[i] for i in idx], [labels[i] for i in idx], loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.05, 1, 1))
lbx.save(fig, "fig7_genre_effects")
plt.close(fig)

print(t23.to_string())
show = t24[[("n films", ""), ("step (raw)", "coef"), ("step", "coef"), ("step", "q_bh"), ("drift after", "coef"),
            ("drift after", "q_bh")]]
print(show.round(4).to_string())
print(t25.round(4).to_string())
print(t26[[("mean change (all)", ""), ("n2021>=200, raw", "coef"), ("n2021>=200, adjusted", "coef"),
           ("n2021>=200, adjusted", "q_bh")]].round(4).to_string())
print(pd.read_csv(lbx.TAB / "T27_review_intensity_genre_controls.csv").round(4).to_string())
