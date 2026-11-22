# Five Years of Letterboxd — analysis code

Analysis pipeline for the paper *"Five years of Letterboxd: a longitudinal audit of
how a rating-rule change alters displayed film scores"* (Y. Mishra), submitted to the
*Journal of Computational Social Science*. The code
reconstructs and validates Letterboxd's pre-June-2023 rating rule, decomposes the
2021→2026 change in displayed scores into a raw-mean change, a change in the old
rule's adjustment, and a 2026 rule contrast, and runs the timing, dominance,
genre/trait, and robustness analyses reported in the paper.

This repository contains **code only**. The collected data are not included (see
[Data](#data)).

## What's here

| File | Purpose |
|------|---------|
| `lbx.py` | Shared library: panel loading and cleaning, archive assembly, external joins (IMDb, canon), stats and plotting helpers, table/figure writers. Imported by every script. |
| `01_panel.py` | Overall 2021→2026 change: sample flow, change by strata, regressions, regression-to-the-mean benchmark (T1–T4; fig1–fig4). |
| `02_jump.py` | The June-2023 step vs organic drift on the Internet Archive sample: step estimators, placebo dates, break-date scans, decomposition, measurement checks (T5–T10, T19–T21; fig5, figS1). Writes `derived/jump_per_film.csv`. |
| `03_selection.py` | Selection from the higher 2026 display threshold: retention, logit of losing the average, bias check (T11–T14; figS2). |
| `04_step_diagnostics.py` | Is the change a clean step? Prediction test, timing traps, step-vs-ramp, model-free trajectory (T15–T18; figS3). |
| `05_genre.py` | Genre effects on the step, post-change drift and total change, with controls and multiple-testing correction (T23–T27; fig7). |
| `06_histograms.py` | Rating histograms: old-rule fit and validation, the measured split across the change, and the panel decomposition (T28–T32). Writes `derived/panel_decomposition.csv`, `derived/pilot_split.csv`. |
| `07_histogram_analysis.py` | New-rule fits, member/rule trait comparisons, spread and polarization, small-film selection, headline bootstrap, reordering (T33–T40; fig8). |
| `08_imdb_crossplatform.py` | Cross-platform agreement with IMDb and whether IMDb predicts the adjustment (T41–T43). |
| `09_new_rule.py` | C-sensitivity, level compression, dominance reversals, persistence and pass-through, whole/half-star shift (T44–T49; fig6, fig9). |
| `10_feedback_canon.py` | Feedback (anchoring) test and the *Sight and Sound* canon comparison (T50–T51). |
| `11_robustness.py` | Alternative samples, review intensity, panel representativeness, break-date bootstrap (T52–T54). |
| `12_revision_checks.py` | Old-rule validation by period/band, decomposition uncertainty, full-histogram trait models, break-date simulation (T55–T58; fig10). |
| `13_genre_description.py` | Genres described in 2021 and 2026, and the genre ranking split into the ratings' part and the rule's part (T59–T60). |

## Requirements

- Python ≥ 3.10
- `numpy`, `pandas`, `scipy`, `statsmodels`, `patsy`, `matplotlib`, `tabulate`

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins the exact versions the paper's results were produced with
(numpy 2.5.4, pandas 3.0.6, scipy 1.18.1, statsmodels 0.15.0, patsy 1.0.3,
matplotlib 3.11.2, tabulate 0.10.0).

## Data

The data are **not** included in this repository. Only film-level aggregates,
collected from public pages, were used; no member-level or personal data were
collected. IMDb inputs are used under IMDb's non-commercial licence and are **not**
redistributed. The aggregate data that back the figures and tables are available
from the author on reasonable request.

The scripts read their inputs from a single data directory. By default that is
`../scrape2026/data` relative to the scripts; override it with the `LBX_DATA`
environment variable:

```bash
export LBX_DATA=/path/to/data
```

Expected inputs in that directory (supplied separately):

- `panel_2021_2026.csv` — the matched 2021↔2026 film panel
- `films_2026.csv` — the October 2026 crawl
- `letterboxd_2026.sqlite`, `archive_history.sqlite`, `histograms.sqlite`,
  `archive_hist_pilot.sqlite` — crawl/archive databases
- `canon_sight_and_sound_2022.csv`, `imdb_leads.csv` — external joins
- `imdb_ratings/title.ratings.tsv.gz` — IMDb public non-commercial dataset (not redistributed)
- `feb2021.json` — raw February 2021 listing (optional; only for the sample-flow count)

Where a database is absent, some loaders fall back to equivalent CSV exports
(`archive_queue.csv`, `archive_readings.csv`, …) placed in the same directory.

## Running

`lbx.py` is a library, not a script. Run the numbered scripts **in order** — later
scripts read files written by earlier ones:

```bash
for s in 01_panel 02_jump 03_selection 04_step_diagnostics 05_genre \
         06_histograms 07_histogram_analysis 08_imdb_crossplatform 09_new_rule \
         10_feedback_canon 11_robustness 12_revision_checks 13_genre_description; do
    python "$s.py"
done
```

Key dependencies between steps:

- `02_jump.py` writes `derived/jump_per_film.csv`, needed by `04` and `05`.
- `06_histograms.py` writes `derived/panel_decomposition.csv`, `derived/pilot_split.csv`
  and the old-rule fit table `T29`, needed by `07`–`13`.
- `12_revision_checks.py` additionally assumes `07` and `09` have run.

## Outputs

Running the pipeline writes:

- `figures/` — each figure as `.png` and `.pdf`
- `tables/` — each table as `.csv` and `.md`
- `<LBX_DATA>/derived/` — intermediate film-level files
  (`jump_per_film.csv`, `panel_decomposition.csv`, `pilot_split.csv`,
  `identity_review.csv`)

These are regenerated from the inputs and are not tracked in git. A full run takes a
few minutes on a laptop.

## Paper figures

The scripts name figures by the order in which they were written. The paper
numbers them differently. `figures/Fig1.pdf`–`Fig8.pdf` in this repository are the
exact files submitted with the paper. Running the pipeline regenerates them under
the names below.

| Paper | Generated file | Script |
|-------|----------------|--------|
| Fig. 1 | `fig1_change_distribution` | `01_panel.py` |
| Fig. 2 | `fig2_change_by_level` | `01_panel.py` |
| Fig. 3 | `fig8_decomposition_by_size` | `07_histogram_analysis.py` |
| Fig. 4 | `fig10_decomposition_spread_by_size` | `12_revision_checks.py` |
| Fig. 5 | `fig9_dominance_reversal_example` | `09_new_rule.py` |
| Fig. 6 | `fig5_formula_jump` | `02_jump.py` |
| Fig. 7 | `fig6_level_members_vs_formula` | `09_new_rule.py` |
| Fig. 8 | `fig7_genre_effects` | `05_genre.py` |

`fig3_change_by_cohort`, `fig4_change_vs_growth` and `figS1`–`figS4` are
supporting figures that do not appear in the paper.

## Notes

- Results are labelled *primary*, *supporting* or *exploratory* in the paper; none
  of the analyses was pre-registered.
- `lbx.KNOWN_MISMATCHES` lists the hand-excluded title-swap cases used to build the
  clean panel (documented inline).

## Citation

> Mishra, Y. (2026). Five years of Letterboxd: a longitudinal audit of how a
> rating-rule change alters displayed film scores. *Journal of Computational
> Social Science* (under review).

## License

Code released under the MIT License — see `LICENSE`. IMDb data referenced by the
pipeline are subject to IMDb's own non-commercial terms and are not included here.
