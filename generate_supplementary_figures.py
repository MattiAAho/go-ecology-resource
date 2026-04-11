"""
generate_supplementary_figures.py — Regenerate all supplementary figures.

Figures saved to figures/final/ as S1.pdf through S7.pdf.
S3 (gate extended, n=225+625) is not yet implemented — see note below.

Figure mapping:
  S1 — Policy sensitivity            (plot_policy_sensitivity.py)
  S2 — Capture dynamics              (plot_capture_summary.py → already PDF)
  S3 — Gate on/off extended          *** NOT YET GENERATED — deferred ***
  S4 — Two-player stone group struct. (plot_eye_diagnostic.py, fig 1)
  S5 — Three-player eye/group struct. (plot_eye_diagnostic.py, fig 2)
  S6 — Genuine coexistence rate       (coexistence_territory_plots.py, plot3)
  S7 — Loser defended territory       (coexistence_territory_plots.py, plot1)

Patches applied:
  - Figure.suptitle → no-op (titles belong in captions)
  - Figure.savefig  → also writes PDF alongside any PNG save

Usage:
    python generate_supplementary_figures.py
"""

from __future__ import annotations

import importlib.util
import pickle
import runpy
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.figure

# ── patch 1: silence suptitle ─────────────────────────────────────────────────
matplotlib.figure.Figure.suptitle = lambda self, *a, **kw: self

# ── patch 2: whenever a PNG is saved, also write PDF alongside it ─────────────
_orig_savefig = matplotlib.figure.Figure.savefig

def _also_pdf(self, fname, **kwargs):
    _orig_savefig(self, fname, **kwargs)
    p = Path(str(fname))
    if p.suffix.lower() == ".png":
        pdf_path = p.with_suffix(".pdf")
        pdf_kwargs = {k: v for k, v in kwargs.items() if k != "dpi"}
        _orig_savefig(self, pdf_path, **pdf_kwargs)

matplotlib.figure.Figure.savefig = _also_pdf

# ── pre-cache analysis_helpers so runpy sub-scripts don't reparse go_engine ───
from analysis_helpers import ci_2p  # noqa: E402

# ── output directory ──────────────────────────────────────────────────────────
FINAL_DIR   = Path("figures/final")
FIGURES_DIR = Path("figures")
RESULTS_DIR = Path("results")
FINAL_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Generating supplementary figures (no suptitles)")
print(f"Output: {FINAL_DIR}/")
print("=" * 60)


def run_and_copy(script: str, src_filename: str, dest_filename: str,
                 src_dir: Path = FIGURES_DIR) -> None:
    print(f"\n── {dest_filename} from {script} ──")
    runpy.run_path(script, run_name="__main__")
    src = src_dir / src_filename
    dst = FINAL_DIR / dest_filename
    dst.unlink(missing_ok=True)   # release any Windows file lock
    shutil.copy(src, dst)
    print(f"  Copied → {dst}")


# ─────────────────────────────────────────────────────────────────────────────
# S1 — Policy sensitivity
# ─────────────────────────────────────────────────────────────────────────────
run_and_copy("plot_policy_sensitivity.py", "fig_policy_sensitivity.pdf", "S1.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# S2 — Capture dynamics (re-run so suptitle patch applies)
# ─────────────────────────────────────────────────────────────────────────────
run_and_copy("plot_capture_summary.py", "fig_capture_summary.pdf", "S2.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# S3 — Gate on/off extended (n=225 + n=625) — NOT YET GENERATED
# ─────────────────────────────────────────────────────────────────────────────
print("\n── S3.pdf — SKIPPED (use plot_gate_extended.py) ──")


# ─────────────────────────────────────────────────────────────────────────────
# S4 + S5 — Two-player and three-player eye/group structure
#            plot_eye_diagnostic.py produces both figures; savefig patch writes PDF
# ─────────────────────────────────────────────────────────────────────────────
run_and_copy("plot_eye_diagnostic.py", "fig_eye_diag_2p.pdf", "S4.pdf")
(FINAL_DIR / "S5.pdf").unlink(missing_ok=True)
shutil.copy(FIGURES_DIR / "fig_eye_diag_3p.pdf", FINAL_DIR / "S5.pdf")
print(f"  Copied → {FINAL_DIR / 'S5.pdf'}")


# ─────────────────────────────────────────────────────────────────────────────
# S6 + S7 — Genuine coexistence rate and loser defended territory
#            Import coexistence_territory_plots as module, call functions directly
#            (avoids argparse; allows custom output paths)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── S6.pdf + S7.pdf from coexistence_territory_plots.py ──")

spec = importlib.util.spec_from_file_location(
    "coexistence_territory_plots",
    "coexistence_territory_plots.py",
)
ctp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ctp)

print("  Loading eye_records_3p.pkl …")
with open(RESULTS_DIR / "eye_records_3p.pkl", "rb") as fh:
    records = pickle.load(fh)
print(f"  {len(records)} records loaded")

# S6 — filter rate (genuine coexistence rate by topology + contest)
print("  Generating S6 (genuine coexistence rate) …")
ctp.plot3_filter_rate(records, FINAL_DIR / "S6.pdf")

# S7 — loser alive groups in genuine coexistence runs
print("  Generating S7 (loser defended territory) …")
ctp.plot1_loser_alive_groups(records, FINAL_DIR / "S7.pdf")


# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Done. Supplementary figures in figures/final/:")
for p in sorted(FINAL_DIR.glob("S*.pdf")):
    print(f"  {p.name}")
print("\n  S3.pdf — NOT generated (deferred; requires plot_gate_extended.py)")
print("=" * 60)
