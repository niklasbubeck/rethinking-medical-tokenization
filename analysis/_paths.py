"""Shared locations for the analysis scripts.

  results/       the released result tables (inputs; override with $RMT_RESULTS)
  analysis_out/  generated figures and LaTeX tables (override with $RMT_ANALYSIS_OUT)
  RUNS           pipeline outputs, <RMT_BASE_DIR>/outputs/<RMT_PROJECT>/, needed only
                 by the scripts that read image dumps (qualitative matrices, dataset
                 variance) or re-aggregate per-run evaluation CSVs.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.environ.get("RMT_RESULTS", os.path.join(ROOT, "results"))
OUT = os.environ.get("RMT_ANALYSIS_OUT", os.path.join(ROOT, "analysis_out"))
RUNS = os.path.join(os.environ.get("RMT_BASE_DIR", ROOT), "outputs",
                    os.environ.get("RMT_PROJECT", "tokenizers"))


def res(name):
    """Path of a released result file."""
    return os.path.join(RESULTS, name)


def out(name):
    """Path for a generated artifact (creates the output folder)."""
    os.makedirs(OUT, exist_ok=True)
    return os.path.join(OUT, name)
