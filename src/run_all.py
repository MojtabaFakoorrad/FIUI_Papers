from __future__ import annotations
import subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/"src"
for script in [
    "01_build_components.py",
    "02_primary_fiui.py",
    "03_content_validity.py",
    "04_nomological_and_sector_tests.py",
    "05_round2_diagnostics.py",
    "99_validate_reported_results.py",
]:
    print(f"\n>>> {script}", flush=True)
    subprocess.run([sys.executable,str(SRC/script)],cwd=ROOT,check=True)
print(f"\nDone. Outputs are in: {ROOT/'outputs'}")
