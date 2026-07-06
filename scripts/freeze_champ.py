"""Freeze the phase-2 single-deck champion as the fixed generalization baseline.
Usage: venv-train\\Scripts\\python scripts\\freeze_champ.py runs/phase2-a/checkpoint-0031.pt champ/sd-champ.pt"""
import shutil, sys
from pathlib import Path

src, dst = sys.argv[1], sys.argv[2]
Path(dst).parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(src, dst)
print("froze", src, "->", dst)
