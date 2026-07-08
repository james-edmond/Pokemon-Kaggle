"""Assemble the self-contained submission bundle in dist/submission/ and zip it.
Run from repo root (base python). Usage: python scripts/make_submission.py"""
import os
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "submission_src")
CG = os.path.join(REPO, "pokemon-tcg-ai-battle", "sample_submission",
                  "sample_submission", "cg")
OUT = os.path.join(REPO, "dist", "submission")
PTCG_MODULES = ["__init__.py", "cards.py", "engine.py", "tracker.py",
                "featurize.py", "model.py", "action.py"]


def main():
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)
    for f in ("main.py", "deck.csv", "policy.pt", "README.md"):
        shutil.copyfile(os.path.join(SRC, f), os.path.join(OUT, f))
    shutil.copytree(CG, os.path.join(OUT, "cg"))
    os.makedirs(os.path.join(OUT, "ptcg"))
    for m in PTCG_MODULES:
        shutil.copyfile(os.path.join(REPO, "ptcg", m),
                        os.path.join(OUT, "ptcg", m))
    # self-containment check: import + deck-selection call from a CLEAN cwd/sys.path,
    # with REPO removed from the path so it can only resolve bundled ptcg/cg.
    check = ("import sys; sys.path=[p for p in sys.path if 'Pokemon-Kaggle' not in p "
             "or p.endswith('submission')]; sys.path.insert(0, '.'); "
             "import importlib.util as u; s=u.spec_from_file_location('m','main.py'); "
             "m=u.module_from_spec(s); s.loader.exec_module(m); "
             "d=m.agent({'select':None,'current':None,'logs':[]}); "
             "assert len(d)==60; print('self-contained OK: deck', len(d))")
    r = subprocess.run([sys.executable, "-c", check], cwd=OUT,
                       capture_output=True, text=True)
    print(r.stdout.strip()); print(r.stderr.strip())
    if r.returncode != 0:
        sys.exit("self-containment check FAILED")
    zip_base = os.path.join(REPO, "dist", "submission")
    shutil.make_archive(zip_base, "zip", OUT)
    print(f"wrote {OUT}/ and {zip_base}.zip")


if __name__ == "__main__":
    main()
