"""Assemble the self-contained submission bundle in dist/submission/ and zip it.
Run from repo root (base python). Usage: python scripts/make_submission.py"""
import os
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "submission_src")
OUT = os.path.join(REPO, "dist", "submission")
CG = os.path.join(REPO, "pokemon-tcg-ai-battle", "sample_submission",
                  "sample_submission", "cg")
PTCG_MODULES = ["__init__.py", "cards.py", "engine.py", "tracker.py",
                "featurize.py", "model.py", "action.py"]


def main():
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)
    for f in ("main.py", "deck.csv", "policy.pt", "README.md"):
        shutil.copyfile(os.path.join(SRC, f), os.path.join(OUT, f))
    os.makedirs(os.path.join(OUT, "ptcg"))
    for m in PTCG_MODULES:
        shutil.copyfile(os.path.join(REPO, "ptcg", m),
                        os.path.join(OUT, "ptcg", m))
    shutil.copytree(CG, os.path.join(OUT, "cg"))
    # Self-containment check: from a subprocess with cwd=OUT and the REPO stripped
    # from sys.path, the bundle must exec main.py the Kaggle way (no __file__),
    # lazily load the native model + bundled cg engine, and return a legal deck.
    check = (
        "import os, sys\n"
        f"repo = {REPO!r}\n"
        "sys.path = [p for p in sys.path if os.path.abspath(p or '.') != repo]\n"
        "sys.path.insert(0, os.getcwd())\n"
        "ns = {}\n"
        "exec(compile(open('main.py').read(), 'main.py', 'exec'), ns)\n"
        "ns['_ensure_model']()\n"
        "assert ns['_MODEL'] is not None\n"
        "d = ns['agent']({'select': None, 'current': None, 'logs': []})\n"
        "assert len(d) == 60\n"
        "print('self-contained OK: no __file__, model+cg from bundle, deck', len(d))\n"
    )
    r = subprocess.run([sys.executable, "-c", check], cwd=OUT,
                       capture_output=True, text=True)
    print(r.stdout.strip())
    print(r.stderr.strip())
    if r.returncode != 0:
        sys.exit("self-containment check FAILED")
    # drop __pycache__ created by the check so it doesn't bloat the zip
    for root, dirs, _ in os.walk(OUT):
        for dname in list(dirs):
            if dname == "__pycache__":
                shutil.rmtree(os.path.join(root, dname))
    zip_base = os.path.join(REPO, "dist", "submission")
    shutil.make_archive(zip_base, "zip", OUT)
    print(f"wrote {OUT}/ and {zip_base}.zip")


if __name__ == "__main__":
    main()
