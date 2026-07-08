"""Assemble the self-contained submission bundle in dist/submission/ and zip it.
Run from repo root (base python). Usage: python scripts/make_submission.py"""
import os
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "submission_src")
OUT = os.path.join(REPO, "dist", "submission")
PTCG_MODULES = ["__init__.py", "cards.py", "engine.py", "tracker.py",
                "featurize.py", "model.py", "action.py"]


def main():
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)
    for f in ("main.py", "deck.csv", "policy.pt", "tables.pkl", "README.md"):
        shutil.copyfile(os.path.join(SRC, f), os.path.join(OUT, f))
    os.makedirs(os.path.join(OUT, "ptcg"))
    for m in PTCG_MODULES:
        shutil.copyfile(os.path.join(REPO, "ptcg", m),
                        os.path.join(OUT, "ptcg", m))
    # Self-containment check: from a subprocess with cwd=OUT and the REPO stripped
    # from sys.path, the bundle must resolve its OWN ptcg (not the repo's), load
    # the precomputed tables.pkl, and return a deck -- all WITHOUT ever importing
    # the native cg engine. A missing bundled module or an accidental cg import
    # raises here -> build FAILS loudly.
    check = (
        "import os, sys\n"
        f"repo = {REPO!r}\n"
        "sys.path = [p for p in sys.path if os.path.abspath(p or '.') != repo]\n"
        "sys.path.insert(0, os.getcwd())\n"
        "import ptcg, ptcg.action, ptcg.featurize, ptcg.model, ptcg.cards, ptcg.tracker\n"
        "assert os.path.abspath(os.path.dirname(ptcg.__file__)).startswith(os.getcwd()), ptcg.__file__\n"
        "ns = {}\n"
        "exec(compile(open('main.py').read(), 'main.py', 'exec'), ns)\n"  # Kaggle-style: NO __file__
        "d = ns['agent']({'select': None, 'current': None, 'logs': []})\n"
        "assert len(d) == 60\n"
        "assert 'cg' not in sys.modules and 'cg.game' not in sys.modules, 'agent loaded cg!'\n"
        "print('self-contained OK: no __file__, cg-free, deck', len(d))\n"
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
