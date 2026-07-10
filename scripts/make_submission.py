"""Assemble the self-contained submission bundle in dist/submission/ and zip it.
Run from repo root (base python).
Usage: python scripts/make_submission.py [--no-search]
--no-search builds dist/submission-nosearch/ with _SEARCH_ENABLED flipped to
False in main.py — a rollback bundle behaviorally identical to the validated
pre-search agent."""
import os
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "submission_src")
CG = os.path.join(REPO, "pokemon-tcg-ai-battle", "sample_submission",
                  "sample_submission", "cg")
PTCG_MODULES = ["__init__.py", "cards.py", "engine.py", "tracker.py",
                "featurize.py", "model.py", "action.py",
                "clock.py", "simsearch.py", "determinize.py", "mcts.py"]


def main(no_search=False):
    name = "submission-nosearch" if no_search else "submission"
    out = os.path.join(REPO, "dist", name)
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(out)
    for f in ("deck.csv", "policy.pt", "README.md"):
        shutil.copyfile(os.path.join(SRC, f), os.path.join(out, f))
    main_src = open(os.path.join(SRC, "main.py")).read()
    if no_search:
        flipped = main_src.replace("_SEARCH_ENABLED = True",
                                   "_SEARCH_ENABLED = False", 1)
        assert flipped != main_src, "_SEARCH_ENABLED literal not found"
        main_src = flipped
    with open(os.path.join(out, "main.py"), "w") as f:
        f.write(main_src)
    os.makedirs(os.path.join(out, "ptcg"))
    for m in PTCG_MODULES:
        shutil.copyfile(os.path.join(REPO, "ptcg", m),
                        os.path.join(out, "ptcg", m))
    shutil.copytree(CG, os.path.join(out, "cg"))
    # Self-containment check: from a subprocess with cwd=out and the REPO
    # stripped from sys.path, the bundle must exec main.py the Kaggle way
    # (no __file__), lazily load the native model + bundled cg engine,
    # return a legal deck, and (search build) create a search arena.
    search_check = (
        "import ptcg.simsearch as ss\n"
        "assert ss.SearchSession().ensure_ptr(), 'AgentStart failed'\n"
        "print('search arena OK')\n"
    ) if not no_search else ""
    check = (
        "import os, sys\n"
        f"repo = {REPO!r}\n"
        "sys.path = [p for p in sys.path if os.path.abspath(p or '.') != repo]\n"
        "sys.path.insert(0, os.getcwd())\n"
        "ns = {}\n"
        "exec(compile(open('main.py').read(), 'main.py', 'exec'), ns)\n"
        f"assert ns['_SEARCH_ENABLED'] is {not no_search}\n"
        "ns['_ensure_model']()\n"
        "assert ns['_MODEL'] is not None\n"
        "d = ns['agent']({'select': None, 'current': None, 'logs': []})\n"
        "assert len(d) == 60\n"
        + search_check +
        "print('self-contained OK: no __file__, model+cg from bundle, deck', len(d))\n"
    )
    r = subprocess.run([sys.executable, "-c", check], cwd=out,
                       capture_output=True, text=True)
    print(r.stdout.strip())
    print(r.stderr.strip())
    if r.returncode != 0:
        sys.exit("self-containment check FAILED")
    for root, dirs, _ in os.walk(out):
        for dname in list(dirs):
            if dname == "__pycache__":
                shutil.rmtree(os.path.join(root, dname))
    zip_base = os.path.join(REPO, "dist", name)
    shutil.make_archive(zip_base, "zip", out)
    print(f"wrote {out}/ and {zip_base}.zip")


if __name__ == "__main__":
    main(no_search="--no-search" in sys.argv[1:])
