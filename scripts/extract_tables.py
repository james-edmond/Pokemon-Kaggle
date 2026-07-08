"""Precompute the card-metadata tables offline (uses the dev cg) and pickle them so
the submission agent never loads the native engine.
Usage: python scripts/extract_tables.py submission_src/tables.pkl"""
import os
import pickle
import sys

from ptcg.cards import build_tables

dst = sys.argv[1] if len(sys.argv) > 1 else "submission_src/tables.pkl"
tables = build_tables()
os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
with open(dst, "wb") as f:
    pickle.dump(tables, f)
# verify round-trip (unpickle without cg being required)
with open(dst, "rb") as f:
    t2 = pickle.load(f)
assert t2.n_rows == tables.n_rows
print(f"wrote {dst}: n_rows={tables.n_rows}")
