"""Plot metrics.csv curves: python scripts/plot_run.py runs/<run-id>"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(run_dir):
    import csv
    rows = list(csv.DictReader(open(Path(run_dir) / "metrics.csv", newline="")))
    train_rows = [r for r in rows if r["kind"] == "train"]
    eval_rows = [r for r in rows if r["kind"] == "eval"]
    out = Path(run_dir) / "plots"
    out.mkdir(exist_ok=True)

    def series(rs, key):
        pts = [(int(r["round"]), float(r[key])) for r in rs if r.get(key)]
        return [p[0] for p in pts], [p[1] for p in pts]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, keys, title in (
            (axes[0][0], ["loss_pg", "loss_v", "loss_critic"], "losses"),
            (axes[0][1], ["entropy"], "entropy"),
            (axes[1][0], ["approx_kl", "ratio_drift"], "kl / drift"),
            (axes[1][1], ["mean_len"], "game length")):
        for k in keys:
            ax.plot(*series(train_rows, k), label=k)
        ax.set_title(title)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / "train.png", dpi=120)

    if eval_rows:
        fig2, ax = plt.subplots(figsize=(8, 5))
        for k in ("wr_random", "wr_ck5", "wr_ck15"):
            x, y = series(eval_rows, k)
            if x:
                ax.plot(x, y, marker="o", label=k)
        ax.axhline(0.5, ls="--", c="gray")
        ax.axhline(0.65, ls=":", c="green")
        ax.set_ylim(0, 1)
        ax.set_title("win rates")
        ax.legend()
        fig2.savefig(out / "eval.png", dpi=120)

        fig3, ax3 = plt.subplots(figsize=(8, 5))
        for k in ("wr_champ_nonsample", "wr_champ_sample", "wr_random_mean"):
            x, y = series(eval_rows, k)
            if x:
                ax3.plot(x, y, marker="o", label=k)
        ax3.axhline(0.60, ls=":", c="green", label="0.60 target")
        ax3.axhline(0.5, ls="--", c="gray")
        ax3.set_ylim(0, 1)
        ax3.set_title("generalization vs SD-champ / portfolio")
        ax3.legend()
        fig3.savefig(out / "generalization.png", dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1])
