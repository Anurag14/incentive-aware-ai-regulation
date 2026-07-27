from __future__ import annotations

import argparse
import os
import pickle

import numpy as np
import pandas as pd

import betting
import data as data_mod
import providers as prov_mod


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="german", choices=["taiwan", "german", "synth"])
    p.add_argument("--T", type=int, default=500, help="audited applicants per run")
    p.add_argument("--runs", type=int, default=30, help="independent audits")
    p.add_argument("--cal-frac", type=float, default=0.2,
                   help="share of the stream used to calibrate alpha (burn-in)")
    p.add_argument("--bias", type=float, default=0.8,
                   help="historical bias in the TRAINING splits only")
    p.add_argument("--alpha-min", type=float, default=betting.ALPHA_MIN,
                   help="most-diluted member of the prohibited segment")
    p.add_argument("--eps", type=float, default=betting.EPS_SMOOTH)
    p.add_argument("--gammas", type=float, nargs="*", default=[1.0, 0.8, 0.6],
                   help="dilution levels for the mixing-attack providers")
    p.add_argument("--C", type=float, default=betting.C_DEFAULT)
    p.add_argument("--R", type=float, default=betting.R_DEFAULT)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--strict", action="store_true")
    p.add_argument("--outdir", default="figures")
    p.add_argument("--plot-providers", type=str, nargs="*",
                   default=["blind", "uses_A"],
                   help="name prefixes to draw in the figure. The dilution "
                        "providers stay in the printed table and the CSV; they "
                        "are omitted from the plot by default to keep it to the "
                        "two curves that carry the result.")
    p.add_argument("--no-plots", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    X, a, y = data_mod.load(args.dataset, strict=args.strict)
    W = prov_mod.fit_world(X, a, y, seed=args.seed, bias=args.bias)
    m = W.meta

    print(f"\n[splits] regulator {m['n_reg']} (biased -> {m['n_reg_biased']}), "
          f"providers {m['n_prov']} (biased -> {m['n_prov_biased']}), "
          f"audit {m['n_audit']}")
    print(f"[models] A-coefficient   P_bad {m['coef_a_bad']:+.3f}   "
          f"uses_A {m['coef_a_uses']:+.3f}")
    print(f"[models] mean log-lik    P_bad {m['loglik_bad']:+.4f}   "
          f"uses_A {m['loglik_uses']:+.4f}   blind {m['loglik_blind']:+.4f}")
    print(f"[setup]  P_0 = alpha P_bad + (1-alpha) U, alpha in "
          f"[{args.alpha_min}, 1];  C={args.C}, R={args.R}, "
          f"B=R/C={args.R / args.C:.2f}")
    print(f"[setup]  T={args.T}, calibration={args.cal_frac:.0%}, "
          f"runs={args.runs}\n")

    panel = [("blind (compliant)", W.p_blind, True),
             ("uses_A (non-compliant)", W.p_uses, False)]
    for g in args.gammas:
        tag = "P_bad itself" if g == 1.0 else f"P_bad diluted gamma={g:g}"
        panel.append((tag, prov_mod.dilute(W.p_bad, g), False))

    rows, trajectories, split = [], {}, None
    print(f"{'provider':<28s}{'alpha*':>8s}{'licence':>10s}{'part':>7s}   verdict")
    for name, q_user, compliant in panel:
        traj, lic, part, alphas = betting.run_audits(
            q_user, W.p_bad, W.y, T=args.T, n_runs=args.runs, rng=rng,
            C=args.C, R=args.R, eps=args.eps, cal_frac=args.cal_frac,
            alpha_min=args.alpha_min)
        trajectories[name] = traj
        split = max(int(args.T * args.cal_frac), 10)
        participates = part.mean() > 0.5
        ok = participates == compliant
        rows.append({"provider": name, "compliant": compliant,
                     "alpha_star": float(alphas.mean()),
                     "licence_mean": float(lic.mean()),
                     "licence_se": float(lic.std(ddof=1) / np.sqrt(len(lic))),
                     "participation_rate": float(part.mean()),
                     "pmo_correct": bool(ok)})
        v = "PARTICIPATES" if participates else "SELF-EXCLUDES"
        print(f"{'ok ' if ok else 'XX '}{name:<25s}{alphas.mean():>8.3f}"
              f"{lic.mean():>10.2f}{part.mean():>7.2f}   {v}")

    summary = pd.DataFrame(rows)
    print(f"\n-> PMO achieved: {bool(summary['pmo_correct'].all())}")

    tag = args.dataset
    summary.to_csv(os.path.join(args.outdir, f"credit_{tag}_summary.csv"), index=False)
    with open(os.path.join(args.outdir, f"credit_{tag}_results.pkl"), "wb") as f:
        pickle.dump({"summary": summary, "trajectories": trajectories,
                     "meta": m, "split": split, "config": vars(args)}, f)

    if not args.no_plots:
        make_plot(trajectories, split, args, tag)
    print(f"\n[done] wrote results to {args.outdir}/")
    return summary


def _style(name):
    if name.startswith("blind"):
        return "tab:blue", "-"
    if name.startswith("uses_A"):
        return "tab:red", (0, (5, 2))
    if "itself" in name:
        return "#7f0000", (0, (1, 1))
    return "tab:orange", (0, (3, 1, 1, 1))


def make_plot(trajectories, split, args, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import scienceplots  # noqa: F401
        plt.style.use(["science", "no-latex"])
    except Exception:
        plt.style.use("default")

    keep = tuple(args.plot_providers)
    fig, ax = plt.subplots(figsize=(5.8, 4.1))
    for name, traj in trajectories.items():
        if keep and not name.startswith(keep):
            continue
        c, ls = _style(name)
        betting.plot_licence(ax, traj, label=name, color=c, split=split, ls=ls)

    ax.axhline(args.C, color="k", ls="--", lw=1.2)
    ax.axhline(args.R, color="k", ls=":", lw=1.0)
    ax.axvline(split, color="gray", lw=1.0, alpha=0.7)
    ax.text(split, args.R * 1.02, " audit begins", fontsize=7, color="gray")
    ax.set_ylim(0, args.R * 1.12)
    ax.set_xlabel("audited applicants $n$")
    ax.set_ylabel(r"licence value $\pi_n$")
    ax.set_title(f"Credit scoring ({tag}): compliant vs attribute-using provider",
                 fontsize=10)
    ax.text(0.99, args.C * 1.06, "entry fee $C$", fontsize=7, ha="right",
            transform=ax.get_yaxis_transform())
    ax.legend(fontsize=8, loc="center right")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.outdir, f"credit_{tag}_licence.{ext}"),
                    dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
