"""
Licence mechanism: faithful port of
``experiments/waterbirds/audit.ipynb::run_martingale_test``.

The credal set
--------------
    P_0 = { alpha * P_bad + (1 - alpha) * U : alpha in [alpha_min, 1] }

the segment between the regulator's prohibited exemplar and the uninformative
distribution U (= 1/2 on the realised binary outcome). A segment between two
points is **closed and convex by construction**, so this is a credal set with
nothing left to prove -- no threshold on a parameter, no absolute value, no
union of half-spaces.

Regulatory reading: *this behaviour is banned, and so is any diluted version of
it.* A provider cannot escape by randomising between the banned scorer and a
coin flip, because every such mixture is itself in P_0. That is the
mixing-robustness of Theorem 3.5 in one sentence.

Two structural details inherited from the Waterbirds audit, both load-bearing:

1. The provider's likelihood is eps-smoothed, ``(1 - eps) q + eps / 2``, while
   the null is ``alpha q_bad + (1 - alpha) / 2``. These are two *different*
   points on the same segment, so auditing a provider that sits on the segment
   gives a ratio that hovers near 1 and fluctuates per applicant -- it is never
   identically 1. Nothing is degenerate.
2. During calibration the licence is held flat at the entry fee C. The
   regulator is choosing its null and has issued no licence yet; the wealth
   only starts moving once the audit proper begins.

alpha is chosen on the calibration segment to *maximise* the null's fit to the
realised outcomes -- the least-favourable member, hardest for the provider to
beat -- subject to the truncation bound max ratio <= B = R / C. This is safe
precisely because the segment is small: mixing with U only ever makes a
predictor worse, so the best member is the P_bad endpoint and the null can
never out-predict a genuinely compliant provider.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

C_DEFAULT = 15.0
R_DEFAULT = 250.0
EPS_SMOOTH = 0.01
ALPHA_MIN = 0.5


def smooth(q, eps: float = EPS_SMOOTH):
    """eps-smooth a likelihood towards the uninformative 1/2."""
    return (1.0 - eps) * np.asarray(q, dtype=float) + eps * 0.5


def null_member(p_bad, alpha: float):
    """Member ``alpha P_bad + (1 - alpha) U`` of the segment (probability scale)."""
    return alpha * np.asarray(p_bad, dtype=float) + (1.0 - alpha) * 0.5


def _bern_kl(p, q):
    """KL( Bern(p) || Bern(q) ), elementwise."""
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    q = np.clip(np.asarray(q, dtype=float), 1e-9, 1 - 1e-9)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


def choose_alpha(p_user_cal, p_bad_cal, alpha_min=ALPHA_MIN):
    """
    Least-favourable member of the segment: the **reverse information
    projection** of the provider's law onto P_0,

        alpha* = argmin_alpha  E[ KL( Bern(p_user) || Bern(p_alpha) ) ].

    This is the choice that makes the licence a valid e-variable *uniformly
    over the whole segment* (Grunwald et al.), and convexity of P_0 is exactly
    what buys that uniformity -- so obedience holds against every dilution, not
    just against the endpoint.

    Note the Waterbirds objective instead maximises the null's fit to the
    realised labels. That expression does not involve the provider at all, so
    it returns essentially the same alpha for every audited provider; it deters
    the exemplar but lets a sufficiently diluted copy through. The projection
    fixes this.

    KL is convex in its second argument and p_alpha is affine in alpha, so the
    objective is convex in alpha and the 1-D solve is exact.
    """
    def objective(alpha):
        a = float(np.clip(alpha, alpha_min, 1.0))
        return float(np.mean(_bern_kl(p_user_cal, null_member(p_bad_cal, a))))

    res = minimize_scalar(objective, bounds=(alpha_min, 1.0), method="bounded",
                          options={"xatol": 1e-12})
    return float(np.clip(res.x, alpha_min, 1.0))


def run_martingale_test(p_user, p_bad, y, C=C_DEFAULT, R=R_DEFAULT,
                        eps=EPS_SMOOTH, cal_frac=0.2, alpha_min=ALPHA_MIN,
                        clip_steps=True):
    """
    One audit. Returns ``(licence_history, info)``.

    Inputs are on the probability scale; the licence uses the likelihood of the
    realised outcome. ``licence_history`` is flat at C through calibration, then
    accumulates ``pi_n = min(C * prod q_user / p*, R)``.
    """
    p_user = np.asarray(p_user, dtype=float)
    p_bad = np.asarray(p_bad, dtype=float)
    y = np.asarray(y, dtype=float)

    n = len(p_user)
    split = max(int(n * cal_frac), 10)

    alpha = choose_alpha(p_user[:split], p_bad[:split], alpha_min=alpha_min)

    # eps-smooth BOTH sides. Smoothing only the numerator (as the Waterbirds
    # code does) inflates every ratio: where q is small the smoothing lifts it
    # a lot and where q is near 1 it lowers it only slightly, so the asymmetry
    # is a systematic upward drift that is not evidence of compliance. It is
    # invisible there because the audited alternative genuinely differs from the
    # null; here, where a provider can sit exactly on the segment, it is the
    # difference between deterring a dilution and licensing it.
    ps_user = smooth(p_user, eps)
    ps_star = smooth(null_member(p_bad, alpha), eps)
    q_user = y * ps_user + (1 - y) * (1 - ps_user)
    q_star = y * ps_star + (1 - y) * (1 - ps_star)

    ratios = q_user[split:] / np.clip(q_star[split:], 1e-9, None)
    if clip_steps:
        # Prop. 4.4 truncates the per-observation licence at the market cap
        ratios = np.clip(ratios, 1.0 / (R / C), R / C)

    licence = [float(C)] * split                     # calibration: wealth held
    w = float(C)
    for r in ratios:
        w *= float(r)
        if w < 1e-4:
            w = 0.0
        licence.append(min(w, R))

    info = {"alpha": alpha, "split": split,
            "mean_log_ratio": float(np.mean(np.log(np.clip(ratios, 1e-12, None)))),
            "max_ratio": float(np.max(ratios))}
    return np.array(licence), info


PART_TOL = 0.01        # provider needs >1% surplus over the fee to enter


def outcome(licence, C=C_DEFAULT, R=R_DEFAULT, tol=PART_TOL):
    """
    Final licence value and participation decision.

    Participation requires a surplus, pi > C (1 + tol). The tolerance is not
    cosmetic. A provider whose law lies exactly in the credal null earns a
    licence that is an exact martingale at C -- obedience is tight there -- so
    pi sits at C up to numerical precision, and with tol = 0 the verdict is
    decided by the alpha-solver's rounding compounded over T rounds. A provider
    indifferent at pi = C gains nothing by paying the entry fee, so requiring a
    genuine surplus is both economically right and numerically stable.
    """
    pi = float(min(licence[-1], R))
    return pi, bool(pi > C * (1.0 + tol))


def run_audits(p_user_pool, p_bad_pool, y_pool, T, n_runs, rng,
               C=C_DEFAULT, R=R_DEFAULT, eps=EPS_SMOOTH, cal_frac=0.2,
               alpha_min=ALPHA_MIN):
    """Repeat the audit on ``n_runs`` fresh applicant streams."""
    n_pool = len(p_user_pool)
    traj, lic, part, alphas = [], [], [], []
    for _ in range(n_runs):
        idx = rng.integers(0, n_pool, size=T)
        h, info = run_martingale_test(p_user_pool[idx], p_bad_pool[idx],
                                      y_pool[idx],
                                      C=C, R=R, eps=eps, cal_frac=cal_frac,
                                      alpha_min=alpha_min)
        pi, p = outcome(h, C=C, R=R)
        traj.append(h)
        lic.append(pi)
        part.append(p)
        alphas.append(info["alpha"])
    return np.array(traj), np.array(lic), np.array(part), np.array(alphas)


def plot_licence(ax, traj, label, color, split=None, **kwargs):
    """
    Mean licence trajectory with a mean +/- 1 standard-error band.

    SE is across audit runs (ddof=1). If ``split`` is given the calibration
    phase is drawn faded, as in the Waterbirds burn-in figure.
    """
    A = np.asarray(traj, dtype=float)
    n_runs = A.shape[0]
    m = A.mean(axis=0)
    se = (A.std(axis=0, ddof=1) / np.sqrt(n_runs)
          if n_runs > 1 else np.zeros_like(m))
    x = np.arange(len(m))
    if split:
        ax.plot(x[:split], m[:split], color=color, lw=2.5, alpha=0.45)
        ax.plot(x[split - 1:], m[split - 1:], color=color, lw=2.5,
                label=label, **kwargs)
    else:
        ax.plot(x, m, color=color, lw=2.5, label=label, **kwargs)
    ax.fill_between(x, np.maximum(m - se, 0.0), m + se, color=color, alpha=0.2)
