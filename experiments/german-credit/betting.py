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
    def objective(alpha):
        a = float(np.clip(alpha, alpha_min, 1.0))
        return float(np.mean(_bern_kl(p_user_cal, null_member(p_bad_cal, a))))

    res = minimize_scalar(objective, bounds=(alpha_min, 1.0), method="bounded",
                          options={"xatol": 1e-12})
    return float(np.clip(res.x, alpha_min, 1.0))


def run_martingale_test(p_user, p_bad, y, C=C_DEFAULT, R=R_DEFAULT,
                        eps=EPS_SMOOTH, cal_frac=0.2, alpha_min=ALPHA_MIN,
                        clip_steps=True):
    p_user = np.asarray(p_user, dtype=float)
    p_bad = np.asarray(p_bad, dtype=float)
    y = np.asarray(y, dtype=float)

    n = len(p_user)
    split = max(int(n * cal_frac), 10)

    alpha = choose_alpha(p_user[:split], p_bad[:split], alpha_min=alpha_min)

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
