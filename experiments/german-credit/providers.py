"""
Providers and the regulator's prohibited exemplar.

Three disjoint splits of the dataset, which is what keeps the audit honest:

    reg    the regulator's own supervisory data. It trains its prohibited
           exemplar P_bad here -- a scorer that uses the protected attribute,
           fitted to biased lending history. This is the regulator's own
           artifact, so knowing it exactly is legitimate.
    prov   the providers' training data. `blind` and `uses_A` are fitted here.
    audit  the applicant population the audit draws from. Untouched by both.

Because P_bad and uses_A are trained on *different* splits, even the
non-compliant provider is not exactly on the prohibited segment -- so no
likelihood ratio is degenerate by construction.

Waterbirds analogue
-------------------
    Waterbirds                       Credit scoring
    -------------------------------  --------------------------------------
    ERM       (uses the shortcut)    P_bad / uses_A  (use the attribute)
    GroupDRO  (does not)             blind           (does not)
    softmax at the true label        P(realised repayment | applicant)
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 0


# --------------------------------------------------------------------------
# likelihood of the realised outcome
# --------------------------------------------------------------------------
def likelihood(p_good, y):
    """P(realised label) for a scorer that returns P(repay) = p_good."""
    p_good = np.asarray(p_good, dtype=float)
    y = np.asarray(y, dtype=float)
    return y * p_good + (1.0 - y) * (1.0 - p_good)


def dilute(p, gamma: float):
    """
    A provider that randomises between a scorer and a coin flip.

    ``gamma * p + (1 - gamma) * 0.5`` on the *probability* scale is the law of a
    provider who uses the scorer with probability gamma and answers
    uninformatively otherwise. These are exactly the members of the prohibited
    segment, so this is the mixing attack: what a provider would try in order to
    dilute a banned model out of the null.
    """
    return gamma * np.asarray(p, dtype=float) + (1.0 - gamma) * 0.5


# --------------------------------------------------------------------------
# historical bias (Waterbirds' training-time shortcut)
# --------------------------------------------------------------------------
def biased_indices(a, y, idx, bias: float, rng, direction: float = 1.0):
    """
    Drop stereotype-inconsistent records from a TRAINING split.

    Discriminatory scorers arise because historical data over-states the
    association between the attribute and repayment, not because the attribute
    is mildly predictive. With direction = +1 the dropped records are group-1
    applicants who defaulted and group-0 applicants who repaid.
    """
    if bias <= 0:
        return idx
    a_i, y_i = np.asarray(a)[idx], np.asarray(y)[idx]
    if direction >= 0:
        inconsistent = ((a_i == 1) & (y_i == 0)) | ((a_i == 0) & (y_i == 1))
    else:
        inconsistent = ((a_i == 1) & (y_i == 1)) | ((a_i == 0) & (y_i == 0))
    keep = idx[~(inconsistent & (rng.random(len(idx)) < bias))]
    if len(keep) < 50 or len(np.unique(np.asarray(y)[keep])) < 2:
        return idx
    return keep


def _fit(X, y, idx, use_a, a=None):
    """Fit a logistic scorer on ``idx``, with or without the attribute."""
    Xv = np.asarray(X, dtype=float)
    if use_a:
        Xv = np.hstack([Xv, np.asarray(a, dtype=float).reshape(-1, 1)])
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    model.fit(Xv[idx], np.asarray(y)[idx])
    return model, Xv


class World:
    """
    Everything evaluated on the audit split.

    Scores are kept on the **probability** scale (p = P(repay)). The projection
    onto the prohibited segment is a divergence between Bernoulli laws, so it
    needs probabilities, not just the likelihood of the realised label.
    """

    def __init__(self, a, y, p_bad, p_blind, p_uses, meta):
        self.a, self.y = a, y
        self.p_bad = p_bad        # regulator's prohibited exemplar
        self.p_blind = p_blind    # compliant provider
        self.p_uses = p_uses      # non-compliant provider
        self.meta = meta

    # likelihoods of the realised outcome, for convenience
    def q(self, p):
        return likelihood(p, self.y)


def fit_world(X, a, y, seed: int = SEED, bias: float = 0.8,
              fracs=(1 / 3, 1 / 3, 1 / 3)):
    """
    Build the three splits, train the regulator's exemplar and both providers,
    and return every likelihood on the audit split.
    """
    y = np.asarray(y)
    a = np.asarray(a)
    idx = np.arange(len(y))

    f_reg, f_prov, _ = fracs
    idx_reg, idx_rest = train_test_split(idx, train_size=f_reg,
                                         random_state=seed, stratify=y)
    idx_prov, idx_audit = train_test_split(
        idx_rest, train_size=f_prov / (1.0 - f_reg),
        random_state=seed, stratify=y[idx_rest])

    rng = np.random.default_rng(seed)
    # regulator's prohibited exemplar: uses A, fitted to biased history
    reg_b = biased_indices(a, y, idx_reg, bias, rng)
    m_bad, Xa = _fit(X, y, reg_b, use_a=True, a=a)

    # providers, trained on their own split
    prov_b = biased_indices(a, y, idx_prov, bias, rng)
    m_uses, _ = _fit(X, y, prov_b, use_a=True, a=a)
    m_blind, Xp = _fit(X, y, idx_prov, use_a=False)

    ya = y[idx_audit]
    p_bad = m_bad.predict_proba(Xa[idx_audit])[:, 1]
    p_uses = m_uses.predict_proba(Xa[idx_audit])[:, 1]
    p_blind = m_blind.predict_proba(Xp[idx_audit])[:, 1]
    q_bad, q_uses, q_blind = (likelihood(p_bad, ya), likelihood(p_uses, ya),
                              likelihood(p_blind, ya))

    meta = {
        "n_reg": len(idx_reg), "n_reg_biased": len(reg_b),
        "n_prov": len(idx_prov), "n_prov_biased": len(prov_b),
        "n_audit": len(idx_audit),
        "coef_a_bad": float(m_bad[-1].coef_[0][-1]),
        "coef_a_uses": float(m_uses[-1].coef_[0][-1]),
        "loglik_bad": float(np.mean(np.log(np.clip(q_bad, 1e-12, None)))),
        "loglik_uses": float(np.mean(np.log(np.clip(q_uses, 1e-12, None)))),
        "loglik_blind": float(np.mean(np.log(np.clip(q_blind, 1e-12, None)))),
    }
    return World(a[idx_audit], ya, p_bad, p_blind, p_uses, meta)
