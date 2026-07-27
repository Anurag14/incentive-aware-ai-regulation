"""
Data loaders for the credit-scoring regulation experiment.

Each loader returns a triple ``(X, a, y)``:
    X : pd.DataFrame  legitimate (non-sensitive) features only
    a : np.ndarray    binary sensitive attribute in {0, 1}
    y : np.ndarray    binary label, 1 = creditworthy / good credit

The sensitive attribute is *excluded* from X by construction: a provider that
"uses the sensitive attribute" must inject it explicitly (see providers.py),
which makes non-compliance a deliberate, controllable act.

Datasets
--------
taiwan : UCI "Default of Credit Card Clients" (id=350, n=30000). A = SEX.
german : UCI "Statlog German Credit Data" (id=144, n=1000). A = sex parsed
         from the personal-status attribute (A91/A93/A94 male, A92/A95 female).
synth  : Reproducible synthetic generator with the same schema. Offline
         fallback so the pipeline is runnable without network access.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 0


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
LAST_FAILURE = None      # reason the most recent real-data load failed


def _fail(msg: str):
    """
    Record and *print* a load failure.

    Uses print rather than warnings.warn: Jupyter shows a given warning only
    once per code location (``__warningregistry__``), so on a re-run of the
    cell the reason would silently vanish and you would see nothing but the
    fallback message.
    """
    global LAST_FAILURE
    LAST_FAILURE = msg
    print(f"[data] !! {msg}")


def _fetch_uci(uci_id: int):
    """Fetch a UCI dataset via ucimlrepo. Returns (features, targets) or None."""
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError:
        _fail("ucimlrepo is not installed -- run `pip install ucimlrepo` "
              "and restart the kernel")
        return None
    try:
        ds = fetch_ucirepo(id=uci_id)
        return ds.data.features.copy(), ds.data.targets.copy()
    except Exception as exc:  # network blocked, proxy, UCI down, ...
        _fail(f"could not fetch UCI id={uci_id} ({type(exc).__name__}: {exc})")
        return None


def _as_1d(targets) -> np.ndarray:
    return np.asarray(targets).reshape(-1)


# --------------------------------------------------------------------------
# Taiwan: default of credit card clients
# --------------------------------------------------------------------------
def load_taiwan():
    """UCI id=350. Sensitive attribute: SEX (1=male -> 0, 2=female -> 1)."""
    got = _fetch_uci(350)
    if got is None:
        return None
    X, targets = got

    sex_col = next((c for c in X.columns if str(c).upper() in ("SEX", "X2")), None)
    if sex_col is None:
        _fail(f"Taiwan: no SEX/X2 column. Columns seen: {list(X.columns)[:12]}")
        return None

    a = (pd.to_numeric(X[sex_col], errors="coerce").fillna(1).astype(int) == 2).astype(int).to_numpy()

    # target: 1 = default next month  ->  y = 1 means *good* credit (no default)
    default = pd.to_numeric(pd.Series(_as_1d(targets)), errors="coerce").fillna(0).astype(int)
    y = (1 - default).to_numpy()

    X_legit = X.drop(columns=[sex_col]).apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return X_legit.reset_index(drop=True), a, y


# --------------------------------------------------------------------------
# German credit (Statlog)
# --------------------------------------------------------------------------
_GERMAN_MALE = {"A91", "A93", "A94"}
_GERMAN_FEMALE = {"A92", "A95"}


def load_german():
    """UCI id=144. Sensitive attribute: sex parsed from personal-status codes."""
    got = _fetch_uci(144)
    if got is None:
        return None
    X, targets = got

    # locate the personal-status/sex column by its A9x code vocabulary
    status_col = None
    for c in X.columns:
        vals = set(map(str, pd.Series(X[c]).dropna().unique()))
        if vals & (_GERMAN_MALE | _GERMAN_FEMALE):
            status_col = c
            break
    if status_col is None:
        _fail(f"German: no personal-status column with A91-A95 codes. "
              f"Columns seen: {list(X.columns)[:12]}")
        return None

    a = X[status_col].astype(str).isin(_GERMAN_FEMALE).astype(int).to_numpy()

    # target: 1 = good, 2 = bad  ->  y = 1 means good credit
    tgt = pd.to_numeric(pd.Series(_as_1d(targets)), errors="coerce").fillna(1).astype(int)
    y = (tgt == 1).astype(int).to_numpy()

    X_legit = pd.get_dummies(X.drop(columns=[status_col]), drop_first=True).astype(float)
    return X_legit.reset_index(drop=True), a, y


# --------------------------------------------------------------------------
# Synthetic fallback (same schema, no network required)
# --------------------------------------------------------------------------
def load_synthetic(n: int = 30000, d: int = 12, seed: int = SEED,
                   beta_a: float = 0.8):
    """
    Synthetic credit data in which the protected attribute is *genuinely
    predictive* -- which is the realistic regulatory situation.

    Group membership both shifts a couple of legitimate features (legitimate
    correlation) and has a direct effect ``beta_a`` on repayment. A model given
    access to A therefore predicts better on the population as a whole, so the
    regulator cannot detect its use from overall accuracy: it must look at the
    counter-stereotypical applicants the shortcut is wrong about.

    Set ``beta_a = 0`` for the degenerate case where A carries no extra signal.
    """
    rng = np.random.default_rng(seed)
    a = rng.binomial(1, 0.5, size=n)

    X = rng.normal(size=(n, d))
    X[:, 0] += 0.35 * a          # legitimate correlation with group
    X[:, 1] -= 0.20 * a

    w = rng.normal(scale=0.6, size=d)
    p_good = 1.0 / (1.0 + np.exp(-(X @ w + beta_a * (2.0 * a - 1.0))))
    y = rng.binomial(1, p_good)

    cols = [f"f{i}" for i in range(d)]
    return pd.DataFrame(X, columns=cols), a.astype(int), y.astype(int)


# --------------------------------------------------------------------------
# dispatch
# --------------------------------------------------------------------------
_LOADERS = {"taiwan": load_taiwan, "german": load_german}


def load(name: str = "taiwan", allow_synthetic_fallback: bool = True,
         strict: bool = False):
    """
    Load a dataset by name ('taiwan', 'german', 'synth').

    If the real dataset cannot be fetched and ``allow_synthetic_fallback`` is
    True, returns the synthetic generator so the pipeline still runs; the
    reason is always printed. Pass ``strict=True`` (or
    ``allow_synthetic_fallback=False``) to raise instead of falling back --
    use this when producing paper figures, so a silent fallback can never be
    mistaken for a real-data run.
    """
    if strict:
        allow_synthetic_fallback = False
    name = str(name).lower()
    if name in ("synth", "synthetic"):
        print("[data] using synthetic generator")
        return load_synthetic()

    if name not in _LOADERS:
        raise ValueError(f"unknown dataset {name!r}; choose from {list(_LOADERS) + ['synth']}")

    out = _LOADERS[name]()
    if out is not None:
        X, a, y = out
        print(f"[data] loaded {name}: n={len(X)}, d={X.shape[1]}, "
              f"P(a=1)={a.mean():.3f}, P(y=1)={y.mean():.3f}")
        return out

    if not allow_synthetic_fallback:
        raise RuntimeError(
            f"could not load {name!r} and fallback is disabled. "
            f"Reason: {LAST_FAILURE}")
    print(f"[data] {name} unavailable -> falling back to synthetic generator")
    return load_synthetic()


def diagnose():
    """Print why real-data loading does or does not work in this environment."""
    print("--- credit_scoring data diagnostics ---")
    try:
        import ucimlrepo
        print(f"ucimlrepo   : installed (v{getattr(ucimlrepo, '__version__', '?')})")
    except ImportError:
        print("ucimlrepo   : NOT INSTALLED  ->  pip install ucimlrepo, then restart the kernel")
        return
    for name, uid in (("taiwan", 350), ("german", 144)):
        got = _fetch_uci(uid)
        if got is None:
            print(f"{name:<12}: FAILED  ({LAST_FAILURE})")
        else:
            X, _ = got
            print(f"{name:<12}: ok, n={len(X)}, columns[:8]={list(X.columns)[:8]}")
