"""Toy preference world: prompts with K candidate responses, each described by a feature vector.

Features (per response): helpful, correct, safe, verbose, hedging.
Hidden "true" human reward is linear in most features but *penalises excess verbosity quadratically*,
so a linear reward model / linear policy is slightly misspecified (room for reward hacking).
"""
from __future__ import annotations

import numpy as np

FEATURES = ["helpful", "correct", "safe", "verbose", "hedging"]
W_TRUE = np.array([1.0, 1.5, 0.7, 0.6, -0.4])
VERBOSE_PENALTY = 0.8   # r* -= VERBOSE_PENALTY * max(0, verbose - 0.5)^2
THETA_REF = np.array([0.4, 0.2, 0.3, 0.9, 0.3])  # "SFT" reference: likes long, hedgy answers


def make_prompts(n: int, k: int, rng) -> np.ndarray:
    """Feature tensor of shape (n_prompts, k_candidates, d)."""
    return rng.normal(0, 1, (n, k, len(FEATURES)))


def true_reward(phi: np.ndarray) -> np.ndarray:
    v = phi[..., FEATURES.index("verbose")]
    return phi @ W_TRUE - VERBOSE_PENALTY * np.maximum(0, v - 0.5) ** 2


def log_softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


def policy_logp(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """log pi(y|x) for a linear-softmax policy over the K candidates of each prompt."""
    return log_softmax(phi @ theta)


def sample_pairs(phi: np.ndarray, theta_ref: np.ndarray, n_pairs: int, rng):
    """Draw two distinct responses from the reference policy and let a Bradley-Terry 'human' pick one.

    Returns (prompt idx, winner idx, loser idx, label_matches_true_order)."""
    p = np.exp(policy_logp(theta_ref, phi))
    r = true_reward(phi)
    P, W, L, agree = [], [], [], []
    for _ in range(n_pairs):
        i = rng.integers(len(phi))
        a, b = rng.choice(phi.shape[1], 2, replace=False, p=p[i])
        pa = 1 / (1 + np.exp(-(r[i, a] - r[i, b])))
        w, l = (a, b) if rng.random() < pa else (b, a)
        P.append(i); W.append(w); L.append(l); agree.append(r[i, w] > r[i, l])
    return np.array(P), np.array(W), np.array(L), np.array(agree)
