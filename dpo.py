"""Bradley-Terry reward model and DPO, both from scratch (NumPy, full-batch Adam).

DPO loss (Rafailov et al., 2023) for a preference (x, y_w, y_l):
    L = -log sigma( beta * [ (log pi(y_w|x) - log pi_ref(y_w|x)) - (log pi(y_l|x) - log pi_ref(y_l|x)) ] )
For a linear-softmax policy log pi(y|x) = theta . phi(x,y) - logZ(x); the logZ terms cancel inside the
difference, so the margin is beta * (theta - theta_ref) . (phi_w - phi_l) and the gradient is
    dL/dtheta = -beta * (1 - sigma(margin)) * (phi_w - phi_l).
"""
from __future__ import annotations

import numpy as np

from prefs import policy_logp


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class Adam:
    def __init__(self, dim: int, lr: float):
        self.lr, self.t, self.m, self.v = lr, 0, np.zeros(dim), np.zeros(dim)

    def step(self, x, g):
        self.t += 1
        self.m = 0.9 * self.m + 0.1 * g
        self.v = 0.999 * self.v + 0.001 * g * g
        return x - self.lr * (self.m / (1 - 0.9 ** self.t)) / (np.sqrt(self.v / (1 - 0.999 ** self.t)) + 1e-8)


def train_reward_model(dphi: np.ndarray, steps: int, lr: float, l2: float = 1e-3):
    """Linear BT reward r(x,y) = w . phi: minimise -log sigma(w . (phi_w - phi_l)) + l2 |w|^2."""
    w, opt, hist = np.zeros(dphi.shape[1]), Adam(dphi.shape[1], lr), []
    for _ in range(steps):
        s = sigmoid(dphi @ w)
        hist.append(float(-np.log(s + 1e-12).mean()))
        w = opt.step(w, -((1 - s)[:, None] * dphi).mean(0) + 2 * l2 * w)
    return w, hist


def dpo_loss_grad(theta, theta_ref, dphi, beta):
    margin = beta * (dphi @ (theta - theta_ref))
    s = sigmoid(margin)
    return float(-np.log(s + 1e-12).mean()), -(beta * (1 - s)[:, None] * dphi).mean(0), margin


def train_dpo(theta_ref, dphi, beta: float, steps: int, lr: float):
    theta, opt, hist = theta_ref.copy(), Adam(len(theta_ref), lr), []
    for _ in range(steps):
        loss, g, _ = dpo_loss_grad(theta, theta_ref, dphi, beta)
        hist.append(loss)
        theta = opt.step(theta, g)
    return theta, hist


def kl_to_ref(theta, theta_ref, phi) -> float:
    """Mean over prompts of KL(pi_theta(.|x) || pi_ref(.|x))."""
    lp, lq = policy_logp(theta, phi), policy_logp(theta_ref, phi)
    return float((np.exp(lp) * (lp - lq)).sum(-1).mean())


def win_rate(theta, theta_ref, phi, r_true) -> float:
    """P(true-reward Bradley-Terry judge prefers y ~ pi_theta over y' ~ pi_ref), exact over the K x K grid."""
    p, q = np.exp(policy_logp(theta, phi)), np.exp(policy_logp(theta_ref, phi))
    pref = sigmoid(r_true[:, :, None] - r_true[:, None, :])
    return float(np.einsum("nk,nkj,nj->n", p, pref, q).mean())


def expected_reward(theta, phi, r_true) -> float:
    return float((np.exp(policy_logp(theta, phi)) * r_true).sum(-1).mean())
