#!/usr/bin/env python3
"""DPO smoke: synthetic preferences -> Bradley-Terry reward model -> DPO beta sweep vs frozen reference -> results/."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np

from dpo import expected_reward, kl_to_ref, train_dpo, train_reward_model, win_rate
from prefs import FEATURES, THETA_REF, W_TRUE, make_prompts, policy_logp, sample_pairs, true_reward
from smoke_plots import make_plots, write_results_md

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SEED = 42
CFG = {"n_train_prompts": 400, "n_test_prompts": 300, "k_candidates": 6, "n_train_pairs": 3000, "n_test_pairs": 1500,
       "rm_steps": 500, "rm_lr": 0.05, "rm_l2": 1e-3, "dpo_steps": 3000, "dpo_lr": 0.05,
       "betas": [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]}


def _compact(js: str) -> str:
    js = re.sub(r"\[\s+([^\[\]{}]*?)\s+\]", lambda m: "[" + re.sub(r"\s+", " ", m.group(1)) + "]", js)
    return re.sub(r"\{\n([^{}\[\]]*?)\n\s*\}", lambda m: "{" + re.sub(r"\s*\n\s*", " ", m.group(1)).strip() + "}", js)


def r4(x) -> float:
    return round(float(x), 4)


def main() -> None:
    t0 = time.perf_counter()
    rng = np.random.default_rng(SEED)
    tr = make_prompts(CFG["n_train_prompts"], CFG["k_candidates"], rng)
    te = make_prompts(CFG["n_test_prompts"], CFG["k_candidates"], rng)
    P, W, L, agree = sample_pairs(tr, THETA_REF, CFG["n_train_pairs"], rng)
    Pt, Wt, Lt, agree_t = sample_pairs(te, THETA_REF, CFG["n_test_pairs"], rng)
    dphi, dphi_t = tr[P, W] - tr[P, L], te[Pt, Wt] - te[Pt, Lt]
    r_te = true_reward(te)
    true_order_t = np.sign(r_te[Pt, Wt] - r_te[Pt, Lt])  # +1 if the labelled winner really is better

    # 1) Bradley-Terry reward model
    w_rm, rm_hist = train_reward_model(dphi, CFG["rm_steps"], CFG["rm_lr"], CFG["rm_l2"])
    rm = {"weights": dict(zip(FEATURES, map(r4, w_rm))), "true_weights": dict(zip(FEATURES, map(float, W_TRUE))),
          "train_pair_acc": r4(((dphi @ w_rm) > 0).mean()), "test_pair_acc": r4(((dphi_t @ w_rm) > 0).mean()),
          "test_true_order_acc": r4((np.sign(dphi_t @ w_rm) == true_order_t).mean()),
          "label_noise_ceiling": r4(agree_t.mean()), "final_loss": r4(rm_hist[-1])}

    # 2) DPO sweep against the frozen reference; compare with the closed-form KL-regularised RLHF optimum
    #    pi*(y|x) ∝ pi_ref(y|x) exp(r_rm(x,y) / beta)  <=>  theta* = theta_ref + w_rm / beta
    ref = {"expected_true_reward": r4(expected_reward(THETA_REF, te, r_te)), "win_rate_vs_ref": 0.5, "kl": 0.0}
    sweep, curves = [], {}
    for beta in CFG["betas"]:
        th, hist = train_dpo(THETA_REF, dphi, beta, CFG["dpo_steps"], CFG["dpo_lr"])
        th_star = THETA_REF + w_rm / beta
        lp, ls = policy_logp(th, te), policy_logp(th_star, te)
        sweep.append({"beta": beta, "win_rate_vs_ref": r4(win_rate(th, THETA_REF, te, r_te)),
                      "kl_to_ref": r4(kl_to_ref(th, THETA_REF, te)),
                      "expected_true_reward": r4(expected_reward(th, te, r_te)),
                      "implicit_rm_test_pair_acc": r4(((dphi_t @ (th - THETA_REF)) > 0).mean()),
                      "kl_dpo_vs_rlhf_optimum": float(f"{(np.exp(lp) * (lp - ls)).sum(-1).mean():.2e}"),
                      "final_dpo_loss": r4(hist[-1]), "theta": [round(float(v), 3) for v in th]})
        curves[str(beta)] = [r4(hist[i]) for i in range(0, len(hist), 100)]
    best = max(sweep, key=lambda r: r["win_rate_vs_ref"])
    b1 = next(r for r in sweep if r["beta"] == 1.0)
    oracle = r4(r_te.max(-1).mean())

    m = {"project": "ai-learn-23-preference-tuning-dpo", "seed": SEED, "config": CFG, "features": FEATURES,
         "theta_ref": THETA_REF.tolist(), "reward_model": rm, "reference": ref, "oracle_best_of_k_reward": oracle,
         "dpo_sweep": sweep, "dpo_loss_curves_every100": curves, "rm_loss_curve_every25": rm_hist[::25],
         "headline": {
             "rm_test_pair_acc": rm["test_pair_acc"], "rm_test_true_order_acc": rm["test_true_order_acc"],
             "label_agreement_with_true_order": rm["label_noise_ceiling"],
             "best_beta": best["beta"], "best_win_rate_vs_ref": best["win_rate_vs_ref"], "best_kl_to_ref": best["kl_to_ref"],
             "beta1_win_rate_vs_ref": b1["win_rate_vs_ref"], "beta1_kl_to_ref": b1["kl_to_ref"],
             "beta10_win_rate_vs_ref": sweep[-1]["win_rate_vs_ref"], "beta10_kl_to_ref": sweep[-1]["kl_to_ref"],
             "ref_expected_true_reward": ref["expected_true_reward"], "best_expected_true_reward": best["expected_true_reward"],
             "oracle_best_of_k_reward": oracle,
         }}
    m["rm_loss_curve_every25"] = [r4(v) for v in m["rm_loss_curve_every25"]]
    m["wall_time_s"] = round(time.perf_counter() - t0, 2)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "metrics.json").write_text(_compact(json.dumps(m, indent=1, ensure_ascii=False)) + "\n")
    shot = {"project": m["project"], "seed": SEED, "config": CFG, "headline": m["headline"]}
    (RESULTS / "JSON.shot").write_text(_compact(json.dumps(shot, indent=2)) + "\n")
    plots = make_plots(RESULTS, m)
    write_results_md(RESULTS / "RESULTS.md", m, plots)
    json.loads((RESULTS / "JSON.shot").read_text())
    print(json.dumps(m["headline"], indent=2), "\nwall", m["wall_time_s"])


if __name__ == "__main__":
    main()
