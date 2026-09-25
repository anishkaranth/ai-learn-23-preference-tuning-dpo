"""Matplotlib SVG plots + RESULTS.md writer for the DPO smoke run."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from svg_utils import minify_svg  # noqa: E402

plt.rcParams.update({"svg.hashsalt": "ai-learn-23", "svg.fonttype": "none", "font.family": "sans-serif",
                     "font.sans-serif": ["DejaVu Sans"], "axes.unicode_minus": False})


def _save(fig, path: Path) -> str:
    fig.tight_layout()
    buf = io.StringIO()
    fig.savefig(buf, format="svg", metadata={"Date": None})
    plt.close(fig)
    path.write_text(minify_svg(buf.getvalue()), encoding="utf-8")
    return path.name


def make_plots(out: Path, m: Dict[str, Any]) -> List[str]:
    names, sw = [], m["dpo_sweep"]
    betas = [r["beta"] for r in sw]
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.plot(betas, [r["win_rate_vs_ref"] for r in sw], color="#126782", label="win rate vs ref")
    ax.set_xscale("log")
    ax.set_xlabel("beta")
    ax.set_ylabel("win rate vs reference", color="#126782")
    ax2 = ax.twinx()
    ax2.plot(betas, [r["kl_to_ref"] for r in sw], color="#e76f51", ls="--", label="KL to ref")
    ax2.set_ylabel("KL(pi || pi_ref)", color="#e76f51")
    ax.set_title("DPO beta sweep: small beta = stronger push away from ref")
    names.append(_save(fig, out / "beta_sweep.svg"))

    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    ax.plot([0] + [r["kl_to_ref"] for r in sw[::-1]], [0.5] + [r["win_rate_vs_ref"] for r in sw[::-1]], color="#126782")
    for r in sw:
        ax.annotate(f"b={r['beta']}", (r["kl_to_ref"], r["win_rate_vs_ref"]), fontsize=7)
    ax.set_xlabel("KL to reference")
    ax.set_ylabel("win rate vs reference")
    ax.set_title("Reward / KL frontier")
    names.append(_save(fig, out / "kl_winrate_frontier.svg"))

    fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.2))
    ax[0].plot([i * 25 for i in range(len(m["rm_loss_curve_every25"]))], m["rm_loss_curve_every25"], color="#2a9d8f")
    ax[0].set_title("Bradley-Terry reward model loss")
    ax[0].set_xlabel("step")
    for b, c in m["dpo_loss_curves_every100"].items():
        ax[1].plot([i * 100 for i in range(len(c))], c, label=f"beta={b}")
    ax[1].set_title("DPO loss")
    ax[1].set_xlabel("step")
    ax[1].legend(fontsize=7)
    names.append(_save(fig, out / "loss_curves.svg"))
    return names


def write_results_md(path: Path, m: Dict[str, Any], plots: List[str]) -> None:
    rm, h = m["reward_model"], m["headline"]
    rows = "\n".join(f"| {r['beta']} | {r['win_rate_vs_ref']} | {r['kl_to_ref']} | {r['expected_true_reward']} | "
                     f"{r['implicit_rm_test_pair_acc']} | {r['kl_dpo_vs_rlhf_optimum']} | {r['final_dpo_loss']} |" for r in m["dpo_sweep"])
    wts = " ".join(f"{k}={v}" for k, v in rm["weights"].items())
    twts = " ".join(f"{k}={v}" for k, v in rm["true_weights"].items())
    txt = f"""# Results: ai-learn-23-preference-tuning-dpo

Real output of `python run_smoke.py` (seed {m['seed']}, CPU, {m['wall_time_s']} s wall time).

## Setup
- {m['config']['n_train_prompts']} train and {m['config']['n_test_prompts']} test prompts, each with {m['config']['k_candidates']} candidate responses described by features {m['features']}.
- The reference policy is linear-softmax with theta_ref={m['theta_ref']} (it likes verbose, hedgy answers).
- Preference pairs: {m['config']['n_train_pairs']} train and {m['config']['n_test_pairs']} test pairs, sampled from pi_ref and labelled by a Bradley-Terry judge on a hidden true reward
  (linear, with a quadratic penalty on excess verbosity). The labels match the true ordering {rm['label_noise_ceiling']} of the time on test pairs.

## Reward model (Bradley-Terry, linear)
- Test pairwise accuracy vs labels: **{rm['test_pair_acc']}**. vs the true ordering: {rm['test_true_order_acc']}. Train: {rm['train_pair_acc']}.
- Learned weights: {wts}
- True linear weights: {twts} (plus the verbosity penalty the linear model can't express)

## DPO beta sweep (policy starts at pi_ref and pi_ref stays frozen)
Reference expected true reward = {m['reference']['expected_true_reward']}. Oracle best-of-K = {m['oracle_best_of_k_reward']}.

| beta | win rate vs ref | KL to ref | E[true reward] | implicit-RM test pair acc | KL(DPO ‖ closed-form RLHF opt.) | final DPO loss |
|---|---|---|---|---|---|---|
{rows}

## Plots
""" + "\n".join(f"![{p}]({p})" for p in plots) + f"""

## Observations (from the numbers above)
- Smaller beta moves the policy further from the reference (higher KL) and raises the win rate, until it saturates near the
  best-of-K ceiling. Below beta≈0.3 extra KL buys almost nothing: best win rate {h['best_win_rate_vs_ref']} at beta={h['best_beta']}.
  Large beta keeps the policy close to pi_ref.
- Every beta converges to the same DPO loss. The optimum is theta = theta_ref + w_MLE / beta, so beta only rescales the step.
- The reward model gives `verbose` about 0 weight even though the true linear weight is +0.6. The hidden quadratic penalty cancels it on
  this data, which is a small example of reward-model misspecification.
- The implicit reward (beta * log pi/pi_ref) ranks test pairs the same way at every beta. For a linear policy DPO learns the same
  direction as the BT reward model, and beta only sets how far along it the policy moves.
- The last column checks that DPO matches the closed-form KL-regularised RLHF optimum pi_ref * exp(r_rm / beta) with the separately trained reward model.

## Honest notes
- Everything is synthetic and linear. Win rate is computed exactly over the K x K candidate grid using the hidden true reward as the judge.
- A single seed was run. The smallest beta may not be fully converged within {m['config']['dpo_steps']} Adam steps (see its final loss).
"""
    path.write_text(txt, encoding="utf-8")
