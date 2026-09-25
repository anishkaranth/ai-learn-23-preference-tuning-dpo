# ai-learn-23-preference-tuning-dpo

Preference tuning from scratch in NumPy. We build a toy world of prompts with candidate responses, collect synthetic
human preferences, train a **Bradley-Terry reward model**, and then tune a softmax policy with **DPO** against a
frozen reference policy. A beta sweep shows the trade-off between win rate and KL.

Part of the AI learning series (after `ai-learn-11-lora-scratch` … `ai-learn-14-end-to-end-ai-assistant`).

## Architecture

```mermaid
flowchart LR
    W[(Prompts x K candidates<br/>feature vectors phi)] --> REF[Frozen reference policy<br/>pi_ref = softmax theta_ref·phi]
    REF -->|sample 2 responses| J[Bradley-Terry 'human' judge<br/>on hidden true reward]
    J --> PP[(Preference pairs<br/>y_w ≻ y_l)]
    PP --> RM[Reward model<br/>-log σ(r_w − r_l)]
    PP --> DPO["DPO loss<br/>-log σ(β[Δlog π − Δlog π_ref])"]
    REF --> DPO
    DPO --> POL[Tuned policy pi_theta]
    RM --> OPT[Closed-form RLHF optimum<br/>pi_ref·exp(r/β)]
    POL --> EV[Win rate vs ref · KL to ref<br/>E true reward]
    OPT -. sanity check .-> EV
```

## What you'll learn
- How a Bradley-Terry reward model turns pairwise preferences into a scalar reward, and why noisy labels cap its pairwise accuracy.
- The DPO objective and its gradient. For a softmax policy the partition function cancels, so DPO is logistic regression on log-ratio margins.
- What beta does: small beta pushes the policy far from the reference (high KL, higher win rate), and large beta keeps it close.
- That DPO recovers the KL-regularised RLHF optimum `pi_ref · exp(r/β)` without running RL.

## Layout
| file | purpose |
|---|---|
| `prefs.py` | Toy preference world: features, hidden true reward, reference policy, pair sampling |
| `dpo.py` | BT reward model, DPO loss/grad and trainer, KL, exact win rate, expected reward |
| `run_smoke.py` | Full experiment: RM → DPO beta sweep → closed-form check → `results/` |
| `smoke_plots.py` | SVG plots + RESULTS.md writer |
| `svg_utils.py` | Makes SVGs smaller so they are easy to diff |
| `notebooks/preference_tuning_dpo.ipynb` | Step-by-step walkthrough |
| `results/` | `RESULTS.md`, `metrics.json`, `JSON.shot`, SVG plots from the real smoke run |

## Run
```bash
pip install -r requirements.txt
python run_smoke.py          # ~2 s on CPU, seed 42, writes results/
jupyter notebook notebooks/preference_tuning_dpo.ipynb
```

## Results (seed 42, from `results/metrics.json`)
| metric | value |
|---|---|
| reward model test pairwise acc (vs noisy labels / vs true order) | 0.822 / 0.916 |
| label agreement with true order (noise ceiling) | 0.838 |
| DPO beta=0.1: win rate vs ref, KL to ref | **0.7497**, 1.50 |
| DPO beta=1: win rate vs ref, KL to ref | 0.6852, 0.539 |
| DPO beta=10: win rate vs ref, KL to ref | 0.5289, 0.011 |
| E[true reward]: ref → best DPO (oracle best-of-6) | 0.634 → 2.451 (2.530) |

See [results/RESULTS.md](results/RESULTS.md) for the full sweep and plots.

## Caveats
- The world is synthetic and linear. Real DPO uses sequence log-probs from an LM, where the partition function does not cancel so conveniently.
- Win rate is computed exactly, using the hidden true reward as the judge. Real evaluations use sampled judges and are noisy.
- One seed. The smallest beta (0.03) needs many Adam steps to converge.

## Next steps
- Add IPO / KTO / ORPO losses on the same pairs.
- On-policy iterated DPO: resample pairs from the current policy.
- Make the policy non-linear and show reward over-optimisation at small beta.
