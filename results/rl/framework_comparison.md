# PyTorch vs TensorFlow: Same PPO, Two Frameworks

Phase 3 of the reinforcement-learning extension. The same PPO algorithm was implemented in PyTorch (`rl/ppo_torch.py`) and TensorFlow / Keras 3 (`rl/ppo_tf.py`). Both agents run in the same environment and use the same shared code for rollouts, advantages, minibatches, evaluation and checkpoints. Only the network code differs.

Full per-seed table: [`framework_evaluation.md`](framework_evaluation.md). Curves: [`frameworks_single.png`](frameworks_single.png), [`frameworks_multi.png`](frameworks_multi.png).

## 1. Are the two implementations really the same algorithm?

`tests/test_framework_equivalence.py` copies the weights of a PyTorch agent into a TensorFlow agent and compares them directly:

| Check | Result |
|---|---|
| Actions, values, log-probabilities | Identical (float32 tolerance) |
| Policy loss, value loss, entropy | Identical (relative 1e-5) |
| Gradients (one plain-SGD step) | Weight changes differ by **9 × 10⁻⁷** (relative) |
| 20 Adam steps | Weight changes differ by **1.2%** |

### A subtle difference: Adam's epsilon

The first version used `eps = 1e-5` in both frameworks, a common PPO setting. Even so, one Adam update on the same minibatch differed by about 59%.

The two libraries put epsilon in different places:

```text
PyTorch:  update = lr · m̂ / (√v̂ + ε)                       (ε added to the bias-corrected √v̂)
Keras:    update = lr · √(1 − β₂ᵗ)/(1 − β₁ᵗ) · m / (√v + ε)  (ε added to the uncorrected √v)
```

Rewritten in PyTorch's form, Keras' effective epsilon is `ε / √(1 − β₂ᵗ)`: **31.6 × larger in the first step**, approaching ε only after thousands of steps. For gradients around 1e-4, that changes the early updates substantially.

The fix is a shared `adam_epsilon` setting with a small value (1e-8, PyTorch's default), where epsilon is negligible in both formulations. All runs below use it. Same name, same value and still not the same algorithm: exactly the kind of difference a cross-framework comparison has to rule out first.

## 2. Learning results

Identical configuration and seeds (0, 1, 2), 1,000,000 steps per run. Mean over 3 seeds, 10 evaluation episodes per track:

| Model | Classic Oval | Tall Oval | Asymmetric Oval | Unseen Narrow Oval | Worst training-track fitness |
|---|---:|---:|---:|---:|---:|
| NE single-track | 10.0 laps | 0.0 | 0.0 | 0.0 | 252 |
| PyTorch single-track | 9.9 laps | 0.0 | 0.0 | 0.0 | -17 |
| TensorFlow single-track | 9.9 laps | 0.0 | 0.0 | 0.0 | 123 |
| NE multi-track | 0.0 | 9.0 laps | 0.0 | 8.0 laps | **3,420** |
| PyTorch multi-track | 1.1 laps | 4.2 laps | 0.0 | 6.3 laps | 2,098 |
| TensorFlow multi-track | 0.0 | 6.0 laps | 0.0 | 6.0 laps | 2,049 |

- **Sample efficiency is identical.** Every single-track run in both frameworks first reached 9 laps at the same evaluation point: 40,960 steps.
- **Final performance is statistically indistinguishable.** The differences between frameworks are smaller than the differences between seeds of the same framework. For example, PyTorch multi-track seeds score 0, 4.6 and 8 laps on the Tall Oval.
- **Best single result in the project:** `torch_multi_seed0` completes **10 laps on the unseen track** (fitness 66,446). The TensorFlow seeds 1 and 2 and the PyTorch seed 2 complete 9 laps there, against 8 for the NE multi-track champion.
- **Seed sensitivity is high for multi-track training.** In each framework, one seed of three produced a policy that crashes on most tracks (`torch_multi_seed1`, `tf_multi_seed0`).

## 3. Speed

### Full training runs (wall clock)

| Configuration | PyTorch | TensorFlow |
|---|---:|---:|
| Single-track, 1M steps | 511 s | 578 s (+13%) |
| Multi-track, 1M steps | 631 s | 753 s (+19%) |

The runs were executed six at a time on a 6-core laptop, three per framework. The multi-track times vary a lot between seeds (TensorFlow: 566 s, 849 s and 845 s), so they reflect CPU contention as much as the framework.

### Network operations only

`python -m rl.benchmark_frameworks`, alternating frameworks, median of 5 rounds:

| Operation | PyTorch | TensorFlow | TF / PT |
|---|---:|---:|---:|
| `act` (8 observations) | 630 µs | 799 µs | 1.27× |
| `value` (1 observation) | 102 µs | 667 µs | 6.6× |
| `predict` (1 observation) | 95 µs | 637 µs | 6.7× |
| `train_step` (256 samples) | 4,236 µs | 2,729 µs | **0.64×** |

Absolute values depend on CPU clock (this laptop was running below its maximum frequency). The ratios are the meaningful part.

**Interpretation**

- **TensorFlow optimizes faster.** The whole PPO update (forward pass, loss, gradient tape, clipping, Adam) is compiled into one graph by `tf.function`. PyTorch executes the same steps eagerly, operation by operation.
- **TensorFlow has a much higher per-call overhead.** Every call into a `tf.function` costs about 0.5 ms regardless of the batch size. For a 2 × 64 network on a single observation, that overhead is almost the entire cost.
- **Evaluation explains the slower TensorFlow training.** Periodic evaluation calls `predict` for one observation at a time: 49 evaluations × 5,400 steps, about 265,000 calls in single-track training. The measured 68 s difference corresponds to about 0.26 ms of extra cost per call. That is the same order of magnitude as the per-call overhead measured above (0.54 ms on the slowed-down CPU), so evaluation, not learning, accounts for the gap.

In short: PyTorch is faster for many small calls, TensorFlow for larger compiled computations. Evaluating several episodes as one batch would remove most of TensorFlow's disadvantage.

## Conclusion

Implemented with care, the same algorithm learns the same way in both frameworks: same sample efficiency, statistically identical results, gradients equal to float32 precision. What differs is the engineering: optimizer conventions that must be aligned explicitly, and very different performance profiles for small eager calls versus compiled graphs.
