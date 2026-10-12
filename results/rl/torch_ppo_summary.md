# PPO in PyTorch: Experiment Summary

Phase 2 of the reinforcement-learning extension. PPO agents were trained in `AIRacing-v0` and compared with the existing neuroevolution (NE) champions using the same evaluator (`rl/evaluate.py`), the same tracks and the same evaluation seeds.

Full per-seed table: [`torch_ppo_evaluation.md`](torch_ppo_evaluation.md). Learning curves: [`torch_learning_curves.png`](torch_learning_curves.png).

## Setup

| Setting | Value |
|---|---|
| Algorithm | PPO, clipped objective, GAE (λ = 0.95), γ = 0.99 |
| Networks | Separate actor and critic, 2 × 64 tanh, Gaussian policy with learned log std |
| Rollouts | 8 environments × 256 steps = 2,048 steps per update |
| Optimization | 10 epochs, 8 minibatches, Adam 3e-4 with linear annealing, gradient clip 0.5 |
| Budget | 1,000,000 environment steps per run |
| Seeds | 3 per configuration (0, 1, 2) |
| Model selection | `best.pt`: highest worst-track evaluation fitness during training (maximin, as in NE) |
| Hardware | Laptop CPU, one thread per run, three to six runs in parallel |

Reward is the per-step change of the NE fitness, so PPO and NE optimize the same objective.

| Configuration | Training tracks | Track sampling |
|---|---|---|
| `torch_single` | Classic Oval | — |
| `torch_multi` | Classic, Tall, Asymmetric | uniform |
| `torch_multiweak` | Classic, Tall, Asymmetric | weakest track with probability 0.5 |

## Results

Mean over 3 seeds, 10 evaluation episodes per track, deterministic actions:

| Model | Classic Oval | Tall Oval | Asymmetric Oval | Unseen Narrow Oval | Worst training-track fitness |
|---|---:|---:|---:|---:|---:|
| NE single-track | 10.0 laps | 0.0 laps | 0.0 laps | 0.0 laps | 252 |
| PPO single-track | 10.0 laps | 0.0 laps | 0.0 laps (67% crash) | 0.0 laps | 24 |
| NE multi-track | 0.0 laps | 9.0 laps | 0.0 laps | 8.0 laps | **3,420** |
| PPO multi-track (uniform) | 0.0 laps | 8.7 laps | 0.0 laps | 7.1 laps | 2,548 |
| PPO multi-track (weakest) | 2.7 laps | 7.7 laps | 0.0 laps | 5.7 laps | 2,588 |

Selected single seeds:

- `torch_multi_seed0` and `seed1`: **9 laps on the unseen track** (fitness 63,317 and 65,407), more than the NE multi-track champion (8 laps, 56,734).
- `torch_multiweak_seed0`: 7.3 laps on Classic **and** 8 laps on Tall, the first model in the project to lap two different training tracks.

## Findings

1. **Single-track PPO matches neuroevolution.** All three seeds complete 10 laps on the Classic Oval without crashing, like the NE champion, and fail on every other track in the same way.

2. **PPO learns from far fewer simulation steps.** PPO reached 9 laps after about 41,000 steps (about 10 seconds of training). The NE champion first reached 9 laps in generation 3, at most 216,000 simulated car-frames (40 cars × 1,800 frames per generation; an upper bound, since crashed cars stop early). The multi-track NE champion used up to 7.6 million car-frames over 35 generations, on top of the single-track model it was seeded from. PPO multi-track runs used 1 million.

3. **Multi-track PPO specializes, just like average-fitness NE.** With uniform track sampling, PPO maximizes the *average* return. Each seed masters one track and drives safe but checkpoint-missing circles on the others. This is the "average-fitness problem" that multi-track NE solved with maximin fitness.

4. **Weakest-track sampling helps the weak track, not the worst one.** Sampling the weakest track more often produced the first model that laps two training tracks. However, it did not raise the worst-track fitness and made unseen-track results more variable. On these tracks, a different track was always the real bottleneck.

5. **The Asymmetric Oval is a shared bottleneck.** No model, NE or PPO, completes a lap there. Its left checkpoint (`x = 140..160`) sits near the outer edge of a 270 px wide lane whose centre is at `x = 215`. Fast policies take the inner line and miss it. Every multi-track method's worst-track fitness is therefore limited to about 2,500–3,800. NE's somewhat higher value comes from lap-less circles that collect more progress fitness, not from laps. This is an exploration problem in the track design, not a weakness of one method.

6. **Training speed.** On a laptop CPU, a 1M-step run took 6.5–8.5 minutes (about 2,000–2,500 steps/s per run, with several runs in parallel).

## Next

- Phase 3: the same PPO agent in TensorFlow, trained with identical settings and seeds.
- Phase 4: a common report covering all three methods, with properly measured NE sample counts.
- Possible experiment: full-width checkpoint gates, so that lap counts no longer depend on one specific racing line.
