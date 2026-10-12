Each cell: mean laps, crash rate and fitness over 10 episodes per track (evaluation seeds 10000-10009, deterministic actions). Worst-track fitness is taken over the three training tracks.

| Model | Classic Oval | Tall Oval | Asymmetric Oval | Unseen Narrow Oval | Worst-track fitness |
|---|---:|---:|---:|---:|---:|
| best_brain_continuous.npz | 10.0 laps, 0% crash, 47,982 | 0.0 laps, 0% crash, 252 | 0.0 laps, 0% crash, 267 | 0.0 laps, 0% crash, 1,198 | 252 |
| best_brain_multitrack.npz | 0.0 laps, 0% crash, 3,420 | 9.0 laps, 0% crash, 42,130 | 0.0 laps, 0% crash, 3,814 | 8.0 laps, 0% crash, 56,734 | 3,420 |
| torch_single (mean of 3 seeds) | 9.9 laps, 0% crash, 46,895 | 0.0 laps, 0% crash, 257 | 0.0 laps, 90% crash, -17 | 0.0 laps, 0% crash, 1,195 | -17 |
|   torch_single_seed0 | 10.0 laps, 0% crash, 47,243 | 0.0 laps, 0% crash, 260 | 0.0 laps, 100% crash, -44 | 0.0 laps, 0% crash, 1,103 | -44 |
|   torch_single_seed1 | 9.9 laps, 0% crash, 46,789 | 0.0 laps, 0% crash, 251 | 0.0 laps, 70% crash, 40 | 0.0 laps, 0% crash, 1,243 | 40 |
|   torch_single_seed2 | 9.9 laps, 0% crash, 46,653 | 0.0 laps, 0% crash, 260 | 0.0 laps, 100% crash, -47 | 0.0 laps, 0% crash, 1,241 | -47 |
| tf_single (mean of 3 seeds) | 9.9 laps, 0% crash, 46,570 | 0.0 laps, 0% crash, 154 | 0.0 laps, 33% crash, 123 | 0.0 laps, 0% crash, 1,187 | 123 |
|   tf_single_seed0 | 10.0 laps, 0% crash, 46,164 | 0.0 laps, 0% crash, 119 | 0.0 laps, 0% crash, 155 | 0.0 laps, 0% crash, 1,089 | 119 |
|   tf_single_seed1 | 10.0 laps, 0% crash, 47,235 | 0.0 laps, 0% crash, 140 | 0.0 laps, 0% crash, 272 | 0.0 laps, 0% crash, 1,237 | 140 |
|   tf_single_seed2 | 9.8 laps, 0% crash, 46,311 | 0.0 laps, 0% crash, 204 | 0.0 laps, 100% crash, -57 | 0.0 laps, 0% crash, 1,235 | -57 |
| torch_multi (mean of 3 seeds) | 1.1 laps, 0% crash, 7,423 | 4.2 laps, 0% crash, 20,187 | 0.0 laps, 33% crash, 2,098 | 6.3 laps, 33% crash, 44,644 | 2,098 |
|   torch_multi_seed0 | 0.0 laps, 0% crash, 3,517 | 4.6 laps, 0% crash, 21,718 | 0.0 laps, 0% crash, 2,531 | 10.0 laps, 0% crash, 66,446 | 2,531 |
|   torch_multi_seed1 | 3.3 laps, 0% crash, 17,048 | 0.0 laps, 0% crash, 1,392 | 0.0 laps, 100% crash, 1,233 | 0.0 laps, 100% crash, 3,046 | 1,233 |
|   torch_multi_seed2 | 0.0 laps, 0% crash, 1,705 | 8.0 laps, 0% crash, 37,452 | 0.0 laps, 0% crash, 2,531 | 9.0 laps, 0% crash, 64,439 | 1,705 |
| tf_multi (mean of 3 seeds) | 0.0 laps, 33% crash, 2,812 | 6.0 laps, 33% crash, 28,529 | 0.0 laps, 33% crash, 2,049 | 6.0 laps, 33% crash, 43,615 | 2,049 |
|   tf_multi_seed0 | 0.0 laps, 100% crash, 2,212 | 0.0 laps, 100% crash, 1,204 | 0.0 laps, 100% crash, 1,175 | 0.0 laps, 100% crash, 1,015 | 1,175 |
|   tf_multi_seed1 | 0.0 laps, 0% crash, 3,051 | 9.0 laps, 0% crash, 42,180 | 0.0 laps, 0% crash, 2,491 | 9.0 laps, 0% crash, 64,397 | 2,491 |
|   tf_multi_seed2 | 0.0 laps, 0% crash, 3,173 | 9.0 laps, 0% crash, 42,202 | 0.0 laps, 0% crash, 2,482 | 9.0 laps, 0% crash, 65,432 | 2,482 |
