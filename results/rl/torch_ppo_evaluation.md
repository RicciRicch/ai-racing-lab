Each cell: mean laps, crash rate and fitness over 10 episodes per track (evaluation seeds 10000-10009, deterministic actions). Worst-track fitness is taken over the three training tracks.

| Model | Classic Oval | Tall Oval | Asymmetric Oval | Unseen Narrow Oval | Worst-track fitness |
|---|---:|---:|---:|---:|---:|
| best_brain_continuous.npz | 10.0 laps, 0% crash, 47,982 | 0.0 laps, 0% crash, 252 | 0.0 laps, 0% crash, 267 | 0.0 laps, 0% crash, 1,198 | 252 |
| best_brain_multitrack.npz | 0.0 laps, 0% crash, 3,420 | 9.0 laps, 0% crash, 42,130 | 0.0 laps, 0% crash, 3,814 | 8.0 laps, 0% crash, 56,734 | 3,420 |
| torch_single (mean of 3 seeds) | 10.0 laps, 0% crash, 46,878 | 0.0 laps, 0% crash, 197 | 0.0 laps, 67% crash, 24 | 0.0 laps, 0% crash, 1,198 | 24 |
|   torch_single_seed0 | 10.0 laps, 0% crash, 47,262 | 0.0 laps, 0% crash, 255 | 0.0 laps, 100% crash, -51 | 0.0 laps, 0% crash, 1,224 | -51 |
|   torch_single_seed1 | 10.0 laps, 0% crash, 46,112 | 0.0 laps, 0% crash, 84 | 0.0 laps, 0% crash, 181 | 0.0 laps, 0% crash, 1,153 | 84 |
|   torch_single_seed2 | 10.0 laps, 0% crash, 47,259 | 0.0 laps, 0% crash, 252 | 0.0 laps, 100% crash, -58 | 0.0 laps, 0% crash, 1,218 | -58 |
| torch_multi (mean of 3 seeds) | 0.0 laps, 0% crash, 3,029 | 8.7 laps, 0% crash, 40,736 | 0.0 laps, 0% crash, 2,548 | 7.1 laps, 0% crash, 51,360 | 2,548 |
|   torch_multi_seed0 | 0.0 laps, 0% crash, 3,519 | 9.0 laps, 0% crash, 42,164 | 0.0 laps, 0% crash, 2,556 | 9.0 laps, 0% crash, 63,317 | 2,556 |
|   torch_multi_seed1 | 0.0 laps, 0% crash, 3,170 | 8.0 laps, 0% crash, 37,521 | 0.0 laps, 0% crash, 2,493 | 9.0 laps, 0% crash, 65,407 | 2,493 |
|   torch_multi_seed2 | 0.0 laps, 0% crash, 2,398 | 9.0 laps, 0% crash, 42,524 | 0.0 laps, 0% crash, 2,595 | 3.2 laps, 0% crash, 25,355 | 2,398 |
| torch_multiweak (mean of 3 seeds) | 2.7 laps, 0% crash, 14,751 | 7.7 laps, 0% crash, 37,465 | 0.0 laps, 0% crash, 2,588 | 5.7 laps, 0% crash, 40,718 | 2,588 |
|   torch_multiweak_seed0 | 7.3 laps, 0% crash, 35,296 | 8.0 laps, 0% crash, 37,506 | 0.0 laps, 0% crash, 2,580 | 4.2 laps, 0% crash, 32,148 | 2,580 |
|   torch_multiweak_seed1 | 0.0 laps, 0% crash, 3,651 | 7.0 laps, 0% crash, 36,310 | 0.0 laps, 0% crash, 2,620 | 8.9 laps, 0% crash, 59,407 | 2,620 |
|   torch_multiweak_seed2 | 0.7 laps, 0% crash, 5,306 | 8.0 laps, 0% crash, 38,578 | 0.0 laps, 0% crash, 2,565 | 3.9 laps, 0% crash, 30,598 | 2,565 |
