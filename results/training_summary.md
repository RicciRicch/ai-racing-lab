# Training Summary

## Overview

AI Racing Lab was developed incrementally from a simple rule-based racing controller into a continuous-control neuroevolution system.

The project was used to investigate:

- neural-network control
- neuroevolution
- continuous steering and throttle
- fitness-function design
- overfitting
- multi-environment training
- generalization
- robustness
- exploration vs exploitation

---

## Single-Track Training

The continuous-control model was first trained on a single racing environment.

The model receives:

- five raycast sensor distances
- normalized vehicle speed

The neural network outputs:

- steering in the range `[-1, 1]`
- throttle in the range `[0, 1]`

The best single-track model achieved strong performance on the original training environment.

### Evaluation Results

- Average laps: **10.00**
- Best laps: **10**
- Worst laps: **10**
- Average checkpoints: **41.80**
- Average normalized speed: **0.985**
- Crash rate: **0.0%**

The policy was therefore highly stable on the environment on which it was trained.

---

## Unseen-Track Evaluation

The same single-track model was evaluated on a previously unseen track.

### Results

- Average laps: **0.00**
- Average checkpoints: **0.20**
- Average frames survived: **360.80**
- Average normalized speed: **0.211**
- Crash rate: **80.0%**

This revealed strong overfitting to the original training environment.

Although the model performed extremely well on the training track, it failed to generalize reliably to substantially different track geometry.

---

## Multi-Track Training

To reduce overfitting, the training process was extended to three environments:

- Classic Oval
- Tall Oval
- Asymmetric Oval

The same neural policy is evaluated on all three tracks during every generation.

---

## Average-Fitness Problem

The first multi-track objective used the average score across all tracks.

This produced an unintended behavior.

A candidate could achieve a high fitness by performing extremely well on one or two tracks while almost completely failing on another.

For example:

```text
Track 1:    150
Track 2: 43,000
Track 3: 39,000
```

Despite failing on Track 1, this candidate still achieved a high average fitness.

This encouraged specialization rather than balanced generalization.

---

## Maximin Fitness

The objective was therefore changed to a maximin strategy:

```text
fitness = minimum(track_scores)
```

The weakest track now determines the candidate's fitness.

Candidate ranking also prioritizes:

```text
1. highest worst-track score
2. highest mean track score as a tie-breaker
```

This encourages the evolutionary algorithm to improve the weakest environment rather than maximizing performance only where the policy is already strong.

---

## Maximin Training Result

During the maximin experiment, worst-track fitness improved from approximately:

```text
2,575 -> 3,546
```

over 35 generations.

The current champion achieved approximately:

- Track 1: **3,546**
- Track 2: **42,130**
- Track 3: **3,816**
- Worst-track fitness: **3,546**
- Crashes: **0 / 3**

This represents measurable progress toward balanced behavior, although substantial performance imbalance remains.

---

## Multi-Track Model Evaluation

The best multi-track model was evaluated across all three training environments.

### Results

- Total episodes: **30**
- Average laps per episode: **3.07**
- Average checkpoints per episode: **14.23**
- Average normalized speed: **0.985**
- Average throttle: **0.996**
- Average absolute steering: **0.574**
- Overall crash rate: **0.0%**

### Per-Track Results

| Track | Average Laps | Crash Rate |
|---|---:|---:|
| Classic Oval | 0.00 | 0.0% |
| Tall Oval | 9.00 | 0.0% |
| Asymmetric Oval | 0.20 | 0.0% |

The model is stable across all three environments, but it remains strongly specialized toward Tall Oval.

---

## Single-Track vs Multi-Track Generalization

Both models were evaluated on the same unseen environment.

| Metric | Single-Track Model | Multi-Track Model |
|---|---:|---:|
| Average laps | 0.00 | 0.00 |
| Average checkpoints | 0.20 | 0.30 |
| Average frames survived | 360.80 | 540.70 |
| Average normalized speed | 0.211 | 0.308 |
| Crash rate | 80.0% | 70.0% |

The multi-track model showed modest improvement in robustness.

Compared with the single-track model, it:

- survived longer
- reached more checkpoints
- maintained a higher average speed
- reduced the crash rate

However, neither model completed a full lap on the unseen track.

This suggests that multi-track neuroevolution improved robustness, but did not yet produce strong generalization to substantially different track geometry.

---

## Current Limitations

The current system still has several important limitations.

The strongest limitation is that the neural network can still specialize heavily toward particular track geometries.

The simulation also currently uses pixel-based raycasting and collision detection, which couples the environment logic to Pygame rendering.

This makes future headless training, faster simulation, and more complex environments harder to implement.

---

## Future Work

Planned improvements include:

- specialist preservation
- generalist-specialist crossover
- diversity-aware evolutionary selection
- procedural track generation
- randomized track geometry
- stronger domain randomization
- evaluation on larger unseen-track sets
- separating physics from rendering
- mathematical collision detection
- headless simulation
- faster-than-real-time training
- Gymnasium-compatible environment
- reinforcement-learning experiments
- comparison with algorithms such as PPO or DQN
