# 🏎️ AI Racing Lab

AI Racing Lab is a 2D autonomous driving simulation built with **Python, Pygame, and NumPy**. A population of neural-network-controlled cars learns to navigate a racing track through **neuroevolution**: fitness evaluation, selection, crossover, mutation, and elitism.

The current policy uses **continuous steering and throttle control**, learning both how to turn and how much to accelerate from five distance sensors and the car's current speed.

The project began with a rule-based controller and has grown into an experimental environment for studying autonomous agents, evolutionary optimization, and driving performance.

## Contents

- [Features](#features)
- [Getting started](#getting-started)
- [Controls and operating modes](#controls-and-operating-modes)
- [Observations and neural network](#observations-and-neural-network)
- [Vehicle and track](#vehicle-and-track)
- [Neuroevolution](#neuroevolution)
- [Fitness and progress](#fitness-and-progress)
- [Adaptive mutation](#adaptive-mutation)
- [Model persistence](#model-persistence)
- [Training history and plotting](#training-history-and-plotting)
- [Project structure](#project-structure)
- [Configuration](#configuration)
- [Development findings](#development-findings)
- [Roadmap](#roadmap)

## Features

- Custom Pygame racing environment with real-time visualization
- Basic vehicle physics with acceleration, friction, rotation, and a speed limit
- Track-boundary collision detection
- Five ray-casting distance sensors
- A small feed-forward neural network implemented directly in NumPy
- Continuous steering and neural-network-controlled throttle
- Population-based training with fitness-based parent selection
- Elitism, crossover, multiple mutation strengths, and random agents
- Stagnation detection and adaptive mutation
- Ordered checkpoints, lap counting, and continuous progress rewards
- Separate **TRAIN** and **DEMO** modes
- Automatic saving and loading of the best model
- Lap timing in DEMO mode
- Per-generation CSV history and Matplotlib training plots

## Getting started

### Requirements

- Python 3 and `pip`
- A desktop environment capable of displaying a Pygame window
- Pygame, NumPy, and Matplotlib

Download or clone this repository, then open a terminal in the `ai-racing-lab` directory containing `main.py`.

Create a virtual environment:

```sh
python -m venv .venv
```

Activate it on **Windows PowerShell**:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on **macOS / Linux**:

```sh
source .venv/bin/activate
```

Install the dependencies:

```sh
python -m pip install pygame numpy matplotlib
```

Run the simulation:

```sh
python main.py
```

The application starts in TRAIN mode. Let at least one generation finish so that generation statistics and a model can be saved. Press `TAB` to watch the saved model in DEMO mode.

To visualize recorded training history:

```sh
python plot_training.py
```

Run these commands from the project directory so relative model and history paths resolve consistently.

## Controls and operating modes

| Control | Action |
|---|---|
| `TAB` | Switch between TRAIN and DEMO |
| `R` | Restart the current mode |
| Window close button | Exit the application |

### TRAIN

TRAIN mode evaluates a population of cars on the same track. Each car has its own neural network. At the end of a generation, the simulation ranks the population and creates the next generation.

Training ends a generation when all cars have crashed or the generation frame budget is exhausted. The default budget is **1,800 frames**.

The interface displays live driving and training information, including generation, living agents, fitness, checkpoints, laps, speed, steering, throttle, mutation mode, and stagnation count. The leading agent is highlighted in yellow.

Restarting TRAIN mode creates a fresh random population. Archive any model and history you want to preserve before running a separate experiment; restarting the population is not the same as managing experiment files.

### DEMO

DEMO mode loads `best_brain_continuous.npz` and runs one autonomous car. It performs inference only: observations pass through the saved network to produce steering and throttle, with no selection or mutation.

DEMO mode tracks lap times. A saved continuous-control model must exist before DEMO can start. Use `R` to restart the demo car.

## Observations and neural network

The policy receives **local observations**, not absolute coordinates, a track map, or the location of the next checkpoint.

Five rays measure the distance to a track boundary:

```text
left_far    left    front    right    right_far
     \        \      |      /        /
      \        \     |     /        /
                    CAR
```

The observation vector contains normalized sensor distances and normalized speed:

```text
[left_far, left, front, right, right_far, speed]
```

### Architecture

```text
5 distance sensors + current speed
                |
            6 inputs
                |
      8 hidden neurons, tanh
                |
       2 outputs, tanh
                |
       steering + throttle
```

The network contains **74 trainable parameters**:

| Parameter | Shape | Count |
|---|---|---:|
| Input-to-hidden weights | `6 × 8` | 48 |
| Hidden biases | `8` | 8 |
| Hidden-to-output weights | `8 × 2` | 16 |
| Output biases | `2` | 2 |

For an observation vector `x`, the forward pass is:

```text
hidden = tanh(x @ W1 + b1)
output = tanh(hidden @ W2 + b2)

steering = output[0]
throttle = (output[1] + 1) / 2
```

| Output | Range | Meaning |
|---|---|---|
| Steering | `[-1, 1]` | Negative turns right; positive turns left; zero keeps heading |
| Throttle | `[0, 1]` | Zero applies no throttle; one applies maximum throttle |

Reduced throttle allows friction to slow the vehicle. Throttle is not a separate brake or reverse command.

The network is intentionally small enough to inspect and evolve without a large machine learning framework. It is trained through evolutionary search, **not backpropagation**.

## Vehicle and track

The environment uses a simple oval track with inner and outer grass boundaries. Cars accelerate, rotate, and move according to their current controls. Sensors and collision checks sample the rendered track.

Checkpoints must be visited in order. With the current starting position at the bottom of the track, facing right, the sequence is:

```text
right → top → left → bottom → repeat
```

Completing the checkpoint sequence increments the lap count. Checkpoint tracking supplies evaluation information to the training system; checkpoint coordinates are not neural-network inputs.

This is a simplified driving simulation, with no detailed tire, suspension, or aerodynamic model.

## Neuroevolution

Each generation follows this cycle:

1. Run the population using each car's neural network.
2. Measure driving progress and calculate fitness.
3. Rank agents by fitness.
4. Save a new best model when the improvement condition is met.
5. Copy elite networks unchanged into the next generation.
6. Select high-fitness parents and combine their parameters through crossover.
7. Mutate offspring using a mixture of mutation strengths.
8. Introduce random agents to maintain diversity.
9. Reset vehicle states and evaluate the new population.

```text
Population → Driving → Fitness → Selection
    ↑                               |
    └── Elites + Mutated offspring + Random agents
```

Elitism preserves successful network parameters while mutation and random agents search for different strategies. A new car starts each evaluation from the initial vehicle state, even when it inherits an elite network.

## Fitness and progress

The fitness objective rewards progress around the track. Its components include checkpoint progress, continuous progress toward the next checkpoint, speed, and a small survival contribution.

- **Checkpoints:** provide a strong signal for completing the route in order.
- **Continuous progress:** measures changes in distance to the next checkpoint, providing feedback between checkpoint crossings.
- **Speed:** encourages useful forward motion alongside route completion.
- **Survival:** contributes a small reward for remaining active.
- **Collisions:** stop the car's evaluation and prevent further progress and reward accumulation.

The scoring implementation in `main.py` is the source of truth for numerical weights and any explicit penalties. Fitness is an optimization score, not a lap time or a universal measure of driving quality.

## Adaptive mutation

The system tracks generations without a meaningful improvement in best fitness. Longer stagnation increases exploration through stronger mutations and more random agents.

| Mode | Purpose |
|---|---|
| `FINE` | Refine successful policies with smaller changes |
| `EXPLORE` | Broaden the search after sustained stagnation |
| `ESCAPE` | Apply stronger changes after longer stagnation |

The current stagnation thresholds are **8** and **18** generations, with `MIN_IMPROVEMENT = 1.0` defining the improvement threshold. Elites remain protected as exploration increases.

A falling population average does not necessarily mean the best policy is getting worse: stronger mutations can produce many weak candidates while elites retain strong behavior. Inspect best fitness, average fitness, laps, and mutation mode together.

Adaptive mutation does not guarantee escape from a plateau. The action space, fitness function, track geometry, speed limit, and frame budget can also constrain performance.

## Model persistence

The continuous-control model is saved to:

```text
best_brain_continuous.npz
```

The NumPy archive stores both layers' weights and biases. A saved model can be loaded for DEMO inference or used to seed further evolution when the application starts.

This preserves a policy, not a complete training-session checkpoint: network parameters alone do not restore the full population, generation counters, or random-number state.

### Compatibility with the earlier controller

The earlier discrete policy selected `LEFT`, `STRAIGHT`, or `RIGHT` and used automatic throttle. Its output layer had three neurons and its model file was `best_brain.npz`.

The current policy has two outputs. **The old and new model architectures are incompatible.** Keep the separate filenames; renaming an old archive does not convert it to continuous control.

## Training history and plotting

Each completed generation records statistics in:

```text
training_history_continuous.csv
```

The recorded metrics include:

| Metric | What it describes |
|---|---|
| Generation | Generation index |
| Best fitness | Fitness of the generation's champion |
| Average fitness | Mean fitness across the population |
| Best checkpoints | Champion's checkpoint count |
| Best laps | Champion's completed laps |
| Frames alive | Champion's active frame count |
| Champion average speed | Speed summary for the champion |
| Stagnation | Generations without meaningful improvement |
| Mutation mode | Current adaptive evolution mode |

`plot_training.py` uses Matplotlib to visualize best and average fitness, best laps, and champion average speed across generations.

### Reading the plots

- Rising best fitness suggests the search is finding better-scoring policies.
- Rising average fitness suggests improvement across more of the population.
- Stable best fitness with a fluctuating average can reflect exploration around a retained champion.
- A stable lap count may indicate a limit imposed by the track and evaluation budget, or a policy that has stopped improving.

Keep separate history files for separate experiments. If the plotting script still targets the earlier `training_history.csv`, update its input to the continuous-control history and use the matching CSV columns.

For a fair comparison, keep track layout, frame budget, physics, and scoring consistent. Do not compare raw fitness values from different scoring functions as if they were equivalent.

## Project Structure

```text
ai-racing-lab/
│
├── brain.py
│   Neural-network policy, mutation, crossover,
│   model saving and loading.
│
├── car.py
│   Vehicle physics, neural control, raycast sensors,
│   collision detection, checkpoints and lap tracking.
│
├── main.py
│   Multi-track neuroevolution training loop,
│   adaptive mutation, selection, elitism and demo mode.
│
├── tracks.py
│   Definitions of training tracks, start positions
│   and checkpoint layouts.
│
├── evaluate_model.py
│   Evaluates the single-track continuous model
│   on the original training environment.
│
├── evaluate_multitrack_model.py
│   Evaluates the multi-track model across all
│   training environments.
│
├── evaluate_unseen_track.py
│   Evaluates a trained model on a previously
│   unseen track to measure generalization.
│
├── plot_training.py
│   Visualizes training metrics with Matplotlib.
│
├── results/
│   Selected experiment summaries and evaluation results.
│
│   ├── single_track_evaluation.txt
│   ├── unseen_track_single_model.txt
│   ├── multitrack_evaluation.txt
│   ├── unseen_track_multitrack_model.txt
│   └── training_summary.md
│
├── requirements.txt
│   Python dependencies.
│
├── .gitignore
│   Excludes virtual environments, caches,
│   trained model files and generated histories.
│
└── README.md
    Project documentation and experimental results.
```

Model archives (`*.npz`) and generated training histories (`training_history*.csv`) are ignored by Git. The five summaries in `results/` are kept under version control.

## Configuration

The current main simulation settings are defined in `main.py`:

| Setting | Default | Purpose |
|---|---:|---|
| `WIDTH`, `HEIGHT` | `1000`, `700` | Window dimensions |
| `FPS` | `60` | Target display/update rate |
| `POPULATION_SIZE` | `40` | Cars evaluated per generation |
| `PARENT_COUNT` | `5` | Top agents used as parents |
| `ELITE_COUNT` | `2` | Networks copied unchanged |
| `MAX_FRAMES_PER_GENERATION` | `1800` | Maximum generation length |
| `STAGNATION_LEVEL_1` | `8` | First exploration threshold |
| `STAGNATION_LEVEL_2` | `18` | Stronger exploration threshold |
| `MIN_IMPROVEMENT` | `1.0` | Minimum meaningful fitness improvement threshold |

Vehicle parameters live in `car.py`; network dimensions and operations live in `brain.py`. Changing network dimensions requires a compatible saved model or a new training run.

## Development findings

The project has evolved through the following stages:

```text
Rule-based driving
  → Discrete neural control
  → Population training and neuroevolution
  → Model persistence and DEMO mode
  → CSV metrics and plots
  → Stagnation detection and adaptive mutation
  → Continuous steering and throttle
```

The earlier policy could choose steering direction but could not directly adjust throttle. Its observed plateau motivated expanding the action space so the agent could learn trajectory and acceleration together.

Development plots discussed after the continuous-control update showed rapid progress to stable multi-lap driving, with best laps reaching approximately **10** and best fitness approaching **48,000** in that run. These are informal observations from a development run, not a repeated benchmark or proof of generalization.

A controlled comparison with the discrete policy remains planned. Evaluation should report lap times, crashes, speed, and completed laps under the same conditions, rather than relying on training fitness alone.

## Evaluation Results

The best continuous-control model was evaluated over 10 episodes with small random perturbations to the starting position and heading.

| Metric | Result |
|---|---:|
| Episodes | 10 |
| Average laps | 10.00 |
| Best laps | 10 |
| Worst laps | 10 |
| Average checkpoints | 41.30 |
| Average frames | 1800.00 |
| Average normalized speed | 0.985 |
| Crash rate | 0.0% |

The model completed all evaluation episodes without crashing while maintaining nearly maximum average speed.

This indicates that the learned policy is highly stable on the training track and robust to small perturbations in the initial state.

## Generalization Experiment

The continuous-control model achieved perfect stability on the training track:

- 10.0 average laps
- 0% crash rate
- 0.985 normalized average speed

However, evaluation on an unseen track revealed poor generalization:

- 0.0 average laps
- 0.5 average checkpoints
- 50% crash rate
- 0.502 normalized average speed

This indicates that the learned policy strongly overfit to the geometry of the training environment.

The next stage of the project focuses on multi-track training and domain randomization to improve generalization.

## Single-Track Model Evaluation

The continuous-control model trained on the original track was evaluated over 10 episodes with small perturbations to the starting position and heading.

| Metric | Result |
|---|---:|
| Episodes | 10 |
| Average laps | 10.00 |
| Best laps | 10 |
| Worst laps | 10 |
| Average checkpoints | 41.80 |
| Average frames | 1800.00 |
| Average normalized speed | 0.985 |
| Crash count | 0 / 10 |
| Crash rate | 0.0% |

The model completed all evaluation episodes without crashing while maintaining nearly maximum average speed.

This confirms that the learned policy is highly stable on the original training track and robust to small perturbations in the initial state.

## Unseen-Track Evaluation

The single-track continuous-control model was also evaluated on a different, previously unseen track.

Results over 10 episodes:

| Metric | Result |
|---|---:|
| Episodes | 10 |
| Average laps | 0.00 |
| Best laps | 0 |
| Worst laps | 0 |
| Average checkpoints | 0.20 |
| Average frames | 360.80 |
| Average normalized speed | 0.211 |
| Crash count | 8 / 10 |
| Crash rate | 80.0% |

The model failed to complete a lap on the unseen track and crashed in most episodes.

This shows that the policy learned on a single track does not generalize reliably to significantly different track geometry.

Some episodes also terminated almost immediately, indicating that evaluation sensitivity to the unseen track's starting position and local geometry should be investigated further.

## Multi-Track Generalization

After observing strong overfitting to the original training track, the project was extended to multi-track neuroevolution.

The same neural policy is now evaluated on three different tracks during every generation.

Initially, fitness was defined as the average score across tracks. This caused specialization: agents could achieve high average fitness by performing extremely well on two tracks while almost completely failing on the third.

The objective was therefore changed to a maximin strategy:

```text
fitness = minimum(track_scores)

This makes the weakest track determine the candidate's fitness and encourages balanced behavior across environments.

In the current experiment, worst-track performance improved from approximately **2,575** to **3,546** over **35 generations**.

The current champion achieved approximately:

- **Track 1:** 3,546
- **Track 2:** 42,130
- **Track 3:** 3,816
- **Worst-track fitness:** 3,546
- **Crashes:** 0 / 3

The results show measurable progress toward generalization, although the policy still exhibits substantial performance imbalance across tracks.

## Single-Track vs Multi-Track Generalization

Both models were evaluated on the same unseen track.

| Metric | Single-Track Model | Multi-Track Model |
|---|---:|---:|
| Average laps | 0.00 | 0.00 |
| Average checkpoints | 0.20 | 0.30 |
| Average frames survived | 360.80 | 540.70 |
| Average normalized speed | 0.211 | 0.308 |
| Crash rate | 80.0% | 70.0% |

The multi-track model showed modest improvements in robustness on the unseen environment.

Compared with the single-track model, it survived longer, reached more checkpoints, maintained a higher average speed, and reduced the crash rate from 80% to 70%.

However, neither model completed a full lap on the unseen track.

This suggests that multi-track neuroevolution improved robustness, but did not yet produce strong generalization to substantially different track geometry.

### Future Work

Future work includes:

- specialist preservation
- improved crossover strategies
- randomized track generation
- evaluation on completely unseen environments

## Roadmap

### Implemented

- [x] Custom 2D track and vehicle movement
- [x] Collision detection and ray-casting sensors
- [x] Rule-based driving prototype
- [x] NumPy feed-forward neural network
- [x] Population evaluation and fitness scoring
- [x] Checkpoints, laps, and continuous progress tracking
- [x] Selection, elitism, crossover, and mutation
- [x] Random agents and multiple mutation strengths
- [x] Adaptive mutation and stagnation detection
- [x] Continuous steering and throttle outputs
- [x] Best-model saving and loading
- [x] TRAIN and DEMO modes
- [x] DEMO lap timing
- [x] CSV generation history and training plots

### Next: dedicated evaluation

- [ ] Create a separate model evaluation script
- [ ] Evaluate saved champions over repeated episodes
- [ ] Report average and best lap times
- [ ] Report crash rate and checkpoints reached before crashing
- [ ] Report average throttle and absolute steering magnitude
- [ ] Report average speed and laps within a fixed frame budget
- [ ] Compare discrete and continuous policies under matched conditions
- [ ] Record experiment settings, seeds, and results together

### Environment and training improvements

- [ ] Add additional track layouts
- [ ] Test generalization on unseen tracks and different starting conditions
- [ ] Separate simulation logic from rendering
- [ ] Support headless and accelerated training
- [ ] Experiment with network size and fitness design
- [ ] Introduce a Gymnasium-style environment interface
- [ ] Compare neuroevolution with reinforcement-learning approaches suited to continuous control

## Long-term goal

Develop AI Racing Lab into a small, understandable environment for building and evaluating autonomous agents: from observations and policy design to optimization, saved models, and reproducible evaluation.

The central question is how changes to the controller, learning process, and environment affect driving behavior—and how to measure those changes reliably.
