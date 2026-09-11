# 🏎️ AI Racing Lab

AI Racing Lab is a 2D autonomous driving simulation built from scratch in Python using Pygame and NumPy.

The project explores how an autonomous agent can learn to navigate a racing track using only local sensor information.

It started as a rule-based driving experiment and has evolved into a small neuroevolution system where a population of neural-network-controlled cars is trained through selection, crossover, mutation, and fitness-based evaluation.

## Current Features

The current version includes:

- Custom 2D racing environment built with Pygame
- Basic vehicle physics:
  - acceleration
  - steering
  - speed limits
  - rotation
- Collision detection with track boundaries
- Five ray-casting distance sensors
- Neural-network-based vehicle control
- Population-based training
- Fitness-based evaluation
- Checkpoint tracking
- Lap counting
- Continuous progress reward
- Genetic selection
- Elitism
- Neural network crossover
- Multiple mutation strengths
- Random agents for population diversity
- Automatic best-model saving
- Loading previously trained models
- Separate training and demo modes
- Lap timing in demo mode
- Real-time sensor visualization
- Live training statistics

## How the Agent Sees the Track

The car does not receive its absolute position or a map of the track.

Instead, it observes the environment through five ray-casting sensors.

```text id="0m9j34"
      \   \   |   /   /
       \   \  |  /   /
        \   \ | /   /
             CAR
```

The sensors represent:

```text id="3nuq15"
left_far
left
front
right
right_far
```

Each sensor measures the distance from the car to the nearest track boundary.

The current speed of the car is also provided to the neural network.

This gives the neural network a total of six inputs.

## Neural Network Architecture

Each car is controlled by a small feed-forward neural network.

```text id="x06y6j"
5 distance sensors + speed
           │
           ▼
       6 inputs
           │
           ▼
     8 hidden neurons
           │
        tanh()
           │
           ▼
       3 outputs
```

The three output neurons represent:

```text id="06bwre"
0 = steer left
1 = drive straight
2 = steer right
```

The action with the highest output score is selected.

The neural network is implemented directly with NumPy instead of using a machine learning framework.

## Neuroevolution

The neural networks are not trained using backpropagation.

Instead, the project currently uses a neuroevolution approach.

A training generation works roughly like this:

```text id="qmes27"
Create population
      │
      ▼
Cars drive the track
      │
      ▼
Calculate fitness
      │
      ▼
Rank the population
      │
      ▼
Select top agents
      │
      ├── Elites survive unchanged
      │
      ├── Crossover creates children
      │
      ├── Mutation modifies weights
      │
      └── Random agents preserve diversity
      │
      ▼
Next generation
```

Over multiple generations, neural networks that produce better driving behavior are more likely to contribute to future populations.

## Fitness Function

Each agent receives a fitness score based primarily on progress around the track.

The current fitness function combines:

- number of checkpoints passed
- continuous progress toward the next checkpoint
- a small survival reward

Conceptually:

```text id="bokz44"
fitness =
    checkpoint reward
    + progress reward
    + small survival reward
```

Checkpoint rewards dominate the score so that an agent is encouraged to actually move around the track instead of simply surviving.

## Selection Strategy

The current population contains 40 agents.

After each generation:

- the population is ranked by fitness
- the top agents become parents
- the best agents survive through elitism
- children are created through crossover
- different mutation rates are used for exploration
- several completely random agents are added to preserve diversity

This creates a balance between:

```text id="21oaay"
exploitation
    +
exploration
```

Some agents stay close to successful solutions, while others explore more aggressive changes.

## Model Persistence

The best neural network discovered during training is automatically saved as:

```text id="hlhguq"
best_brain.npz
```

The file contains the trained neural network weights and biases.

When the program is started again, the saved model can be loaded and used as the starting point for further training.

The trained model file is ignored by Git and is not committed to the repository.

## Train Mode

In `TRAIN` mode, a population of neural-network-controlled cars is evaluated and evolved.

The simulation displays:

- current generation
- number of surviving agents
- current best fitness
- completed laps
- passed checkpoints
- generation frame count
- all-time records

The best current agent is displayed in yellow.

## Demo Mode

Press `TAB` to switch to `DEMO` mode.

Demo mode loads the saved `best_brain.npz` model and runs a single autonomous car.

No evolution happens in demo mode.

This represents the inference stage:

```text id="zx5o7k"
saved neural network
        │
        ▼
sensor observations
        │
        ▼
neural network
        │
        ▼
steering decision
```

Demo mode also tracks:

- completed laps
- checkpoint progress
- current lap time
- best lap time
- crash status

## Controls

| Key | Action |
|---|---|
| `TAB` | Switch between TRAIN and DEMO mode |
| `R` | Restart current mode |
| Window close button | Exit simulation |

## Project Structure

```text id="089r0d"
ai-racing-lab/
│
├── main.py
├── car.py
├── brain.py
├── README.md
├── .gitignore
│
└── best_brain.npz   # generated locally, ignored by Git
```

### `main.py`

Contains:

- simulation loop
- racing track
- population management
- fitness evaluation
- evolutionary algorithm
- TRAIN / DEMO modes
- model saving and loading
- HUD and training statistics

### `car.py`

Contains:

- vehicle movement
- collision detection
- sensor ray casting
- checkpoint tracking
- lap tracking
- progress calculation
- neural-network-based control

### `brain.py`

Contains:

- feed-forward neural network
- forward propagation
- neural network weights and biases
- mutation
- crossover
- model saving
- model loading

## Tech Stack

- Python
- Pygame
- NumPy

Concepts explored in the project include:

- neural networks
- neuroevolution
- genetic algorithms
- autonomous agents
- simulation
- ray casting
- fitness / reward design
- exploration vs. exploitation
- model persistence
- training vs. inference

## Running the Project

Clone the repository:

```bash id="l3s3gp"
git clone https://github.com/YOUR_USERNAME/ai-racing-lab.git
cd ai-racing-lab
```

Create a virtual environment:

```bash id="0z1tke"
python -m venv .venv
```

Activate it.

Windows PowerShell:

```powershell id="8e95dz"
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash id="dhawpe"
pip install pygame numpy
```

Run the simulation:

```bash id="yr6bl0"
python main.py
```

If no trained model exists, training starts with randomly initialized neural networks.

Once a better model is discovered, it is automatically saved locally as:

```text id="q7sugc"
best_brain.npz
```

## Current Training Architecture

The current neural network architecture is intentionally small:

```text id="mohnxe"
Input layer:   6
Hidden layer:  8
Output layer:  3
```

Because the network is small, the project makes it easier to inspect and understand how the complete learning system works instead of hiding the logic behind a large machine learning framework.

## Roadmap

Completed:

- [x] Build a basic racing environment
- [x] Implement vehicle movement
- [x] Add collision detection
- [x] Implement ray-casting sensors
- [x] Add checkpoints
- [x] Add lap tracking
- [x] Build a rule-based autonomous controller
- [x] Implement a neural network controller
- [x] Run multiple agents simultaneously
- [x] Design a fitness function
- [x] Add continuous progress rewards
- [x] Implement neuroevolution
- [x] Add elitism
- [x] Add crossover
- [x] Add multiple mutation rates
- [x] Preserve population diversity
- [x] Save trained models
- [x] Load trained models
- [x] Separate training and inference
- [x] Add demo mode
- [x] Measure autonomous lap times

Planned:

- [ ] Save training statistics to files
- [ ] Plot fitness across generations
- [ ] Add multiple track layouts
- [ ] Test whether trained agents generalize to unseen tracks
- [ ] Improve the fitness function
- [ ] Experiment with different neural network architectures
- [ ] Add adjustable simulation speed
- [ ] Train without rendering for faster experiments
- [ ] Compare neuroevolution with reinforcement learning
- [ ] Implement a Gymnasium-style environment
- [ ] Experiment with algorithms such as DQN or PPO

## Long-Term Goal

The long-term goal of AI Racing Lab is to gradually evolve from a simple visual simulation into a more complete autonomous-agent experimentation environment.

The project is intended to explore the full pipeline:

```text id="n6psf8"
environment
    ↓
observations
    ↓
agent
    ↓
actions
    ↓
reward / fitness
    ↓
optimization
    ↓
trained policy
    ↓
inference
```

Future versions will compare different approaches to autonomous decision-making, including neuroevolution and reinforcement learning.

The main objective is not only to build a car that can complete a track, but to understand how autonomous learning systems are designed, trained, evaluated, and improved.