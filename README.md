# 🏎️ AI Racing Lab

AI Racing Lab is a 2D autonomous driving simulation built from scratch in Python using Pygame.

The goal of the project is to explore how an autonomous agent can learn to navigate a racing track using only local sensor information. The project starts with a manually designed sensor-based controller and will gradually evolve toward neural networks and reinforcement learning.

## Current Features

The current version includes:

- Custom 2D racing environment built with Pygame
- Basic vehicle physics including acceleration, braking, steering, friction, and rotation
- Collision detection with track boundaries
- Five ray-casting distance sensors that allow the car to detect nearby track boundaries
- Human driving mode using keyboard controls
- Autonomous driving using a rule-based controller
- Checkpoint system for measuring progress around the track
- Lap counter and lap timer
- Best lap tracking
- Real-time visualization of the vehicle's sensors
- Ability to switch between human and autonomous driving

## How the Autonomous Driver Works

The car does not have access to its absolute position on the track.

Instead, it observes the environment using five ray-casting sensors:

```text
        \    |    /
         \   |   /
          \  |  /
           \ | /
            CAR
```

Each sensor measures the distance between the car and the nearest track boundary.

The controller receives approximately the following state:

```text
left_far
left
front
right
right_far
speed
```

The current autonomous controller uses manually designed heuristics to determine steering decisions based on these sensor readings.

This controller will serve as a baseline for future learning-based agents.

## Controls

| Key | Action |
|---|---|
| ↑ | Accelerate |
| ↓ | Brake / Reverse |
| ← | Steer left |
| → | Steer right |
| SPACE | Switch between Human and AI mode |
| R | Restart |

## Project Structure

```text
ai-racing-lab/
│
├── main.py
├── car.py
└── README.md
```

## Tech Stack

- Python
- Pygame
- Mathematics / basic vehicle kinematics
- Ray casting

## Roadmap

The project will gradually evolve from a rule-based autonomous driver into a learning-based racing agent.

Planned milestones include:

- [x] Build a basic racing environment
- [x] Implement vehicle movement and steering
- [x] Add collision detection
- [x] Implement ray-casting sensors
- [x] Add checkpoints and lap timing
- [x] Create a rule-based autonomous driver
- [ ] Implement a neural network controller
- [ ] Run multiple agents simultaneously
- [ ] Design a fitness/reward function
- [ ] Implement evolutionary training
- [ ] Visualize training across generations
- [ ] Experiment with reinforcement learning
- [ ] Add more complex racing tracks
- [ ] Track and compare trained models

## Long-Term Goal

The long-term goal is to build an environment where autonomous agents can learn driving strategies without being explicitly programmed how to navigate the track.

Instead of manually defining rules such as:

```text
if obstacle is close on the right:
    steer left
```

a learning agent will receive sensor observations and learn its own driving policy through interaction with the environment.

This project is being built as an exploration of autonomous systems, neural networks, reinforcement learning, simulation, and software engineering.