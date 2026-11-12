# Reinforcement Learning in C

A from-scratch reinforcement learning implementation written in C. The project trains a small neural policy with REINFORCE to play Snake on a grid, without PyTorch, TensorFlow, or another machine-learning framework.

## Overview

The implementation is split into focused C modules:

- `arena.c` and `arena.h`: custom memory arena allocator
- `random.c` and `random.h`: xorshift-style pseudo-random number generator
- `environment.c` and `environment.h`: growing Snake grid and batched environment
- `common.h`: shared types and configuration
- `matrix.c` and `matrix.h`: matrix operations and linear algebra
- `model.c` and `model.h`: neural-network autograd and policy model
- `training.c` and `training.h`: REINFORCE rollout and training loop
- `Snake.c`: training entry point and persistent policy interface
- `snake.py`: Pygame game with human and C-policy modes

The source trains in a 16x16 environment using batches of 32 independent trajectories.

## Requirements

- A C11-compatible compiler
- GCC or MinGW-w64 on Windows
- A command line with access to the math library
- Python 3.13 and Pygame for the graphical game
- Docker Engine/Desktop for the containerized build

## Build

Run the following command in the project directory:

```powershell
gcc -std=c11 -Wall -Wextra -O3 Snake.c arena.c random.c environment.c matrix.c model.c training.c -lm -o Snake.exe
```

The arena aligns memory using a pointer-width integer, so this build avoids pointer-size conversion warnings on both 32-bit and 64-bit Windows. With GCC or MinGW installed under a different command name, replace `gcc` with that compiler executable.

On Linux, use the same source list and name the output `Snake`:

```sh
gcc -std=c11 -Wall -Wextra -O3 Snake.c arena.c random.c environment.c matrix.c model.c training.c -lm -o Snake
```

## Docker

The image uses a build stage to compile the C agent once. The runtime stage includes Python/Pygame and the prebuilt agent, but not GCC, so starting the game does not trigger another compilation. A named volume keeps trained weights between disposable containers.

```sh
docker build -t snake-rl .
```

The default container command launches the AI game. A host display server must be running and configured to accept connections from Docker.

On Windows with Docker Desktop and an X server such as VcXsrv:

```powershell
docker run --rm -v snake-policy:/data -e DISPLAY=host.docker.internal:0.0 snake-rl
```

On Linux with X11:

```sh
docker run --rm -v snake-policy:/data -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix snake-rl
```

To run only C training without a display:

```sh
docker run --rm -v snake-policy:/data --entrypoint ./Snake snake-rl
```

## Run

```powershell
.\Snake.exe
```

To play the graphical game, create and activate a Python environment, install Pygame, and launch the game:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install pygame
python snake.py
```

To launch the AI game without a console window, double-click `run_snake.vbs`.

The game opens in human mode. Choose **C POLICY** to train or load the C policy and watch it play. Locally, weights are saved to `Snake-policy.bin`. In Docker, `SNAKE_POLICY_PATH` points to `/data/Snake-policy.bin`; use the documented `snake-policy` volume to keep weights across container runs. Use `python snake.py --ai --retrain` to force training again, or pass an epoch count such as `python snake.py --ai --retrain --epochs 50`.

Use the arrow keys or WASD to steer, Space to pause, R to restart, and + or - to adjust the speed. The game runs at 45 steps per second by default.

## Developer Notes

- `Snake.c` initializes the policy and exposes the persistent C agent used by Python.
- `training.c` runs batched REINFORCE episodes; grid size, batch size, episode length, and default epoch count are in `common.h`.
- `environment.c` owns movement, collision handling, reward calculation, and observation encoding.
- Live action selection filters collisions and uses flood fill to prefer moves that keep more of the board reachable; the tail coordinate is sent explicitly because its cell may vacate on a non-growth move.
- `snake.py` implements the Pygame UI and sends one observation per move to the C process.
- The 266-value observation contains a 256-cell body map, normalized head and food coordinates, a four-value direction one-hot vector, and normalized tail coordinates. Keep the C and Python layouts in sync when changing it.
- Outside Docker, the Python bridge automatically rebuilds the C agent when a source or header is newer than the executable. In Docker, `SNAKE_PREBUILT=1` requires the build-stage executable and skips that runtime compiler path.
- The checkpoint includes the grid and model dimensions; incompatible checkpoints are ignored and retrained. `SNAKE_POLICY_PATH` can override its path, which Docker sets to `/data/Snake-policy.bin`.

The program prints training progress after each epoch, including:

- Average return
- Number of foods collected
- Number of training samples
- Return standard deviation

## Default Training Configuration

The current training loop uses:

| Setting | Value |
|---|---:|
| Grid | 16x16 |
| Batch size | 32 |
| Episode length | 60 steps |
| Discount factor (`gamma`) | 0.99 |
| Learning rate | 0.05 |
| Epochs | 25 |

Grid size, batch size, episode length, and default epochs are defined in `common.h`; the remaining training settings are in `training.c`.

## Project Structure

```text
.
|-- arena.c       Memory arena implementation
|-- arena.h       Memory arena interface
|-- common.h      Shared types and configuration
|-- .gitignore    Excludes local environments and generated binaries
|-- Dockerfile    Container build for the C trainer and Pygame app
|-- .dockerignore Excludes local environments and generated artifacts
|-- environment.c Snake environment implementation
|-- environment.h Snake environment interface
|-- random.c      Pseudo-random number generator implementation
|-- random.h      Pseudo-random number generator interface
|-- matrix.c      Matrix operations implementation
|-- matrix.h      Matrix operations interface
|-- model.c       Neural-network and autograd implementation
|-- model.h       Neural-network and autograd interface
|-- training.c    REINFORCE training implementation
|-- training.h    Training interface
|-- Snake.c       Training entry point and C policy interface
|-- snake.py     Pygame game and C-policy bridge
|-- run_snake.vbs Hidden-console AI game launcher for Windows
```

## Notes

This is a small educational implementation. It demonstrates core reinforcement-learning and neural-network concepts directly in C, but it does not include the optimizations, stability features, or tooling found in production ML frameworks.
   