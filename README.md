# Reinforcement Learning in C

A from-scratch reinforcement learning implementation written in C. The project trains a small neural policy with REINFORCE to play Snake on a grid, without PyTorch, TensorFlow, or another machine-learning framework.

## Overview

The implementation is split into focused C modules:

- `arena.c` and `arena.h`: custom memory arena allocator
- `random.c` and `random.h`: xorshift-style pseudo-random number generator
- `environment.c` and `environment.h`: Snake grid and batched environment
- `common.h`: shared types and configuration
- `matrix.c` and `matrix.h`: matrix operations and linear algebra
- `model.c` and `model.h`: neural-network autograd and policy model
- `training.c` and `training.h`: REINFORCE rollout and training loop
- `Snake.c`: program entry point

The source trains in a 6x6 environment using batches of 64 independent trajectories.

## Requirements

- A C11-compatible compiler
- GCC or MinGW-w64 on Windows
- A command line with access to the math library

## Build

Run the following command in the project directory:

```powershell
gcc -std=c11 -Wall -Wextra -O2 Snake.c arena.c random.c environment.c matrix.c model.c training.c -lm -o Snake.exe
```

With GCC or MinGW installed under a different command name, replace `gcc` with that compiler executable.

## Run

```powershell
./Snake.exe
```

The program prints training progress after each epoch, including:

- Average return
- Number of foods collected
- Number of training samples
- Return standard deviation

## Default Training Configuration

The current training loop uses:

| Setting | Value |
|---|---:|
| Grid | 6x6 |
| Batch size | 64 |
| Episode length | 100 steps |
| Discount factor (`gamma`) | 0.99 |
| Learning rate | 0.05 |
| Epochs | 2000 |

Batch size and episode length are defined in `common.h`; the remaining training settings are in `training.c`.

## Project Structure

```text
.
|-- Snake.c       Program entry point
|-- arena.c       Memory arena implementation
|-- arena.h       Memory arena interface
|-- common.h      Shared types and configuration
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
|-- README.md     Project documentation
```

## Notes

This is a small educational implementation. It demonstrates core reinforcement-learning and neural-network concepts directly in C, but it does not include the optimizations, stability features, or tooling found in production ML frameworks.
   