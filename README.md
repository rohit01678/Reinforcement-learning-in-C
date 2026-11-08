# Reinforcement Learning in C

A from-scratch reinforcement learning implementation written in C. The project trains a small neural policy with REINFORCE to play Snake on a grid, without PyTorch, TensorFlow, or another machine-learning framework.

## Overview

`Snake.c` contains the complete implementation:

- Custom memory arena allocator
- Xorshift-style pseudo-random number generator
- Matrix operations and batched matrix multiplication
- Small neural-network autograd engine
- Actor network with ReLU and softmax layers
- Batched Snake environment and REINFORCE training loop

The source trains in a 6x6 environment using batches of 64 independent trajectories.

## Requirements

- A C11-compatible compiler
- GCC or MinGW-w64 on Windows
- A command line with access to the math library

## Build

Run the following command in the project directory:

```powershell
gcc -std=c11 -Wall -Wextra -O2 Snake.c -lm -o Snake.exe
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

These values are defined near the batching and training sections in `Snake.c` and can be changed to experiment with training behavior.

## Project Structure

```text
.
|-- Snake.c       Main implementation and training entry point
|-- README.md     Project documentation
```

## Notes

This is a small educational implementation. It demonstrates core reinforcement-learning and neural-network concepts directly in C, but it does not include the optimizations, stability features, or tooling found in production ML frameworks.
   