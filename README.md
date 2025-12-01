# EvoPixels

**Evolutionary Image Reconstruction**

EvoPixels is a Python project that reconstructs a target image using a genetic algorithm. An image is treated as an individual in a population, and reconstruction is achieved by evolving candidate images through selection, mutation, and crossover, guided by a fitness function that measures visual similarity to the target. Over successive generations, complex structure emerges from simple primitives.

![Evolution GIF](output/evolution_1.gif)

---

## Core Concept

- Each candidate solution is a full image
- Fitness is computed using mean squared error (MSE) against the target
- Better candidates are selected and combined
- Random mutations introduce variation
- Iterative evolution drives the population toward visual similarity

This project focuses on search-based optimization in high-dimensional visual space.

---

## Features

- Genetic algorithm–driven image reconstruction
- Shape-based mutations (ellipse, rectangle, line)
- Patch-level crossover between candidate images
- Elitism-based selection
- Downsampled fitness evaluation for efficiency
- Real-time visualization and result export

---

## Project Structure

```
.
├── main.py        # Entry point and evolutionary loop
├── evolution.py   # Mutation, crossover, and fitness functions
├── canvas.py      # Canvas setup and preprocessing
├── gui.py         # Visualization and frame capture
├── config.py      # All configuration and GA parameters
├── assets/        # Input images
└── output/        # Generated evolution GIFs
```

---

## How It Works

1. Load and resize the target image
2. Initialize a population of simple candidate images
3. For each generation:
   - Evaluate fitness of all candidates
   - Preserve top-performing individuals
   - Generate new candidates via crossover and mutation
   - Replace the population and repeat until convergence or termination

---

## Requirements

Install dependencies (recommended in a virtual environment):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

**`requirements.txt` includes:**

- numpy
- opencv-python
- Pillow
- pygame

---

## Running

From the project root:

```powershell
python main.py
```

The program will begin evolving candidate images toward the target.

---

## Configuration

All parameters are defined in `config.py`, including:

- Target image path
- Population size and number of generations
- Mutation and crossover settings
- Fitness downsampling factor
- Output paths

---

## Notes

- The implementation prioritizes clarity and extensibility over aggressive optimization
- Mutation operators and fitness functions are intentionally modular and easy to modify

---

## Possible Extensions

- Additional mutation primitives (polygons, textures)
- Alternative fitness metrics (SSIM, edge-based loss)
- Parallel or batched fitness evaluation
- Headless execution mode for automated runs

---

## License

This project is open source. See LICENSE file for details.

---
