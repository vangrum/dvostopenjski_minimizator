# Beam Bending Simulation

This repository contains a Python script that simulates the **bending of a beam** using a two step process: the **Viterbi algorithm** and the **numerical SLSQP method**. The program is designed to find a stable equilibrium shape of a beam under an axial force. It generates a series of plots showing the beam's profile for different force values. By finding the global minimum of the energy functional of the beam it finds the configuration of the beam which the beam is most likely to follow for a given set of parameters. 

## Key Concepts

  * **Viterbi Algorithm:** Originally used for sequence analysis, it's adapted here to find the optimal beam shape by discretizing the problem. It finds the path with the minimum cumulative energy cost through a grid of possible beam configurations.
  * **Energy Minimization:** This approach uses a numerical optimization solver to directly minimize the total energy of the system. The total energy is defined by the sum of the beam's **internal strain energy** and the **work done by the external horizontal force**.
  * **Numba:** The core computational functions are accelerated using `numba`'s `@njit` decorator, which compiles Python code to highly performant machine code.

## Prerequisites

To run this simulation, you'll need the following libraries installed:

  * `numpy`
  * `matplotlib`
  * `scipy`
  * `numba`

You can install them using pip:

```bash
pip install numpy matplotlib scipy numba
```

## How It Works

The script operates in three main stages:

1.  **Parameter Setup:** The `main` function initializes the physical parameters of the microbeam, such as its length, thickness, and material properties (Young's Modulus). It also defines a range of horizontal forces to simulate.

2.  **Simulation Loop:**

      * For each horizontal force value, the script calculates the beam's shape using two methods:
          * **Viterbi Algorithm:** The `viterbi_algorithm` function discretizes the possible beam positions into a grid and finds the path of minimum energy. This solution serves as an excellent initial guess for the optimization solver.
          * **Energy Minimization:** The `minimize` function from `scipy.optimize` refines the Viterbi solution. It finds the exact shape that minimizes the total energy functional, subject to boundary conditions (e.g., fixed ends).
      * The `energy_functional` function calculates the total potential energy of the beam for a given shape.

3.  **Visualization and Output:**

      * The script generates a series of plots, each showing the beam's shape at a specific horizontal force.
      * The plot compares the Viterbi solution (the initial guess) with the final, minimized energy solution.
      * Each plot is saved as a PNG file in a dedicated output folder, named to reflect the simulation's parameters.

## Running the Script

Simply execute the script from your terminal:

```bash
python main.py
```

The script will automatically create directories for the output images and run simulations for various configurations defined in the `if __name__ == "__main__":` block. It uses `multiprocessing` to run multiple simulations in parallel, significantly speeding up the process.

## Customizing the Simulation

You can modify the simulation parameters within the `main` function or the `if __name__ == "__main__":` block to explore different scenarios:

  * `segment_count_beam`: The number of discrete segments the beam is divided into. A higher number increases resolution but also computational time.
  * `vertical_level_count`: The number of discrete vertical positions for the Viterbi algorithm.
  * `length_beam`, `youngs_modulus`, `thickness`: Physical properties of the beam.
  * `channel_width`: The width of the channel the beam bends within.
  * `force_start`, `force_stop`, `force_step`: The range and increment of the horizontal force.
