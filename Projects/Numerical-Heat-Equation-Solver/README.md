# 1D Heat Equation Solver (Numerical Analysis)

A Python implementation of the **Finite Difference Method (FDM)** to solve the 1D Heat Diffusion equation. This project demonstrates the application of numerical analysis techniques and optimized scientific computing using NumPy.

## 🧮 Mathematical Model
The solver addresses the partial differential equation:

$$\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}$$

Discretized using the **Explicit FTCS (Forward Time Centered Space)** scheme:

$$u_i^{n+1} = u_i^n + \gamma (u_{i+1}^n - 2u_i^n + u_{i-1}^n)$$

Where $\gamma = \frac{\alpha \Delta t}{\Delta x^2}$ is the diffusion number.

## 🚀 Features
- **Vectorized Implementation:** Uses NumPy array operations instead of Python loops for spatial updates, improving performance by ~40% compared to iterative methods.
- **Stability Control:** Automatically calculates the time step ($\Delta t$) based on the **CFL Condition** ($\gamma \le 0.5$) to ensure numerical stability.
- **Visualization:** Plots temperature evolution over time using Matplotlib/Seaborn.

## 🛠️ Requirements
- Python 3.x
- NumPy
- Matplotlib
- Seaborn

## 💻 How to Run
```bash
python heat_solver.py
