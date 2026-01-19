"""
1D Heat Equation Solver using Finite Difference Method (Explicit Scheme)
Author: Hamza Lboukhari
Description: Numerical simulation of heat diffusion in a 1D rod.
             Optimized using NumPy vectorization and includes CFL stability checks.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class HeatEquationSolver:
    def __init__(self, L=1.0, T=0.5, nx=50, alpha=0.01):
        """
        Initialize the simulation parameters.
        L: Length of the rod
        T: Total simulation time
        nx: Number of spatial grid points
        alpha: Thermal diffusivity constant
        """
        self.L = L
        self.T = T
        self.nx = nx
        self.dx = L / (nx - 1)
        self.alpha = alpha
        
        # Spatial grid
        self.x = np.linspace(0, L, nx)
        
        # Temperature array
        self.u = np.zeros(nx)

    def set_initial_conditions(self):
        """Sets the initial temperature profile (e.g., a Gaussian pulse in the center)."""
        # Example: Pulse in the middle of the rod
        self.u = np.exp(-100 * (self.x - self.L / 2)**2)
        
        # Boundary conditions (Dirichlet: Fixed temperature at ends)
        self.u[0] = 0
        self.u[-1] = 0

    def solve(self):
        """
        Solves the Heat Equation using Explicit FDM.
        Check CFL condition for stability first.
        """
        # Calculate maximum stable time step (CFL condition: alpha * dt / dx^2 <= 0.5)
        dt_stable = 0.5 * (self.dx ** 2) / self.alpha
        self.dt = dt_stable * 0.9  # Use 90% of the max stable step for safety
        
        nt = int(self.T / self.dt) # Number of time steps
        gamma = self.alpha * self.dt / (self.dx ** 2)

        print(f"Simulation Info:")
        print(f"- Grid points: {self.nx}")
        print(f"- Time steps: {nt}")
        print(f"- CFL Number (gamma): {gamma:.4f} (Must be <= 0.5)")

        # Store history for visualization
        history = [self.u.copy()]

        # Time-stepping loop
        u_curr = self.u.copy()
        
        for n in range(nt):
            # Vectorized Update: u[i] = u[i] + gamma * (u[i+1] - 2*u[i] + u[i-1])
            # Slicing [1:-1] avoids boundaries, [2:] is i+1, [:-2] is i-1
            u_curr[1:-1] = u_curr[1:-1] + gamma * (u_curr[2:] - 2*u_curr[1:-1] + u_curr[:-2])
            
            # Save state every 50 steps for plotting to save memory
            if n % 50 == 0:
                history.append(u_curr.copy())

        return history

    def plot_results(self, history):
        """Visualizes the heat diffusion over time."""
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")
        
        # Plot initial state, middle states, and final state
        steps_to_plot = np.linspace(0, len(history)-1, 6, dtype=int)
        
        for idx in steps_to_plot:
            plt.plot(self.x, history[idx], label=f'Step {idx*50}')

        plt.title(f'1D Heat Diffusion (FDM) - alpha={self.alpha}')
        plt.xlabel('Position (x)')
        plt.ylabel('Temperature (u)')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.show()

if __name__ == "__main__":
    # Run the simulation
    solver = HeatEquationSolver(L=1.0, T=0.2, nx=100, alpha=0.1)
    solver.set_initial_conditions()
    data = solver.solve()
    solver.plot_results(data)
