import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation as animate
import argparse
import time
import random
from numba import njit
from joblib import Parallel, delayed
import pandas as pd

@njit
def gauss_seidel_algorithm(size, A, J):
    """Run Gauss-Seidel algorithm for one iteration to converge on a solution for the
    potential. Difficult to vectorise (except for black-red), so instead 
    speed it up with numba"""
    #Original A to compare distances
    A_original = A.copy()

    #Update A iteratively
    for i in range(1, size - 1):
        for j in range(1, size - 1):
            for k in range(1, size - 1):
                A[i,j,k] = (A[i-1, j, k] + A[i+1, j, k] +
                              A[i, j-1, k] + A[i, j+1, k] +
                              A[i, j, k-1] + A[i, j, k+1] +
                              J[i, j, k]) / 6
                
    A[0, :, :] = 0
    A[-1, :, :] = 0
    A[:, 0, :] = 0
    A[:, -1, :] = 0
    A[:, :, 0] = 0
    A[:, :, -1] = 0
                
    distance = np.max(np.abs(A - A_original))

    return A, distance

@njit
def SOR(size, A, J, omega):
    """Runs one instance of the Successive Over-Relaxation 
    algorithm"""
    A_original = A.copy()

    for i in range(1, size - 1):
        for j in range(1, size - 1):
            for k in range(1, size - 1):
                A_gs = (A[i-1, j, k] + A[i+1, j, k] +
                          A[i, j-1, k] + A[i, j+1, k] +
                          A[i, j, k-1] + A[i, j, k+1] +
                          J[i, j, k]) / 6

                A[i, j, k] = (1 - omega) * A[i, j, k] + omega * A_gs

    A[0, :, :] = 0
    A[-1, :, :] = 0
    A[:, 0, :] = 0
    A[:, -1, :] = 0
    A[:, :, 0] = 0
    A[:, :, -1] = 0

    distance = np.max(np.abs(A - A_original))

    return A, distance

class PoissonMagnetic:
    """Solves the magnetic Poisson equation."""

    def __init__(self, J = "thinwire", size = 50, tolerance = 0.000001):
        """?"""
        if J == "thinwire":
            self.J = np.zeros((size, size, size))
            #Normalise current density to be 1 (and positive)
            self.J[size//2, size//2, :] = 1

        #Initialise potential as zero everywhere
        self.A = np.zeros((size, size, size))
        self.tolerance = tolerance
        self.space_step = 1
        self.J_str = J
        self.size = size

    def jacobi_algorithm(self):
        """The Jacobi algorithm updates the potential in the numerical Poisson equation,
        getting it closer to solution"""
        #Copy potential lattice to update
        A_n = self.A
        A_n_plus_1 = A_n.copy()
    
        A_n_plus_1[1:-1, 1:-1, 1:-1] = (A_n[2:,   1:-1, 1:-1] + A_n[:-2,  1:-1, 1:-1] +
                                            A_n[1:-1, 2:,   1:-1] + A_n[1:-1, :-2,  1:-1] +
                                            A_n[1:-1, 1:-1, 2:] + A_n[1:-1, 1:-1, :-2] +
                                            self.J[1:-1, 1:-1, 1:-1]) / 6

        A_n_plus_1[0, :, :] = 0
        A_n_plus_1[-1, :, :] = 0
        A_n_plus_1[:, 0, :] = 0
        A_n_plus_1[:, -1, :] = 0
        
        self.A = A_n_plus_1

        #Distance between successive iterations will be mean of distance
        #Between elements, to compare with self.tolerance

        distance = np.max(np.abs(A_n_plus_1-A_n))

        return self.A, distance
    
    def gauss_seidel_self(self):
        """Call numba compiled Gauss-Seidel algorithm"""

        self.A, distance = gauss_seidel_algorithm(self.size, self.A, self.J)

        return self.A, distance

    def success_over_relax_self(self, omega):
        """Calls numba compiled SOR algorithm"""

        self.A, distance = SOR(self.size, self.A, self.J, omega)

        return self.A, distance
    
    def solve_for_potential(self, omega, algorithm, max_iter = 10000):
        """Runs the Jacobi algorithm until the potential converges
        to a solution which does not change with reiteration more than
        the tolerance. Will fail if solution does not converge within 10000
        iterations"""

        #Run Jacobi algorithm until distance is less than tolerance

        for i in range(max_iter):
            if algorithm == "Jacobi":
                distance = self.jacobi_algorithm()[1]
            elif algorithm == "Gauss-Seidel":
                distance = self.gauss_seidel_self()[1]
            elif algorithm == "SOR":
                distance = self.success_over_relax_self(omega)[1]
            
            if distance < self.tolerance:
                print(f"Potential converged in {i} iterations")
                return self.A, i
    
        print(f"Potential did not converge in {max_iter} iterations, \
              reached distance of {distance}")
        #Best estimate if still not converged
        return self.A, max_iter
    
    def plot_magnetic_potential(self):
        """Plot midplane of magnetic potential that is the solution to the Poisson
        equation as a heatmap. Also plot magnetic potential as a function of distance
        from the thin wire"""

        midplane = self.A.shape[2] // 2
        A_midplane = self.A[:, :, midplane]
        #The potential should be symmetric for a thin wire, so this shouldn't matter

        plt.imshow(A_midplane, cmap = "plasma")
        plt.colorbar(label = r"Magnetic potential (A)")
        plt.title(f"Magnetic Potential Steady State", fontsize = 16)
        plt.savefig(f"magneticpotential.png")
        plt.show()

        #Save to CSV
        np.savetxt("magneticpotential.csv", A_midplane, delimiter=",")

        #Plot potential as function of distance from charge
        #Only possible if just a thin wire is present
        if self.J_str == "thinwire": 
            #If potential cylindrically symmetric, just need a single 1D cut
            cut = A_midplane.shape[0] // 2
            radial = np.arange(A_midplane.shape[0] // 2)
            potential_cut = A_midplane[cut, cut:] 

            #Potential should be ~ln(r)
            fit = -potential_cut[1] * np.log(radial) + potential_cut[1]

            plt.plot(radial, fit, markersize = 0, color = "red", label = "ln(r)")
            plt.plot(radial, potential_cut, markersize = 5, marker = "o", color = "blue")
            plt.grid(which="both")
            plt.xlabel(f"Radial distance from charge (space steps)", fontsize = 12)
            plt.ylabel(r"magnetic Potential A", fontsize = 12)
            plt.title(f"Magnetic Potential for a Thin Wire", fontsize = 16)
            plt.tick_params(axis = "both", labelsize = 12)
            plt.xscale("log")
            plt.tight_layout()
            plt.legend()
            plt.savefig(f"Magneticpotentialrad.png")
            plt.show()

            #Save to CSV file
            data = np.column_stack((radial, potential_cut))
            np.savetxt("Magneticpotentialdist.csv", data, delimiter=",", header = "Distance, Potential")


    
    def compute_magnetic_field(self):
        """Numerically compute the vector B-field as the
        numerically discretised curl of the magnetic potential (A)"""
    
        A_left = np.roll(self.A, shift = 1, axis = 0)
        A_up = np.roll(self.A, shift = 1, axis = 1)

        d_A_d_x = ((A_left - self.A) / self.space_step)
        d_A_d_y = ((A_up - self.A) / self.space_step)

        B_y = -d_A_d_x
        B_x = d_A_d_y
        #B_z = 0 - cylindrically symmetric
        B = np.stack((B_x, B_y), axis = -1)
        return B
    
    def plot_magnetic_field(self, B_field):
        """Plot the magnetic field of the box in the (x-y) midplane 
        through the wire."""

        midplane = B_field.shape[2] // 2
        #B_x in midplane
        B_x_midplane = B_field[: , : , midplane , 0]
        #B_y in midplane
        B_y_midplane = B_field[: , : , midplane , 1]

        #B-field_magnitude
        B_size = np.sqrt(B_x_midplane**2 + B_y_midplane**2)

        plt.quiver(B_y_midplane, B_x_midplane, scale = 0.5)
        plt.title(f"Discretised Magnetic Field", fontsize = 16)
        plt.savefig(f"Bfield.png")
        plt.show()

        #Save magnetic field as CSV
        np.savetxt("Bxfield.csv", B_x_midplane, delimiter=",")
        np.savetxt("Byfield.csv", B_y_midplane, delimiter=",")

        #Save magnetic field strength
        np.savetxt("Bfieldstrength.csv", B_size, delimiter=",")

        #Plot & save magnetic field strength as function of radial distance for thin wire
        if self.J_str == "thinwire": 
            #If potential spherically symmetric, just need a single 1D cut
            cut = B_size.shape[0] // 2
            radial = np.arange(B_size.shape[0] // 2)
            bfield_cut = B_size[cut, cut:] 

            #magnetic field should be 1/r
            fit = bfield_cut[1]/radial

            plt.plot(radial, fit, markersize = 0, color = "red", label = "1/r")
            plt.plot(radial, bfield_cut, markersize = 5, marker = "o", color = "blue")
            plt.grid(which="both")
            plt.xlabel(f"Radial distance from wire (space steps)", fontsize = 12)
            plt.ylabel(r"Magnetic Field B", fontsize = 12)
            plt.title(f"Magnetic Field Strength for a Thin Wire", fontsize = 16)
            plt.tick_params(axis = "both", labelsize = 12)
            plt.tight_layout()
            plt.legend()
            plt.savefig(f"magneticfieldrad.png")
            plt.show()

            #Save to CSV file
            data = np.column_stack((radial, bfield_cut))
            np.savetxt("magneticfielddist.csv", data, delimiter=",", header = "Distance, Potential")


def main():
    parser = argparse.ArgumentParser(description="Solve the Poisson Equation for an magnetic potential")
    parser.add_argument("size", type = int, default=50)
    parser.add_argument("J", type = str, default = "thinwire")
    parser.add_argument("tolerance", type = float, default=0.000001)
    parser.add_argument("algorithm", type = str, choices=["Jacobi", "Gauss-Seidel", "SOR"])
    parser.add_argument("omega", type = float, default=1.87)
    parser.add_argument("--potential", type = str, choices=["Y", "N"], default="N")
    parser.add_argument("--field", type=str, choices=["Y", "N"], default="N")

    args = parser.parse_args()

    time_0 = time.time()

    pm = PoissonMagnetic(size = args.size, J = args.J, tolerance=args.tolerance)

    pm.solve_for_potential(omega = args.omega, algorithm=args.algorithm)

    if args.field == "Y":
        B = pm.compute_magnetic_field()
        pm.plot_magnetic_field(B)

    if args.field == "Y":
        pm.plot_magnetic_potential()

    time_1 = time.time()

    print(f"Complete in {(time_1 - time_0)/60} minutes")

if __name__ == "__main__":
    main()

