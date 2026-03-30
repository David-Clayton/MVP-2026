import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation as animate
import argparse
import time
from numba import njit
from joblib import Parallel, delayed
import pandas as pd

@njit
def gauss_seidel_algorithm(size, phi, rho):
    """Run Gauss-Seidel algorithm for one iteration to converge on a solution for the
    potential. Difficult to vectorise (except for black-red), so instead 
    speed it up with numba"""
    #Original phi to compare distances
    phi_original = phi.copy()

    #Update phi iteratively
    for i in range(1, size - 1):
        for j in range(1, size - 1):
            for k in range(1, size - 1):
                phi[i,j,k] = (phi[i-1, j, k] + phi[i+1, j, k] +
                              phi[i, j-1, k] + phi[i, j+1, k] +
                              phi[i, j, k-1] + phi[i, j, k+1] +
                              rho[i, j, k]) / 6
                
    phi[0, :, :] = 0
    phi[-1, :, :] = 0
    phi[:, 0, :] = 0
    phi[:, -1, :] = 0
    phi[:, :, 0] = 0
    phi[:, :, -1] = 0
                
    distance = np.max(np.abs(phi - phi_original))

    return phi, distance

@njit
def SOR(size, phi, rho, omega):
    """Runs one instance of the Successive Over-Relaxation 
    algorithm"""
    phi_original = phi.copy()

    for i in range(1, size - 1):
        for j in range(1, size - 1):
            for k in range(1, size - 1):
                phi_gs = (phi[i-1, j, k] + phi[i+1, j, k] +
                          phi[i, j-1, k] + phi[i, j+1, k] +
                          phi[i, j, k-1] + phi[i, j, k+1] +
                          rho[i, j, k]) / 6

                phi[i, j, k] = (1 - omega) * phi[i, j, k] + omega * phi_gs

    phi[0, :, :] = 0
    phi[-1, :, :] = 0
    phi[:, 0, :] = 0
    phi[:, -1, :] = 0
    phi[:, :, 0] = 0
    phi[:, :, -1] = 0

    distance = np.max(np.abs(phi - phi_original))

    return phi, distance

class PoissonElectric:
    """Solves the Poisson equation."""

    def __init__(self, rho = "monopole", size = 50, tolerance = 0.000001):
        """?"""
        if rho == "monopole":
            self.rho = np.zeros((size, size, size))
            #Normalise point charge in to be 1 (and positive)
            self.rho[size//2, size//2, size//2] = 1

        #Initialise potential as zero everywhere
        self.phi = np.zeros((size, size, size))
        self.tolerance = tolerance
        self.space_step = 1
        self.rho_str = rho
        self.size = size

    def jacobi_algorithm(self):
        """The Jacobi algorithm updates the potential in the numerical Poisson equation,
        getting it closer to solution"""
        #Copy potential lattice to update
        phi_n = self.phi
        phi_n_plus_1 = phi_n.copy()
    
        phi_n_plus_1[1:-1, 1:-1, 1:-1] = (phi_n[2:,   1:-1, 1:-1] + phi_n[:-2,  1:-1, 1:-1] +
                                            phi_n[1:-1, 2:,   1:-1] + phi_n[1:-1, :-2,  1:-1] +
                                            phi_n[1:-1, 1:-1, 2:] + phi_n[1:-1, 1:-1, :-2] +
                                            self.rho[1:-1, 1:-1, 1:-1]) / 6

        phi_n_plus_1[0, :, :] = 0
        phi_n_plus_1[-1, :, :] = 0
        phi_n_plus_1[:, 0, :] = 0
        phi_n_plus_1[:, -1, :] = 0
        phi_n_plus_1[:, :, 0] = 0
        phi_n_plus_1[:, :, -1] = 0
        
        self.phi = phi_n_plus_1

        #Distance between successive iterations will be mean of distance
        #Between elements, to compare with self.tolerance

        distance = np.max(np.abs(phi_n_plus_1-phi_n))

        return self.phi, distance
    
    def gauss_seidel_self(self):
        """Call numba compiled Gauss-Seidel algorithm"""

        self.phi, distance = gauss_seidel_algorithm(self.size, self.phi, self.rho)

        return self.phi, distance

    def success_over_relax_self(self, omega):
        """Calls numba compiled SOR algorithm"""

        self.phi, distance = SOR(self.size, self.phi, self.rho, omega)

        return self.phi, distance
    
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
                return self.phi, i
    
        print(f"Potential did not converge in {max_iter} iterations, \
              reached distance of {distance}")
        #Best estimate if still not converged
        return self.phi, max_iter
    
    def plot_electric_potential(self):
        """Plot midplane of electric potential that is the solution to the Poisson
        equation as a heatmap. Also plot electric potential as a function of distance
        from the point charge"""

        midplane = self.phi.shape[2] // 2
        phi_midplane = self.phi[:, :, midplane]
        #The potential should be symmetric for a point charge, so this shouldn't matter

        plt.imshow(phi_midplane, cmap = "plasma")
        plt.colorbar(label = r"Electric potential ($\phi$)")
        plt.title(f"Electric Potential Steady State", fontsize = 16)
        plt.savefig(f"Electricpotential.png")
        plt.show()

        #Save to CSV
        np.savetxt("Electricpotential.csv", phi_midplane, delimiter=",")

        #Plot potential as function of distance from charge
        #Only possible if just a point charge is present
        if self.rho_str == "monopole":
            #If potential spherically symmetric, just need a single 1D cut
            cut = phi_midplane.shape[0] // 2
            radial = np.arange(phi_midplane.shape[0] // 2)
            potential_cut = phi_midplane[cut, cut:] 

            #Potential should be ~1/r
            fit = potential_cut[1]/radial

            plt.plot(radial, fit, markersize = 0, color = "red", label = "1/r")
            plt.plot(radial, potential_cut, markersize = 5, marker = "o", color = "blue")
            plt.grid(which="both")
            plt.xlabel(f"Radial distance from charge (space steps)", fontsize = 12)
            plt.ylabel(r"Electric Potential $\phi$", fontsize = 12)
            plt.title(f"Electric Potential for a Point Charge", fontsize = 16)
            plt.tick_params(axis = "both", labelsize = 12)
            plt.xscale("log")
            plt.yscale("log")
            plt.tight_layout()
            plt.legend()
            plt.savefig(f"Electricpotentialrad.png")
            plt.show()

            #Save to CSV file
            data = np.column_stack((radial, potential_cut))
            np.savetxt("Electricpotentialdist.csv", data, delimiter=",", header = "Distance, Potential")


    
    def compute_electric_field(self):
        """Numerically compute the vector E-field as the negative of the
        numerically discretised gradient of the electric potential (phi)"""
    
        d_phi_d_x, d_phi_d_y, d_phi_d_z = np.gradient(self.phi)

        E_x = -d_phi_d_x
        E_y = -d_phi_d_y
        E_z = -d_phi_d_z

        E = np.stack((E_x, E_y, E_z), axis = -1)
        return E
    
    def plot_electric_field(self, E_field):
        """Plot the electric field of the box in the (x-y) midplane 
        through the charge."""

        midplane = E_field.shape[2] // 2
        #E_x in midplane
        E_x_midplane = E_field[: , : , midplane , 0]
        #E_y in midplane
        E_y_midplane = E_field[: , : , midplane , 1]
        #E_z in midplane
        E_z_midplane = E_field[: , : , midplane , 2]

        #E-field_magnitude
        E_size = np.sqrt(E_x_midplane**2 + E_y_midplane**2 + E_z_midplane**2)

        plt.quiver(E_y_midplane, E_x_midplane)
        plt.title(f"Discretised Electric Field", fontsize = 16)
        plt.savefig(f"Efield.png")
        plt.show()

        #Save electric field as CSV
        np.savetxt("Exfield.csv", E_x_midplane, delimiter=",")
        np.savetxt("Eyfield.csv", E_y_midplane, delimiter=",")
        np.savetxt("Ezfield.csv", E_z_midplane, delimiter=",")

        #Save electric field strength
        np.savetxt("Efieldstrength.csv", E_size, delimiter=",")

        #Plot & save electric field strength as function of radial distance for point charge
        if self.rho_str == "monopole":
            #If potential spherically symmetric, just need a single 1D cut
            cut = E_size.shape[0] // 2
            radial = np.arange(E_size.shape[0] // 2)
            efield_cut = E_size[cut, cut:] 

            #Electric field should be 1/r^2
            fit = efield_cut[1]/(radial**2)

            plt.plot(radial, fit, markersize = 0, color = "red", label = "1/r^2")
            plt.plot(radial, efield_cut, markersize = 5, marker = "o", color = "blue")
            plt.grid(which="both")
            plt.xlabel(f"Radial distance from charge (space steps)", fontsize = 12)
            plt.ylabel(r"Electric Field E", fontsize = 12)
            plt.title(f"Electric Field Strength for a Point Charge", fontsize = 16)
            plt.tick_params(axis = "both", labelsize = 12)
            plt.xscale("log")
            plt.yscale("log")
            plt.tight_layout()
            plt.legend()
            plt.savefig(f"Electricfieldrad.png")
            plt.show()

            #Save to CSV file
            data = np.column_stack((radial, efield_cut, efield_cut))
            np.savetxt("Electricfielddist.csv", data, delimiter=",", header = "Distance, Potential")

    def find_optimal_omega(self):
        """Plot the number of iterations the potential takes to converge as a
        function of omega"""

        #Empty arrays for data
        no_iter_data = np.zeros(100)

        omega_range = np.linspace(1, 2, 100)

        for i, omega in enumerate(omega_range):
            #Reinitialise in each iteration
            restart = PoissonElectric(rho=self.rho_str, size=self.size, tolerance=self.tolerance)
            _, no_iter = restart.solve_for_potential(omega, algorithm="SOR")
            no_iter_data[i] = no_iter
            
        plt.plot(omega_range, no_iter_data, marker = "o", markersize = 5, color = "green")
        plt.xlabel(r"$\omega$", fontsize = 12)
        plt.ylabel(f"No. of iteration for convergence", fontsize = 12)
        plt.title(f"Optimisation of SOR Convergence", fontsize = 16)
        plt.tick_params(axis = "both", labelsize = 12)
        plt.ylim(0, 1000)
        plt.tight_layout()
        plt.savefig(f"SORconvergence.png")
        plt.show()

        data = np.column_stack((omega_range, no_iter_data))
        np.savetxt("SORconvergence.csv", data, delimiter=",", header="omega, no_iterations")


def main():
    
    parser = argparse.ArgumentParser(description="Solve the Poisson Equation for an electric potential")
    parser.add_argument("size", type = int, default=50)
    parser.add_argument("rho", type = str, default = "monopole")
    parser.add_argument("tolerance", type = float, default=0.000001)
    parser.add_argument("algorithm", type = str, choices=["Jacobi", "Gauss-Seidel", "SOR"])
    parser.add_argument("omega", type = float, default=1.87)
    parser.add_argument("--potential", type = str, choices=["Y", "N"], default="N")
    parser.add_argument("--field", type=str, choices=["Y", "N"], default="N")
    parser.add_argument("--sorconv", type=str, choices=["Y", "N"], default="N")

    args = parser.parse_args()

    time_0 = time.time()

    pe = PoissonElectric(size = args.size, rho = args.rho, tolerance=args.tolerance)

    pe.solve_for_potential(omega = args.omega, algorithm=args.algorithm)

    if args.field == "Y":
        E = pe.compute_electric_field()
        pe.plot_electric_field(E)

    if args.field == "Y":
        pe.plot_electric_potential()

    if args.sorconv == "Y":
        pe.find_optimal_omega()

    time_1 = time.time()

    print(f"Complete in {(time_1 - time_0)/60} minutes")

if __name__ == "__main__":
    main()
        



