import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation as animate
import argparse
import time
import random
from numba import njit
from joblib import Parallel, delayed
import pandas as pd

class Poisson:
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
        self.space_step = 2
        self.rho_str = rho

    
    def compute_electric_field(self):
        """Numerically compute the vector E-field as the negative of the
        numerically discretised gradient of the electric potential (phi)"""
        phi_left = np.roll(self.phi, shift = 1, axis = 0)
        phi_up = np.roll(self.phi, shift = 1, axis = 1)
        phi_forward = np.roll(self.phi, shift = 1, axis = 2)
        d_phi_d_x = ((phi_left - self.phi) / self.space_step)
        d_phi_d_y = ((phi_up - self.phi) / self.space_step)
        d_phi_d_z = ((phi_forward - self.phi) / self.space_step)

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

        #Cap size of vectors
        cap_threshold = np.percentile(E_size, 95)
        cap = np.minimum(1, cap_threshold/E_size)

        E_x_midplane_plot = cap * E_x_midplane
        E_y_midplane_plot = cap * E_y_midplane

        plt.quiver(E_x_midplane_plot, E_y_midplane_plot)
        plt.title(f"Discretised Electric field", fontsize = 16)
        plt.savefig(f"Efield.png")
        plt.show()

        #Save electric field as CSV
        np.savetxt("Exfield.csv", E_x_midplane, delimiter=",")
        np.savetxt("Eyfield.csv", E_x_midplane, delimiter=",")
        np.savetxt("Ezfield.csv", E_x_midplane, delimiter=",")

        #Save electric field strength
        np.savetxt("Efieldstrength.csv", E_size, delimiter=",")

        #Plot & save electric field strength as function of radial distance for point charge
        if self.rho_str == "monopole":
            #If potential spherically symmetric, just need a single 1D cut
            cut_x = E_size.shape[0] // 2
            cut_y = E_size.shape[1] // 2
            radial = np.arange(E_size.shape[0] // 2)
            efield_cut_x = E_size[cut_x, cut_x:] 
            efield_cut_y = E_size[cut_y, cut_y:] 

            #Electric field should be 1/r^2
            fit = efield_cut_x[1]/(radial**2)

            plt.plot(radial, fit, markersize = 0, color = "red", label = "1/r^2")
            plt.plot(radial, efield_cut_x, markersize = 5, marker = "o", color = "blue", label = "x-axis")
            plt.plot(radial, efield_cut_y, markersize = 5, marker = "o", color = "orange", label = "y-axis")
            plt.grid(which="both")
            plt.xlabel(f"Radial distance from charge (space steps)", fontsize = 12)
            plt.ylabel(r"Electric Field E", fontsize = 12)
            plt.title(f"Electric field strength for a point charge", fontsize = 16)
            plt.tick_params(axis = "both", labelsize = 12)
            plt.tight_layout()
            plt.legend()
            plt.savefig(f"Electricfieldrad.png")
            plt.show()

            #Save to CSV file
            data = np.column_stack((radial, efield_cut_x, efield_cut_y))
            np.savetxt("Electricfielddist.csv", data, delimiter=",", header = "Distance, Potential_xaxis, Potential_yaxis")


    def jacobi_algorithm(self):
        """The Jacobi algorithm updates the potential in the numerical Poisson equation,
        getting it closer to solution"""
        #Copy potential lattice to update
        phi_n = self.phi
        phi_n_plus_1 = phi_n.copy()

        phi_left = np.roll(phi_n, shift = 1, axis = 1)
        phi_right = np.roll(phi_n, shift = -1, axis = 1)
        phi_up = np.roll(phi_n, shift = 1, axis = 0)
        phi_down = np.roll(phi_n, shift = -1, axis = 0)
        phi_forward = np.roll(phi_n, shift = 1, axis = 2)
        phi_back = np.roll(phi_n, shift = -1, axis = 2)

        phi_n_plus_1[1:-1, 1:-1, 1:-1] = (phi_left[1:-1, 1:-1, 1:-1] + phi_right[1:-1, 1:-1, 1:-1] 
                                        + phi_up[1:-1, 1:-1, 1:-1] + phi_down[1:-1, 1:-1, 1:-1] 
                                        + phi_forward[1:-1, 1:-1, 1:-1] + phi_back[1:-1, 1:-1, 1:-1]
                                        + self.rho[1:-1, 1:-1, 1:-1]) / 6
        
        
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
    
    def solve_for_potential(self, max_iter = 10000):
        """Runs the Jacobi algorithm until the potential converges
        to a solution which does not change with reiteration more than
        the tolerance. Will fail if solution does not converge within 10000
        iterations"""

        #Run Jacobi algorithm until distance is less than tolerance

        for i in range(max_iter):
            distance = self.jacobi_algorithm()[1]
            if distance < self.tolerance:
                print(f"Potential converged in {i} iterations")
                return self.phi
    
        print(f"Potential did not converge in {max_iter} iterations, \
              reached distance of {distance}")
        #Best estimate if still not converged
        return self.phi
    
    def plot_electric_potential(self):
        """Plot midplane of electric potential that is the solution to the Poisson
        equation as a heatmap. Also plot electric potential as a function of distance
        from the point charge"""

        midplane = self.phi.shape[2] // 2
        phi_midplane = self.phi[:, :, midplane]
        #The potential should be symmetric for a point charge, so this shouldn't matter

        plt.imshow(phi_midplane, cmap = "plasma")
        plt.colorbar(label = r"Electric potential ($\phi$)")
        plt.title(f"Electric potential steady state", fontsize = 16)
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
            plt.title(f"Electric potential for a point charge", fontsize = 16)
            plt.tick_params(axis = "both", labelsize = 12)
            plt.tight_layout()
            plt.legend()
            plt.savefig(f"Electricpotentialrad.png")
            plt.show()

            #Save to CSV file
            data = np.column_stack((radial, potential_cut))
            np.savetxt("Electricpotentialdist.csv", data, delimiter=",", header = "Distance, Potential")


def main():
    c = Poisson()
    c.solve_for_potential()
    c.plot_electric_potential()
    E = c.compute_electric_field()
    c.plot_electric_field(E)

main()
        



