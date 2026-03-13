import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation as animate
import argparse
import time
import random
from numba import njit
from joblib import Parallel, delayed

class CahnHilliard:
    """Determine the numerical solution for the Cahn-Hilliard equation"""

    def __init__(self, phi_0, parameter, space_step, time_step, size = 100):

        #Lattice of phi centred around phi_0 with small random noise
        self.lattice = np.random.normal(phi_0, scale = 0.01, size = (size,size))
        self.size = size
        self.space_step = space_step
        self.time_step = time_step
        self.parameter = parameter

    def calc_chem_pot(self):
        """Calculate the (dimensionless) chemical potential (mu) across the lattice
        using the discretised solution"""
       
        #Spatial nearest neighbours of phi for laplacian
        phi = self.lattice
        phi_left = np.roll(self.lattice, shift = 1, axis = 1)
        phi_right = np.roll(self.lattice, shift = -1, axis = 1)
        phi_up = np.roll(self.lattice, shift = 1, axis = 0)
        phi_down = np.roll(self.lattice, shift = -1, axis = 0)
        
        mu = phi * (phi ** 2 - 1) - (self.parameter / self.space_step ** 2) * (phi_right + phi_left + phi_up + phi_down - 4*phi)
        
        return mu
    
    def calc_order_param(self):
        """Update the order parameter (phi) from timestep n to timestep n+1 with the discretised
        solution"""
        phi_n = self.lattice.copy()

        mu = self.calc_chem_pot()
        mu_left = np.roll(mu, shift = 1, axis = 1)
        mu_right = np.roll(mu, shift = -1, axis = 1)
        mu_up = np.roll(mu, shift = 1, axis = 0)
        mu_down = np.roll(mu, shift = -1, axis = 0)

        phi_n_plus_1 = phi_n + (self.time_step / self.space_step**2) * (mu_left + mu_right + mu_up + mu_down - 4*mu)

        self.lattice = phi_n_plus_1

        return self.lattice
    
    def animate_lattice(self, number_of_frames = 10000, interval = 50):
        """Animates evolution of system over time"""

        iter_per_frame = self.size ** 2
        fig, ax = plt.subplots()
        figure = ax.imshow(self.lattice, cmap = "viridis")
        fig.colorbar(figure, ax=ax)
        ax.set_title(f"Evolution of phase seperated system (Working title)")
        
        def update(frame): #Needs frame arg as animate passes current frame no. to update()

            """Function to update figure for FuncAnimation."""
            for step in range(iter_per_frame):
                self.calc_order_param()

            figure.set_data(self.lattice)

            return [figure]
        
        animation = animate(fig, update, frames = number_of_frames, interval = interval, blit = True)

        plt.show()
        return animation


def main():
    c = CahnHilliard(phi_0=0, size=100, parameter=1, space_step=1, time_step=0.01)
    c.animate_lattice()

main()









