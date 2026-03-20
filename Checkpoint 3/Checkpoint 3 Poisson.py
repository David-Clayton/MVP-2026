import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation as animate
import argparse
import time
import random
from numba import njit
from joblib import Parallel, delayed

class Poisson:
    """Solves the Poisson equation."""

    def __init__(self):
        """?"""
        self.space_step = 1
    
    def compute_electric_field(self, phi):
        """Numerically compute the vector E-field as the negative of the
        numerically discretised gradient of the electric potential (phi)"""
        phi_left = np.roll(phi, shift = 1, axis = 2)
        phi_up = np.roll(phi, shift = 1, axis = 1)
        phi_forward = np.roll(phi, shift = 1, axis = 0)
        d_phi_d_x = ((phi_left - phi) / self.space_step)
        d_phi_d_y = ((phi_up - phi) / self.space_step)
        d_phi_d_z = ((phi_forward - phi) / self.space_step)

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

        plt.quiver(E_x_midplane, E_y_midplane)
        plt.title(f"Discretised Electric field", fontsize = 12)
        plt.show()

        plt.imshow(E_size)
        plt.show()

    def jacobi_algorithm(self):
        """The Jacobi algorithm solves the Poisson equation for phi """

def main():
    phi = np.random.random((50,50,50))
    c = Poisson()
    E = c.compute_electric_field(phi)
    c.plot_electric_field(E)

main()
        



