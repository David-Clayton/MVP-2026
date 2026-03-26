import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation as animate
import argparse
import time
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
        self.phi_0 = phi_0

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

        iter_per_frame = self.size
        fig, ax = plt.subplots()
        figure = ax.imshow(self.lattice, cmap = "viridis")
        fig.colorbar(figure, ax=ax)
        ax.set_title(f"Evolution of phase seperated system (Working title)")
        
        def update(frame): #Needs frame arg as animate passes current frame no. to update()

            """Function to update figure for FuncAnimation."""
            for step in range(iter_per_frame):
                self.calc_order_param()

            figure.set_data(self.lattice)
            figure.set_clim(vmin=1, vmax=-1)

            return [figure]
        
        animation = animate(fig, update, frames = number_of_frames, interval = interval, blit = True)

        plt.show()
        return animation

    def calculate_free_energy(self):
        """Calculate the free energy density (f) of the system at a
        single time step at order parameter phi"""

        #Neighbours of each phi lattice point
        phi = self.lattice
        phi_left = np.roll(self.lattice, shift = 1, axis = 1)
        phi_right = np.roll(self.lattice, shift = -1, axis = 1)
        phi_up = np.roll(self.lattice, shift = 1, axis = 0)
        phi_down = np.roll(self.lattice, shift = -1, axis = 0)

        #Calculate f with discretised free energy equation
        f = -0.5*(phi**2) + 0.25*(phi**4) + ((self.parameter/2)*(phi_left**2 + phi_down**2 + 2*phi**2 - 2*phi_left*phi - 2*phi_down*phi)/self.space_step**2)

        return f
    
    def free_energy_plot(self, phi_0, no_meas = 2000, iter_per_meas = 1000):
        """Plot the evolution of free energy of the system over time"""
        #Array of timesteps
        time_data = self.time_step * np.linspace(0, no_meas*iter_per_meas, no_meas)
        #Initialise empty array for free energy data
        f_data = np.zeros(no_meas)

        for i in range(no_meas):
            for j in range(iter_per_meas):
                self.calc_order_param()
            f = self.calculate_free_energy()
            f_data[i] = np.mean(f)
            if i % 100 == 0:
                print(f"Free energy computed for measurement {i}/{no_meas}.")

        plt.plot(time_data, f_data, markersize = 2, marker = "o", color = "orange")
        plt.grid(which = "both")
        plt.xlabel(r"Time", fontsize = 16)
        plt.ylabel(r"Mean free energy density (f)", fontsize = 16)
        plt.title(f"Time evolution of free energy density for $\\phi_0$ = {self.phi_0}", fontsize = 12)
        plt.tick_params(axis = "both", labelsize = 12)
        plt.tight_layout()
        plt.savefig(f"Free_energy_{phi_0}.png")
        plt.show()

        data = np.column_stack((time_data, f_data))
        np.savetxt(f"Free_energy_{phi_0}.csv", data, delimiter=",", header = "time, f")

def main():
    parser = argparse.ArgumentParser(description="Solve the Cahn-Hilliard Eq. numerically")
    parser.add_argument("size", type = int, default = 100)
    parser.add_argument("phi_0", type = float)
    parser.add_argument("parameter", type = float, default=1.0)
    parser.add_argument("space_step", type = float, default = 1.0)
    parser.add_argument("time_step", type = float, default = 0.02)
    parser.add_argument("--free_energy", type=str, choices=["Y", "N"], default="N")

    args = parser.parse_args()

    time_0 = time.time()

    ch = CahnHilliard(size = args.size, phi_0=args.phi_0, parameter=args.parameter, 
                      space_step=args.space_step, time_step=args.time_step)

    ch.animate_lattice()

    if args.free_energy == "Y":
        ch.free_energy_plot(phi_0=args.phi_0)

    time_1 = time.time()

    print(f"Complete in {(time_1 - time_0)/60} minutes")

if __name__ == "__main__":
    main()









