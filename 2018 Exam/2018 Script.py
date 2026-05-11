import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation as animate
import argparse
import time

class Fisher:
    """Determine the numerical solution for the Fisher equation"""

    def __init__(self, R, k, time_step, size):

        #Lattice of phi=1 within radius R around centre of grid and phi=0 at points of radius >R

        #Uncomment for part a)
        """xcoord_lattice, ycoord_lattice = np.meshgrid(np.arange(0, size), np.arange(0, size))
        lattice_centre = np.full((size,size), size // 2)
        distance_to_centre = np.sqrt((xcoord_lattice - lattice_centre)**2 + (ycoord_lattice - lattice_centre)**2)
        self.lattice = (distance_to_centre < R).astype(int)"""

        #Uncomment for part c)
        """
        coord_lattice = np.arange(0, size, 1)
        self.lattice = (coord_lattice < size / 10).astype(int)
        """

        #Uncomment for part d)
        coord_lattice = np.arange(0, size, 1)
        self.lattice = np.exp(-k*coord_lattice)

        self.k = k
        self.R = R
        self.size = size
        self.space_step = 0.1
        self.time_step = time_step
        self.alpha = 1
        self.D = 1
    
    def solve_fisher_eqn_pt_a(self):
        """Numerically solve the Fisher equation for phi for the case of a N x N lattice with 
        boundary conditions stipulated in part a)."""

        phi_n = self.lattice.copy() 

        #Roll phi to get nearest neighbours
        phi_left = np.roll(phi_n, shift = 1, axis = 1)
        phi_right = np.roll(phi_n, shift = -1, axis = 1)
        phi_up = np.roll(phi_n, shift = 1, axis = 0)
        phi_down = np.roll(phi_n, shift = -1, axis = 0)

        phi_n_plus_1 = (phi_n + self.D * (self.time_step / self.space_step**2) * (phi_left + phi_right + phi_up + phi_down - 4*phi_n)
                        + (self.alpha * self.time_step * phi_n * (1 - phi_n)))

        self.lattice = phi_n_plus_1

        return self.lattice
    
    def solve_fisher_eqn_pt_c(self):
        """Numerically solve the Fisher equation for phi for the case of the N x 1 lattice with
        boundary conditions stipulated in part c). This function is also reusable for part d)"""

        phi_n = self.lattice.copy()

        phi_left = np.roll(phi_n, shift = 1, axis = 0)
        phi_right = np.roll(phi_n, shift = -1, axis = 0)

        phi_n_plus_1 = (phi_n + self.D * (self.time_step / self.space_step**2) * (phi_left + phi_right - 2*phi_n)
                        + (self.alpha * self.time_step * phi_n * (1 - phi_n)))
        
        #Enforce boundary conditions
        phi_n_plus_1[0] = 1
        phi_n_plus_1[-1] = phi_n_plus_1[-2]

        self.lattice = phi_n_plus_1

        return self.lattice

    def integrate_phi_for_part_c(self, iter_per_meas = 100, no_meas = 100):
        """Compute the integrated value for phi across the 1D lattice over time and plot the
        results"""

        time_data = np.arange(0, no_meas)
        phi_data = np.zeros(len(time_data))

        for i in range(no_meas):
            for j in range(iter_per_meas):
                phi = self.solve_fisher_eqn_pt_c()

            integ_phi = np.sum(phi)
            phi_data[i] = integ_phi

        plt.plot(time_data, phi_data, marker = "o", color = "orange")
        plt.xlabel("Time (arb. units)", fontsize = 12)
        plt.ylabel(f"$\\int\\phi dx$", fontsize = 12)
        plt.title(f"Time evolution of total $\\phi$", fontsize = 16)
        plt.tight_layout()
        plt.savefig("Phiintegpartc.png")
        plt.show()

        """Numerically calculate the speed of the increase of phi"""
        #Time is in arbitrary units - set == 1
        velocity = np.roll(phi_data, shift = -1) - phi_data

        #Remove final data point as np.roll results in a wild speed
        plt.plot(time_data[:-1], velocity[:-1], marker = "o", color = "orange")
        plt.xlabel("Time (arb. units)", fontsize = 12)
        plt.ylabel(f"Speed (arb. units)", fontsize = 12)
        plt.title(f"Time evolution of speed of integral change", fontsize = 16)
        plt.tight_layout()
        plt.savefig("Integral_speed_partc.png")
        plt.show()

    def integrate_phi_for_part_d(self, iter_per_meas = 100, no_meas = 100):
        """Compute the integrated value for phi across the 1D lattice over time and plot the
        results"""

        time_data = np.arange(0, no_meas)
        phi_data = np.zeros(len(time_data))

        for i in range(no_meas):
            for j in range(iter_per_meas):
                phi = self.solve_fisher_eqn_pt_c()

            integ_phi = np.sum(phi)
            phi_data[i] = integ_phi

        plt.plot(time_data, phi_data, marker = "o", color = "orange")
        plt.xlabel("Time (arb. units)", fontsize = 12)
        plt.ylabel(f"$\\int\\phi dx$", fontsize = 12)
        plt.title(f"Time evolution of total $\\phi$", fontsize = 16)
        plt.tight_layout()
        plt.savefig("Phi_integral_part_d.png")
        plt.show()



    def animate_lattice(self, number_of_frames = 10000, interval = 50):
        """Animates evolution of system over time"""

        iter_per_frame = self.size
        fig, ax = plt.subplots()
        figure = ax.imshow(self.lattice, cmap = "viridis")
        fig.colorbar(figure, ax=ax)
        ax.set_title(f"Time evolution of diffusion of chemical \n as per the Fisher equation")
        
        def update(frame): #Needs frame arg as animate passes current frame no. to update()

            """Function to update figure for FuncAnimation."""
            for step in range(iter_per_frame):
                self.solve_fisher_eqn_pt_a()

            figure.set_data(self.lattice)
            figure.set_clim(vmin=1, vmax=-1)

            return [figure]
        
        animation = animate(fig, update, frames = number_of_frames, interval = interval, blit = True)

        plt.show()
        return animation
    
class CahnHilliard:
    """Determine the numerical solution for the Cahn-Hilliard equation"""

    def __init__(self, phi_0, time_step, size = 100):

        #Lattice of phi centred around phi_0 with small random noise
        self.lattice = np.random.normal(phi_0, scale = 0.01, size = (size,size))
        self.size = size
        self.space_step = 1
        self.time_step = time_step
        self.a = 0.1
        self.k = 0.1
        self.M = 0.1
        self.alpha = 0.0003
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
        
        mu = self.a * phi * (phi ** 2 - 1) * (phi ** 2 - 2) - (self.k / self.space_step ** 2) * (phi_right + phi_left + phi_up + phi_down - 4*phi)
        
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

        phi_n_plus_1 = (phi_n + self.M * (self.time_step / self.space_step**2) * (mu_left + mu_right + mu_up + mu_down - 4*mu) 
                        + self.alpha * self.time_step * phi_n * (1 - phi_n))

        self.lattice = phi_n_plus_1

        return self.lattice
    
    def animate_lattice(self, number_of_frames = 10000, interval = 50):
        """Animates evolution of system over time"""

        iter_per_frame = self.size
        fig, ax = plt.subplots()
        figure = ax.imshow(self.lattice, cmap = "viridis")
        fig.colorbar(figure, ax=ax)
        ax.set_title(f"Time evolution of phase seperated system")
        
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

def main():
    parser = argparse.ArgumentParser(description="Solve the Cahn-Hilliard Eq. numerically")
    parser.add_argument("size", type = int)
    parser.add_argument("--R", type = float, default = 10)
    parser.add_argument("time_step", type = float)
    parser.add_argument("--IID_solve", type=str, choices=["Y", "N"], default="N")
    parser.add_argument("--ID_solve", type=str, choices=["Y", "N"], default="N")
    parser.add_argument("--ID_phi_behav", type=str, choices = ["None", "exp", "step"], default="None")
    parser.add_argument("--CH", type=str, choices=["Y", "N"], default="N")
    parser.add_argument("--k", type=float, default=0)
    parser.add_argument("--phi_0", type=float, default=1)

    args = parser.parse_args()

    time_0 = time.time()

    fe = Fisher(size = args.size, R = args.R, k = args.k, time_step=args.time_step)

    if args.IID_solve == "Y":
        fe.animate_lattice()

    if args.ID_solve == "Y":
        if args.ID_phi_behav == "step":
            fe.integrate_phi_for_part_c()
        elif args.ID_phi_behav == "exp":
            fe.integrate_phi_for_part_d()

    ch = CahnHilliard(phi_0=args.phi_0, time_step=args.time_step, size=args.size)

    if args.CH == "Y":
        ch.animate_lattice()

    time_1 = time.time()

    print(f"Complete in {(time_1 - time_0)/60} minutes")

if __name__ == "__main__":
    main()









