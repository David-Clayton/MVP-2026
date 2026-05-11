import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation as animate
import argparse
import time
from numba import njit

class CahnHilliard:
    """Determine the numerical solution for the Cahn-Hilliard equation"""

    def __init__(self, phi_0, v_0, space_step, time_step, size = 100):

        #Lattice of phi centred around phi_0 with small random noise
        self.lattice = np.random.uniform(low = phi_0 - 0.1, high = phi_0 + 0.1, size = (size,size))
        self.size = size
        self.space_step = space_step
        self.time_step = time_step
        self.a = 0.1
        self.M = 0.1
        self.q_0 = 0.5
        self.phi_0 = phi_0
        self.v_0 = v_0
    
    def laplacian(self, grid):
        """Calculate the Laplacian of a 2D array"""
        grid_left = np.roll(grid, shift = 1, axis = 1)
        grid_right = np.roll(grid, shift = -1, axis = 1)
        grid_up = np.roll(grid, shift = 1, axis = 0)
        grid_down = np.roll(grid, shift = -1, axis = 0)

        laplacian = 1 / (self.space_step ** 2) * (grid_down + grid_up + grid_left + grid_right - 4*grid)

        return laplacian

    def calc_chem_pot(self):
        """Calculate the (dimensionless) chemical potential (mu) across the lattice
        using the discretised solution"""
       
        #Spatial nearest neighbours of phi for laplacian
        phi = self.lattice

        laplacian_term = 2 * self.q_0 * self.laplacian(phi)
        double_laplacian_term = self.laplacian(self.laplacian(phi))

        mu = -self.a * phi + phi ** 3 + (self.q_0 ** 4) * phi + laplacian_term + double_laplacian_term

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

        phi_n_plus_1 = phi_n + self.M * (self.time_step / self.space_step**2) * (mu_left + mu_right + mu_up + mu_down - 4*mu)

        self.lattice = phi_n_plus_1

        return self.lattice
    
    def calc_order_param_w_advection(self):
        """Update the order parameter (phi) from timestep n to timestep n+1 with the discretised
        solution, including a spatially dependent velocity term"""

        phi_n = self.lattice.copy()
        phi_left = np.roll(phi_n, shift = 1, axis = 1)
        phi_right = np.roll(phi_n, shift = -1, axis = 1)
        phi_up = np.roll(phi_n, shift = 1, axis = 0)
        phi_down = np.roll(phi_n, shift = -1, axis = 0)

        mu = self.calc_chem_pot()
        mu_left = np.roll(mu, shift = 1, axis = 1)
        mu_right = np.roll(mu, shift = -1, axis = 1)
        mu_up = np.roll(mu, shift = 1, axis = 0)
        mu_down = np.roll(mu, shift = -1, axis = 0)

        #sin function as an array
        counting = np.arange(0, self.size, 1)
        #1d sinusoidal array
        one_d_sin = -self.v_0 * np.sin(2*np.pi*counting/self.size)
        #Expand 1d array to 2d along x-axis
        two_d_sin = np.tile(one_d_sin, (self.size, 1))

        phi_n_plus_1 = phi_n - ((self.time_step / self.space_step) * two_d_sin * (phi_down-phi_n)) + self.M * (self.time_step / self.space_step**2) * (mu_left + mu_right + mu_up + mu_down - 4*mu)

        self.lattice = phi_n_plus_1

        return self.lattice
    
    def animate_lattice(self, number_of_frames = 10000, interval = 50, v_field = False):
        """Animates evolution of system over time"""

        iter_per_frame = self.size
        fig, ax = plt.subplots()
        figure = ax.imshow(self.lattice, cmap = "viridis")
        fig.colorbar(figure, ax=ax)
        ax.set_title(f"Time evolution of phase-separated system \n with $\\phi_0$ = {self.phi_0} and $v_0$ = {self.v_0}")
        
        def update(frame): #Needs frame arg as animate passes current frame no. to update()

            """Function to update figure for FuncAnimation."""
            for step in range(iter_per_frame):
                if v_field == False:
                    self.calc_order_param()
                else:
                    self.calc_order_param_w_advection()

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

    def calc_spatial_variance(self):
        """Calculate the spatial variance of the lattice at a single
        time step"""
        phi = self.lattice

        var = np.mean(phi**2) - (np.mean(phi))**2

        return var
    
    def calc_var_convergence(self, no_meas = 50, tolerance = 0.001):
        """Run the time evolution of the lattice until a
        steady state of the variance is reached. Run for 1000 steps,
        checking for convergence in the variance. If the steady state is 
        reached early, break"""

        iter_per_step = 5000

        var_data = []
        
        for i in range(no_meas):
            for j in range(iter_per_step):
                self.calc_order_param()
            var = self.calc_spatial_variance()
            var_data.append(var)

            if i % 1 == 0:
                print(f"Step {i} complete in convergence test, var_{i} = {var}") 

            if i > 0:
                if abs(var_data[i] - var_data[i-1]) < tolerance:
                    print(f"Variance converged at step {i}")
                    break

        else:
            print(f"Variance did not converge in {no_meas} steps.")
        
        return var_data[i]

    def plot_var_phi_0_relation(self):
        """Calculate and plot the variation of the steady state 
        variance with phi_0"""

        phi_0_range = np.arange(0, 0.2625, 0.0125)
        equilib_var_data = np.zeros(len(phi_0_range))

        for i, phi_0 in enumerate(phi_0_range):
            #Reinitialise lattice with new phi_0
            self.lattice = np.random.uniform(low = phi_0 - 0.1, high = phi_0 + 0.1, size = (self.size,self.size))
            equilib_var_data[i] = self.calc_var_convergence()
            print(f"Variance plot complete for phi_0 = {phi_0}")

        plt.plot(phi_0_range, equilib_var_data, marker = "o", color = "purple")
        plt.xlabel(f"$\\phi_0$", fontsize = 12)
        plt.ylabel(f"Steady-state Var($\\phi$)", fontsize = 12)
        plt.tick_params(axis="both", labelsize = 12)
        plt.tight_layout()
        plt.savefig(f"Steadystatevariance.png")
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Solve the Cahn-Hilliard Eq. numerically")
    parser.add_argument("size", type = int, default = 100)
    parser.add_argument("phi_0", type = float)
    parser.add_argument("v_0", type = float)
    parser.add_argument("space_step", type = float, default = 1.0)
    parser.add_argument("time_step", type = float, default = 0.02)
    parser.add_argument("--var_plot", type=str, choices=["Y", "N"], default="N")
    parser.add_argument("--velo_field", type = str, choices= ["Y", "N"], default = "Y")

    args = parser.parse_args()

    time_0 = time.time()

    ch = CahnHilliard(size = args.size, phi_0=args.phi_0, v_0 = args.v_0,
                      space_step=args.space_step, time_step=args.time_step)

    if args.velo_field == "Y":
        ch.animate_lattice(v_field = True)
    else:
        ch.animate_lattice()

    if args.var_plot == "Y":
        ch.plot_var_phi_0_relation()

    time_1 = time.time()

    print(f"Complete in {(time_1 - time_0)/60} minutes")

if __name__ == "__main__":
    main()









