import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation as animate
import argparse 
import time 
from numba import njit

@njit
def glauber_dynamics_numba(kT, mu, lattice, size):
    """Numba-suitable Glauber dynamics function. See docstring for glauber_dynamics
    in class for details."""

    i = np.random.randint(0, size)
    j = np.random.randint(0, size)

    s_i = lattice[i, j] #Inital spin
        
    nearest_neighbours = np.array([lattice[(i+1) % size, j] , lattice[(i-1) % size, j],
                                       lattice[i, (j+1) % size] , lattice[i, (j-1) % size]]) 
        
    #Energy change equation = 2 * sum of multiplied NN spins

    energy_change = 2 * np.sum(s_i * nearest_neighbours)

    #Change in N due to change in lattice point
    if s_i == 0:
        N_change = 1
    else:
        N_change = -1

    #Combined system energy change with chemical potential change
    combined_change = energy_change - mu*N_change

    #Accept new state conditional on energy_change

    boltzmann_factor = np.exp(-(combined_change) / kT)
    if combined_change <= 0 and s_i == 0:
        lattice[i,j] = 1
    elif combined_change <= 0 and s_i == 1:
        lattice[i,j] = 0
    elif combined_change > 0 and s_i == 0: 
        r = np.random.random()
        if r < boltzmann_factor:
            lattice[i,j] = 1
    elif combined_change > 0 and s_i == 1: 
        r = np.random.random()
        if r < boltzmann_factor:
            lattice[i,j] = 0
        
    return lattice


class IsingModel:

    """This class creates a 2D simulation of a fluid at temperature T based off an Ising model in contact with
    a particle reservoir of chemical potential mu."""

    def __init__(self, kT, mu, size):

        """Initialise lattice. The distribution of spins in the lattice is based off a random number generator. Set J = 1"""

        #Replace spins S = {+1,-1} with occupation nos n = {0,1}
        self.lattice = np.random.choice(a = [0, 1], size = (size,size))
        self.kT = kT
        self.mu = mu
        self.size = size
        self.N = np.sum(self.lattice)
    
    def compute_total_N(self):

        """Calculate the total number of particles in the lattice.
        
        Inputs: None
        Outputs: total lattice magnetisation"""

        total_magnetisation = np.sum(self.lattice)

        return total_magnetisation
    
    def glauber_dynamics(self, kT, mu):

        """Take inital lattice, swap a spin on a random point, calculate the
        energy change of that swap, and use Metropolis algorithm to determine
        whether lattice is altered.
        
        Inputs: none
        Outputs: self.lattice (possibly changed)"""

        self.lattice = glauber_dynamics_numba(kT, mu, self.lattice, self.size)
        
    def animate_lattice(self, number_of_frames = 500, interval = 50):
        """Animates lattice evolution over time
        
        Inputs: 
                Number of frames in animation
                Interval between frames (ms)
        Outputs: Shows and returns animation"""
        
        #Number of iterations of algorithm per frame (1 frame per sweep)
        iter_per_frame = self.size ** 2 
        fig, ax = plt.subplots()
        figure = ax.imshow(self.lattice, cmap = "viridis")
        ax.set_title(f"Ising model using Glauber dynamics at kT = {self.kT} \n and mu = {self.mu} for a {self.size}x{self.size} lattice.")

        def update(frame): #Needs frame arg as animate passes current frame no. to update()

            """Function to update figure for FuncAnimation."""
            for step in range(iter_per_frame):
                    self.glauber_dynamics(self.kT, self.mu)

            figure.set_data(self.lattice)

            return [figure]
        
        animation = animate(fig, update, frames = number_of_frames, interval = interval, blit = True)

        plt.show()
        return animation
    
    def equilibriate(self, mu, kT, no_sweeps = 100):
        """Run equilibriation of the lattice before any measurements are taken so the lattice
        loses memory of its initial conditions.
        
        Inputs: Dynamical model (default Glauber)
                Number of sweeps before equilibriation (default 100)
        Outputs: self.lattice (equilibriated)"""

        iter_per_sweep = self.size ** 2
        for i in range(no_sweeps):
            for j in range(iter_per_sweep):
                    self.glauber_dynamics(kT, mu)
        #Return equilibriated lattice
        return self.lattice
        
    
    def compute_average_N_and_density(self, plot = True):
        """Calculates the average of N, the average of the N^2, and
        the density from measurements from a large
        number of sweeps of the system. Does this over a range of mu = -2.5 -> -1.5 
        in increments of 0.1, keeping kT = 0.4
        
        Inputs: Whether to plot density (default True)
        Outputs: Range of mu, arrays of: density, averaged
        N, and averaged square N over mu range. """

        #First, equilibriate system before measurements with 5000 sweeps at minimum chemical potential
        self.equilibriate(no_sweeps = 2000, mu = -2.5)
        print(f"Initial equilibriation complete")
        #Temperature range
        mu_data = np.arange(-2.5, -1.4, 0.1) 

        iter_per_sweep = self.size ** 2
        #Number of sweeps to be done at each temperature for measurements
        number_of_sweeps = 5000 
        #Number of sweeps to be done inbetween measurements
        sweeps_bw_meas = 10
        #Empty arrays for average data as function of T
        rho = np.zeros(len(mu_data))
        N_avg = np.zeros(len(mu_data))
        N_sq_avg = np.zeros(len(mu_data))
        #Empty array for error calculation
        #First column will be mu data, each row afterwards will be N data at each mu
        N_error = np.zeros((len(mu_data) , number_of_sweeps//sweeps_bw_meas + 1))
        mu_index = 0
        for mu in mu_data: 
            #Empty array for N data
            N = np.zeros(number_of_sweeps//sweeps_bw_meas)
            data_index = 0
            #Run for necessary number of sweeps
            for i in range(number_of_sweeps):
                #Run dynamics algorithm for 1 sweep
                for j in range(iter_per_sweep):
                    self.glauber_dynamics(kT = 1.0, mu = mu)
                #Only take measurement every 10th sweep to avoid correlation between measurements
                if i % sweeps_bw_meas == 0:
                    N_meas = self.compute_total_N()
                    #Add measurements to array
                    N[data_index] = N_meas
                    data_index += 1
                if i % 500 == 0:
                    print(f"Sweep {i} completed at mu = {mu}")

            #Add the average of these sets of measurements of this temp to the final data
            rho[mu_index] = (np.mean(N)/self.size**2)
            N_avg[mu_index] = (np.mean(N))
            N_sq_avg[mu_index] = (np.mean(N**2))

            #Append data to error array
            N_error[mu_index, 0] = mu
            N_error[mu_index, 1:] = N

            mu_index += 1
            #Re-equilibriate with 100 sweeps after measurements taken before system passed to next T
            self.equilibriate(mu = mu)
            

        if plot == True:
            plt.plot(mu_data, rho, marker = "o", color = "g")
            plt.grid(which = "both")
            plt.xlabel(f"$\\mu$/K", fontsize = 16)
            plt.ylabel(f"Density ($\\rho$)", fontsize = 16)
            plt.title(f"Density vs. chemical potential for a {self.size}x{self.size} \n lattice at kT = 1.0", fontsize = 12)
            plt.tight_layout()
            plt.savefig(f"Partfdensity")
            plt.show()

        #Write data to csv
        data = np.column_stack((mu_data, rho))
        np.savetxt("Densitydatapartf.csv", data, delimiter=",", header = "mu, rho")

        
        return mu_data, rho, N_avg, N_sq_avg, N_error


    def compute_isothermal_compressability(self, mu_data, N_avg, N_sq_avg):

        """Calculates the magnetic susceptibility (chi) of the system in the range of 
        temperatures kT = 1 -> 3 in increments of 0.1.

        Inputs: Range of kT, average magnetisation over T range, average square
        of magnetisation over T range 
        Outputs: Susceptibility over T range
        """
        kT = 1.0
        #Compute magnetic susceptibility
        kappa_data = (N_sq_avg - N_avg ** 2) / (kT*mu_data)

        return kappa_data
    
    def compute_compressibility_errors(self, array, k = 1000):
        """Calculate the errors on susceptibility using the bootstrap method.
        
        Inputs: 2D Array of magnetisations obtained from measurements where each column is the
        set of values measured at each temperature in the range. 
        Number of iterations of bootstrap algorithm (default = 250)
        
        Outputs: Errors on chi over T range"""
    
        kappa_error_mu = np.zeros(np.size(array, axis = 0))
        kappa_index = 0
        for i in range(np.size(array, axis = 0)): #For each temperature. Make sure kT is along this axis
            kappa_data = np.zeros(k)
            kappa_data_index = 0
            for j in range(k):
                #Resample measurements at each kT
                resample = np.random.choice(array[i,1:], size = len(array[i,1:]))
                #Get mean and mean of squares of M to calculate values
                avg_N = np.mean(resample)
                avg_sq_N = np.mean(resample**2)
                #Get temperature
                mu = array[i,0] 
                #Get susceptibility
                kappa = self.compute_isothermal_compressability(mu, avg_N, avg_sq_N)
                #Append to set of calculated chis
                kappa_data[kappa_data_index] = kappa
                kappa_data_index += 1
            
            #Calculate overall error on chi for each T
            kappa_error = np.sqrt(np.mean(kappa_data ** 2) - (np.mean(kappa_data))**2)
            kappa_error_mu[kappa_index] = (kappa_error)
            kappa_index += 1
            print(f"Errors for mu = {array[i,0]} done")
        
        return kappa_error_mu
    
    def plot_compressibility(self, mu_data, N_avg, N_sq_avg, errors, plot = True):
        """Plots the magnetic susceptibility against kT with error bars
        
        Inputs: kT range, averaged magnetisation over T range, average square of 
        magnetisation over T range, array to calculate susceptibility errors. Whether to
        plot (default = True)
        Outputs: Susceptibilty data over T range and errors over T range"""

        kappa_data = self.compute_isothermal_compressability(mu_data, N_avg, N_sq_avg)
        kappa_errors = self.compute_compressibility_errors(errors)
        if plot == True:
            plt.errorbar(mu_data, kappa_data, yerr= kappa_errors, marker = "o", color = "purple")
            plt.grid(which = "both")
            plt.xlabel(f"$\\mu$/J", fontsize = 16)
            plt.ylabel(f"Variance $\\kappa$", fontsize = 16)
            plt.title(f"Compressiblity vs. chemical potential for a {self.size}x{self.size} \n lattice with kT = 1.0", fontsize = 12)
            plt.tight_layout()
            plt.savefig(f"Partfvariance")
            plt.show()

        data = np.column_stack((mu_data, kappa_data, kappa_errors))
        np.savetxt("Compressibilitydatapartf.csv", data, delimiter=",", header = "mu, kappa, kappa_error")

    def compute_average_N_and_density_with_T(self, plot = True):
        """Calculates the average of N, the average of the N^2, and
        the density from measurements from a large
        number of sweeps of the system. Does this over a range of mu = -2.5 -> -1.5 
        in increments of 0.1, keeping kT = 0.4
        
        Inputs: Whether to plot density (default True)
        Outputs: Range of mu, arrays of: density, averaged
        N, and averaged square N over mu range. """

        #First, equilibriate system before measurements with 5000 sweeps at minimum chemical potential
        self.equilibriate(no_sweeps = 2000, kT = 0.75, mu = -2)
        print(f"Initial equilibriation complete")
        #Temperature range
        kT_data = np.linspace(0.75, 0.4, 20) 
    
        iter_per_sweep = self.size ** 2
        #Number of sweeps to be done at each temperature for measurements
        number_of_sweeps = 5000 
        #Number of sweeps to be done inbetween measurements
        sweeps_bw_meas = 10
        #Empty arrays for average data as function of T
        rho = np.zeros(len(kT_data))
        N_avg = np.zeros(len(kT_data))
        N_sq_avg = np.zeros(len(kT_data))
        #Empty array for error calculation
        #First column will be mu data, each row afterwards will be N data at each mu
        N_error = np.zeros((len(kT_data) , number_of_sweeps//sweeps_bw_meas + 1))
        kT_index = 0
        for kT in kT_data: 
            #Empty array for N data
            N = np.zeros(number_of_sweeps//sweeps_bw_meas)
            data_index = 0
            #Run for necessary number of sweeps
            for i in range(number_of_sweeps):
                #Run dynamics algorithm for 1 sweep
                for j in range(iter_per_sweep):
                    self.glauber_dynamics(kT = kT, mu = -2.0)
                #Only take measurement every 10th sweep to avoid correlation between measurements
                if i % sweeps_bw_meas == 0:
                    N_meas = self.compute_total_N()
                    #Add measurements to array
                    N[data_index] = N_meas
                    data_index += 1
                if i % 500 == 0:
                    print(f"Sweep {i} completed at kT = {kT}")

            #Add the average of these sets of measurements of this temp to the final data
            rho[kT_index] = (np.mean(N)/self.size**2)
            N_avg[kT_index] = (np.mean(N))
            N_sq_avg[kT_index] = (np.mean(N**2))

            #Append data to error array
            N_error[kT_index, 0] = kT
            N_error[kT_index, 1:] = N

            kT_index += 1
            #Re-equilibriate with 100 sweeps after measurements taken before system passed to next T
            self.equilibriate(mu = -2.0, kT = kT)
            
        #Flip arrays so kT increases
        kT_data = np.flip(kT_data)
        N_avg = np.flip(N_avg)
        rho = np.flip(rho)
        N_sq_avg = np.flip(N_sq_avg)
        N_error = np.flip(N_error, axis=0)

        if plot == True:
            plt.plot(kT_data, rho, marker = "o", color = "g")
            plt.grid(which = "both")
            plt.xlabel(f"kT/K", fontsize = 16)
            plt.ylabel(f"Density ($\\rho$)", fontsize = 16)
            plt.title(f"Density vs. temperature for a {self.size}x{self.size} \n lattice at mu = -2.0", fontsize = 12)
            plt.tight_layout()
            plt.savefig(f"Partgdensity")
            plt.show()

        #Write data to csv
        data = np.column_stack((kT_data, rho))
        np.savetxt("Densitydatapartg.csv", data, delimiter=",", header = "kT, rho")

        
        return kT_data, rho, N_avg, N_sq_avg, N_error


    def compute_isothermal_compressability_with_T(self, kT_data, N_avg, N_sq_avg):

        """Calculates the magnetic susceptibility (chi) of the system in the range of 
        temperatures kT = 1 -> 3 in increments of 0.1.

        Inputs: Range of kT, average magnetisation over T range, average square
        of magnetisation over T range 
        Outputs: Susceptibility over T range
        """
        mu = 1.0
        #Compute magnetic susceptibility
        kappa_data = (N_sq_avg - N_avg ** 2) / (mu*kT_data)

        return kappa_data
    
    def compute_compressibility_errors_with_T(self, array, k = 1000):
        """Calculate the errors on susceptibility using the bootstrap method.
        
        Inputs: 2D Array of magnetisations obtained from measurements where each column is the
        set of values measured at each temperature in the range. 
        Number of iterations of bootstrap algorithm (default = 250)
        
        Outputs: Errors on chi over T range"""
    
        kappa_error_kT = np.zeros(np.size(array, axis = 0))
        kappa_index = 0
        for i in range(np.size(array, axis = 0)): #For each temperature. Make sure kT is along this axis
            kappa_data = np.zeros(k)
            kappa_data_index = 0
            for j in range(k):
                #Resample measurements at each kT
                resample = np.random.choice(array[i,1:], size = len(array[i,1:]))
                #Get mean and mean of squares of M to calculate values
                avg_N = np.mean(resample)
                avg_sq_N = np.mean(resample**2)
                #Get temperature
                mu = array[i,0] 
                #Get susceptibility
                kappa = self.compute_isothermal_compressability(mu, avg_N, avg_sq_N)
                #Append to set of calculated chis
                kappa_data[kappa_data_index] = kappa
                kappa_data_index += 1
            
            #Calculate overall error on chi for each T
            kappa_error = np.sqrt(np.mean(kappa_data ** 2) - (np.mean(kappa_data))**2)
            kappa_error_kT[kappa_index] = (kappa_error)
            kappa_index += 1
            print(f"Errors for kT = {array[i,0]} done")
        
        return kappa_error_kT
    
    def plot_compressibility_with_T(self, kT_data, N_avg, N_sq_avg, errors, plot = True):
        """Plots the magnetic susceptibility against kT with error bars
        
        Inputs: kT range, averaged magnetisation over T range, average square of 
        magnetisation over T range, array to calculate susceptibility errors. Whether to
        plot (default = True)
        Outputs: Susceptibilty data over T range and errors over T range"""

        kappa_data = self.compute_isothermal_compressability_with_T(kT_data, N_avg, N_sq_avg)
        kappa_errors = self.compute_compressibility_errors_with_T(errors)
        if plot == True:
            plt.errorbar(kT_data, kappa_data, yerr= kappa_errors, marker = "o", color = "purple")
            plt.grid(which = "both")
            plt.xlabel(f"kT/J", fontsize = 16)
            plt.ylabel(f"Variance $\\kappa$", fontsize = 16)
            plt.title(f"Compressiblity vs. temperature for a {self.size}x{self.size} \n lattice with mu = -2.0", fontsize = 12)
            plt.tight_layout()
            plt.savefig(f"Partgvariance")
            plt.show()

        data = np.column_stack((kT_data, kappa_data, kappa_errors))
        np.savetxt("Compressibilitydatapartg.csv", data, delimiter=",", header = "kT, kappa, kappa_error")

def main():
    
    #Arguments to run animation and measurements
    parser = argparse.ArgumentParser(description="Animate & take measurements from the Ising Model")
    parser.add_argument("size", type=int)
    parser.add_argument("kT", type = float)
    parser.add_argument("mu", type = float)
    parser.add_argument("--run_N_mu", type = str, choices=["Y" , "N"], default = "N")
    parser.add_argument("--run_N_kT", type = str, choices=["Y" , "N"], default = "N")

    args = parser.parse_args()

    lattice = IsingModel(kT = args.kT, size = args.size, mu = args.mu)
    lattice.animate_lattice()

    time_0 = time.time()
    if args.run_N_mu == "Y":
        N_model = IsingModel(kT = 1.0, mu = -2.5, size = 50)

        mu_data, rho, N_avg, N_sq_avg, N_error = N_model.compute_average_N_and_density()
        N_model.plot_compressibility(mu_data, N_avg, N_sq_avg, N_error)

    if args.run_N_kT == "Y":
        N_model = IsingModel(kT = 0.75, mu = -2, size = 50)

        kT_data, rho, N_avg, N_sq_avg, N_error = N_model.compute_average_N_and_density_with_T()
        N_model.plot_compressibility_with_T(kT_data, N_avg, N_sq_avg, N_error)


    time_4 = time.time()
    print(f"Measurements done in {(time_4 - time_0)/60} minutes.")
    
       
if __name__ == "__main__":

    main()


    

            


            
            



    





        

        


        



        
        