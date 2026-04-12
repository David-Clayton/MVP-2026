import numpy as np
import random 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation as animate
import argparse 
import time 
from numba import jit

class IsingModel:

    """This class creates a 50x50 2D Ising Model where lattice points are attributed spin up/down, with the inital
    lattice state based on a Boltzmann distribution, and its evolution governed by Glauber and Kawasaki dynamics."""

    def __init__(self, kT, size):

        """Initialise Ising lattice. The distribution of spins in the lattice is based off a random number generator. Set J = 1"""

        self.lattice = np.random.choice(a = [-1, 1], size = (size,size))
        self.kT = kT
        self.size = size
    
    def compute_total_magnetisation(self):

        """Calculate the total magnetisation of the lattice by summing all spins.
        
        Inputs: None
        Outputs: total lattice magnetisation"""

        total_magnetisation = np.sum(self.lattice)

        return total_magnetisation
    
    def compute_total_energy(self):

        """Calculate the total energy of the lattice by summing the four
        array products of the initial array and the array shifted one
        element with np.roll
        
        Inputs: None
        Outputs: total lattice energy"""

        shift_up = np.roll(self.lattice, shift = 1, axis = 0)
        shift_down = np.roll(self.lattice, shift = -1, axis = 0)
        shift_left = np.roll(self.lattice, shift = 1, axis = 1)
        shift_right = np.roll(self.lattice, shift = -1, axis = 1)

        #Energy double counted - divide by 2
        total_energy = -1*np.sum(self.lattice * (shift_left + shift_right + shift_up + shift_down)) / 2

        return total_energy
    
    def glauber_dynamics(self, kT):

        """Take inital lattice, swap a spin on a random point, calculate the
        energy change of that swap, and use Metropolis algorithm to determine
        whether lattice is altered.
        
        Inputs: none
        Outputs: self.lattice (possibly changed)"""

        i = np.random.randint(0, self.size)
        j = np.random.randint(0, self.size)

        s_i = self.lattice[i, j] #Inital spin
        
        nearest_neighbours = np.array([self.lattice[(i+1) % self.size, j] , self.lattice[(i-1) % self.size, j],
                                       self.lattice[i, (j+1) % self.size] , self.lattice[i, (j-1) % self.size]]) 
        
        #Energy change equation = 2 * sum of multiplied NN spins

        energy_change = 2 * np.sum(s_i * nearest_neighbours)

        #Accept new state conditional on energy_change

        boltzmann_factor = np.exp(-energy_change / kT)
        if energy_change <= 0:
            self.lattice[i,j] = -self.lattice[i,j]
        elif energy_change > 0: 
            r = random.uniform(0, 1)
            if r < boltzmann_factor:
                self.lattice[i,j] = -self.lattice[i,j]
        
        return self.lattice
    
    def kawasaki_dynamics(self, kT):

        """Take initial lattice, exchange the spins of two randomly chosen sites, calculate the
        energy change of this action, and apply Metropolis to determine if this
        lattice is altered.
        
        Inputs: None
        Outputs: self.lattice (possibly altered)"""

        [i_1, j_1] = [np.random.randint(0, self.size), np.random.randint(0, self.size)]
        [i_2, j_2] = [np.random.randint(0, self.size), np.random.randint(0, self.size)]

        if i_1 == i_2 and j_1 == j_2: #Rejects if same site chosen twice.
            return self.lattice
        
        elif self.lattice[i_1, j_1] == self.lattice[i_2, j_2]: #Rejects if swapped spins equal because it means nothing.
            return self.lattice
        
        #Distances between lattice sites including periodicity
        delta_i = min(abs(i_1 - i_2), self.size - abs(i_1 - i_2))
        delta_j = min(abs(j_1 - j_2), self.size - abs(j_1 - j_2))
        
        
        if (delta_i > 1 or delta_j > 1) or (delta_i == 1 and delta_j == 1): #If spins are not adjacent
            s_1 = self.lattice[i_1, j_1]  
            s_2 = self.lattice[i_2, j_2]

            #Basically do Glauber dynamics twice simultaneously

            nearest_neighbours_1 = np.array([self.lattice[(i_1+1) % self.size, j_1] , self.lattice[(i_1-1) % self.size, j_1],
                                       self.lattice[i_1, (j_1+1) % self.size] , self.lattice[i_1, (j_1-1) % self.size]]) 
            
            nearest_neighbours_2 = np.array([self.lattice[(i_2+1) % self.size, j_2] , self.lattice[(i_2-1) % self.size, j_2],
                                       self.lattice[i_2, (j_2+1) % self.size] , self.lattice[i_2, (j_2-1) % self.size]]) 

            energy_change_1 = 2 * np.sum(s_1 * nearest_neighbours_1)
            energy_change_2 = 2 * np.sum(s_2 * nearest_neighbours_2)
        
            tot_energy_change = energy_change_1 + energy_change_2
 
            #Accept or reject state based on tot_energy_change

            boltzmann_factor = np.exp(-tot_energy_change / kT)
            if tot_energy_change <= 0:
                self.lattice[i_1,j_1] = -self.lattice[i_1,j_1]
                self.lattice[i_2,j_2] = -self.lattice[i_2,j_2]
            elif tot_energy_change > 0:
                r = random.uniform(0, 1)
                if r < boltzmann_factor:
                    self.lattice[i_1,j_1] = -self.lattice[i_1,j_1]
                    self.lattice[i_2,j_2] = -self.lattice[i_2,j_2]

            return self.lattice
        
        elif (delta_i == 1 and delta_j == 0) or (delta_i == 0 and delta_j == 1): #Spins adjacent

            s_1 = self.lattice[i_1, j_1]  
            s_2 = self.lattice[i_2, j_2]

            nearest_neighbours_1 = np.array([self.lattice[(i_1+1) % self.size, j_1] , self.lattice[(i_1-1) % self.size, j_1],
                                       self.lattice[i_1, (j_1+1) % self.size] , self.lattice[i_1, (j_1-1) % self.size]]) 
            
            nearest_neighbours_2 = np.array([self.lattice[(i_2+1) % self.size, j_2] , self.lattice[(i_2-1) % self.size, j_2],
                                       self.lattice[i_2, (j_2+1) % self.size] , self.lattice[i_2, (j_2-1) % self.size]]) 

            #Calculate energy change not yet considering adjacency

            energy_change_1 = 2 * np.sum(s_1 * nearest_neighbours_1)
            energy_change_2 = 2 * np.sum(s_2 * nearest_neighbours_2)
        
            tot_energy_change = energy_change_1 + energy_change_2

            #Correct for adjacency 
            
            tot_energy_change = tot_energy_change - 4 * s_1 * s_2

            #Accept or reject state based on tot_energy_change

            boltzmann_factor = np.exp(-tot_energy_change / kT)
            if tot_energy_change <= 0:
                self.lattice[i_1,j_1] = -self.lattice[i_1,j_1]
                self.lattice[i_2,j_2] = -self.lattice[i_2,j_2]
            elif tot_energy_change > 0:
                r = random.uniform(0, 1)
                if r < boltzmann_factor:
                    self.lattice[i_1,j_1] = -self.lattice[i_1,j_1]
                    self.lattice[i_2,j_2] = -self.lattice[i_2,j_2]

            return self.lattice
        
    def animate_lattice(self, dynamics, number_of_frames = 500, interval = 50):
        """Animates lattice evolution over time
        
        Inputs: Dynamical model (Glauber/Kawasaki)
                Number of frames in animation
                Interval between frames (ms)
        Outputs: Shows and returns animation"""
        
        #Number of iterations of algorithm per frame (1 frame per sweep)
        iter_per_frame = self.size ** 2 
        fig, ax = plt.subplots()
        figure = ax.imshow(self.lattice, cmap = "viridis")
        ax.set_title(f"Ising model using {dynamics} dynamics at kT = {self.kT}.")

        def update(frame): #Needs frame arg as animate passes current frame no. to update()

            """Function to update figure for FuncAnimation."""
            for step in range(iter_per_frame):
                if dynamics == "Glauber":
                    self.glauber_dynamics(self.kT)
                elif dynamics == "Kawasaki":
                    self.kawasaki_dynamics(self.kT)
                else:
                    raise ValueError(f"Please input either Glauber or Kawasaki.")

            figure.set_data(self.lattice)

            return [figure]
        
        animation = animate(fig, update, frames = number_of_frames, interval = interval, blit = True)

        plt.show()
        return animation
    
    def equilibriate(self, kT, dynamics = "Glauber", no_sweeps = 100):
        """Run equilibriation of the lattice before any measurements are taken so the lattice
        loses memory of its initial conditions.
        
        Inputs: Dynamical model (default Glauber)
                Number of sweeps before equilibriation (default 100)
        Outputs: self.lattice (equilibriated)"""

        iter_per_sweep = self.size ** 2
        for i in range(no_sweeps):
            for j in range(iter_per_sweep):
                if dynamics == "Glauber": #Default
                    self.glauber_dynamics(kT)
                elif dynamics == "Kawasaki":
                    self.kawasaki_dynamics(kT)
        #Return equilibriated lattice
        return self.lattice
        
    
    def compute_average_magnetisation(self, plot = True):
        """Calculates the average of the magnetisation, the average of the absolute value of the magnetisation,
        and the average of the square of the magnetisation of the system from measurements from a large
        number of sweeps of the system. Does this over a range of temperatures
        kT = 1 -> 3 in increments of 0.1. Uses Glauber dynamics only.
        
        Inputs: Whether to plot abs. value of magnetisation (default True)
        Outputs: Range of kT, arrays of: averaged magnetisation, averaged
        absolute magnetisation, and averaged square magnetisation over T range. """

        #First, equilibriate system before measurements with 5000 sweeps at maximum temperature
        self.equilibriate(no_sweeps = 2000, kT = 3)
        print(f"Initial equilibriation complete")
        #Temperature range
        kT_data = np.arange(3, 0.9, -0.1) 

        iter_per_sweep = self.size ** 2
        #Number of sweeps to be done at each temperature for measurements
        number_of_sweeps = 5000 
        #Number of sweeps to be done inbetween measurements
        sweeps_bw_meas = 10
        #Empty arrays for average data as function of T
        m_avg_T = np.zeros(len(kT_data))
        abs_m_avg_T = np.zeros(len(kT_data))
        m_sq_avg_T = np.zeros(len(kT_data))
        #Empty array for error calculation
        #First column will be T data, each row afterwards will be m data at each T
        m_error = np.zeros((len(kT_data) , number_of_sweeps//sweeps_bw_meas + 1))
        T_index = 0
        for kT in kT_data: 
            #Empty array for magnetisation data
            m = np.zeros(number_of_sweeps//sweeps_bw_meas)
            data_index = 0
            #Run for necessary number of sweeps
            for i in range(number_of_sweeps):
                #Run dynamics algorithm for 1 sweep
                for j in range(iter_per_sweep):
                    self.glauber_dynamics(kT)
                #Only take measurement every 10th sweep to avoid correlation between measurements
                if i % sweeps_bw_meas == 0:
                    mag_meas = self.compute_total_magnetisation()
                    #Add measurements to array
                    m[data_index] = mag_meas
                    data_index += 1
                if i % 500 == 0:
                    print(f"Sweep {i} completed at kT = {kT}")

            #Add the average of these sets of measurements of this temp to the final data
            m_avg_T[T_index] = (np.mean(m))
            abs_m_avg_T[T_index] = (np.mean(np.abs(m)))
            m_sq_avg_T[T_index] = (np.mean(m**2))

            #Append data to error array
            m_error[T_index, 0] = kT
            m_error[T_index, 1:] = m

            T_index += 1
            #Re-equilibriate with 100 sweeps after measurements taken before system passed to next T
            self.equilibriate(kT = kT)
            
        
        #Flip arrays so kT increases
        kT_data = np.flip(kT_data)
        m_avg_T = np.flip(m_avg_T)
        abs_m_avg_T = np.flip(abs_m_avg_T)
        m_sq_avg_T = np.flip(m_sq_avg_T)
        m_error = np.flip(m_error, axis=0)

        if plot == True:
            plt.plot(kT_data, abs_m_avg_T, marker = "o", color = "g")
            plt.grid(which = "both")
            plt.xlabel("kT/J", fontsize = 16)
            plt.ylabel(f"Absolute magnetisation |M|", fontsize = 16)
            plt.title(f"|M| vs. thermal energy for a {self.size}x{self.size} \n lattice using Glauber dynamics", fontsize = 12)
            plt.tight_layout()
            plt.savefig(f"Abs_mag_Glauber")
            plt.show()
        
        return kT_data, m_avg_T, abs_m_avg_T, m_sq_avg_T, m_error


    def compute_magnetic_susceptibility(self, kT_data, m_avg_T, m_sq_avg_T):

        """Calculates the magnetic susceptibility (chi) of the system in the range of 
        temperatures kT = 1 -> 3 in increments of 0.1.

        Inputs: Range of kT, average magnetisation over T range, average square
        of magnetisation over T range 
        Outputs: Susceptibility over T range
        """
        n = self.size ** 2
        #Compute magnetic susceptibility
        chi_data = (m_sq_avg_T - m_avg_T ** 2) / (n*kT_data)

        return chi_data
    
    def compute_susceptibility_errors(self, array, k = 1000):
        """Calculate the errors on susceptibility using the bootstrap method.
        
        Inputs: 2D Array of magnetisations obtained from measurements where each column is the
        set of values measured at each temperature in the range. 
        Number of iterations of bootstrap algorithm (default = 250)
        
        Outputs: Errors on chi over T range"""
    
        chi_error_T = np.zeros(np.size(array, axis = 0))
        chi_T_index = 0
        for i in range(np.size(array, axis = 0)): #For each temperature. Make sure kT is along this axis
            chi_data = np.zeros(k)
            chi_index = 0
            for j in range(k):
                #Resample measurements at each kT
                resample = np.random.choice(array[i,1:], size = len(array[i,1:]))
                #Get mean and mean of squares of M to calculate values
                avg_m = np.mean(resample)
                avg_sq_m = np.mean(resample**2)
                #Get temperature
                temp = array[i,0] 
                #Get susceptibility
                chi = self.compute_magnetic_susceptibility(temp, avg_m, avg_sq_m)
                #Append to set of calculated chis
                chi_data[chi_index] = chi
                chi_index += 1
            
            #Calculate overall error on chi for each T
            chi_error = np.sqrt(np.mean(chi_data ** 2) - (np.mean(chi_data))**2)
            chi_error_T[chi_T_index] = (chi_error)
            chi_T_index += 1
            print(f"Errors for kT = {array[i,0]} done")
        
        return chi_error_T
    
    def plot_susceptibility(self, kT_data, m_avg_T, m_sq_avg_T, errors, plot = True):
        """Plots the magnetic susceptibility against kT with error bars
        
        Inputs: kT range, averaged magnetisation over T range, average square of 
        magnetisation over T range, array to calculate susceptibility errors. Whether to
        plot (default = True)
        Outputs: Susceptibilty data over T range and errors over T range"""

        chi_data = self.compute_magnetic_susceptibility(kT_data, m_avg_T, m_sq_avg_T)
        chi_errors = self.compute_susceptibility_errors(errors)
        if plot == True:
            plt.errorbar(kT_data, chi_data, yerr= chi_errors, marker = "o", color = "purple")
            plt.grid(which = "both")
            plt.xlabel(f"kT/J", fontsize = 16)
            plt.ylabel(r"Magnetic susceptibility $\chi$", fontsize = 16)
            plt.title(f"Susceptibility vs. thermal energy for a {self.size}x{self.size} \n lattice using Glauber Dynamics", fontsize = 12)
            plt.tight_layout()
            plt.savefig(f"Chi_Glauber")
            plt.show()

        #Return susceptibility data for datafile writing
        return chi_data, chi_errors
    
    def compute_average_energy(self, dynamics, plot = True):
        """Calculates the average of the energy and the average of the square of the energy of the system. 
        Does this over a range of temperatures kT = 1 -> 3 in increments of 0.1, from measurements from a large
        number of sweeps of the system. Uses both Glauber and Kawasaki dynamics.
        
        Inputs: Dynamical model 
        Outputs: kT range, average energy over T range, average square of energy over
        T range, array of measured energies to determine errors"""

        #First, equilibriate system before measurements with 5000 sweeps
        self.equilibriate(no_sweeps = 2000, dynamics=dynamics, kT = 3)
        #Temperature range
        kT_data = np.arange(3, 0.9, -0.1) 

        iter_per_sweep = self.size ** 2
        #Number of sweeps to be done at each temperature for measurements
        number_of_sweeps = 5000 
        #Number of sweeps to be done inbetween measurements
        sweeps_bw_meas = 10
        #Empty arrays for average data as function of T
        e_avg_T = np.zeros(len(kT_data))
        e_sq_avg_T = np.zeros(len(kT_data))
        #Data array to compute errors with
        e_errors = np.zeros((len(kT_data), number_of_sweeps//sweeps_bw_meas + 1))
        T_index = 0
        for kT in kT_data: 
            #Empty arrays for energy data
            e = np.zeros(number_of_sweeps//sweeps_bw_meas)
            data_index = 0
            #Run for necessary number of sweeps
            for i in range(number_of_sweeps):
                #Run dynamics algorithm for 1 sweep
                for j in range(iter_per_sweep):
                    if dynamics == "Glauber":
                        self.glauber_dynamics(kT)
                    elif dynamics == "Kawasaki":
                        self.kawasaki_dynamics(kT)
                #Only take measurement every 10th sweep to avoid correlation between measurements
                if i % sweeps_bw_meas == 0:
                    energy_meas = self.compute_total_energy()
                    #Add measurements to arrays
                    e[data_index] = energy_meas
                    data_index += 1
                if i % 500 == 0:
                    print(f"Sweep {i} completed at kT = {kT}")

            #Add the average of these sets of measurements of this temp to the final data
            e_avg_T[T_index] = (np.mean(e))
            e_sq_avg_T[T_index] = (np.mean(e**2))

            #Append data to error array
            e_errors[T_index, 0] = kT
            e_errors[T_index, 1:] = e
    

            T_index += 1
            #Re-equilibriate with 100 sweeps after measurements taken before system passed to next T
            self.equilibriate(dynamics = dynamics, kT = kT)
            
        #Flip arrays so kT increases
        kT_data = np.flip(kT_data)
        e_avg_T = np.flip(e_avg_T)
        e_sq_avg_T = np.flip(e_sq_avg_T)
        e_errors = np.flip(e_errors, axis = 0)

        if plot == True:

            plt.plot(kT_data, e_avg_T, marker = "o", color = "black")
            plt.grid(which = "both")
            plt.xlabel("kT/J", fontsize = 16)
            plt.ylabel(f"Average energy <E>", fontsize = 16)
            plt.title(f"<E> vs. thermal energy for a {self.size}x{self.size} \n lattice using {dynamics} dynamics", fontsize = 12)
            plt.tight_layout()
            plt.savefig(f"Avg_E_{dynamics}")
            plt.show()

        return kT_data, e_avg_T, e_sq_avg_T, e_errors
    
    def compute_heat_capacity(self, kT_data, e_avg_T, e_sq_avg_T):
        """Calculates the heat capacity (c) of the system in the range of 
        temperatures kT = 1 -> 3 in increments of 0.1. 

        Inputs: kT range, average energy over T range, average square of energy
        over T ramge
        Outputs: heat capacity data over T range
        """
    
        n = self.size ** 2
        #Compute heat capacity
        #Set Boltzmann's constant = 1
        c_data = (e_sq_avg_T - e_avg_T ** 2) / (n*(kT_data)**2)

        return c_data
    
    def compute_heat_capacity_errors(self, array, k = 1000):
        """Calculate the errors on heat capacity using the bootstrap method.
        
        Inputs: 2D Array of heat capacities obtained from measurements where each column are the
        values measured at each temperature in the range, Number of iterations 
        of bootstrap algorithm (default = 250)
        
        Outputs: Errors on heat capacity over T range"""

        c_error_T = np.zeros(np.size(array, axis = 0))
        c_T_index = 0
        for i in range(np.size(array, axis = 0)): #For each temperature. Make sure kT is along this axis
            c_data = np.zeros(k)
            c_index = 0
            for j in range(k):
                #Resample measurements at each kT
                resample = np.random.choice(array[i,1:], size = len(array[i,1:]))
                #Get mean and mean of square of E to calculate values
                avg_e = np.mean(resample)
                avg_sq_e = np.mean(resample**2)
                #Get temperature
                temp = array[i,0] 
                #Get heat capacity
                c = self.compute_heat_capacity(temp, avg_e, avg_sq_e)
                #Append to set of calculated heat capacities
                c_data[c_index] = c
                c_index += 1
            
            #Calculate overall error on heat capacity for each T
            c_error = np.sqrt(np.mean(c_data ** 2) - (np.mean(c_data))**2)
            c_error_T[c_T_index] = (c_error)
            c_T_index += 1
            print(f"Errors for kT = {array[i,0]} done")
        
        return c_error_T
    
    def plot_heat_capacity(self, dynamics, kT_data, e_avg_T, e_sq_avg_T, errors, plot = True):
        """Plots the heat capacity against kT with error bars
        
        Inputs: Dynamical model, kT range, averaged energy over T range, average square of 
        energy over T range, array to calculate heat capacity errors. Whether to
        plot (default = True)
        Outputs: Heat capacity data over T range and errors over T range"""

        c_data = self.compute_heat_capacity(kT_data, e_avg_T, e_sq_avg_T)
        c_errors = self.compute_heat_capacity_errors(errors)
        if plot == True:
            plt.errorbar(kT_data, c_data, yerr= c_errors, marker = "o", color = "orange")
            plt.grid(which = "both")
            plt.xlabel(f"kT/J", fontsize = 16)
            plt.ylabel(f"Heat capacity C", fontsize = 16)
            plt.title(f"Heat capacity vs. thermal energy for a {self.size}x{self.size} \n lattice using {dynamics} Dynamics", fontsize = 12)
            plt.tight_layout()
            plt.savefig(f"Heat_cap_{dynamics}")
            plt.show()

        #Return heat capacity for datafile writing
        return c_data, c_errors
       

def main():
    
    #Arguments to run animation and measurements
    parser = argparse.ArgumentParser(description="Animate & take measurements from the Ising Model")
    parser.add_argument("size", type=int, default = 50)
    parser.add_argument("kT", type = float)
    parser.add_argument("dynamics", type = str, choices=["Glauber" , "Kawasaki"])
    parser.add_argument("--run_mag", type = str, choices=["Y" , "N"], default = "N")
    parser.add_argument("--run_therm_Glauber", type = str, choices=["Y" , "N"], default = "N")
    parser.add_argument("--run_therm_Kawasaki", type = str, choices=["Y" , "N"], default = "N")

    args = parser.parse_args()

    lattice = IsingModel(kT = args.kT, size = args.size)
    lattice.animate_lattice(dynamics = args.dynamics)

    time_0 = time.time()
    if args.run_mag == "Y":
        Mag_model = IsingModel(kT = 3, size = 50)

        #Magnetic data
        kT_data, m_avg_data, abs_m_data, m_sq_avg_data, errors = Mag_model.compute_average_magnetisation()
        chi_data, chi_error_data = Mag_model.plot_susceptibility(kT_data, m_avg_data, m_sq_avg_data, errors) 

        time_1 = time.time()
        print(f"Magnetisation data collected in {(time_1 - time_0)/60} minutes.")

        #Write data to csv file
        all_data = np.column_stack((kT_data, abs_m_data, chi_data, chi_error_data))
        np.savetxt("IsingDataMag.csv", all_data, delimiter=",", header = "kT, |M|, chi, chi_error") 

    if args.run_therm_Glauber == "Y":

        Therm_model_g = IsingModel(kT = 3, size = 50)
        #Thermal data (Glauber)
        kT_data, e_avg_G, e_sq_avg_G, e_errors_G = Therm_model_g.compute_average_energy(dynamics="Glauber")
        c_data_G, c_errors_G = Therm_model_g.plot_heat_capacity("Glauber", kT_data, e_avg_G, e_sq_avg_G, e_errors_G)

        time_2 = time.time()
        print(f"Glauber thermal data collected in {(time_2 - time_0)/60} minutes.")

        #Write data to csv file
        all_data = np.column_stack((kT_data, e_avg_G, c_data_G, c_errors_G))
        np.savetxt("IsingDataGlauber.csv", all_data, delimiter=",", header = "kT, E(G), C(G), C(G)_error")

    if args.run_therm_Kawasaki == "Y":

        Therm_model_k = IsingModel(kT=3, size= 50)
        #Thermal data (Kawasaki)
        kT_data, e_avg_K, e_sq_avg_K, e_errors_K = Therm_model_k.compute_average_energy(dynamics="Kawasaki")
        c_data_K, c_errors_K = Therm_model_k.plot_heat_capacity("Kawasaki", kT_data, e_avg_K, e_sq_avg_K, e_errors_K)

        time_3 = time.time()
        print(f"Kawasaki thermal data collected in {(time_3 - time_0)/60} minutes.")

        #Write data to csv file
        all_data = np.column_stack((kT_data, e_avg_K, c_data_K, c_errors_K))
        np.savetxt("IsingDataKawasaki.csv", all_data, delimiter=",", header = "kT, E(K), C(K), C(K)_error")

    time_4 = time.time()
    print(f"Measurements done in {(time_4 - time_0)/60} minutes.")
    
       
if __name__ == "__main__":

    main()


    

            


            
            



    





        

        


        



        
        