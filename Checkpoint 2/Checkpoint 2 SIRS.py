import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation as animate
from matplotlib.colors import ListedColormap
import argparse
import time
import pandas as pd
from numba import njit

@njit
def sirs_step(lattice, size, p_si, p_ir, p_rs):
        """Run the SIRS rules onne on a random lattice point:
        1: S cell turns I with prob p_si if at least one neighbour is I
        2: I cell turns R with prob p_ir
        3: R cell turns S with prob p_rs
        4: Immune R cell never changes
        And updates lattice
        Inputs: lattice
                size
                p_si
                p_ir
                p_rs
        Outputs: lattice (changed after one iteration)"""

        i = np.random.randint(0, size)
        j = np.random.randint(0, size)

        state = lattice[i,j]

        nearest_neighbours = np.array([lattice[(i+1) % size, j] , lattice[(i-1) % size, j],
                                       lattice[i, (j+1) % size] , lattice[i, (j-1) % size]]) 
        
        #Test if any nearest neighbour is infected
        is_neighbour_infected = False
        for n in nearest_neighbours:
            if n == 1:
                is_neighbour_infected = True
                break

        #Change state based on probability and infection condition
        if state == 0 and is_neighbour_infected:
            if np.random.random() < p_si:      
                lattice[i, j] = 1

        elif state == 1:
            if np.random.random() < p_ir:
                lattice[i, j] = 2

        elif state == 2:
            if np.random.random() < p_rs:
                lattice[i, j] = 0

        elif state == 3:
            return lattice

        return lattice

class SIRS:
    """Runs the SIRS model for epidemics on a 50x50 lattice."""

    def __init__(self, size, p_si, p_rs, p_ir = 0.5, f_im = 0):
        
        """Initialises population with random number of people susceptible,
        infected, and recovered, and immune
        Inputs: Size
                probability that susceptibile cell become infected
                probability that recovered cell becomes susceptible
                probability that infected cell becomes recovered (default = 0.5)
                fraction of initial cells that are immune (default = 0)
        
        NB susceptible = 0, infected = 1, recovered = 2, immune = 3"""
        #Probability of site being in state 3 is f_im. 
        #Probability of site being in states 0,1 or 2 is (1-f_im)/3 as they are all equally probable
        prob = (1-f_im)/3
        self.lattice = np.random.choice(a = [0,1,2,3], size = (size,size), p=[prob,prob,prob,f_im])
        self.size = size
        self.p_si = p_si
        self.p_rs = p_rs
        self.p_ir = p_ir
        self.f_im = f_im

    def sirs_rules(self):
        """Call numba compiled function
        Inputs: None
        Outputs: self.lattice"""
        self.lattice = sirs_step(self.lattice, self.size, 
                                  self.p_si, self.p_ir, self.p_rs)
        return self.lattice

    def animate_lattice(self, no_frames = 10000, interval = 50):
        """Animates epidemic over time
        Inputs: Number of frames (default = 10000)
                Interval between frames (default = 50ms)
        Outputs: Returns and runs animation"""

        iter_per_frame = self.size ** 2
        fig, ax = plt.subplots()
        cmap = ListedColormap(["darkblue", "red", "lightblue", "green"])
        figure = ax.imshow(self.lattice, cmap = cmap, vmin = 0, vmax = 3)
        ax.set_title(f"SIRS model on a {self.size}x{self.size} lattice")
        cbar = plt.colorbar(figure, ax=ax, ticks=[0, 1, 2, 3])
        cbar.ax.set_yticklabels(["Susceptible", "Infected", "Recovered", "Immune"])
        
        def update(frame): #Needs frame arg as animate passes current frame no. to update()

            """Function to update figure for FuncAnimation."""
            for step in range(iter_per_frame):
                self.sirs_rules()

            figure.set_data(self.lattice)

            return [figure]
        
        animation = animate(fig, update, frames = no_frames, interval = interval, blit = True)

        plt.show()
        return animation

    def equilibriate(self, no_sweeps = 100):
        """Run SIRS model for a number of sweeps before taking any measurements
        so system forgets its initial conditions.
        Inputs: Number of sweeps to run (default = 100)
        Outputs: Equlibriated lattice
        """
    
        iter_per_sweep = self.size ** 2
        for i in range(no_sweeps):
            for j in range(iter_per_sweep):
                self.sirs_rules()
        #Return equilibriated lattice
        return self.lattice

    def phase_diagram(self):
        """Create phase diagram of the average number of infected cells over
        a number of measurements (1 meas per sweep) over the 2D parameter
        space of p_si and p_rs, setting p_ir = 0.5.
        Inputs: None
        Outputs: Heatmap of results
                Heatmap data written to csv"""

        #Probability ranges
        p_si_range = np.arange(0, 1.05, 0.05)
        p_rs_range = np.arange(0, 1.05, 0.05)
        #Empty array to take average infection fractiona
        avg_inf_frac = np.zeros((len(p_rs_range), len(p_si_range)))
    
        #No. iterations of algorithm per sweep
        iter_per_sweep = self.size ** 2
        #Number of measurments to be made per p_si/p_rs combination
        no_of_sweeps = 10000

        #Number of lattice points in array
        n = self.size ** 2

        for i, p_rs in enumerate(p_rs_range):
            for j, p_si in enumerate(p_si_range):

                #Reset lattice and update parameters for each combination
                self.lattice = np.random.choice(a=[0,1,2], size=(self.size, self.size))
                self.p_si = p_si
                self.p_rs = p_rs
                self.p_ir = 0.5

                #Equlibriate system before taking measurements
                self.equilibriate()

                no_infected_arr = np.zeros(no_of_sweeps)
                
                #Run sweeps to take measurments
                for k in range(no_of_sweeps):
                    for l in range(iter_per_sweep):
                        self.sirs_rules()
                    no_inf = np.sum(self.lattice == 1)
                    no_infected_arr[k] = no_inf
                

                avg_inf_frac[i,j] = np.mean(no_infected_arr) / n
                print(f"Infected fraction at p_rs={p_rs}, p_si={p_si} complete.")

        #Create heatmap plot
        fig, ax = plt.subplots()
        im = ax.imshow(avg_inf_frac, cmap = "viridis", origin="lower", extent=[p_si_range[0], p_si_range[-1], 
                              p_rs_range[0], p_rs_range[-1]], aspect="auto")
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(r"$\langle I \rangle/N$")

        ax.set_xlabel(r"$p(S\rightarrow I)$", fontsize = 12)
        ax.set_ylabel(r"$p(R\rightarrow S)$", fontsize = 12)
        ax.set_title(f"SIRS model heatmap with ($p({{I→R}})$ = 0.5 \n for a {self.size}x{self.size} lattice", fontsize = 12)
        
        plt.tight_layout()
        plt.savefig('Phasediagram.png')
        plt.show()

        df = pd.DataFrame(avg_inf_frac, 
                         index=p_rs_range,
                         columns=p_si_range)
        df.index.name = "p_rs / p_si"
        df.to_csv("Phasediagram.csv")

        return avg_inf_frac

    def infection_variance(self):
        """Calculate the variance on the average infection fraction ((<I^2> - <I>^2)/N)
        for a range of p_si = 0.2 - 0.5, with fixed p_ir = p_rs = 0.5.
        Inputs: None
        Outputs: Variance for each p_si (array)
                Number of infected sites for every sweep (for error calc., 2D array)
                Range of p_si
                Average fraction of infected cells for each p_si"""

        #No. iterations of algorithm per sweep
        iter_per_sweep = self.size ** 2
        #Number of measurments to be made per p_si/p_rs combination
        no_of_sweeps = 10000

        p_si_range = np.arange(0.2, 0.51, 0.01)

        #Need average infection number and average square of infection number
        avg_inf_arr = np.zeros(len(p_si_range))
        avg_inf_sq_arr = np.zeros(len(p_si_range))
        #Empty array for error computation
        inf_errors = np.zeros((len(p_si_range), no_of_sweeps + 1))

        for i, p_si in enumerate(p_si_range):
            #Reset lattice and update parameters for each combination
            self.lattice = np.random.choice(a=[0,1,2], size=(self.size, self.size))
            self.p_si = p_si
            self.p_rs = 0.5
            self.p_ir = 0.5

            #Equlibriate system before taking measurements
            self.equilibriate()

            no_infected_arr = np.zeros(no_of_sweeps)

            #Run sweeps to take measurments
            for k in range(no_of_sweeps):
                for l in range(iter_per_sweep):
                    self.sirs_rules()
                no_inf = np.sum(self.lattice == 1)
                no_infected_arr[k] = no_inf

            avg_inf_arr[i] = np.mean(no_infected_arr)
            avg_inf_sq_arr[i] = np.mean(no_infected_arr ** 2)

            #Append to error array
            inf_errors[i, 0] = p_si
            inf_errors[i, 1:] = no_infected_arr

            print(f"Variance computation for p_si = {p_si} complete")

        n = self.size **2

        variance = (avg_inf_sq_arr - avg_inf_arr**2) / n
        avg_inf_frac = avg_inf_arr / n

        return variance, inf_errors, p_si_range, avg_inf_frac
    
    def infection_variance_errors(self, array, k = 1000):
        """Compute errors for the variance plot using the bootstrap method.
        Inputs: 2D Array of infection data (from infection_variance)
                Number of times to iterate bootstrap method (default = 1000)
        Outputs: Errors on variance"""

        n = self.size ** 2

        var_errors = np.zeros(np.size(array, axis = 0)) #Length is total number of probs. sampled
        #For each p_si value:
        for i in range(len(var_errors)): 
            #Variances calculated from each sampled list of I
            bootstr_vars = np.zeros(k) 
            #For k times
            for j in range(k):
                #Resample number of infected sites
                resample = np.random.choice(array[i,1:], size = len(array[i,1:]))
                #Get mean and mean of square of number of inf. sites
                avg_inf = np.mean(resample)
                avg_inf_sq = np.mean(resample**2)
                
                #Get variance
                variance = (avg_inf_sq - avg_inf **2)/n
                bootstr_vars[j] = variance

            #Calculate std dev of list of variances
            #MINUS SIGN!!! THE PROBLEM WITH CP 1 WAS THE SIGN HERE WAS A PLUS
            error = np.sqrt(np.mean(bootstr_vars**2) - (np.mean(bootstr_vars)**2))
            var_errors[i] = error

            print(f"Errors from p_si = {array[i,0]} calculated")

        return var_errors
    
    def plot_variance(self):
        """Plot variance on error fraction with bootstrap-generated
        error bars
        Inputs: None
        Outputs: Plot of variance data with errorbars
                csv file of average infection rate, variance and variance errors"""
        variance, variance_errors_data, p_si, inf_frac = self.infection_variance()
        variance_errors = self.infection_variance_errors(variance_errors_data)

        plt.errorbar(p_si, variance, yerr = variance_errors, marker = "o", color = "purple")
        plt.grid(which = "both")
        plt.xlabel(r"p(S$\rightarrow$I)", fontsize = 16)
        plt.ylabel(r"$\frac{\langle I^2 \rangle - \langle I \rangle ^2}{N}$", fontsize = 16)
        plt.title(f"Variance on the infection rate with respect to $p({{S→I}})$ \n with $p({{I→R}})$ = $p({{R→S}})$ = 0.5 for a {self.size}x{self.size} lattice", fontsize = 12)
        plt.tick_params(axis = "both", labelsize = 12)
        plt.tight_layout()
        plt.savefig(f"Infection_variance.png")
        plt.show()

        #Read out results to datafile
        data = np.column_stack((p_si, inf_frac, variance, variance_errors))
        np.savetxt("Infectionvariance.csv", data, delimiter=",", header="p(s->i), avg_no_infections, inf_variance, variance_errors")

    def calculate_immunity(self):
        """Calculate the average infection rate for a varying f_im, fixing p_ir = p_rs = p_si = 0.5,
        and plot
        Inputs: None
        Outputs: Plot with average infection fraction vs. f_im
                CSV file of plot data"""

        f_im_range = np.arange(0, 1.05, 0.05)
        #Empty arrays to store average infection rate
        avg_inf_frac = np.zeros(len(f_im_range))

        iter_per_sweep = self.size ** 2
        n = self.size ** 2
        #Number of sweeps (measurements) to do per f_im
        no_sweeps = 10000
        for i, f_im in enumerate(f_im_range):
            #Reset lattice for each f_im value
            prob = (1 - f_im) / 3
            self.lattice = np.random.choice(a=[0,1,2,3], size=(self.size, self.size), p=[prob,prob,prob,f_im])
            self.p_si = 0.5
            self.p_rs = 0.5
            self.p_ir = 0.5
            self.f_im = f_im

            #Equlibriate system before taking measurements
            self.equilibriate()
            #Empty array to store number of infected in each sweep
            no_infected_arr = np.zeros(no_sweeps)
            #Run sweeps to take measurments
            for j in range(no_sweeps):
                for k in range(iter_per_sweep):
                    self.sirs_rules()
                no_inf = np.sum(self.lattice == 1)
                no_infected_arr[j] = no_inf

            avg_inf_frac[i] = np.mean(no_infected_arr) / n
            print(f"Infected fraction f_im = {f_im} complete.")

        #plot
        plt.plot(f_im_range, avg_inf_frac, marker = "o", color = "darkblue")
        plt.xlabel(r"Fraction of immune cells $f_{im}$", fontsize = 12)
        plt.ylabel(r"Average infection rate $\frac{\langle I \rangle}{N}$", fontsize = 12)
        plt.title(f"Effect of immunity on infection rate \n with p(s$\\rightarrow$ i) = p(i$\\rightarrow$ r) = p(r$\\rightarrow$ s) = 0.5", fontsize = 12)
        plt.tick_params(axis="both", labelsize = 12)
        plt.tight_layout()
        plt.savefig(f"Immunity.png")
        plt.show()

        #Readout to datafile

        data = np.column_stack((f_im_range, avg_inf_frac))
        np.savetxt("Immunity.csv", data, delimiter=",", header="f_im, avg_infect_rate")

def main():

    #Arguments to run animation and measurements
    parser = argparse.ArgumentParser(description="Simulate an epidemic with the SIRS model")
    parser.add_argument("size", type=int, default = 50)
    parser.add_argument("p_si", type=float)
    parser.add_argument("p_ir", type=float, default=0.5)
    parser.add_argument("p_rs", type=float)
    parser.add_argument("f_im", type=float, default=0.0)
    parser.add_argument("--run_heatmap", type=str, choices=["Y", "N"], default="N")
    parser.add_argument("--run_var", type=str, choices=["Y", "N"], default="N")
    parser.add_argument("--run_immun", type=str, choices=["Y", "N"], default="N")
    
    args = parser.parse_args()

    sirs = SIRS(size=args.size, p_si=args.p_si, p_ir=args.p_ir, p_rs=args.p_rs, f_im=args.f_im)
    sirs.animate_lattice()

    time_0 = time.time()

    if args.run_heatmap == "Y":
        sirs.phase_diagram()
    
    if args.run_var == "Y":
        sirs.plot_variance()

    if args.run_immun == "Y":
        sirs.calculate_immunity()

    time_1 = time.time()

    print(f"Selected functions done in {(time_1 - time_0)/60} minutes.")

if __name__ == "__main__":

    main()