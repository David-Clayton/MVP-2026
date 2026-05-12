import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation as animate
import argparse
import time
from numba import njit
import pandas as pd

@njit
def gol_rules(lattice, num_live_nbrs, size, p_1, p_2):
    """Apply Game of Life rules with numba optimization
    
    Inputs: lattice
            number of live neighbours for each lattice point (array)
            lattice size
            
    Outputs: Test lattice"""

    #Copy self.lattice such that when iterating GoL, updated cells will not affect cells being checked
    test_lattice = lattice.copy()
    
    for i in range(size):
        for j in range(size):
            if lattice[i, j] == 1:
                if np.random.uniform(0,1) < p_1:
                    test_lattice[i, j] = 0
            else: 
                if num_live_nbrs[i, j] == 2 and np.random.uniform(0,1) < p_2:
                    test_lattice[i, j] = 1
    
    return test_lattice

class GameOfLife:

    """This class runs Conway's Game of Life on a 50x50 lattice with periodic
    boundary conditions"""

    def __init__(self, size, p_1, p_2, init = "random"):

        """Initialise lattice with size and set of initial conditions. 
        Input: size - lattice width on each side
                init - set of initial conditions. One of "random", "blinker",
                "glider", "beehive".
                position - initial coordinates of object. One of "random" or "middle"."""

        #0 is dead. 1 is alive
        if init == "random":
            self.lattice = np.random.choice(a = [0, 1], size = (size,size))

        self.size = size
        self.p_1 = p_1
        self.p_2 = p_2
        self.initial_lattice = self.lattice.copy()
       
    def run_rules(self):

        """Evolve the initialised lattice according to the rules of the GoL.

        1: Death
        2: Taxes

        Or actually:
        1: A live cell with 2 or 3 live neighbours stays alive
        2: A live cell with <2 or >3 live neighbours dies
        3: A dead cell with 3 live neighbours comes to life

        Inputs: None
        Outputs: self.lattice
        """
        #Create lattices of neighbours
        top_neighb = np.roll(self.lattice, shift = 1, axis = 0)
        bottom_neighb = np.roll(self.lattice, shift = -1, axis = 0)
        left_neighb = np.roll(self.lattice, shift = 1, axis = 1)
        right_neighb = np.roll(self.lattice, shift = -1, axis = 1)
        tl_neighb = np.roll(top_neighb, shift = 1, axis = 1)
        tr_neighb = np.roll(top_neighb, shift = -1, axis = 1)
        bl_neighb = np.roll(bottom_neighb, shift = 1, axis = 1)
        br_neighb = np.roll(bottom_neighb, shift = -1, axis = 1)

        #Add neighbour lattices to get number of live neighbours for each cell

        num_live_nbrs = top_neighb + bottom_neighb + left_neighb + right_neighb + tl_neighb + tr_neighb + bl_neighb + br_neighb

        
        #Now can update lattice calling numba optimised function
        self.lattice = gol_rules(self.lattice, num_live_nbrs, self.size, self.p_1, self.p_2)
        return self.lattice

    def animate_lattice(self, number_of_frames = 10000, interval = 50):

        """Animates GoL evolution over time
        Inputs: Number of frames (default = 10000)
                Interval between frames (default = 50ms)
        Outputs: Returns and runs animation"""

        fig, ax = plt.subplots()
        figure = ax.imshow(self.lattice, cmap = "viridis")
        ax.set_title(f"Conway's Game of Life evolution on a {self.size}x{self.size} lattice")
        
        def update(frame): #Needs frame arg as animate passes current frame no. to update()

            """Function to update figure for FuncAnimation."""
            self.run_rules()

            figure.set_data(self.lattice)

            return [figure]
        
        animation = animate(fig, update, frames = number_of_frames, interval = interval, blit = True)

        plt.show()
        return animation

    def compute_avg_live_frac(self, no_sweeps=100):
        """Calculate the average fraction of a lattice that it spends
        alive over its lifetime, ranging from p_1 = 0.1->1.0 and p_2=0.1->1.0, as
        well as the variance of the fraction"""
        #Probability ranges
        p_1_range = np.arange(0.1, 1.1, 0.1)
        p_2_range = np.arange(0.1, 1.1, 0.1)
        #Empty data array
        prob_data = np.zeros((len(p_1_range), len(p_2_range)))
        var_data = np.zeros((len(p_1_range), len(p_2_range)))
        #No iterations per sweep
        iter_per_sweep = self.size 


        #Iterate over each combination of p_1 and p_2
        for i, p_1 in enumerate(p_1_range):
            for j, p_2 in enumerate(p_2_range):

                #Reinitialise lattice with new probability parameter
                self.lattice = np.random.choice(a = [0, 1], size = (self.size, self.size))
                self.p_1 = p_1
                self.p_2 = p_2
            
                #Store data as a 3D array - initialise empty array
                data = np.zeros((self.size, self.size, no_sweeps))
                for k in range(no_sweeps):
                    for l in range(iter_per_sweep):
                        self.run_rules()
    
                    data[:, :, k] = self.lattice

                #Average alive fraction over time and over all cells
                avg_live_frac = np.mean(data)
                #Take variance
                frac_var = np.mean(data**2) - np.mean(data)**2 

                prob_data[i ,j] = avg_live_frac
                var_data[i,j] = frac_var

                print(f"Data collection complete for p_1 = {p_1} and p_2 = {p_2}")

        #Create heatmap plot for average fraction
        fig, ax = plt.subplots()
        im = ax.imshow(prob_data, cmap = "viridis", origin="lower", extent=[p_1_range[0], p_1_range[-1], 
                              p_2_range[0], p_2_range[-1]], aspect="auto")
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(f"Average live fraction")

        ax.set_xlabel(f"$p_1$", fontsize = 12)
        ax.set_ylabel(f"$p_2$", fontsize = 12)
        ax.set_title(f"Average fraction of {self.size}x{self.size} lattice spend alive over its \n lifetime as function of $p_1$ and $p_2$", fontsize = 12)
        
        plt.tight_layout()
        plt.savefig('Partdheatmap.png')
        plt.show()

        df = pd.DataFrame(prob_data, 
                         index=p_1_range,
                         columns=p_2_range)
        df.index.name = "p_1 / p_2"
        df.to_csv("Partdheatmap.csv")  

        #Create heatmap plot for variance
        fig, ax = plt.subplots()
        im = ax.imshow(var_data, cmap = "viridis", origin="lower", extent=[p_1_range[0], p_1_range[-1], 
                              p_2_range[0], p_2_range[-1]], aspect="auto")
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(f"Average live fraction variance")

        ax.set_xlabel(f"$p_1$", fontsize = 12)
        ax.set_ylabel(f"$p_2$", fontsize = 12)
        ax.set_title(f"Variance on average fraction of {self.size}x{self.size} live lattice cells \n as function of $p_1$ and $p_2$", fontsize = 12)
        
        plt.tight_layout()
        plt.savefig('Partfheatmap.png')
        plt.show()

        df = pd.DataFrame(prob_data, 
                         index=p_1_range,
                         columns=p_2_range)
        df.index.name = "p_1 / p_2"
        df.to_csv("Partdheatmap.csv")  

        


def main():
    #Arguments to run animation and measurements
    parser = argparse.ArgumentParser(description="Run Conway's Game of Life")
    parser.add_argument("size", type=int)
    parser.add_argument("p_1", type=float)
    parser.add_argument("p_2", type=float)
    parser.add_argument("--init", type=str, choices = ["random", "beehive", "glider", "blinker"], default = "random")
    parser.add_argument("--heatmap", type=str, choices = ["Y", "N"], default = "N")

    args = parser.parse_args()

    game = GameOfLife(size = args.size, p_1=args.p_1, p_2=args.p_2, init = args.init)
    game.animate_lattice()

    time_0 = time.time()

    if args.heatmap == "Y":
        game.compute_avg_live_frac()

    time_1 = time.time()

    print(f"Selected functions run in {(time_1 - time_0)/60} minutes") 

if __name__ == "__main__":
    main()
