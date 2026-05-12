import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation as animate
import argparse
import time
from numba import njit

@njit
def gol_rules(lattice, num_live_nbrs, size, n):
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
                    test_lattice[i, j] = 0
            else: 
                if num_live_nbrs[i, j] == n:
                    test_lattice[i, j] = 1
    
    return test_lattice

class GameOfLife:

    """This class runs Conway's Game of Life on a 50x50 lattice with periodic
    boundary conditions"""

    def __init__(self, size, n, init = "random"):

        """Initialise lattice with size and set of initial conditions. 
        Input: size - lattice width on each side
                init - set of initial conditions. One of "random", "blinker",
                "glider", "beehive".
                position - initial coordinates of object. One of "random" or "middle"."""

        #0 is dead. 1 is alive
        if init == "random":
            self.lattice = np.random.choice(a = [0, 1], size = (size,size), p = [0.99, 0.01])

        elif init == "block": 
            self.lattice = np.zeros((size,size))
            self.lattice[size//2 - 10:size//2 + 10, size//2 - 10:size//2 + 10] = 1

        self.size = size
        self.initial_lattice = self.lattice.copy()
        #Number of live neighbours for a dead cell to switch to a live one
        self.n = n
       
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
        self.lattice = gol_rules(self.lattice, num_live_nbrs, self.size, self.n)
        return self.lattice

    def animate_lattice(self, number_of_frames = 10000, interval = 50):

        """Animates GoL evolution over time
        Inputs: Number of frames (default = 10000)
                Interval between frames (default = 50ms)
        Outputs: Returns and runs animation"""

        fig, ax = plt.subplots()
        figure = ax.imshow(self.lattice, cmap = "viridis")
        ax.set_title(f"Conway's Game of Life evolution on a {self.size}x{self.size} lattice \n where n = {self.n}")
        
        def update(frame): #Needs frame arg as animate passes current frame no. to update()

            """Function to update figure for FuncAnimation."""
            self.run_rules()

            figure.set_data(self.lattice)

            return [figure]
        
        animation = animate(fig, update, frames = number_of_frames, interval = interval, blit = True)

        plt.show()
        return animation

    def count_live_cells(self, no_sweeps = 300):
        """Count the number of live cells in the lattice as the 
        simulation evolves and plot the result wrt time
        Inputs: number of sweeps to iterate the GoL (default = 10000)
        Outputs: No. live cells over time (array)"""

        time_data = np.arange(0, no_sweeps)
        num_live_cells = np.zeros(no_sweeps)
        for i in range(no_sweeps):
            lattice = self.run_rules()
            live_cell_count = np.sum(lattice)
            num_live_cells[i] = live_cell_count

        #plot
        plt.plot(time_data, num_live_cells, marker = ".", color = "darkblue")
        plt.xlabel(f"Time (arb. units)", fontsize = 12)
        plt.ylabel(f"Number of live cells", fontsize = 12)
        plt.title(f"Time evolution of GoL for {self.size}x{self.size} lattice \n for n={self.n}", fontsize = 16)
        plt.tick_params(axis="both", labelsize = 12)
        plt.tight_layout()
        plt.savefig(f"Livecellstrack.png")
        plt.show()

    
def main():
    #Arguments to run animation and measurements
    parser = argparse.ArgumentParser(description="Run Conway's Game of Life")
    parser.add_argument("size", type=int)
    parser.add_argument("n", type = float)
    parser.add_argument("--init", type=str, choices = ["random", "block"], default = "random")
    parser.add_argument("--plot_no_live", type=str, choices=["Y", "N"], default="N")

    args = parser.parse_args()

    game = GameOfLife(size = args.size, n = args.n, init = args.init)
    game.animate_lattice()

    time_0 = time.time()
    
    if args.plot_no_live == "Y":
        game.count_live_cells()

    time_1 = time.time()

    print(f"Selected functions run in {(time_1 - time_0)/60} minutes") 

if __name__ == "__main__":
    main()
