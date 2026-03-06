import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation as animate
import argparse
import time
import random
from numba import njit

@njit
def gol_rules(lattice, num_live_nbrs, size):
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
                if num_live_nbrs[i, j] < 2 or num_live_nbrs[i, j] > 3:
                    test_lattice[i, j] = 0
            else: 
                if num_live_nbrs[i, j] == 3:
                    test_lattice[i, j] = 1
    
    return test_lattice

class GameOfLife:

    """This class runs Conway's Game of Life on a 50x50 lattice with periodic
    boundary conditions"""

    def __init__(self, size, init = "random", position = "random"):

        """Initialise lattice with size and set of initial conditions. 
        Input: size - lattice width on each side
                init - set of initial conditions. One of "random", "blinker",
                "glider", "beehive".
                position - initial coordinates of object. One of "random" or "middle"."""

        #0 is dead. 1 is alive
        if init == "random":
            self.lattice = np.random.choice(a = [0, 1], size = (size,size))

        if init == "blinker":
            if position == "middle":
                [i,j] = [size//2, size//2]
            else:
                i = np.random.randint(0, size)
                j = np.random.randint(0, size)
                self.lattice = np.zeros(shape = (size,size))
                self.lattice[i,j] = 1
                self.lattice[i, (j+1) % size] = 1
                self.lattice[i, (j+2) % size] = 1

        elif init == "glider":
            if position == "middle":
                [i,j] = [size//2, size//2]
            else:
                i = np.random.randint(0, size)
                j = np.random.randint(0, size)
            self.lattice = np.zeros(shape=(size,size))
            self.lattice[i,j] = 1
            self.lattice[(i+1) % size, (j-1) % size] = 1
            self.lattice[(i+1) % size, (j-2) % size] = 1
            self.lattice[i, (j-2) % size] = 1
            self.lattice[(i-1) % size, (j-2) % size] = 1

        elif init == "beehive":
            if position == "middle":
                [i,j] = [size//2, size//2]
            else:
                i = np.random.randint(0, size)
                j = np.random.randint(0, size)
                self.lattice = np.zeros(shape=(size,size))
                self.lattice[i,j] = 1
                self.lattice[(i+1) % size, (j+1) % size] = 1
                self.lattice[(i+1) % size, (j+2) % size] = 1
                self.lattice[i, (j+3) % size] = 1
                self.lattice[(i-1) % size, (j+1) % size] = 1
                self.lattice[(i-1) % size, (j+2) % size] = 1

        self.size = size
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
        self.lattice = gol_rules(self.lattice, num_live_nbrs, self.size)
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

    def count_live_cells(self, no_sweeps = 5000):
        """Count the number of live cells in the lattice as the 
        simulation evolves
        Inputs: number of sweeps to iterate the GoL (default = 10000)
        Outputs: Time for GoL to come to steady state"""

        num_live_cells = []
        stdy_state_time = None #Default if it doesn't reach steady state
        for i in range(no_sweeps):
            lattice = self.run_rules()
            live_cell_count = np.sum(lattice)
            num_live_cells.append(live_cell_count)

            if len(num_live_cells) >= 10:
                if len(set(num_live_cells[-10:])) == 1:  
                    stdy_state_time = i
                    break

        return stdy_state_time

    def eq_time_histogram(self, no_sims = 1000):
        """Derive the equilibriation time for a large number of 
        simulations and plot its distribution on a histogram.
        Inputs: number of simulations to run (default = 1000)
        Outputs: Histogram of plot
                CSV of histogram data"""

        stdy_state_times = []
        for j in range(no_sims):
            #Reinitialise a random lattice
            self.lattice = np.random.choice(a = [0, 1], size = (self.size, self.size))
            time = self.count_live_cells()
            if time is not None:
                stdy_state_times.append(time)
            if j % 10 == 0:
                print(f"Simulation {j} complete")

        stdy_state_times = np.array(stdy_state_times)
     
        #Plot stdy_state_times

        plt.hist(stdy_state_times, bins = 50, density=True, color = "darkgreen")
        plt.xlabel(f"Time taken to reach steady state", fontsize = 12)
        plt.ylabel(f"Normalised frequency", fontsize = 12)
        plt.title(f"Histogram of equilibriation times with \n random initial lattice", fontsize = 16)
        plt.tick_params(axis="both", labelsize = 12)
        plt.tight_layout()
        plt.savefig(f"GoLHistogram")
        plt.show()

        #Get histogram data into array format to save to datafile
        counts, bin_edges = np.histogram(stdy_state_times, bins=100, density=True)
        lower_edges = bin_edges[:-1]
        upper_edges = bin_edges[1:]

        hist_data = np.column_stack((lower_edges, upper_edges, counts))
        np.savetxt("GoLHistogram.csv", hist_data, delimiter=",", header="Lower_bound, Upper_bound, frequency")
        
    def glider_speed(self, no_sweeps = 100):
        """Calculate the speed of a glider by tracking its centre of mass
        over time
        Inputs: number of sweeps to track glider over (default = 100)
        Outputs: Plot of glider motion
                CSV of displacement of glider with time"""

        self.lattice = self.initial_lattice.copy()
        
        #NB: A glider always has 5 live cells

        #Iterate the GoL, and calculate the distance of the CoM from the origin

        time_data = np.arange(0, no_sweeps, 1)
        position = np.zeros(len(time_data))

        pos_index = 0
        for i in time_data:
            #Find where glider is (where live cells are)
            live_cells = np.argwhere(self.lattice == 1)
            #Get CoM by averaging coordinates
            centre_of_mass = np.sum(live_cells, axis = 0)/5
            #distance from centre of lattice
            #Glider is moving along line of sight of centre of lattice
            distance_from_origin = np.sqrt((centre_of_mass[0]-self.size//2)**2 + (centre_of_mass[1]-self.size//2)**2)

            position[pos_index] = distance_from_origin

            pos_index += 1

            self.run_rules()

        #Calculate speed of glider from 100 sweep motion track
        max_distance = np.max(position)
        max_time = np.argmax(position)
        min_distance = np.min(position)
        min_time = np.argmin(position)

        speed = (max_distance - min_distance)/(max_time - min_time)

        #Plot results
        plt.plot(time_data, position, label = f"Speed={speed}")
        plt.xlabel(f"Time", fontsize = 12)
        plt.ylabel(f"Distance from centre of lattice", fontsize = 12)
        plt.title(f"Motion of CoM of glider", fontsize = 16)
        plt.tick_params(axis="both", labelsize = 12)
        plt.tight_layout()
        plt.legend()
        plt.savefig(f"Glidermotionzoomed")
        plt.show()

        #Read data to csv file
        speed_data = np.column_stack((time_data, position))
        np.savetxt("GliderSpeed.csv", speed_data, delimiter=",", header = "Time, Displacement")

        return speed

def main():
    #Arguments to run animation and measurements
    parser = argparse.ArgumentParser(description="Run Conway's Game of Life")
    parser.add_argument("size", type=int, default = 50)
    parser.add_argument("init", type=str, choices = ["random", "beehive", "glider", "blinker"], default = "random")
    parser.add_argument("pos", type=str, choices = ["random", "middle"], default = "middle")
    parser.add_argument("--run_hist", type=str, choices={"Y", "N"}, default="N")
    parser.add_argument("--get_speed", type=str, choices={"Y", "N"}, default="N")

    args = parser.parse_args()

    game = GameOfLife(size = args.size, init = args.init, position = args.pos)
    game.animate_lattice()

    time_0 = time.time()

    if args.run_hist == "Y":
        game.eq_time_histogram()

    if args.get_speed == "Y":
        game.glider_speed()

    time_1 = time.time()

    print(f"Selected functions run in {(time_1 - time_0)/60} minutes") 

if __name__ == "__main__":
    main()
