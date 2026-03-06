##Checkpoint 2 GoL.py

#Python script to run a version of Conway's Game of Life

##Inputs:

- "size": The size of the square lattice on one side. Defaults to "50"
- "init": The initial state of cells in the lattice. Defaults to "random". Also can create specific objects with "blinker", "beehive" or "glider"
- "pos": If not "random" chosen for "init", where the object is initially located in the lattice. Defaults to "random"

##Outputs:

- Animation of the Game of Life
- Histogram of equilibriation times for lattice (optional)
- Motion plot of glider (optional)

#
#
#

##Checkpoint 2 SIRS.py

#Python script to simulate an epidemic with the SIRS model

##Inputs:

- "size": The size of the square lattice on one side. Defaults to "50"
- "p_si": Probability of a Susceptible cell becoming Infected
- "p_ir": Probability of an Infected cell become Recovered. Defaults to 0.5
- "p_rs": Probability of a Recovered cell becoming Susceptible
- "f_im" : Fraction of cell that are Immune. Defaults to 0.0

##Outputs:

- Animation of SIRS modelled epidemic
- Heatmap of how average infection rate varies with p_si & p_rs, where p_ir = 0.5 (optional)
- Variance plot of infection rate from p_si = 0.2 -> 0.5, where p_ir = p_rs = 0.5 (optional)
- Plot of how average infection rate varies with f_im ranging from 0 -> 1 with p_ir = p_rs = p_si = 0.5 (optional)