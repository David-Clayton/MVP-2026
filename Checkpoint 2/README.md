Requirements:
numpy v. 2.3.4 or later
numba v. 0.62.1 or later
matplotlib v. 3.10.6 or later
pandas v. 2.3.3 or later

##Checkpoint 2 GoL.py

#Python script to run a version of Conway's Game of Life

##Inputs:
#Non-optional:
- "size": The size of the square lattice on one side. Defaults to "50"
- "init": The initial state of cells in the lattice. Defaults to "random". Also can create specific objects with "blinker", "beehive" or "glider"
- "pos": If not "random" chosen for "init", where the object is initially located in the lattice. Defaults to "random"

#Optional:
- "--run_hist": Whether to obtain the time for 1000 simulations to reach steady state (equilibriation time) and plot the distribution as a histogram. Defaults to "N"
- "--get_speed": Whether to calculate the speed of a glider. Defaults to "N"

##Outputs:

- Animation of the Game of Life
- if "--run_hist" == Y:
    GoLHistogram.png (Plot)
    GoLHistogram.csv (Raw data)
- if "--get_speed" == Y:
    Glidermotion.png (Plot of displacement from lattice centre vs. time with estimated speed)
    GliderSpeed.csv (Raw displacement data)

##Example terminal input
python "Checkpoint 2 GoL.py" 50 glider middle --run_hist N --get_speed Y

#
#
#

##Checkpoint 2 SIRS.py

#Python script to simulate an epidemic with the SIRS model

##Inputs:
#Non-optional:
- "size": The size of the square lattice on one side. Defaults to "50"
- "p_si": Probability of a Susceptible cell becoming Infected
- "p_ir": Probability of an Infected cell become Recovered. Defaults to 0.5
- "p_rs": Probability of a Recovered cell becoming Susceptible
- "f_im" : Fraction of cell that are Immune. Defaults to 0.0

#Optional (all default to "N"):
- "--run_heatmap": Whether to compute and plot a heatmap of the avg. fraction of infected cells     w.r.t p_rs and p_si when p_ir = 0.5
- "--run_var": Whether to compute and plot the variance on the average infection fraction w.r.t p_si = 0.2 -> 0.5, where p_ir = p_rs = 0.5
- "--run_immun": Whether the compute the average infection fraction w.r.t f_im where p_rs = p_si = p_ir = 0.5

##Outputs:

- Animation of SIRS modelled epidemic
- if "--run_heatmap" == Y:
    Phasediagram.png (Heatmap)
    Phasediagram.csv (Raw data)
- if "--run_var" == Y:
    Infection_variance.png (Plot)
    Infectionvariance.csv (Raw data)
- if "--run_immum" == Y:
    Immunity.png (Plot)
    Immunity.csv (Raw data)

##Example terminal input
python "Checkpoint 2 SIRS.py" 50 0.3 0.4 0.3 0.0 --run_heatmap Y --run_var N --run_immun Y
