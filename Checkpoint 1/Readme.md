Requirements:
numpy v. 2.3.4 or later
numba v. 0.62.1 or later
matplotlib v. 3.10.6 or later

##Checkpoint 1.py

Python script to simulate the time evolution of a 2D Ising model with either Glauber or Kawasaki dynamics, with a Monte Carlo Metropolis algorithm.

##Inputs:
#Non - optional
- "size": Number of lattice points on a side, default = 50 (NB: Lattice is always square).
- "kT": Thermal energy of lattice (in units of J).
- "dynamics": Dynamical model, either "Glauber" or "Kawasaki"

#Optional (all default = "N")
- "--run_mag": Whether to compute specific magnetic susceptibilty per lattice pt. and avg. magnetisation as function of kT with Glauber dynamics, either "Y" or "N"
- "--run_therm_Glauber": Whether to compute average energy and specific heat capacity per lattice pt. with Glauber dynamics, either "Y" or "N"
- "--run_therm_Kawasaki": Whether to compute average energy and specific heat capacity per lattice pt. with Kawasaki dynamics, either "Y" or "N"

##Outputs:
Animation of the Ising model

if "--run_mag" == "Y":
    Abs_mag_Glauber.png (Magnetisation vs. thermal energy)
    Chi_Glauber.png (Susceptibility vs. thermal energy)
    IsingDataMag.csv (Combined data file)

if "--run_therm_Glauber" == "Y":
    Avg_E_Glauber.png (Energy vs. thermal energy)
    Heat_cap_Glauber.png (Heat capacity vs. thermal energy)
    IsingDataGlauber.csv (Combined data file)

if "--run_therm_Kawasaki" == "Y":
    Avg_E_Kawasaki.png (Energy vs. thermal energy)
    Heat_cap_Kawasaki.png (Heat capacity vs. thermal energy)
    IsingDataKawasaki.csv (Combined data file)

##Command line example

python "Checkpoint 1.py" 20 2 Kawasaki --run_mag Y --run_therm_Glauber Y --run_therm_Kawasaki N





