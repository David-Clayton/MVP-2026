##Ising model with Monte Carlo

Python script to simulate a 2D Ising model as a lattice and its time evolution with either Glauber or Kawasaki dynamics. The time 
evolution itself is controlled by a Monte Carlo Metropolis 
algorithm.

##Checkpoint 1.py

Python script with all functions to run the simulation. The script contains methods to run an animation of the lattice with options to control the lattice size, thermal energy kT (in units of coupling constant J), and dynamical model (Glauber/Kawasaki). 

Additional methods can compute important physical properties of the model: The absolute magnetisation of the lattice, magnetic susceptbility per lattice point, energy of the lattice, and heat capacity per lattice point, all over a range of thermal energies ranging from 1 -> 3 (in units of J). Susceptibilities and heat capacities are also returned with errors based on the bootstrap algoritm. Magnetic data is calculated with Glauber dynamics. Thermal data (energy & heat capacity) can be calculated with Glauber or Kawasaki dynamics.

##Inputs:
#Non - optional
- "size": Number of lattice points on a side, default = 50 (NB: Lattice is always square).
- "kT": Thermal energy of lattice (in units of J).
- "dynamics": Dynamical model, either "Glauber" or "Kawasaki"

#Optional (all default = "N")
- "--run_mag": Whether to compute magnetic data, either "Y" or "N"
- "--run_therm_Glauber": Whether to compute thermal data with Glauber dynamics, either "Y" or "N"
- "--run_therm_Kawasaki": Whether to compute thermal data with Kawasaki dynamics, either "Y" or "N"

##Outputs:
if "--run_mag" == "Y":
    Abs_mag_Glauber.png (Magnetisation vs. thermal energy)
    Chi_Glauber.png (Susceptibility vs. thermal energy)
    IsingDataMag.csv 

if "--run_therm_Glauber" == "Y":
    Avg_E_Glauber.png (Energy vs. thermal energy)
    Heat_cap_Glauber.png (Heat capacity vs. thermal energy)
    IsingDataGlauber.csv 

if "--run_therm_Kawasaki" == "Y":
    Avg_E_Kawasaki.png (Energy vs. thermal energy)
    Heat_cap_Kawasaki.png (Heat capacity vs. thermal energy)
    IsingDataKawasaki.csv 

else:
    None

##Command line example

python "Checkpoint 1.py" 20 2 Kawasaki --run_mag Y --run_therm_Glauber Y --run_therm_Kawasaki N





