Requirements:
numpy v. 2.3.4 or later
numba v. 0.62.1 or later
matplotlib v. 3.10.6 or later

#Examscript.py

#This script simulates a 2D fluid with temperature T in contact with a particle reservoir of chemical potential mu.

#NB: This script was adapted from CP1. Therefore, most of the docstrings and comments may not be accurate.

#Inputs:
- size :- Length of system on one side. 
- kT: Thermal energy
- mu: Chemical potential
- --run_N_mu (optional): Whether to generate density and isothermal compressibility plots at a constant kT but ranging mu 
- --run_N_kT (optional): Whether to generate density and isothermal compressibility plots at a constant mu but ranging run_N_kT

#Outputs:
- Animation
- Optional plots and datafiles

#Example terminal input
python Examscript.py" 50 1 0.1 --run_N_mu Y --run_N_kT N

