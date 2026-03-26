Requirements:
numpy v. 2.3.4 or later
numba v. 0.62.1 or later
matplotlib v. 3.10.6 or later

##Checkpoint 3 Cahn-Hilliard.py

#Solves the Cahn-Hilliard equation to model a phase seperated emulsion with choosable initial fractions

##Inputs

- "size" : Size of square lattice on one side. Defaults to 100
- "phi_0": Global order parameter for the system
- "parameter": Dimensionless paramter for the system. Defaults to 1
- "space_step": Discretisations of space for algorithm. Defaults to 1
- "time_step": Discretisations of time for algorithm. Defaults to 0.02
- "--free_energy": Whether to plot free energy evolution. Defaults to N

##Outputs

- Animation of time evolution of system
- Plot of free energy evolution (optional)

##Example terminal prompt 

python "Checkpoint 3 Cahn-Hilliard.py" 100 0 1 1 0.02


##Checkpoint 3 Poisson-Electric.py

#Solves the Poisson equation for an electric potential

##Inputs

- "size": size of cubic lattice on one side. Defaults to 50
- "rho": Description of charge distribution. Defaults to monopole
- "tolerance": The minimum precision of the algorithm's convergence before it stops. Defaults to 0.000001
- "algorithm": The algorithm with which to solve the equation. Defaults to Jacobi
- "omega": If using SOR algorithm, the value of omega to use. Defaults to 1.87

Optional inputs (default to N)
- "--potential": Whether to plot potential behaviour.
- "--field": Whether to plot field behaviour.
- "--sorconv": Whether to plot SOR convergence behaviour

##Outputs

- Contour plot of potential at midpoint of box (Optional)
- Plot of radial behaviour of potential (Optional)
- Vector plot of electric field (Optional)
- Plot of radial behaviour of field strength (optional)
- Plot of SOR convergence behaviour w.r.t omega (optional)

##Example terminal prompt

python "Checkpoint 3 Poisson-Electric.py" 50 monopole 0.00001 Jacobi 1.87 --potential Y --field Y --sorconv Y


##Checkpoint 3 Poisson-Magnetic.py

#Solves the Poisson equation for an magnetic potential

##Inputs

- "size": size of cubic lattice on one side. Defaults to 50
- "J": Description of current distribution. Defaults to thinwire
- "tolerance": The minimum precision of the algorithm's convergence before it stops. Defaults to 0.000001
- "algorithm": The algorithm with which to solve the equation. Defaults to Jacobi
- "omega": If using SOR algorithm, the value of omega to use. Defaults to 1.87

Optional inputs (default to N)
- "--potential": Whether to plot potential behaviour.
- "--field": Whether to plot field behaviour.

##Outputs

- Contour plot of potential at midpoint of box (Optional)
- Plot of radial behaviour of potential (Optional)
- Vector plot of magnetic field (Optional)
- Plot of radial behaviour of field strength (optional)

##Example terminal prompt

python "Checkpoint 3 Poisson-Magnetic.py" 50 thinwire 0.00001 Jacobi 1.87 --potential Y --field Y



