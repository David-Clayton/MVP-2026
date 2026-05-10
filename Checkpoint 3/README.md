Requirements:
numpy v. 2.3.4 or later
numba v. 0.62.1 or later
matplotlib v. 3.10.6 or later
scipy v. 1.16.2 or later

##Checkpoint 3 Cahn-Hilliard.py

#Solves the Cahn-Hilliard equation to model a phase seperated emulsion with choosable initial fractions

##Inputs
#Non-optional
- "size" : Size of square lattice on one side. Defaults to 100
- "phi_0": Global order parameter for the system
- "parameter": Dimensionless paramter for the system. Defaults to 1
- "space_step": Discretisations of space for algorithm. Defaults to 1
- "time_step": Discretisations of time for algorithm. Defaults to 0.02
#Optional
- "--free_energy": Whether to plot free energy time evolution. Defaults to "N"

##Outputs

- Animation of time evolution of system
- if "--free_energy" == "N":
    Free_energy_{phi_0}.png (Plot)
    Free_energy_{phi_0}.png (Plot)


##Example terminal input

python "Checkpoint 3 Cahn-Hilliard.py" 100 0 1 1 0.02 --free_energy Y

#
#
#

##Checkpoint 3 Poisson-Electric.py

#Solves the Poisson equation for an electric potential, with either the Jacobi, Gauss-Seidel or Successive Over-Relation (SOR) algorithms

##Inputs
#Non-optional
- "size": size of cubic lattice on one side. Defaults to 50
- "rho": Description of charge distribution. Defaults to monopole
- "tolerance": The minimum precision of the algorithm's convergence before it stops. Defaults to 0.000001
- "algorithm": The algorithm with which to solve the equation. Defaults to Jacobi
- "omega": If using SOR algorithm, the value of omega to use. Defaults to 1.87

#Optional (default to "N")
- "--potential": Whether to plot potential behaviour.
- "--field": Whether to plot field behaviour.
- "--sorconv": Whether to plot SOR convergence behaviour

##Outputs
- if  "--potential" == Y:
    Electricpotential.png (Heatmap of potential at midpoint of box)
    Electricpotential.csv (Raw data)
    if "rho" == "monopole"
    Electricpotentialrad.png (Plot of radial behaviour of potential, with theoretical ~1/r proportionality relation)
    Electricpotentialdist.csv (Raw data)
- if "--field" == Y:
    Efield.png (Vector map of electric field at midpoint in x-y plane)
    Exfield.csv (Raw data by component)
    Eyfield.csv
    Ezfield.csv
    Efieldstrength.csv (Raw data of magnitude)
    if "rho" == "monopole"
    Electricfieldrad.png (Plot of radial behaviour of field, with theoretical ~1/r^2 proportionality relation)
    Electricfielddist.csv (Raw data)

- if "--sorconv" == Y:
    SORconvergence.png (Plot of convergence behaviour w.r.t omega)
    SORconvergence.csv (Raw data)

##Example terminal input

python "Checkpoint 3 Poisson-Electric.py" 50 monopole 0.00001 Jacobi 1.87 --potential Y --field Y --sorconv Y

#
#
#

##Checkpoint 3 Poisson-Magnetic.py

#Solves the Poisson equation for an magnetic potential, with either the Jacobi, Gauss-Seidel or Successive Over-Relation (SOR) algorithms.

##Inputs
#Non-optional
- "size": size of cubic lattice on one side. Defaults to 50
- "J": Description of current distribution. Defaults to thinwire
- "tolerance": The minimum precision of the algorithm's convergence before it stops. Defaults to 0.000001
- "algorithm": The algorithm with which to solve the equation. Defaults to Jacobi
- "omega": If using SOR algorithm, the value of omega to use. Defaults to 1.87

Optional (default to N)
- "--potential": Whether to plot potential behaviour.
- "--field": Whether to plot field behaviour.

##Outputs

 if  "--potential" == Y:
    Magneticpotential.png (Heatmap of potential at midpoint of box)
    Magneticicpotential.csv (Raw data)
    if "J" == "thinwire":
    Magneticpotentialrad.png (Plot of radial behaviour of potential, with theoretical ~ln(r) proportionality relation)
    Magneticpotentialdist.csv (Raw data)
- if "--field" == Y:
    Bfield.png (Vector map of magnetic field at midpoint in x-y plane)
    Bxfield.csv (Raw data by component)
    Byfield.csv
    Bfieldstrength.csv (Raw data of magnitude)
    if "J" == "thinwire"
    Magneticfieldrad.png (Plot of radial behaviour of field, with theoretical ~1/r proportionality relation)
    Magneticicfielddist.csv (Raw data)

##Example terminal input

python "Checkpoint 3 Poisson-Magnetic.py" 50 thinwire 0.00001 Jacobi 1.87 --potential Y --field Y



