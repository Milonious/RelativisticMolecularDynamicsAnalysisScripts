import numpy
import cmath
import matplotlib
import math
from math import erf
from matplotlib import pyplot
from cmath import pi, sqrt 
from numpy import mgrid, power, exp, asarray, dot, trapz
import scipy
from scipy.optimize.zeros import RootResults
from scipy.special import kn
from scipy.optimize import curve_fit
from numpy.core.function_base import linspace
from numpy.lib.type_check import real



""" 
Constant variables 
"""
n_atoms = 500
boltz = 1.380649*(10**(-23))
timestep_to_time = 10**(-12) #in seconds, so converting from picoseconds.  
timestep = 1e-6 
c = 2.998*(10**8)
c_metal = 2.998e+6
box_size = 23.0
mass = 39.95 #g/mol
mass_to_kg = (10**(-3))/(6.022*10**23)
length_to_m = 10**(-10)
mass_kg = mass*mass_to_kg
rest_mass_energy = mass_kg*(c**2)
box_size_m = length_to_m*box_size
print(str(n_atoms) + " particles of mass: " + str(mass_kg) + " kg in a box of size: " + str(box_size_m) + "m per side")
mean_density = n_atoms*mass_kg/(box_size_m**3)
print("mean system density = " + str(mean_density) + " kg/m^3")

filler_lines = 9
hist_file = open("500_atoms_0_0001c_beta_histogram_data", "r")
raw_data = []
counter = 0
for line in hist_file:
    raw_data.append(line.split())

bins = []
vals = []
for i in range(0, len(raw_data)):
    bins.append(float(raw_data[i][0])) #multiply by c to get velocity in ms^-1. 
    vals.append(float(raw_data[i][1]))


bins = asarray(bins)
vals = asarray(vals)


def Maxwell_Juttner(beta, theta, A): #theta is kT/mc^2. A fixes the normalisation that I can't calculate via algebra. 
    gamma = (1-beta**2)**(-0.5)
    return(A*((((gamma**5)*(beta**2))/(theta*kn(2, 1/theta)))*exp(-gamma/theta)))

guessed_juttner = []
for i in range(0, len(bins)):
    guessed_juttner.append(Maxwell_Juttner(bins[i], 0.085, 0.04))

mj_fit_vals, mj_fit_covar = curve_fit(Maxwell_Juttner, bins, vals, p0 = [0.085, 0.04])
fitted_juttner = []
for i in range(0, len(bins)):
    fitted_juttner.append(Maxwell_Juttner(bins[i], mj_fit_vals[0], mj_fit_vals[1]))

print("Temperature from the Fit = " + str(mj_fit_vals[0]*rest_mass_energy/boltz) + " with error " + str(mj_fit_covar[0][0]*rest_mass_energy/boltz))


pyplot.plot(bins, vals, "b*", label = "Simulation Speed Distribution")
pyplot.plot(bins, fitted_juttner, "ro", label = "Fitted MB distribution")
pyplot.xlabel("Atom Beta")
pyplot.ylabel("Probability")
pyplot.title("Speed Distribution From Simulation at 0.1c")
pylot.legend(loc = "upper right")
pyplot.show()
