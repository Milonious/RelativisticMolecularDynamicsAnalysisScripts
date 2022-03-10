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
box_size = 28.982
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

#normalising the velocity distribution so I can do the fitting.



bins = asarray(bins)
vals = asarray(vals)
data_norm = trapz(vals, bins)
print(data_norm)
for i in range(0, len(bins)):
    vals[i] = vals[i]/data_norm

def Maxwell_Boltzmann(vel, alpha, A):#alpha = mass*c^2/2kT multiply by c^2 to get velocity in ms^-1 since we are plotting from beta. 
    return(A * alpha**(3/2) * (vel**2) * exp(-alpha*(vel**2)))



mb_fit, mb_covar = curve_fit(Maxwell_Boltzmann, bins, vals, p0 = [rest_mass_energy/(2*boltz*1300000), 1]) #You have to help curve fit along with a guess or it shits the bed. 

MB_fit = []
MB_manual = []


for i in range(0, len(bins)):
    MB_fit.append(Maxwell_Boltzmann(bins[i], mb_fit[0], mb_fit[1]))

for i in range(0, len(bins)):
    MB_manual.append(Maxwell_Boltzmann(bins[i], rest_mass_energy/(2*boltz*1300000), 1))


MB_fit_norm = trapz(MB_fit, bins)
MB_manual_norm = trapz(MB_manual, bins)


data_norm = trapz(vals, bins)

print("MB Fit Norm is " + str(MB_fit_norm))
print("MB Manual Norm is " + str(MB_manual_norm))


for i in range(0, len(bins)):
    MB_fit[i] = MB_fit[i]/MB_fit_norm
    MB_manual[i] = MB_manual[i]/MB_manual_norm

    vals[i] = vals[i]/data_norm
    bins[i] = bins[i]*c

print("Temp from fit = " + str(rest_mass_energy/(mb_fit[0]*2*boltz)) + " with error " + str(rest_mass_energy/(mb_covar[0][0]*2*boltz)) + " K")


pyplot.title("Speed Distribution From Simulation with Rahman's Conditions")
pyplot.plot(bins, MB_fit, "gx", label = "Fitted MB distribution")
#plot.plot(bins, MB_manual, "r*")
pyplot.plot(bins, vals, "bo", label = "Simulation Speed Distribution")
pyplot.legend(loc = "upper right")
pyplot.xlabel("Atom Speed (m/s)")
pyplot.ylabel("Probability")
pyplot.show()