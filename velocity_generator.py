import numpy
import cmath
import matplotlib
import random
from matplotlib import pyplot
from scipy.special import kn
from cmath import cos, pi, sin, sqrt, exp
from numpy import power
from random import random, uniform

from numpy.lib.type_check import real

c_metal = 2.998e+6
c = 2.998*(10**8)
mass = 39.95 #g/mol
mass_to_kg = (10**(-3))/(6.022*10**23)
mass_kg = mass*mass_to_kg
rest_mass_energy = mass_kg*(c**2)


ratio = 0.5
n_atoms = 500
theta = 0
phi = 0
velocities = []
temp = []
#Random velocities:

for i in range(0, n_atoms):
    theta = uniform(0, 180)
    phi = uniform(0, 360)
    temp.append(c_metal * ratio * sin(theta) * cos(phi)) #calculating vx
    temp.append(c_metal * ratio * sin(theta) * sin(phi)) #calculating vy
    temp.append(c_metal * ratio * cos(theta)) #calculating vz
    velocities.append(temp)
    temp = []

"""
#Juttner-ised Velocities.
def maxwell_juttner(beta, theta, A): #theta is kT/mc^2. A fixes the normalisation that I can't calculate via algebra. 
    gamma = (1-beta**2)**(-0.5)
    return(A*((((gamma**5)*(beta**2))/(theta*kn(2, 1/theta)))*exp(-gamma/theta))) 
"""


for i in range(0, n_atoms):
    print("set    " + "atom " + str(i + 1) + "  vx " + str(real(velocities[i][0])) + " vy " + str(real(velocities[i][1])) + " vz "  + str(real(velocities[i][2])))