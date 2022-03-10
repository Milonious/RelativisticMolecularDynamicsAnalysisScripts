import numpy
import cmath
import matplotlib
from matplotlib import pyplot
from cmath import pi, sqrt
from numpy import mgrid, power, exp, asarray, trapz, cos
import scipy
from scipy.optimize.optimize import vecnorm
from scipy.special import kn
from scipy.optimize import curve_fit
from numpy.core.function_base import linspace

from numpy.lib.type_check import real

#Diffusion from the Green Kubo relation for diffusion. 
#0_0001c has T = Gas Temperature From the MB Fit is: 1375424597.76K +/- 0.0155632223992%

""" 
Constant variables 
"""
n_atoms = 500
boltz = 1.380649*(10**(-23))
timestep_to_time = 10**(-12) #in seconds, so converting from picoseconds.  
timestep = 0.000001 #in metal units, so picoseconds. 0.000001
c = 2.998*(10**8)
c_metal = 2.998e+6
box_size = 28.982 #angstrom
mass = 39.95 #g/mol
mass_to_kg = (10**(-3))/(6.022*10**23)
length_to_m = 10**(-10)
metal_v_to_si = length_to_m/timestep_to_time #10^2 m*s^-1
mass_kg = mass*mass_to_kg
rest_mass_energy = mass_kg*(c**2)
box_size_m = length_to_m*box_size


print(str(n_atoms) + " particles of mass: " + str(mass_kg) + " kg in a box of size: " + str(box_size_m) + "m per side")
mean_density = n_atoms*mass_kg/(box_size_m**3)
print("mean system density = " + str(mean_density) + " kg/m^3")


#extracting and splitting the data.
vac_measure_rate = 100
#File 1  
filler_lines = 2
lammps_vacf_1 = open("tmp.nonrelvacf1.txt", "r") #contains lammps vacf

temp = []
vacf1 = []
normed_vacf1 = []
time_axis1 = []
counter = 0
time = 0
for line in lammps_vacf_1:
    if counter > 1:
        temp = line.split()
        vacf1.append(float(temp[1])*(metal_v_to_si**2))
        normed_vacf1.append(float(temp[1])*(metal_v_to_si**2)/(vacf1[0]))
        time_axis1.append(time*vac_measure_rate*timestep*timestep_to_time)
        time += 1
    counter += 1


#File 2
filler_lines = 2
lammps_vacf_2 = open("tmp.nonrelvacf2.txt", "r") #contains lammps vacf

temp = []
vacf2 = []
normed_vacf2 = []
time_axis2 = []
counter = 0
time = 0
for line in lammps_vacf_2:
    if counter > 1:
        temp = line.split()
        vacf2.append(float(temp[1])*(metal_v_to_si**2))
        normed_vacf2.append(float(temp[1])*(metal_v_to_si**2)/(vacf2[0]))
        time_axis2.append(time*vac_measure_rate*timestep*timestep_to_time)
        time += 1
    counter += 1


#File 3
filler_lines = 2
lammps_vacf_3 = open("tmp.nonrelvacf3.txt", "r") #contains lammps vacf

temp = []
vacf3 = []
normed_vacf3 = []
time_axis3 = []
counter = 0
time = 0
for line in lammps_vacf_3:
    if counter > 1:
        temp = line.split()
        vacf3.append(float(temp[1])*(metal_v_to_si**2))
        normed_vacf3.append(float(temp[1])*(metal_v_to_si**2)/(vacf3[0]))
        time_axis3.append(time*vac_measure_rate*timestep*timestep_to_time)
        time += 1
    counter += 1

#File 4
filler_lines = 2
lammps_vacf_4 = open("tmp.nonrelvacf4.txt", "r") #contains lammps vacf

temp = []
vacf4 = []
normed_vacf4 = []
time_axis4 = []
counter = 0
time = 0 
for line in lammps_vacf_4:
    if counter > 1:
        temp = line.split()
        vacf4.append(float(temp[1])*(metal_v_to_si**2))
        normed_vacf4.append(float(temp[1])*(metal_v_to_si**2)/(vacf4[0]))
        time_axis4.append(time*vac_measure_rate*timestep*timestep_to_time)
        time += 1
    counter += 1


#File 5
filler_lines = 2
lammps_vacf_5 = open("tmp.nonrelvacf5.txt", "r") #contains lammps vacf

temp = []
vacf5 = []
normed_vacf5 = []
time_axis5 = []
counter = 0
time = 0
for line in lammps_vacf_5:
    if counter > 1:
        temp = line.split()
        vacf5.append(float(temp[1])*(metal_v_to_si**2))
        normed_vacf5.append(float(temp[1])*(metal_v_to_si**2)/(vacf5[0]))
        time_axis5.append(time*vac_measure_rate*timestep*timestep_to_time)
        time += 1
    counter += 1


#Averaging the vacf. _____________________________________________________________________________________

vacf_readings = len(vacf5) #last vacf run as that is the limiting length. 
vacf_averaged = []
normed_vacf_averaged = []

for i in range(0, vacf_readings):
    vacf_averaged.append((vacf1[i] + vacf2[i] + vacf3[i] + vacf4[i] + vacf5[i])/5)
    normed_vacf_averaged.append((normed_vacf1[i] + normed_vacf2[i] + normed_vacf3[i] + normed_vacf4[i] + normed_vacf5[i])/5)

#fitting an exponential to the vacf to make integration easier. 
def vacf_exp(t, A, a):
    return(A * exp(-a*t))

vacf_averaged = asarray(vacf_averaged)
time_axis_fit = asarray(time_axis5)
exp_fit, exp_covar = curve_fit(vacf_exp, time_axis_fit, vacf_averaged, p0=[10**16, 10**(-12)])
fitted_vacf = []
for i in range(0, len(time_axis_fit)):
    fitted_vacf.append(vacf_exp(time_axis_fit[i], exp_fit[0], exp_fit[1]))

print(exp_fit[0])
print(exp_fit[1])

#Integrating to get the diffusion coefficient. ______________________________________________________________ 

D_evolution = []
time_axis_diffusion = []
D_measurements = 1000

sample_rate = vacf_readings/D_measurements
Current_D = 0
for i in range(0, D_measurements-1):
    Current_D = trapz(vacf_averaged[0:i*sample_rate], time_axis5[0:i*sample_rate])
    D_evolution.append(Current_D/3)
    time_axis_diffusion.append(time_axis5[i*sample_rate])





#Plotting Zone._______________________________________________________________________________________________ 
pyplot.plot(time_axis5[0:10000], vacf_averaged[0:10000], "k")
pyplot.figure()
#pyplot.plot(time_axis5[0:50000], fitted_vacf[0:50000], "ro")
pyplot.title("VACF for Argon at 94.66K")
pyplot.xlabel("Elapsed Time (seconds)")
pyplot.ylabel("VACF m^2/s^2")
pyplot.figure()
pyplot.plot(time_axis5[0:50000], normed_vacf_averaged[0:50000], "r.")
pyplot.title("Normalised Vacf for Rahman Conditions")
pyplot.xlabel("Elapsed Time (seconds)")
pyplot.ylabel("vacf/(initial value)")
pyplot.figure()
pyplot.plot(time_axis_diffusion, D_evolution, "b.")
pyplot.title("Evolution of Measured D over Measurement Time")
pyplot.xlabel("Elapsed Time (Seconds)")
pyplot.ylabel("Diffusion Coefficient (m^2/s)")
pyplot.show()