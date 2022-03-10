import numpy
import cmath
import matplotlib
import math
from math import erf
from matplotlib import pyplot
from cmath import pi, sqrt 
from numpy import mgrid, power, exp, asarray, dot, trapz
import scipy
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
timestep = 1e-11 
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
"""
data_extraction

"""

"""actual data starts after line 9. every n_atom lines, you have to skip another 9"""

filler_lines = 9
speeds_and_positions = open("dump.nonrelveloc.txt", "r")
raw_data = []
counter = 0
for line in speeds_and_positions:
    if counter > filler_lines-1 and counter < (filler_lines + n_atoms - 1):
        raw_data.append(line.split())
    elif counter > filler_lines + n_atoms - 1:
        counter = 0    
    counter += 1




mag_v = []
temp1 = []
temp2 = []
line = []
clean_velocity_data = []
clean_position_data = []
clean_epair_data = []


#splitting velocity, position, epair data._________________________________________________________________________________
for line in raw_data:
    for j in range(0, len(line)):
        if j < 3:
            temp1.append(float(line[j]))
        elif j >= 3 and j < 6:
            temp2.append(float(line[j]))
        elif j == 6:
            clean_epair_data.append(float(line[j]))
    clean_velocity_data.append(temp1)
    clean_position_data.append(temp2)
    temp1 = []
    temp2 = []
second_counter = 0
counter = 0
for line in raw_data:
    if second_counter > 8:
        if counter < n_atoms:
            if j < 3:
                temp1.append(float(line[j]))
            elif j >= 3 and j < 6:
                temp2.append(float(line[j]))
            elif j == 6:
                clean_epair_data.append(float(line[j]))
            counter += 1
        if counter == n_atoms:
            counter = 0
    second_counter += 1
print("length of clean pair energy data " + str(len(clean_epair_data)))
print("length of velocity data " + str(len(clean_velocity_data)))
print("ratio of the 2: " + str(len(clean_velocity_data)/len(clean_epair_data)))

#Saving the previous Run for input into the new:
previous_velocities = clean_velocity_data[-n_atoms:]
previous_positions = clean_position_data[-n_atoms:]
counter = 1
with open("previous_velocities.txt", "w") as f:
    for line in previous_velocities:
        f.write("set    atom " + str(counter) + "  vx " + str(line[0]) + " vy " + str(line[1]) + " vz " + str(line[2]) + "")
        f.write("\n")
        counter += 1
counter = 1
with open("previous_positions.txt", "w") as f:
    for line in previous_positions:
        f.write("set    atom " + str(counter) + "  x " + str(line[0]) + " y " + str(line[1]) + " z " + str(line[2]) + "")
        f.write("\n")
        counter += 1
#Making speed data. _______________________________________________________________________________________________________________

holder = 0
counter = 0
temp = []
mag_v_by_timestep = []
gamma = []
for i in range(0, len(clean_velocity_data)):
    if counter == n_atoms:
        mag_v_by_timestep.append(temp)
        temp = []
        counter = 0
    for value in clean_velocity_data[i]:
        holder += value**2
    mag_v.append(float(real(sqrt(holder)))/c_metal)
    gamma.append(sqrt(1/(1-((float(real(sqrt(holder)))/c_metal)**2))))
    temp.append(float(real(sqrt(holder)))/c_metal)
    holder = 0
    counter += 1

no_start_mag_v = mag_v[n_atoms*1:] #*number of measurement timesteps to skip. 
no_start_mag_v.sort()
mag_v.sort()
v_hist_vals = []
v_hist_bins = []
n_v_bins = 50
dv = max(mag_v)/n_v_bins #max magv for low speed, 1.0 for high speed. 
#making the velocity histrograms to fill. 
for i in range(0, n_v_bins):
    v_hist_bins.append(dv*i)
    v_hist_vals.append(0)

for i in range(0, n_v_bins-1):  #doing the no start mag v to avoid 0.7c spike. 
    for j in range(0, len(no_start_mag_v)):
        if no_start_mag_v[j] <= v_hist_bins[i+1] and no_start_mag_v[j] > v_hist_bins[i]:
            v_hist_vals[i] += 1

#calculating the normalization factor.
v_norm = trapz(v_hist_vals, v_hist_bins)             


for i in range(0, n_v_bins):
    v_hist_vals[i] = v_hist_vals[i]/v_norm

v_norm_check = trapz(v_hist_vals, v_hist_bins)
#checking to make sure that the normalisation worked:


print("velocity data norm = " + str(v_norm_check) + " should = 1")

#saving the histogram 
with open(str(n_atoms) + "_atoms_0_0001c_beta_histogram_data", "w") as f:
    for i in range(0, len(v_hist_bins)):
        f.write(str(v_hist_bins[i]) + "  " + str(v_hist_vals[i]))
        f.write("\n")
#LAMMPS Version Energy Conservation. ---------------------------------------------------------------------
energies = open("tmp.nonrelenergy.txt", "r")
#data starts after line 0, 1 --> data
raw_data = []
counter = 0
for line in energies: #timestep e_pe e_ke
    if counter > 1:    
        raw_data.append(line.split())
    counter += 1
e_time = []
e_pe = []
e_ke = []
e_tot = []
for line in raw_data:
    e_time.append(float(line[0]))
    e_pe.append(float(line[1]))
    e_ke.append(-float(line[2]))
    e_tot.append(float(line[1]) - float(line[2]))


#LAMMPS G(r)______________________________________________________________________________________________________________
radial_data = open("tmp.nonrelrdf.txt", "r")
#first 0 1 2 3 lines have garbage in them. after that, 0->n_gbins-1 have data, and then the ngbins line is another line of garbage. 
raw_data = []
counter = 0
second_counter = 0
n_g_bins = 100
for line in radial_data:
    if second_counter > 3:
        if counter < n_g_bins:
            raw_data.append(line.split())
        if counter == n_g_bins:
            counter = -1
        counter += 1
    second_counter += 1

g_hist_bins = []
g_hist_vals = []
#data now has 0:step  1:bin  2:rdf 3:????
for line in raw_data[0:n_g_bins]:
    g_hist_bins.append(float(line[1]))
    g_hist_vals.append(0.0)


counter = 0
line_number = 0
for line in raw_data:
    if counter < n_g_bins:
        g_hist_vals[counter] += float(line[2])
        counter += 1
    if counter == n_g_bins:
        counter = 0
        line_number += 1

#print(g_hist_vals)
for i in range(0, len(g_hist_vals)):
    g_hist_vals[i] = g_hist_vals[i]/line_number #normalising by number of histogram measurements. 

#Fitting MJ distribution to the velocity histogram, ___________________________________________________

v_fitting_bins = asarray(v_hist_bins)
v_fitting_vals = asarray(v_hist_vals)

"""
def maxwell_juttner(beta, theta, A): #theta is kT/mc^2. A fixes the normalisation. I think I have an algebra mistake along the way. 
    gamma = (1-beta**2)**(-0.5)
    return(((((A*beta*(gamma**2))/theta*kn(2, 1/theta))*exp(-gamma/theta)))*(beta/((1-beta**2)**1.5)))



fit, covar = curve_fit(maxwell_juttner, v_fitting_bins, v_fitting_vals)
v_fit_vals = maxwell_juttner(v_fitting_bins, fit[0], fit[1])
print(max(v_fitting_bins))
"""

def maxwell_juttner_2(beta, theta, A): #theta is kT/mc^2. A fixes the normalisation that I can't calculate via algebra. 
    gamma = (1-beta**2)**(-0.5)
    return(A*((((gamma**5)*(beta**2))/(theta*kn(2, 1/theta)))*exp(-gamma/theta)))

def maxwell_boltzmann_beta(beta, alpha): #alpha = (m*c^2)/(2*kb*T)
    return((alpha**(3/2)) * (4/(pi**(1/2))) * ((c*beta)**2) * exp(-alpha*(beta**2)))

#MJ Fit. ____________________________________________________________________________
fit, covar = curve_fit(maxwell_juttner_2, v_fitting_bins, v_fitting_vals)
v_fit_vals = maxwell_juttner_2(v_fitting_bins, fit[0], fit[1])
print(max(v_fitting_bins))
print(fit)
print(covar)

v_norm_check = 0
#checking to make sure that the normalisation worked:
for i in range(0, n_v_bins):
    v_norm_check += v_fitting_bins[i]*v_fit_vals[i]

print("fitted velocity data norm = " + str(v_norm_check) + " should = 1")
v_fit_vals = v_fit_vals/v_norm_check
perr = (numpy.diag(covar))**0.5
error_on_A = perr[1]/fit[1]
error_on_sigma = perr[0]/fit[0]
#T = sigma*restmass/K_b
temperature = fit[0]*rest_mass_energy/boltz

print("Error on sigma = " + str(error_on_sigma) + " %")
print("Error on A = " + str(error_on_A) + " %")
print("Gas Temperature From the MJ Fit is: " + str(temperature) + "K +/- " + str(error_on_sigma) + "%")

#Maxwell-Boltzmann Fit. _________________________________________________________________

#in terms of beta
mb_fit, mb_covar = curve_fit(maxwell_boltzmann_beta, v_fitting_bins, v_fitting_vals)
mb_v_fit_vals = maxwell_boltzmann_beta(v_fitting_bins, mb_fit[0])

#Checking to make sure my mb function is actually reasonable.


mb_v_norm_check = 0
#checking to make sure that the normalisation worked:
for i in range(0, n_v_bins):
    mb_v_norm_check += v_fitting_bins[i]*mb_v_fit_vals[i]

print("MB fitted velocity data norm = " + str(mb_v_norm_check) + " should = 1")
mb_v_fit_vals = mb_v_fit_vals/mb_v_norm_check
mb_perr = (numpy.diag(mb_covar))**0.5
mb_error_on_sigma = mb_perr[0]/mb_fit[0]
mb_temperature = rest_mass_energy/(2*boltz*mb_fit[0])

print("MB Error on sigma = " + str(mb_error_on_sigma) + " %")
print("Gas Temperature From the MB Fit is: " + str(mb_temperature) + "K +/- " + str(mb_error_on_sigma) + "%")


#in terms of velocity________________________________________________________________________________________________________________________
v_v_hist_bins = []
for i in range(0, len(v_hist_bins)):
    v_v_hist_bins.append(v_hist_bins[i]*c_metal)

#renormalising the measured v histogram.
v_v_norm = 0
for i in range(0, len(v_hist_bins)):
    v_v_norm += v_v_hist_bins[i]*v_hist_vals[i]

v_v_hist_vals = []
for i in range(0, len(v_hist_bins)):
    v_v_hist_vals.append(v_hist_vals[i]/v_v_norm)

v_norm_check = 0 #just checking to make sure that ive normalised everything properly. 
for i in range(0, len(v_hist_bins)):
    v_norm_check += v_v_hist_bins[i]*v_v_hist_vals[i]

print("v v norm = " + str(v_norm_check) + ", should be 1")
v_v_fitting_bins = asarray(v_v_hist_bins) #bins in velocity
v_v_fitting_vals = asarray(v_v_hist_vals)

def maxwell_boltzmann_velo(vel, a): #alpha = (m*c^2)/(2*kb*T) #Still not normalised correctly... FUCKKKK
    return((4*(a**(3/2))/(pi**(1/2))) * (vel**2) * exp((vel**2)*(-a)))

v_mb_fit, mb_covar = curve_fit(maxwell_boltzmann_velo, v_v_fitting_bins, v_v_fitting_vals, p0 = [150000])
v_mb_v_fit_vals = maxwell_boltzmann_velo(v_v_fitting_bins, v_mb_fit[0])


print("T from the velocity fit = " + str((mass_kg)/(2*boltz*v_mb_fit[0])) + " K")


v_v_fit_norm = trapz(v_mb_v_fit_vals, v_v_fitting_bins)
print("v v fit norm = " + str(v_v_fit_norm) + ", should be 1")

#Checking to make sure that my mbv distribution makes sense:

expected_v_vals = maxwell_boltzmann_velo(v_v_fitting_bins, (mass_kg/(2*boltz*130000)))
expected_v_norm = trapz(expected_v_vals, v_v_fitting_bins)


print("expected distribution has norm: " + str(expected_v_norm))
for i in range(0, len(expected_v_vals)):
    expected_v_vals[i] = expected_v_vals[i]/expected_v_norm

fixed_v_norm = trapz(expected_v_vals, v_v_fitting_bins)

print("normalised expected distribution has norm: " + str(fixed_v_norm))

#Plotting zone. ________________________________________________________________________________________________________________

pyplot.plot(e_time, e_tot)
pyplot.xlabel("Simulation Time (" + str(timestep*timestep_to_time) + " seconds)")
pyplot.ylabel("Energy (eV)")
pyplot.title("total energy")
pyplot.savefig("Total_Energy_" + str(n_atoms) + "_atoms.png")
pyplot.figure()
pyplot.plot(e_time, e_pe)
pyplot.title("potential energy")
pyplot.xlabel("Simulation Time (" + str(timestep*timestep_to_time) + " seconds)")
pyplot.ylabel("Energy (eV)")
pyplot.savefig("Potential_Energy_"+ str(n_atoms) + "_atoms.png")
pyplot.figure()
pyplot.plot(e_time, e_ke)
pyplot.title("kinetic energy")
pyplot.xlabel("Simulation Time (" + str(timestep*timestep_to_time) + " seconds)")
pyplot.ylabel("Energy (eV)")
pyplot.savefig("Kinetic_Energy_"+ str(n_atoms) + "_atoms.png")
pyplot.figure()
pyplot.plot(e_time, e_ke, label = "total kinetic energy (relativistic)")
pyplot.plot(e_time, e_pe, label = "total potential energy")
pyplot.title("pe vs ke comparison")
pyplot.xlabel("Simulation Time (" + str(timestep*timestep_to_time) + " seconds)")
pyplot.ylabel("Energy (eV)")
pyplot.legend()
pyplot.savefig("Kinetic_vs_Potential_"+ str(n_atoms) + "_atoms.png")
pyplot.figure()

#making beta histogram.
pyplot.plot(v_hist_bins, v_hist_vals, "ro")
pyplot.plot(v_fitting_bins, v_fit_vals, "bx")
pyplot.plot(v_fitting_bins, mb_v_fit_vals, "k*")
pyplot.title("Beta Distribution")
pyplot.ylabel("P(v/c)")
pyplot.xlabel("v/c_metal")
pyplot.savefig("beta_distribution_"+ str(n_atoms) + "_atoms.png")
pyplot.figure()

#velocity histograms 
pyplot.plot(v_v_fitting_bins, v_v_hist_vals, "ro")
pyplot.plot(v_v_fitting_bins, v_mb_v_fit_vals, "bx")
pyplot.plot(v_v_fitting_bins, expected_v_vals, "k*")
pyplot.title("velocity Distribution")
pyplot.ylabel("P(v/c)")
pyplot.xlabel("v (ms^-1)")
pyplot.savefig("velo_distribution_"+ str(n_atoms) + "_atoms.png")
pyplot.figure()

#making g(r) histogram.
pyplot.plot([3.7, 7.0, 10.4], [2.8, 1.25, 1.1], "rx", markersize = 20, label = "Rahman g(r) Peaks")
pyplot.plot(g_hist_bins, g_hist_vals, "bo", label = "Measured RDF")
pyplot.title("Radial Distribution Function at 94.6K")
pyplot.ylabel("g(r)")
pyplot.xlabel("Distance (Angstrom)")
pyplot.legend(loc = "upper right")
pyplot.savefig("Radial_Distribution_" + str(n_atoms) + "_atoms.png")

pyplot.show()
