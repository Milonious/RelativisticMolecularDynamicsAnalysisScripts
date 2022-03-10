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
speed_to_si = 10**2
box_size = 28.982
mass = 39.95 #g/mol
mass_to_kg = (10**(-3))/(6.022*10**23)
length_to_m = 10**(-10)
mass_kg = mass*mass_to_kg
rest_mass_energy = mass_kg*(c**2)
box_size_m = length_to_m*box_size

epsilon = 0.01034 #ev
sigma = 3.4*(10**(-10)) #m
r_cut = sigma*(2**(1/6))



print(str(n_atoms) + " particles of mass: " + str(mass_kg) + " kg in a box of size: " + str(box_size_m) + "m per side")
mean_density = n_atoms*mass_kg/(box_size_m**3)
print("mean system density = " + str(mean_density) + " kg/m^3")
"""
data_extraction

"""

"""actual data starts after line 9. every n_atom lines, you have to skip another 9"""

filler_lines = 9
speeds_and_positions = open("dump.relcorrectioncheck.txt", "r")
timestepped_raw_data = []
timestep_temp = []
counter = 0
number_of_timesteps = 0
for line in speeds_and_positions:
    if counter > filler_lines-1 and counter < (filler_lines + n_atoms - 1): #loading all of the data points from the same timestep into one array. 
        timestep_temp.append(line.split())
    elif counter > filler_lines + n_atoms - 1: #resetting the collection after ever timestep for ease of formatting later.  
        counter = 0
        timestepped_raw_data.append(timestep_temp)
        number_of_timesteps += 1
        timestep_temp = []
    counter += 1


#raw data has id = 0 vx = 1 vy = 2 vz = 3 x = 4 y = 5 z = 6 for each atom, seperated with all data from a single timestep in the same sub array. [timestep][atom]
 
F_ratio = []
F_classical_by_timestep = []
F_classical_temp = 0
F_retarded_by_timestep = []
F_retarded_temp = 0

for time in range(0, 100):
    for i in range(0, len(timestepped_raw_data[time])): #in my note's jargon, i == atom 1, j == atom 2. 
        for j in range(0, len(timestepped_raw_data[time])):
            if i != j:
                #do math. make computer go brrrrrr. Converting metal units to SI as we go so i don't forget. 
                rx = (float(timestepped_raw_data[time][j][4]) - float(timestepped_raw_data[time][i][4]))*length_to_m
                if rx > box_size_m/2:
                    rx = rx - box_size_m/2
                
                ry = (float(timestepped_raw_data[time][j][5]) - float(timestepped_raw_data[time][i][5]))*length_to_m
                if ry > box_size_m/2:
                    ry = ry - box_size_m/2

                rz = (float(timestepped_raw_data[time][j][6]) - float(timestepped_raw_data[time][i][6]))*length_to_m
                if rz > box_size_m/2:
                    rz = rz - box_size_m/2
                
                r_mag = real(sqrt((rx**2 + ry**2 + rz**2)))
                if r_mag <= r_cut:
                    #force calculation, following my notes on my ipad.
                    v_2 = speed_to_si*[float(timestepped_raw_data[time][j][1]), float(timestepped_raw_data[time][j][2]), float(timestepped_raw_data[time][j][3])]
                    r_unit = [rx/r_mag, ry/r_mag, rz/r_mag] #the unit vector along the interatom seperation. 
                    r_vec = [rx, ry, rz]
                    delta_t = (r_mag/c) * (1 + r_unit[0]*(v_2[0]/c) + r_unit[1]*(v_2[1]/c) + r_unit[2]*(v_2[2]/c)) #to first order. Can change if needed.
                    #the classical lennard jones force: 
                    F_lj = 24*epsilon*((2*((sigma**12)/(r_mag**13))) - ((sigma**6)/(r_mag**7))) #magnitude of the classical force in eV/m
                    F_classical_vec = [F_lj*r_unit[0], F_lj*r_unit[1], F_lj*r_unit[1]]  #classical force vector. 

                    #calculating the retarded force vector. 
                    F_lj_grad = 24*epsilon*((-26*((sigma**12)/(r_mag**14))) - ((-7)*((sigma**6)/(r_mag**8))))
                    r_dot_v = (v_2[0]*r_vec[0]) + (v_2[1]*r_vec[1]) + (v_2[2]*r_vec[2])
                    ret_term_1_vec = [-delta_t*(r_dot_v/(r_mag**2))*F_lj_grad*r_vec[0], -delta_t*(r_dot_v/(r_mag**2))*F_lj_grad*r_vec[1], -delta_t*(r_dot_v/(r_mag**2))*F_lj_grad*r_vec[2]]#added minus signs here to see if that changes anything. 
                    ret_term_2_vec = [delta_t*(r_dot_v/(r_mag**3))*F_lj*r_vec[0], delta_t*(r_dot_v/(r_mag**3))*F_lj*r_vec[1], delta_t*(r_dot_v/(r_mag**3))*F_lj*r_vec[2]]  
                    ret_term_3_vec = [-delta_t*(F_lj/r_mag)*v_2[0], -delta_t*(F_lj/r_mag)*v_2[1], -delta_t*(F_lj/r_mag)*v_2[2]] 
                    tot_ret_vec = [ret_term_1_vec[0] + ret_term_2_vec[0] + ret_term_3_vec[0], ret_term_1_vec[1] + ret_term_2_vec[1] + ret_term_3_vec[1], ret_term_1_vec[2] + ret_term_2_vec[2] + ret_term_3_vec[2]] 
                    tot_ret_mag = real(sqrt(((tot_ret_vec[0]**2) + (tot_ret_vec[1]**2) + (tot_ret_vec[2]**2)))) #magnitude of the retarded force in eV/m
                    
                    F_classical_temp += F_lj
                    F_retarded_temp += tot_ret_mag
                
    print("made it through timestep: " + str(time+1) + " out of " + str(100) + " . yay!")
    print("Correction term: " + str(F_retarded_temp))
    print("Classical term: " + str(F_classical_temp))
    print("Ratio of Correction to Classical: " + str(F_retarded_temp/F_classical_temp))
    print("")
    F_retarded_by_timestep.append(F_retarded_temp)
    F_classical_by_timestep.append(F_classical_temp)
    F_ratio.append(F_retarded_temp/F_classical_temp)
    F_classical_temp = 0
    F_retarded_temp = 0

pyplot.plot(F_classical_by_timestep[:100], "b.")
pyplot.plot(F_retarded_by_timestep[:100], "r.")
pyplot.figure()
pyplot.plot(F_ratio, "k*")
pyplot.show()                    
   