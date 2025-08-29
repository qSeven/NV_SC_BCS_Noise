"""
Solve_Gap_Eq_updated.py

This module provides functions to numerically solve the BCS superconducting gap equation as a
function of temperature. It uses the scipy library for numerical integration and root finding,
and matplotlib for plotting results. The main functionality includes calculating the
superconducting energy gap for a range of temperatures, plotting the results, and optionally
saving the data to a CSV file.

Functions:
-----------
gap_integrand(y, gap, temp)
    Compute the integrand for the superconducting gap equation at a given energy, gap, and
    temperature.

gap_equation(gap, var_nv, temp, debey_energy)

gap(var_nv, temp, debey_energy)
"""
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.integrate import quad
from scipy.optimize import fsolve


import cycler

color=np.roll(["#746198","#4781a7","#589a5d","#b3af38","#b27d58","#a75051"],1)
mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)

def gap_integrand(y, gap, temp):
    """
    Compute the integrand for the superconducting gap equation at given energy, gap,
    and temperature.
    """
    return np.tanh(0.5 * 1 / temp * np.sqrt(y**2 + gap**2)) / np.sqrt(y**2 + gap**2)

def gap_equation(gap, var_nv, temp, debey_energy):
    """
    Computes the BCS gap equation residual for a given gap, density of states, 
    temperature, and Debye energy.
    """
    integral, _ = quad(gap_integrand, 0, debey_energy, args=(gap, temp))
    print(f"integral={integral}")
    print(f"result={1/var_nv-integral}")
    return 1/var_nv-integral

def gap(var_nv, temp, debey_energy):
    """
    Solves for the superconducting energy gap using the given parameters.
    """
    initial_guess = 8
    gap_solution = fsolve(gap_equation, initial_guess, args=(var_nv, temp, debey_energy))
    if gap_solution[0]>0.02:
        return gap_solution[0]
    else:
        return 0.000000

CRITICAL_TEMP = 9
DEBYE_TEMP = 200.0  # in Kelvin
NV = -1/np.log(CRITICAL_TEMP/1.13/DEBYE_TEMP) # "density of staes times interaction strength"
print(f"NV:{NV}")
print(f"TC={1.13*DEBYE_TEMP*np.exp(-1/NV)}")

# Generate temperature and gap data
CRITICAL_TEMP = 9.035
temperature_below_Tc = np.concatenate((
    np.linspace(0.01, CRITICAL_TEMP - 1, 20),  # Linear scale for the lower range
    (CRITICAL_TEMP - np.logspace(-3, -0.1, 30))[::-1]  # Log scale close to T_c
))

# Adding points above T_c
temperature_above_Tc = np.linspace(CRITICAL_TEMP, CRITICAL_TEMP + 1, 5)
print(temperature_above_Tc)
print(temperature_below_Tc)
temperature = np.concatenate((temperature_below_Tc, temperature_above_Tc))


# Calculate the gap for each temperature value
gap_values = [gap(NV, T, DEBYE_TEMP) for T in temperature]
gapBeta = np.divide(gap_values, temperature)

HOME_FOLDER = ('/Users/shanekelly/Documents/Academic/Projects/Superconducting Experiment /'
               'NV_SC_BCS_Noise')
DATA_FILE = HOME_FOLDER+'/Gap Data/self_consistent_gap.csv'
FIG_FILE  = HOME_FOLDER+'/Gap_data_n_generation/self_consistent_gap.png'

# Plot the gap as a function of temperature
plt.plot(temperature, gapBeta, marker='o', linestyle='-', color='b')
plt.xlabel('Temperature (K)')
plt.ylabel('gap/K_B/T')
plt.title('Plot of Gap as a function of Temperature')
plt.show()

plt.plot(temperature, gap_values)
plt.xlabel('Temperature (K)')
plt.ylabel('gap (K)')
plt.title('Plot of Gap as a function of Temperature')
plt.show()

DATA_FILE = HOME_FOLDER+'/Gap_data_n_generation/self_consistent_gap.csv'

# Save data to CSV
SAVE_DATA = False
if SAVE_DATA:
    with open(DATA_FILE, 'w', newline='',encoding='utf8') as file:
        writer = csv.writer(file)
        for T, g in zip(temperature, gap_values):
            writer.writerow([T, g])
