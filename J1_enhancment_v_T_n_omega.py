"""
For Figure 3 of main text.

This script calculates the enhancement of current noise J(q, omega)
due to a superconducting film as a function of temperature for
various NV frequencies (omega). It reads the superconducting gap as a
function of temperature from a CSV file, computes the current noise enhancement
for a fixed wavevector q and a range of temperatures, and saves
 the results to a csv file.
"""
import csv
import numpy as np
import matplotlib.pyplot as plt

from bcs_noise_functions import current_noise_enhancement

# Parameters for fixed gap
ZERO_T_GAP = 16.0 # K
FERMI_ENERGY  = 20e4 # kelvin
TEMPERATURE = 7.7 # kelvin
#hbar_omega = 1e-1 # kelvin
FERMI_WAVEVECTOR = 1/3.65e-5 # mu m^-1
HEIGHT = 0.1 #mu meter


FIXED_WAVEVECTOR_Q=1e2

# Save and load files
HOME_FOLDER = '/Users/shanekelly/Documents/Academic/Projects/Superconducting Experiment /NV_SC_BCS_Noise'

TRIAL_NAME = 'self_consistent' 
GAP_FILE = HOME_FOLDER + '/Gap Data/' + 'self_consistent_gap.csv'

CURRENT_NOISE_FILE = HOME_FOLDER+'/Results/RJ_vs_omega_and_temp_'+TRIAL_NAME+'.csv'
TEMP_FIG = HOME_FOLDER+'/Results/RJ_vs_omega_and_temp_'+TRIAL_NAME+'.png'

# Print lengths
COHERENCE_LENGTH = 1/FERMI_WAVEVECTOR*FERMI_ENERGY/ZERO_T_GAP #mu m
print(f"Coherence length (lc): {COHERENCE_LENGTH} μm")

# read gap:
temperature = []
gap_read = []

with open(GAP_FILE, 'r', encoding='utf-8') as file:
    reader = csv.reader(file)
    for row in reader:
        temperature.append(float(row[0]))
        gap_read.append(float(row[1]))

print("Temperature:", temperature)
print("Gap (read):", gap_read)
gap_K = gap_read
print("Gap (K):", gap_K)


J_enhancement_vs_omega = []
for hbar_omega_over_T_c in [0.0001,0.001,0.01,0.1]:
    NV_FREQUENCY = hbar_omega_over_T_c*9.031 # kelvin
    J_enhancement_vs_T = [] 
    for gap, TEMPERATURE in zip (gap_K, temperature):
        if gap>0:
            COHERENCE_LENGTH = 1/FERMI_WAVEVECTOR*FERMI_ENERGY/gap #mu m
            print(f"Coherence length (lc): {COHERENCE_LENGTH} μm")
        else:
            print(gap==0)

        NORMALIZED_WAVEVECTOR_Q = FIXED_WAVEVECTOR_Q/FERMI_WAVEVECTOR # q_tilde
        J_enhancement_vs_T.append(current_noise_enhancement(NORMALIZED_WAVEVECTOR_Q, NV_FREQUENCY, gap, TEMPERATURE, FERMI_ENERGY))
    J_enhancement_vs_omega.append(J_enhancement_vs_T)
    
with open(CURRENT_NOISE_FILE, 'w', newline='', encoding='utf8') as file:
    writer = csv.writer(file)
    writer.writerow(["Temperature (K)"] + temperature)
    for row in J_enhancement_vs_omega:
        writer.writerow(row)

# Plot T1_enhancement vs temperature
plt.figure(figsize=(10, 6))
for NV_FREQUENCY, JevT in zip([0.001,0.01,0.1,1], J_enhancement_vs_omega):
    plt.plot(temperature, np.array(JevT), linestyle='-', label=f'omega={NV_FREQUENCY}')
plt.xlabel('Temperature (K)')
plt.ylabel(r'$\frac{T_{1,SC}^{-1}}{T_{1,N}^{-1}}$')
plt.title('height: 100nm, fermi energy: 18eV, kf=2.7e4 1/(μm) ')
plt.grid(True)
plt.savefig(TEMP_FIG, dpi=300,bbox_inches='tight')
plt.show()