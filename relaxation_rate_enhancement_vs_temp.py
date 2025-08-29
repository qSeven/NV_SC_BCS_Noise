"""
Script generates data for Figure 1 theory paper, theory data in experimental paper
and last Figure in SM.

This script calculates the enhancement of the NV center T1 relaxation rate 
due to a superconducting film.  It reads the superconducting gap as a 
function of temperature from a CSV file, computes the current noise enhancement 
and T1 relaxation rate enhancement for a range of temperatures, and saves
 the results to csv files.

For a fixed fermi energy, NV frequency, and NV height,
the coherence length can be varied to see its effect on the results.
"""
import csv
import numpy as np
import matplotlib.pyplot as plt

from bcs_noise_functions import current_noise_enhancement, rate_enhancement, coherence_length

# SC film Parameters for fixed gap
FERMI_ENERGY  = 60e3 # kelvin
FERMI_WAVEVECTOR = 30000 # mu m^-1

# NV Parameters
NV_FREQUENCY_K = 1e-1 # kelvin
HEIGHT = 0.1 #mu meter

# Data Files to files
HOME_FOLDER = ('/Users/shanekelly/Documents/Academic/Projects/Superconducting Experiment /'
               'NV_SC_BCS_Noise')

TRIAL_NAME = 'self_consistent_'
GAP_FILE = HOME_FOLDER + '/Gap Data/' + 'self_consistent_gap.csv'
CONVERT_TO_KELVIN = False
ZERO_T_GAP = 16.0 # K


# Print lengths
ENABLE_VARIABLE_LC = False
if ENABLE_VARIABLE_LC:
    COHERENCE_LENGTH = 250e-3 # mu
    FERMI_WAVEVECTOR = 2/COHERENCE_LENGTH*FERMI_ENERGY/ZERO_T_GAP/np.pi #mu m
    MAX_LENGTH_SCALE =  1/FERMI_WAVEVECTOR *FERMI_ENERGY/NV_FREQUENCY_K #mu m
    J_ENHANCEMENT_FILE = (HOME_FOLDER + f'/Results/J_enhancement_{TRIAL_NAME}_lc_'
                         f'{COHERENCE_LENGTH:.3f}um.csv')
    T1_ENHANCEMENT_FILE = (HOME_FOLDER + f'/Results/T1_enhancement_{TRIAL_NAME}_lc_'
                          f'{COHERENCE_LENGTH:.3f}um.csv')
    RT1_FIG = HOME_FOLDER+'/Results/'+TRIAL_NAME+'.png'
else:
    COHERENCE_LENGTH = coherence_length(FERMI_WAVEVECTOR, FERMI_ENERGY, ZERO_T_GAP) # mu m
    MAX_LENGTH_SCALE =  1/FERMI_WAVEVECTOR *FERMI_ENERGY/NV_FREQUENCY_K #mu m
    J_ENHANCEMENT_FILE = HOME_FOLDER+'/Results/J_enhancement_'+TRIAL_NAME+'.csv'
    T1_ENHANCEMENT_FILE = HOME_FOLDER+'/Results/T1_enhancement_' + TRIAL_NAME +'.csv'
    RT1_FIG = HOME_FOLDER+'/Results/'+TRIAL_NAME+'.png'

print(f"Coherence length (lc): {COHERENCE_LENGTH} μm")
print(f"Maximum length (lmax): {MAX_LENGTH_SCALE} μm")

# Momentum range for integration in current noise enhancement J(q, omega)
qmax = FERMI_WAVEVECTOR*2
qmin = 1/MAX_LENGTH_SCALE/2/100
qs = np.logspace(np.log10(qmin), np.log10(qmax), num=100)

# READ GAP DATA
temperature = []
gap_read = []

with open(GAP_FILE, 'r', encoding='utf-8') as file:
    reader = csv.reader(file)
    for row in reader:
        temperature.append(float(row[0]))
        gap_read.append(float(row[1]))

if CONVERT_TO_KELVIN:
    gap_K = 11.605*np.array(gap_read)
else:
    gap_K = gap_read

# Calculate T1 enhancement vs temperature
T1_enhancement_vs_T = []
J_enhancement_vs_T = []
for gap, kbT in zip (gap_K, temperature):
    if gap>0:
        COHERENCE_LENGTH = coherence_length(FERMI_WAVEVECTOR, FERMI_ENERGY, gap) # mu m
        print(f"Coherence length (lc): {COHERENCE_LENGTH} μm")
    else:
        print(gap==0)

    J_enhancement = []
    for q in qs:
        q_tilde = q/FERMI_WAVEVECTOR
        J_enhancement.append(current_noise_enhancement(q_tilde, NV_FREQUENCY_K, gap,
                                                       kbT, FERMI_ENERGY))
    J_enhancement_vs_T.append(J_enhancement)
    RT1en = rate_enhancement(HEIGHT, qs, J_enhancement)
    T1_enhancement_vs_T.append(RT1en)

# Save data to CSV
with open(J_ENHANCEMENT_FILE, 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["Temperature (K)"] + temperature)
    for row in J_enhancement_vs_T:
        writer.writerow(row)

with open(T1_ENHANCEMENT_FILE, 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["Temperature (K)", "T1 Enhancement"])
    for temp, T1_enh in zip(temperature, T1_enhancement_vs_T):
        writer.writerow([temp, T1_enh])

# Plot T1_enhancement vs temperature
plt.figure(figsize=(10, 6))
plt.plot(temperature, np.array(T1_enhancement_vs_T), marker='o', linestyle='-', color='b')
plt.xlabel('Temperature (K)')
plt.ylabel(r'$\frac{T_{1,SC}^{-1}}{T_{1,N}^{-1}}$')
plt.title('height: 100nm, NV frequency: 2.08 GHz (1K), fermi energy: 18eV, kf=2.7e4 1/(μm) ')
plt.grid(True)
plt.savefig(RT1_FIG, dpi=300,bbox_inches='tight')
plt.show()
