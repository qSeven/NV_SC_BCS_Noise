"""
Figure 4 main text.
"""
import csv
import numpy as np
import matplotlib.pyplot as plt

from bcs_noise_functions import current_noise_enhancement, rate_enhancement

# SC parameters
FERMI_ENERGY = 61700  # kelvin
FERMI_WAVEVECTOR = 1 / 0.06e-3  # mu m^-1
FINITE_T_GAP = 10  # kelvin
TEMP = 7.7  # kelvin

# NV parameters
BASE_NV_FREQUENCY = 1e-1  # kelvin


# Length scales
COHERENCE_LENGTH = 1 / FERMI_WAVEVECTOR * FERMI_ENERGY / FINITE_T_GAP  # mu m
MAX_LENGTH_SCALE = 1 / FERMI_WAVEVECTOR * FERMI_ENERGY / BASE_NV_FREQUENCY  # mu m
MAX_WAVEVECTOR = FERMI_WAVEVECTOR * 2
MIN_WAVEVECTOR = 1 / MAX_LENGTH_SCALE / 2 / 100
qs = np.logspace(np.log10(MIN_WAVEVECTOR), np.log10(MAX_WAVEVECTOR), num=100)

# Data storage
data = []

# Plotting Hebel-Slichter Factor vs q_tilde
# Label is hbar_omega/gap
plt.figure(figsize=(10, 6))
for relative_frequency_factor, label in zip([10, 1, 0.1, 0.01], [r'$10^{-1}$', r'$10^{-2}$', r'$10^{-3}$', r'$10^{-4}$']):
    HSfw = [current_noise_enhancement(q / FERMI_WAVEVECTOR, BASE_NV_FREQUENCY * relative_frequency_factor, FINITE_T_GAP, TEMP, FERMI_ENERGY) for q in qs]
    plt.plot(qs, HSfw, label=label)
    data.append((label, qs, HSfw))

plt.xscale('log')
plt.legend()
plt.axvline(x=1 / np.sqrt(COHERENCE_LENGTH) * np.sqrt(FERMI_WAVEVECTOR), color='g', linestyle='--')
plt.axvline(x=FERMI_WAVEVECTOR, color='r', linestyle='--')
plt.axvline(x=1 / MAX_LENGTH_SCALE, color='black', linestyle='--')
plt.axvline(x=1 / COHERENCE_LENGTH, color='b', linestyle='--')
plt.ylabel('HS factor (log scale)')
plt.title('Hebel-Slichter Factor vs q_tilde')
plt.grid(True, which="both", ls="--")
plt.show()

# Plotting T1^-1*d vs d
plt.figure(figsize=(10, 6))
for relative_frequency_factor, label in zip([10, 1, 0.1, 0.01], [r'$10^{-1}$', r'$10^{-2}$', r'$10^{-3}$', r'$10^{-4}$']):
    HSfw = [current_noise_enhancement(q / FERMI_WAVEVECTOR,
                                       BASE_NV_FREQUENCY * relative_frequency_factor, FINITE_T_GAP, TEMP, FERMI_ENERGY) for q in qs]
    Rew = [rate_enhancement(d, qs, HSfw) for d in 1 / qs]
    plt.plot(1 / qs, Rew, label=label)
    data.append((label, 1 / qs, Rew))

plt.xscale('log')
plt.legend()
plt.axvline(x=COHERENCE_LENGTH, color='b', linestyle='--')
plt.axvline(x=np.sqrt(COHERENCE_LENGTH) / np.sqrt(FERMI_WAVEVECTOR), color='g', linestyle='--')
plt.axvline(x=1 / FERMI_WAVEVECTOR, color='r', linestyle='--')
plt.grid(True, which="both", ls="--")
plt.xlabel('d')
plt.ylabel('T1^-1*d')
plt.show()

# Writing data to CSV
with open('Results/height_dependence.csv', 'w', newline='', encoding='utf8') as csvfile:
    csvwriter = csv.writer(csvfile)
    for label, x_values, y_values in data:
        csvwriter.writerow([label])
        csvwriter.writerow(['x'] + list(x_values))
        csvwriter.writerow(['y'] + list(y_values))
        csvwriter.writerow([])  # Blank line for separation
