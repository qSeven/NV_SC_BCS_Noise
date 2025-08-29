"""Figure for T1 enhancement vs temperature for multiple coherence lengths.
Used in Supplementary Information and referee response."""
import csv
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True})
plt.rcParams['font.size'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['lines.linewidth'] = 0.9
plt.rc('legend', fontsize=8)

# Define file paths for the new data files
HOME_FOLDER = ('/Users/shanekelly/Documents/Academic/Projects/Superconducting Experiment '
               '/NV_SC_BCS_Noise')
file_paths = [
    HOME_FOLDER + '/Results/T1_enhancement_self_consistent__lc_0.050um.csv',
    HOME_FOLDER + '/Results/T1_enhancement_self_consistent__lc_0.100um.csv',
    HOME_FOLDER + '/Results/T1_enhancement_self_consistent__lc_0.250um.csv'
]

# Corresponding coherence lengths (lc) in micrometers
coherence_lengths = ['50', '100', '250']

# Initialize data storage
temperature = []
data_by_lc = []

# Read data from each file
for file_path in file_paths:
    temperature = []
    with open(file_path, 'r',encoding='utf8') as file:
        reader = csv.reader(file)
        header = next(reader)
        print(header)
        row = next(reader)
        print(row)
        T1_enhancement_vs_T = []
        for row in reader:
            T1_enhancement_vs_T.append(float(row[1]))
            temperature.append(float(row[0]))
        data_by_lc.append(T1_enhancement_vs_T)

# Plot T1_enhancement vs temperature for each coherence length
plt.figure(figsize=(3.2, 1.05), constrained_layout=True)
CRITICAL_TEMPERATURE = 9.031

for lc, data in zip(coherence_lengths, data_by_lc):
    plt.plot(np.array(temperature) / CRITICAL_TEMPERATURE, np.array(data),
             label=rf'$\xi_0 = {lc} $ nm')

plt.xlabel(r'$T/T_c$')
plt.ylabel(r'$R_{1}$')
plt.yticks([0, 1, 2])
plt.grid(True, which='both')
plt.legend()

# Save and show the plot
OUTPUT_FIGURE = HOME_FOLDER + '/Results/T1_enhancement_vs_T_multiple_lc.pdf'
plt.savefig(OUTPUT_FIGURE, dpi=300, bbox_inches='tight')
plt.show()
