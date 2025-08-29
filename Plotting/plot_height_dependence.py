"""Figure 4 of main text."""
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cycler
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from bcs_noise_functions import coherence_length

plt.rcParams.update({
    "text.usetex": True})
plt.rcParams["font.family"] = ["Helvetica"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams['font.size'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['lines.linewidth'] = 0.9
#plt.rcParams["figure.figsize"] = (3.375*1.5, 3.375/2*1.5)
plt.rc('legend',fontsize=8)

# Colors
color = np.roll(["#746198", "#4781a7", "#589a5d", "#b3af38", "#b27d58", "#a75051"], 1)
mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)

# Data Paramters
HOME_FOLDER = ('/Users/shanekelly/Documents/Academic/Projects/Superconducting Experiment /'
               'NV_SC_BCS_Noise')
FIG_LOCATION = HOME_FOLDER+'/Results/height_dependence.pdf'

# SC Parameters
FINITET_GAP    = 11  # kelvin
FERMI_ENERGY   = 60000  # kelvin
FERMI_MOMENTUM = 30000  # mu m^-1
COHERENCE_LENGTH = coherence_length(FERMI_MOMENTUM, FERMI_ENERGY, FINITET_GAP)  # mu m

# NV Parameters
NV_FREQUENCY   = 1e-1  # kelvin

# Define momentum range
MAX_LENGTH_SCALE = 1 / FERMI_MOMENTUM * FERMI_ENERGY / NV_FREQUENCY  # mu m
Q_MAX = FERMI_MOMENTUM * 2
Q_MIN = 1 / MAX_LENGTH_SCALE / 2 / 100
qs = np.logspace(np.log10(Q_MIN), np.log10(Q_MAX), num=100)

# Function to read data from CSV
def read_csv(filename):
    """
    Reads a custom-formatted CSV file and extracts labeled x and y data arrays.
    Returns a list of tuples: (label, x_values, y_values) as numpy arrays.
    Assumes each data block is separated by an empty line in the CSV.
    """
    data = []
    with open(filename, 'r',encoding='utf8') as csvfile:
        csvreader = csv.reader(csvfile)
        label, x_values, y_values = None, [], []
        for row in csvreader:
            if not row:
                if label:
                    data.append((label, np.array(x_values, dtype=float),
                                 np.array(y_values, dtype=float)))
                label, x_values, y_values = None, [], []
            elif len(row) == 1:
                label = row[0]
            elif row[0] == 'x':
                x_values = row[1:]
            elif row[0] == 'y':
                y_values = row[1:]
        if label:
            data.append((label, np.array(x_values, dtype=float), np.array(y_values, dtype=float)))
    return data


# Read data from CSV
data = read_csv(HOME_FOLDER+'/Results/height_dependence.csv')

PLOT_HS_V_Q = False
if PLOT_HS_V_Q:
    # Plotting Hebel-Slichter Factor vs q_tilde
    plt.figure(figsize=(10, 6))
    for label, x_values, y_values in data[:3]:
        plt.plot(x_values, y_values, label=label)

    plt.xscale('log')
    plt.legend()
    plt.axvline(x=1 / np.sqrt(COHERENCE_LENGTH) * np.sqrt(FERMI_MOMENTUM), color='g',linestyle='--')
    plt.axvline(x=FERMI_MOMENTUM, color='r', linestyle='--')
    plt.axvline(x=1 / MAX_LENGTH_SCALE, color='black', linestyle='--')
    plt.axvline(x=1 / COHERENCE_LENGTH, color='black', linestyle='--')
    plt.ylabel('HS factor (log scale)')
    plt.title('Hebel-Slichter Factor vs q_tilde')
    plt.grid(True, which="both", ls="--")
    plt.show()

# Plotting T1^-1*d vs d
plt.figure(figsize=(3.375, 3.375/2*0.9), constrained_layout=True)
tick_labels = {
    "$10^{-1}$": 1/10,
    "$10^{-2}$": 1/100,
    "$10^{-3}$": 1/1000,
    "$10^{-4}$": 1/10000
}
for label, x_values, y_values in data[4:]:
    if label != "$10^{-4}$":
        plt.plot(x_values[:-10], y_values[:-10], label=label)
        NV_FREQUENCY = tick_labels[label]
        qmin_kc = np.sqrt(10*NV_FREQUENCY * FINITET_GAP / (2*FERMI_ENERGY**2 ))* FERMI_MOMENTUM
        d = 1 / qmin_kc
        print(label)

plt.xlim(3e-4,1e4)
plt.xscale('log')
plt.legend()
#plt.axvline(x=0.231, color='black', linestyle='--')
plt.axvline(x=COHERENCE_LENGTH, color='black', linestyle='--')
plt.grid(True, which="both", ls="--")
plt.xlabel(r'$d$ ($\mu m$)')
plt.ylabel(r'$R_{1}$')
plt.tight_layout()
plt.savefig(FIG_LOCATION)
plt.show()
