"""
Makes Figure 3 of main text.
"""
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl
import cycler

plt.rcParams.update({
    "text.usetex": True})
#plt.rcParams["font.family"] = ["Helvetica"]
#plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams['font.size'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['lines.linewidth'] = 0.9
#plt.rcParams["figure.figsize"] = (3.375*1.5, 3.375/2*1.5)
plt.rc('legend',fontsize=8) 

color=np.roll(["#746198","#4781a7","#589a5d","#b3af38","#b27d58","#a75051"],1)
mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)


HOME_FOLDER = ('/Users/shanekelly/Documents/Academic/Projects/Superconducting Experiment /'
               'NV_SC_BCS_Noise')
#name = 'niobian_first_principles'
TRIAL_NAME = 'self_consistent_'
RJ1_V_Q_FIG=HOME_FOLDER+'/Results/RJ1_vs_q_'+TRIAL_NAME+'.pdf'
TRIAL_NAME = 'self_consistent'
J_V_T_FILE = HOME_FOLDER+'/Results/RJ_vs_omega_and_temp_'+TRIAL_NAME+'.csv'
RJ1_V_T = HOME_FOLDER+'/Results/RJ1_v_T'+TRIAL_NAME+'.pdf'

# Parameters for fixed gap
FINITE_T_GAP = 11 #kelvin
TEMP = 8 # kelvin
FERMI_ENERGY  = 20e4 # kelvin
NV_FREQUENCY = 1e-1 # kelvin
FERMI_WAVEVECTOR = 1/3.65e-5 # mu m^-1

# Read J_enhancement data
J_enhancement_vs_T = []
J_enhancement_vs_omega = []

with open(J_V_T_FILE, 'r',encoding='utf8') as file:
    reader = csv.reader(file)
    header = next(reader)
    temperature = [float(temp) for temp in header[1:]]
    for row in reader:
        J_enhancement_vs_omega.append([float(value) for value in row])

# old color def
cmap = get_cmap('viridis')
norm = plt.Normalize(min(temperature), max(temperature))
colors = cmap(norm(temperature))

# Define the custom color palette
colors = ["#746198","#4781a7","#589a5d","#b3af38","#b27d58","#a75051"]
cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)
norm = plt.Normalize(min(temperature), max(temperature))



COHERENCE_LENGTH = 1/FERMI_WAVEVECTOR*FERMI_ENERGY/FINITE_T_GAP #mu m
print(f"Coherence length (lc): {COHERENCE_LENGTH} μm")
MAX_LENGTH_SCALE =  1/FERMI_WAVEVECTOR *FERMI_ENERGY/NV_FREQUENCY #mu m
print(f"Maximum length (lmax): {MAX_LENGTH_SCALE} μm")

# mometa to consider
Q_MAX = FERMI_WAVEVECTOR*2
Q_MIN = 1/MAX_LENGTH_SCALE/2/100
qs = np.logspace(np.log10(Q_MIN), np.log10(Q_MAX), num=100)


if False:
    plt.figure(figsize=(3.375, 3.375/2), constrained_layout=True)
    for i, J_enhancement in reversed(list(enumerate(J_enhancement_vs_T))):
        if i%4==1:
            plt.plot(qs[20:], J_enhancement[20:],  color=cmap(norm(temperature[i])), label=f'T={temperature[i]:.1f}K')
    plt.xscale('log')
    #plt.yscale('log')
    plt.xlabel(r'$q$ ($\mu m^{-1}$)')
    plt.ylabel(r'$R_J(q)$')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    #sm.set_array([])
    #plt.legend()
    plt.colorbar(sm, label=r'$T (K)$', ax=plt.gca())
    plt.grid(True)
    plt.savefig(RJ1_V_Q_FIG, dpi=300, bbox_inches='tight')
    plt.show()


# Plot T1_enhancement vs temperature
#plt.figure(figsize=(3.375, 3.375/2), constrained_layout=True)
CRIT_TEMP=9.031
plt.figure(figsize=(3.375, 3.375/2))
for NV_FREQUENCY, JevT in zip([r'$10^{-4}$',r'$10^{-3}$',r'$10^{-2}$',r'$10^{-1}$'],
                               J_enhancement_vs_omega):
    # legend is hbar_omega/k_b Tc
    plt.plot(np.array(temperature)/CRIT_TEMP, np.array(JevT), linestyle='-', label=NV_FREQUENCY)
plt.xlabel(r'$T/T_c$')
plt.legend()
plt.ylabel(r'$R_J$')
plt.xlim(0,1.1)
plt.grid(True)
plt.subplots_adjust(right=0.93)  # Adjust the right padding

plt.tight_layout()
plt.savefig(RJ1_V_T)
plt.show()

