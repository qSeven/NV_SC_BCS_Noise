"""Makes Figure 1 of main text."""
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

# Parameters for fixed gap
FERMI_ENERGY  = 20e4 # kelvin
NV_FREQUENCY = 1e-1 # kelvin
FERMI_MOMENTUM = 1/3.65e-5 # mu m^-1

HOME_FOLDER = (
    '/Users/shanekelly/Documents/Academic/Projects/Superconducting Experiment '
    '/NV_SC_BCS_Noise'
)
#name = 'niobian_first_principles'
TRIAL_NAME = 'self_consistent_'
J_ENHANCEMENT_FILE = HOME_FOLDER+'/Results/J_enhancement_'+TRIAL_NAME+'.csv'
T1_ENHANCEMENT_FILE = HOME_FOLDER+'/Results/T1_enhancement_' + TRIAL_NAME +'.csv'
RT1_V_T_FIGURE=HOME_FOLDER+'/Results/RT1_vs_T_'+TRIAL_NAME+'.png'
RJ1_V_Q_FIGURE=HOME_FOLDER+'/Results/RJ1_vs_q_'+TRIAL_NAME+'.png'

# Read J_enhancement data
temperature = []
J_enhancement_vs_T = []
with open(J_ENHANCEMENT_FILE, 'r',encoding='utf8') as file:
    reader = csv.reader(file)
    header = next(reader)
    temperature = [float(temp) for temp in header[1:]]
    for row in reader:
        J_enhancement_vs_T.append([float(value) for value in row])

# Read T1_enhancement data
T1_enhancement_vs_T = []
with open(T1_ENHANCEMENT_FILE, 'r',encoding='utf8') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header
    for row in reader:
        T1_enhancement_vs_T.append(float(row[1]))

# Plot J_enhancement vs q for different temperatures
qs = np.logspace(np.log10(1/(1/3.65e-5*20e4/1e-1)/2/100), np.log10(1/3.65e-5*2), num=100)

# old color def
cmap = get_cmap('viridis')
norm = plt.Normalize(min(temperature), max(temperature))
colors = cmap(norm(temperature))

# Define the custom color palette
colors = ["#746198","#4781a7","#589a5d","#b3af38","#b27d58","#a75051"]
cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)
norm = plt.Normalize(min(temperature), max(temperature))



MAX_LENGTH_SCALE =  1/FERMI_MOMENTUM *FERMI_ENERGY/NV_FREQUENCY #mu m
print(f"Maximum length (lmax): {MAX_LENGTH_SCALE} μm")

# mometa to consider
Q_MAX = FERMI_MOMENTUM*2
Q_MIN = 1/MAX_LENGTH_SCALE/2/100
qs = np.logspace(np.log10(Q_MIN), np.log10(Q_MAX), num=100)



plt.figure(figsize=(3+3/8, 1.4))
for i, J_enhancement in reversed(list(enumerate(J_enhancement_vs_T))):
    if i%4==1:
        plt.plot(qs[20:], J_enhancement[20:],  color=cmap(norm(temperature[i])),
                  label=f'T={temperature[i]:.1f}K')
plt.xscale('log')
plt.xlabel(r'$q$ ($\mu m^{-1}$)')
plt.ylabel(r'$R_J(q)$')

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
plt.colorbar(sm, label='Temperature (K)', ax=plt.gca())
plt.grid(True)
plt.savefig(RJ1_V_Q_FIGURE, dpi=300, bbox_inches='tight')
plt.show()


plt.rcParams['lines.linewidth'] = 1.1
# Plot T1_enhancement vs temperature
plt.figure(figsize=(3.2, 1.05), constrained_layout=True)
CRITICAL_TEMPERATURE=9.031
plt.plot(np.array(temperature)/CRITICAL_TEMPERATURE, np.array(T1_enhancement_vs_T),
           linestyle='-', color="#4781a7")
plt.xlabel(r'$T/T_c$')
plt.ylabel(r'$R_{1}$')
#plt.title('T1 enhancement vs Temperature')
plt.yticks([0, 1, 2])  # Set y-ticks to 0, 1, and 2


plt.grid(True,which='both')
#plt.tight_layout()
plt.savefig(RT1_V_T_FIGURE.replace('.png', '.pdf'))
#plt.savefig(RT1_v_T_figure, dpi=300, bbox_inches='tight')
plt.tight_layout()
plt.show()
