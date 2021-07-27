import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx
import os
import pickle as pk
from functions import *

# %%
getcontext().prec = 200
mpl.style.use('seaborn-deep')
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['text.latex.preamble'] = r'\usepackage{bm}'

# %% Figure and Data paths
FIG_PATH = 'figs/'
if not os.path.isdir(FIG_PATH):
    os.makedirs(FIG_PATH)

DATA_PATH = 'data/'
if not os.path.isdir(DATA_PATH):
    os.makedirs(DATA_PATH)

# %% Setup
N = 10
M = 3
np.random.seed(0)

#%% Graph
G = np.random.choice([0.0, 1.0], size=(N, N), p=[0.5, 0.5])
G = G + np.eye(N)
G = (G > 0) * 1.0

A = np.zeros((N,N))
A = G / G.sum(axis=0)

# %%Check that A^i converges
print('A is primitive: ', np.all(np.isclose(np.linalg.matrix_power(A, 100),
                                            np.linalg.matrix_power(A, 101))))

# %% Hypotheses
theta = np.array([1., 2., 3.]) * 0.6
var = 1
x = np.linspace(-10, 10, 1000)
dt = (max(x) - min(x)) / len(x)

#%% Initialization
np.random.seed(0)
mu_0 = np.random.rand(N,M)
mu_0 = mu_0 / np.sum(mu_0, axis = 1)[:, None]

#%% Monte Carlo runs
delta = 0.2
N_MC = 10000
N_ITER = 100

MU_mc1 = []
for i in range(N_MC):
    csi = []
    for l in range(0, N):
        csi.append(theta[0] + np.sqrt(var) * np.random.randn(N_ITER))
    csi = np.array(csi)
    if i % 10000 == 0:
        print(i)
    m_aux = np.array(asl(mu_0, csi, A, N_ITER, theta, var, delta, is_gaussian=True))
    MU_mc1.append((np.argmax(m_aux, axis=2) != 0) * 1)

#%% Compute prob. of error for agents 1 and 10
p1 = sum(MU_mc1)[:,0] / N_MC
p9 = sum(MU_mc1)[:,9] / N_MC

# %% Save data
# pk.dump((p1, p9), open( DATA_PATH + "p_error.p", "wb" ) )
# p1, p9 = pk.load( open( DATA_PATH + "p_error.p", "rb" ) )

# %% Plot prob. of error
p1[0] = 1.0
p9[0] = 1.0
f, ax = plt.subplots(1, 1, figsize=(6,3))
ax.plot(p1, color='C5', linewidth=2)
ax.plot(p9, color='C0', linewidth=2)
ax.tick_params(axis='both', which='major', labelsize=18)
ax.set_yscale('log')
ax.set_xlim([0, len(p1) - 1])
ax.set_ylim([3e-4,2])
ax.set_xlabel(r'$i$', fontsize=18)
ax.set_ylabel(r'$p_{k,i}^{(\delta)}$', fontsize=18)
ax.legend(['Agent 1', 'Agent 10'], fontsize=18, ncol=2)
ax.annotate(r'$p^{(\delta)}_{1}=7.7 \times 10^{-3}$', (.65, p1[-1]-0.0008),
            xycoords=('axes fraction', 'data'), color='k', fontsize=16, va='center')
ax.annotate(r'$p^{(\delta)}_{10}=1.8 \times 10^{-3}$', (.65, p9[-1]+0.01),
            xycoords=('axes fraction', 'data'), color='k', fontsize=16, va='center')
f.savefig(FIG_PATH + '3.pdf', bbox_inches='tight')
