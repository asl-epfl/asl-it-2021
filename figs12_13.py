import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from functions import *

# %%
getcontext().prec = 200
mpl.style.use('seaborn-deep')
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['text.latex.preamble'] = r'\usepackage{bm}'

# %% Figure path
FIG_PATH = 'figs/'
if not os.path.isdir(FIG_PATH):
    os.makedirs(FIG_PATH)

# %% Setup
N = 10
M = 3
np.random.seed(0)

SETUP = 'SLOW'
# SETUP = 'FAST'

# Transition matrices
if SETUP == 'SLOW':
    N_ITER = 10000
    dx = 1e-4
    fig_name = ['13a.pdf', '13b.pdf', '13c.pdf']
    seed = [1234, 4, 12]
    ylim_plot = [0.18, .5]

else:
    N_ITER = 1000
    dx = 1e-3
    fig_name = ['12a.pdf', '12b.pdf', '12c.pdf']
    seed = [100, 15, 6]
    ylim_plot = [0.16, .54]

T_hyp = np.array([[1-5*dx, 5*dx, 0], [5*dx, 1-10*dx, 5*dx], [0, 5*dx, 1-5*dx]])
T_mat = np.array([[1-dx, dx], [dx, 1-dx]])
T_var = np.array([[1-dx, dx, 0], [dx, 1-2*dx, dx], [0, dx, 1-dx]])

# ASL step size
delta = 1e2 * dx

# Noise profiles
var_list = np.array([0, 5, 500])

# %% Graph generation
G = np.random.choice([0.0, 1.0], size=(N, N), p=[0.5, 0.5])
G = G + G.T + np.eye(N)
G = (G > 0) * 1.0

# %% Averaging Rule -> Left-stochastic matrix
A_ls = np.zeros((N, N))
A_ls = G / G.sum(axis = 0)
A_ls_dec = np.array([[Decimal(x) for x in y] for y in A_ls])

# %% Laplacian Rule -> Doubly-stochastic matrix
A_ds = np.zeros((N, N))
deg = G.sum(axis = 0)
dmax = np.max(deg)
A_ds[G > 0] = 1 / dmax
deg_n = deg - np.diag(G)
np.fill_diagonal(A_ds, 1 - deg_n * 1/ dmax)

A_ds_dec = decimal_array(A_ds)

# %%Check that A^i converges
print('Left-stochastic A is primitive: ',
      np.all(np.isclose(np.linalg.matrix_power(A_ls, 100), np.linalg.matrix_power(A_ls, 101))),
      ' Doubly-stochastic A is primitive: ',
      np.all(np.isclose(np.linalg.matrix_power(A_ds, 100), np.linalg.matrix_power(A_ds, 101))))

# %% Hypotheses
theta = np.arange(1, 4) * 1.0
theta_dec = decimal_array(theta)

b = 1

x = np.linspace(-10, 10, 1000)
x_dec = decimal_array(x)

dt = (max(x)-min(x))/len(x)
dt_dec = (max(x_dec)-min(x_dec))/len(x_dec)

#%% Initialization
np.random.seed(0)
mu_0 = np.random.rand(N, M)
mu_0 = mu_0 / np.sum(mu_0, axis = 1)[:, None]

#%% Markov sequences for the hypothesis, comb. matrix and noise profile:
# Seed values are chosen empirically to allow a clear illustration of the changing variables

# Hypothesis
np.random.seed(seed[0])
theta_vector = []
for i in range(N_ITER):
    if i == 0:
        theta_vector.append(np.random.randint(0, 3))
    else:
        theta_vector.append(np.random.choice([0, 1, 2], 1, p = list(T_hyp[theta_vector[i-1]]))[0])


# Comb. matrix
np.random.seed(seed[1])
A_list = [A_ls, A_ds]

A_vector, state_mat = [], []
for i in range(N_ITER):
    if i == 0:
        state_mat.append(np.random.randint(0, 2))
    else:
        state_mat.append(np.random.choice([0, 1], 1, p = list(T_mat[state_mat[i-1]]))[0])
    A_vector.append(A_list[state_mat[i]])


# Noise profile
np.random.seed(seed[2])

state_var = []
for i in range(N_ITER):
    if i == 0:
        state_var.append(np.random.randint(0, 3))
    else:
        state_var.append(np.random.choice([0, 1, 2], 1, p = list(T_var[state_var[i-1]]))[0])

# %% Generate observations
csi = []
for l in range(0, N):
    csi.append(np.random.laplace(theta[theta_vector], b) +
               np.sqrt(var_list[state_var]) * np.random.randn(len(theta_vector)))
csi = np.array(csi)
csidec = decimal_array(csi)

#%% Simulate ASL
MU = asl_markov(mu_0, csi, A_vector, N_ITER, theta, b, delta)

#%% Plot ASL
f, ax = plt.subplots(1, 1, figsize=(8, 3))
ax.plot(np.array([MU[k][4] for k in range(len(MU))]))
ax.set_xlim([0, N_ITER])
ax.set_ylim(ylim_plot)
ax.tick_params(axis='both', which='major', labelsize=16)
ax.set_xlabel(r'$i$', fontsize=16)
ax.set_ylabel(r'$\bm{\mu}_{1,i}(\theta)$', fontsize=16)
plt.legend([r'$\theta=1$', r'$\theta=2$', r'$\theta=3$'],
           fontsize=16, ncol=3, handlelength=1, loc='center', bbox_to_anchor=[0.5, -0.47])
plt.subplots_adjust(bottom=0.35)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(FIG_PATH + fig_name[0])

#%% Simulate SL
MU_sl = sl_markov(mu_0, csidec, A_vector, N_ITER, theta_dec, b)

#%%
f, ax = plt.subplots(1,1, figsize=(8,3))
ax.plot(np.array([MU_sl[k][0] for k in range(len(MU_sl))]))
ax.set_xlim([0, N_ITER])
ax.tick_params(axis='both', which='major', labelsize=16)
ax.set_xlabel(r'$i$', fontsize=16)
ax.set_ylabel(r'$\bm{\mu}_{1,i}(\theta)$', fontsize=16)
plt.legend([r'$\theta=1$', r'$\theta=2$', r'$\theta=3$'],
           fontsize=16, ncol=3, handlelength=1, loc = 'center', bbox_to_anchor=[0.5,-0.47])
plt.subplots_adjust(bottom=0.35)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.savefig(FIG_PATH + fig_name[2])


#%% Prob. of error
N_MC = 1000
MU_all = []
for n in range(N_MC):
    csi = []
    for l in range(0, N):
        csi.append(np.random.laplace(theta[theta_vector], b) +
                   np.sqrt(var_list[state_var]) * np.random.randn(len(theta_vector)))
    csi = np.array(csi)

    MU_all.append(asl(mu_0, csi, A_vector, N_ITER, theta, b, delta))
    if n % 100 == 0:
        print(n)

mu_pe = np.array([[MU_all[j][k][0] for k in range(len(MU))] for j in range(N_MC)])
acc = np.sum(np.equal(np.argmax(mu_pe, axis=2)[:, :-1], np.array(theta_vector)), axis=0) / N_MC

#%% Plot prob. of error
f, ax = plt.subplots(1, 1, figsize=(8,2))
ax.plot(1 - acc)
ax.set_xlim([0, N_ITER])
ax.tick_params(axis='both', which='major', labelsize=16)
ax.set_xlabel(r'$i$', fontsize=16)
ax.set_ylabel(r'$p^{(0.01)}_{1,i}(\theta)$', fontsize=16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(FIG_PATH + fig_name[1], bbox_inches='tight')

