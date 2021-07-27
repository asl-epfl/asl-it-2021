import numpy as np
import networkx as nx
import os
import pickle as pk
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.stats as stats
from functions import *

#%%
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

# %% Graph
G = np.random.choice([0.0, 1.0],size=(N, N), p=[0.5, 0.5])
G = G + np.eye(N)
G = (G > 0) * 1.0

# Averaging Rule -> Left-stochastic matrix
lamb = .5
A = np.zeros((N, N))
A = G / G.sum(axis=0)

# %% Plot network
Gr = nx.from_numpy_array(A)
pos = nx.shell_layout(Gr, scale=0.9)
f,ax = plt.subplots(1, 1, figsize=(5, 2.5))
plt.axis('off')
plt.xlim([-1.1, 1.1])
plt.ylim([-1.1, 1.1])
nx.draw_networkx_nodes(Gr, pos=pos, node_color= 'C5',nodelist=range(0,N), node_size=400, edgecolors='k', linewidths=.5)
nx.draw_networkx_labels(Gr, pos,{i: i+1 for i in range(N)}, font_size=16, font_color='black', alpha = 1)
nx.draw_networkx_edges(Gr, pos = pos, node_size=400, alpha=1, arrowsize=6, width=1);
plt.tight_layout()
plt.savefig(FIG_PATH + '6.pdf', bbox_inches='tight', pad_inches = 0)
# %%Check that A^i converges
print('A is primitive: ', np.all(np.isclose(np.linalg.matrix_power(A, 100),
                                            np.linalg.matrix_power(A, 101))))

# %% Hypotheses
theta = np.arange(1,4) * 0.1
b = 1
x = np.linspace(-10, 10, 1000)
dt = (max(x) - min(x)) / len(x)
# %% Likelihoods
L0 = laplace(x, theta[0], b)
L1 = laplace(x, theta[1], b)
L2 = laplace(x, theta[2], b)
L = np.array([L0, L1, L2])
#%%
plt.figure(figsize=(6, 3))
plt.plot(x, L0, linewidth=2)
plt.plot(x, L1, linewidth=2)
plt.plot(x, L2, linewidth=2)
plt.xlim([-3, 3])
plt.tick_params(axis='both', which='major', labelsize=16)
plt.xlabel(r'$\xi$', fontsize=16)
plt.ylabel(r'$f_n(\xi)$', fontsize=16)
plt.legend([r'$n=1$', r'$n=2$', r'$n=3$'], fontsize=16)
plt.savefig(FIG_PATH + '7.pdf', bbox_inches='tight')

# %% Initialization
np.random.seed(0)
mu_0 = np.random.rand(N, M)
mu_0 = mu_0/np.sum(mu_0, axis = 1)[:, None]
delta = 0.1

#%% Compute Cave and mave
_, pv = np.linalg.eig(A)
pv = np.real(pv[:,0] / sum(pv[:,0]))

dklv = np.array([0]*3+[DKL(L0, L2, dt)]*3+[DKL(L0, L1, dt)]*4)
ave1 = pv.dot(dklv)
print('m_{ave}(theta=2)=', ave1)

dklv = np.array([DKL(L0, L2, dt)]*3+[DKL(L0, L2, dt)]*3+[0]*4)
ave2 = pv.dot(dklv)
print('m_{ave}(theta=3)=', ave2)

Ave = np.array([ave1, ave2])
cave = 0.5 * np.array([[(pv**2).dot(np.array([Cave(L, 0, 0, dt)]*3 + [Cave(L, 2, 2, dt)]*3 + [Cave(L, 1, 1, dt)]*4)),
                    (pv**2).dot(np.array([Cave(L, 0, 2, dt)]*3 + [Cave(L, 2, 2, dt)]*3 + [Cave(L, 1, 0, dt)]*4))],
                   [(pv**2).dot(np.array([Cave(L, 2, 0, dt)]*3+[Cave(L, 2, 2, dt)]*3+[Cave(L, 0, 1, dt)]*4)),
                    (pv**2).dot(np.array([Cave(L, 2, 2, dt)]*3+[Cave(L, 2, 2, dt)]*3+[Cave(L, 0, 0, dt)]*4))]])
print('C_{ave}=', cave)
#%% ##################################GAUSSIAN APPROXIMATION######################################
N_MC = 100
N_ITER = 10000
delta = [0.5, 0.1, 0.01, 0.001]

#%% Monte Carlo Simulations
MU_delta = []
for d in delta:
    MU_mc1 = []
    for i in range(N_MC):
        csi=[]
        for l in range(0, N):
            csi.append(np.random.laplace(theta[0], b, size=N_ITER))
        csi = np.array(csi)

        MU_mc1.append(asl(mu_0, csi, A, N_ITER, theta, b, d))
    MU_delta.append(MU_mc1)

#%% Save data
# pk.dump(MU_delta, open(DATA_PATH + "ex_gaussian.p", "wb" ) )
# MU_delta = pk.load(open(DATA_PATH + "ex_gaussian.p", "rb" ) )

#%% Plot Gaussian approximation
data = np.array([[MU_delta[j][i][-1][0] for i in range(N_MC)] for j in range(len(delta))])
#%%
f, ax = plt.subplots(2, 2, figsize=(8, 7))
for di, d in enumerate(delta):
    bv1 = np.log(data[di][:, 0] / data[di][:, 1])
    bv2 = np.log(data[di][:, 0] / data[di][:, 2])
    Lbv = np.array([bv1, bv2])
    Cdelta = np.cov(Lbv)
    Avedelta = np.mean(Lbv, axis=1)

    confidence_ellipse(Lbv[0], Lbv[1], ax[di//2, di%2], n_std=1, edgecolor='C0', linewidth=2, linestyle='dashed')
    h2 = confidence_ellipse(Lbv[0], Lbv[1], ax[di//2, di%2], n_std=2, edgecolor='C0', linewidth=2, linestyle='dashed')
    gaussian_ellipse(d*cave, Ave, ax[di//2, di%2], n_std=1, edgecolor='C2', linewidth=2, linestyle='dotted')
    h3 = gaussian_ellipse(d*cave, Ave, ax[di//2, di%2], n_std=2, edgecolor='C2', linewidth=2, linestyle='dotted')

    h1 = ax[di//2, di%2].scatter(bv1, bv2, color = 'C5')
    ax[di//2, di%2].scatter(Avedelta[0], Avedelta[1], color='C0', marker='o', facecolors='None', s=100, linewidth=2)
    ax[di//2, di%2].scatter(ave1, ave2, color='C2', marker='+', s=100, linewidth=2)
    ax[di//2, di%2].tick_params(axis='both', which='major', labelsize=18)
    ax[di//2, di%2].set_xlabel(r'$\bm{\lambda}^{(\delta)}_{1, i}(\theta=2)$', fontsize=18)
    ax[di//2, di%2].set_ylabel(r'$\bm{\lambda}^{(\delta)}_{1, i}(\theta=3)$', fontsize=18)
    ax[di//2, di%2].set_title(r'$\delta={}$'.format(d), fontsize=20)

    if di == 0:
        ax[di//2, di%2].set_xlim(Ave[0]-.07, Ave[0]+.07)
        ax[di//2, di%2].set_ylim(Ave[1]-.12, Ave[1]+.12)

    if di == 1:
        ax[di//2, di%2].set_xlim(Ave[0]-.025, Ave[0]+.025)
        ax[di//2, di%2].set_ylim(Ave[1]-.05, Ave[1]+.05)

    if di == 2:
        ax[di//2, di%2].set_xlim(Ave[0]-.008,Ave[0]+.008)
        ax[di//2, di%2].set_ylim(Ave[1]-.012,Ave[1]+.012)

    if di == 3:
        ax[di//2, di%2].set_xlim(Ave[0]-.0025, Ave[0]+.0025)
        ax[di//2, di%2].set_ylim(Ave[1]-.0045, Ave[1]+.0045)

f.legend([h1, h2, h3],['Data \nsamples', 'Empirical \nGaussian distribution', 'Limiting \nGaussian distribution'],
         ncol=3, loc='center', bbox_to_anchor=(0.5, 0.06), fontsize=17, handlelength=1)
f.tight_layout(rect=(0, 0.1, 1, 1))
f.savefig(FIG_PATH + '9.pdf', bbox_inches='tight')
#%% ################################CONSISTENCY#####################################
np.random.seed(0)
mu_0 = np.random.rand(N,M)
mu_0 = mu_0/np.sum(mu_0, axis=1)[:, None]
N_ITER = 10000
delta =np.logspace(-3, 0, 50)

#%% Run Monte Carlo simulations
MU_mc = []
for di in delta:
    csi=[]
    for l in range(0, N):
        csi.append(np.random.laplace(theta[0], b, size= N_ITER))
    csi=np.array(csi)
    MU_mc.append(asl(mu_0, csi, A, N_ITER, theta, b, di))

#%% Save data
# pk.dump( MU_mc, open( DATA_PATH+"ex_consist.p", "wb" ) )
# MU_mc = pk.load( open( DATA_PATH+"ex_consist.p", "rb" ) )
#%%
beliefv = np.array([MU_mc[i][-1][0] for i in range(len(MU_mc))])
beliefv9 = np.array([MU_mc[i][-1][9] for i in range(len(MU_mc))])

logbv1 = np.log(beliefv[:,0] / beliefv[:,1])
logbv2 = np.log(beliefv[:,0] / beliefv[:,2])

f, ax = plt.subplots(1,1, figsize=(6,3))
ax.set_xlim(1e-3, 1)
ax.set_ylim(-0.07, .1)
ax.set_xscale('log')
ax.invert_xaxis()
ax.plot(delta, ave1 * np.ones(len(delta)), color='k', linewidth=2, linestyle=':')
ax.plot(delta, ave2 * np.ones(len(delta)), color='k', linewidth=2, linestyle=':')
h1 = ax.plot(delta, logbv1, linewidth=2, color='C1', linestyle='-', marker='o')
h2 = ax.plot(delta, logbv2, linewidth=2, color='C2', linestyle='-', marker='o')
ax.legend([h1[0], h2[0]],[r'$\theta=2$',r'$\theta=3$'], ncol=2, fontsize=16, handlelength=1, loc='lower right')
ax.annotate(r'${\sf m}_{\sf ave}(\theta)=%0.4f$' % ave1, (.65, ave1-0.025),
            xycoords=('axes fraction', 'data'), color='k', fontsize=16)
ax.annotate(r'${\sf m}_{\sf ave}(\theta)=%0.4f$' % ave2, (.65, ave2+0.015),
            xycoords=('axes fraction', 'data'), color='k', fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=16)
ax.set_xlabel(r'$\delta$', fontsize=16)
ax.set_ylabel(r'$\bm{\lambda}^{(\delta)}_{1,i}(\theta)$', fontsize=16)
f.savefig(FIG_PATH + '8.pdf', bbox_inches='tight')

#%% ################################LARGE DEVIATIONS#####################################
N_MC = 20000
N_ITER = 3000
idelta = np.linspace(10, 200, 20)
delta = 1 / idelta

#%% Run Monte Carlo simulations
MU_huge =[]
for d in delta:
    MU_mc1 = []
    for i in range(N_MC):
        csi = np.random.laplace(theta[0], b, size=(N, N_ITER))
        MU_mc1.append(asl(mu_0, csi, A, N_ITER, theta, b, d)[-1])
    MU_huge.append(MU_mc1)

#%% Save data
# pk.dump(idelta, N_ITER, N_MC, MU_huge, open( DATA_PATH + "large_deviation.p", "wb" ) )
# idelta, N_ITER, N_MC, MU_huge = pk.load( open( DATA_PATH + "large_deviation_code.p", "rb" ) )

#%% Compute error exponent
t = np.linspace(-30, 30, 10000)
w = np.sum(np.array([0 for pk in pv[:3]] + [lmgf(theta[0], theta[2], pk * t) for pk in pv[3:6]]+
                    [lmgf(theta[0], theta[1], pk * t) for pk in pv[6:]]), axis=0)
w2 = np.sum(np.array([lmgf(theta[0], theta[2], pk * t) for pk in pv[:3]]+
                     [lmgf(theta[0], theta[2], pk * t) for pk in pv[3:6]]+[0 for pk in pv[6:]]), axis=0)
aux = w / t * (max(t) - min(t)) / len(t)
aux2 = w2 / t * (max(t) - min(t)) / len(t)
int1 = [np.sum(aux[10000//2:i]) - np.sum(aux[i:10000//2]) for i in range(len(t))]
int2 = [np.sum(aux2[10000//2:i]) - np.sum(aux2[i:10000//2]) for i in range(len(t))]
phi1 = -min(int1)
phi2 = -min(int2)
phi = min(-min(int1), -min(int2))

#%% Compute approximations
# Large deviations approximation
iy = np.linspace(1, 201, 50)
y = np.exp(-phi * iy)
y2 = np.exp(-phi2 * iy)
ys = y + y2

# Normal approximation
rv = [stats.multivariate_normal(Ave, cave * 1 / d) for d in iy]
norm_ap = np.array([rv[i].cdf(np.array([10, 0])) + rv[i].cdf(np.array([0, 10])) -
                    rv[i].cdf(np.zeros(2)) for i in range(len(iy))])

#%% Plot probability of error
m = ['o', 'v', 's', '*', 'x']
p = np.sum(np.argmax(np.array(MU_huge), axis=3) != 0, axis=1) / N_MC

f, ax = plt.subplots(1, 1, figsize=(6,5))
ax.plot(iy, 0.1 * y, color='red', linewidth=2, linestyle='dashed', zorder=0)
ax.plot(iy, norm_ap, color='blue', linewidth=2.5, linestyle='dotted', zorder=0)
ax.set_yscale('log')
ax.set_xlim(9, 201)
ax.tick_params(axis='both', which='major', labelsize=16)
ax.tick_params(axis='both', which='minor', labelsize=14)
ax.set_xlabel(r'$1/\delta$', fontsize=16)
ax.set_ylabel(r'$p_k^{(\delta)}$', fontsize=16)

h=[]
for k in [0, 2, 5, 6]:
    h.append(ax.scatter(idelta, p[:,k], marker = m[k//2], color='C%d' %(k//2), s=100, linewidth=2, facecolors='None'))
h.append(ax.scatter(idelta, p[:,8], marker = m[9//2], color='C%d' %(9//2), s=100, linewidth=2))

ax.legend(h,['Agent %d' %(d+1) for d in [0, 2, 5, 6, 8]], fontsize=16, loc='upper right')
ax.annotate('Markers: simulation\n Dots: Gaussian approx. (Th. 3) \n Dashes: slope $\Phi$ (Th. 4)',
            (0.02, 0.000046), xycoords=('axes fraction', 'data'), color='k', fontsize=16, bbox=dict(facecolor='none'))
f.savefig(FIG_PATH + '10.pdf', bbox_inches='tight')
