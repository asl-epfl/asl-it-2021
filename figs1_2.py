import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import pickle as pk
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
DATA_PATH = 'data/'
if not os.path.isdir(DATA_PATH):
    os.makedirs(DATA_PATH)

# %% Aux functions
def plot_agent(MU, ag=0, ax=[]):
    '''
    Plot the evolution of beliefs for one agent.
    MU: beliefs evolution for all agents
    N_ITER: number of iterations
    ag: chosen agent
    ax: axis specification
    '''
    vec = np.array([MU[k][ag] for k in range(len(MU))])
    d = np.zeros(2)
    if abs(vec[-1,1] - vec[-1,0])< 0.05:
        d[0] = .1
    if abs(vec[-1,2] - vec[-1,0])  or  abs(vec[-1,2] - vec[-1,1]) < .05:
        d[1] = .1

    if ax:
        ax.plot(vec[:,0], linewidth='2', color = 'C0', label=r'$\theta=1$', alpha=0.8)
        ax.plot(vec[:,1], linewidth='2', color = 'C1', label = r'$\theta=2$', alpha=0.8)
        ax.plot(vec[:,2], linewidth='2', color = 'C2', label = r'$\theta=3$', alpha=0.8)
        ax.set_xlim([0,N_ITER])
        ax.set_ylim([-0.1,1.1])
        ax.set_xlabel(r'$i$', fontsize = 18)
        ax.set_ylabel(r'$\bm{{\mu}}_{{{l},i}}(\theta)$'.format(l=1), fontsize = 18)
        ax.tick_params(axis='both', which='major', labelsize=18)


def plot_agent_dec(MU, ag=0, ax=[]):
    '''
    Plot the evolution of beliefs for one agent using Decimal values.
    MU: beliefs evolution for all agents
    N_ITER: number of iterations
    ag: chosen agent
    ax: axis specification
    '''
    vec = np.array([MU[k][ag] for k in range(len(MU))])
    d = np.array([Decimal(0), Decimal(0)])
    if abs(vec[-1,1] - vec[-1,0])< Decimal(.05):
        d[0] = Decimal(.1)
    if abs(vec[-1,2] - vec[-1,0])  or  abs(vec[-1,2] - vec[-1,1]) < Decimal(.05):
        d[1] = Decimal(.1)

    if ax:
        ax.plot(vec[:,0], linewidth='2', color = 'C0', label=r'$\theta=1$', alpha=0.8)
        ax.plot(vec[:,1], linewidth='2', color = 'C1', label = r'$\theta=2$', alpha=0.8)
        ax.plot(vec[:,2], linewidth='2', color = 'C2', label = r'$\theta=3$', alpha=0.8)
        ax.set_xlim([0,N_ITER])
        ax.set_ylim([-0.1,1.1])
        ax.set_xlabel(r'$i$', fontsize = 18)
        ax.set_ylabel(r'$\bm{{\mu}}_{{{l},i}}(\theta)$'.format(l=1), fontsize = 18)
        ax.tick_params(axis='both', which='major', labelsize=18)

# %% Setup
N = 10
M = 3
np.random.seed(0)

# %% Graph
G = np.random.choice([0.0, 1.0], size=(N, N), p=[0.5, 0.5])
G = G + np.eye(N)
G = (G > 0) * 1.0

# Averaging Rule -> Left-stochastic matrix
lamb = .5
A = np.zeros((N, N))
A = G / G.sum(axis=0)
A_dec = np.array([[Decimal(x) for x in y] for y in A])

# %%Check that A^i converges
print('A is primitive: ', np.all(np.isclose(np.linalg.matrix_power(A, 100),
                                            np.linalg.matrix_power(A, 101))))

# %% Hypotheses
np.random.seed(20)
N_ITER = 600
theta = np.array([1., 2., 3.]) * 0.6
thetadec = decimal_array(theta)
var = 1

x = np.linspace(-10, 10, 1000)
x_dec = decimal_array(x)
dt = (max(x) - min(x)) / len(x)
dtdec = (max(x_dec) - min(x_dec)) / len(x_dec)

# %% Initialization
mu_0 = np.random.rand(N,M)
mu_0 = mu_0/np.sum(mu_0, axis=1)[:, None]
mu_0dec = decimal_array(mu_0)

delta = 0.1

csi=[]
for l in range(0, N):
    csi.append(np.hstack([theta[0] + np.sqrt(var) * np.random.randn(200),
                          theta[2] + np.sqrt(var) * np.random.randn(N_ITER - 200)]))
csi=np.array(csi)
csidec = decimal_array(csi)

MU_adapt = asl(mu_0, csi, A, N_ITER, theta, var, delta, is_gaussian=True)
MU = sl(mu_0dec, csidec, A_dec, N_ITER, thetadec, var, is_gaussian=True)

# %%
dec_sl = np.argmax(np.array(MU), axis=2)[:, 0]
dec_asl = np.argmax(np.array(MU_adapt), axis=2)[:, 0]

# %%
fig, ax = plt.subplots(2, 1, figsize=(6, 5), gridspec_kw={'height_ratios': [1, 1]})
plot_agent_dec(MU, 0, ax[0])
ax[0].set_ylabel('Belief \nof Agent 1')
ax[0].set_xlabel('Iteration')
ax[1].scatter(np.argwhere(dec_sl == 0)[:, 0], np.ones(len(np.argwhere(dec_sl == 0)[:, 0])),
              s=20, marker='.', color='C0')
ax[1].scatter(np.argwhere(dec_sl == 1)[:, 0], 2*np.ones(len(np.argwhere(dec_sl == 1)[:, 0])),
              s=20, marker='.', color='C1')
ax[1].scatter(np.argwhere(dec_sl == 2)[:, 0], 3*np.ones(len(np.argwhere(dec_sl == 2)[:, 0])),
              s=20, marker='.', color='C2')
ax[1].set_xlim(0, N_ITER)
ax[1].set_ylabel('Opinion \nof Agent 1', fontsize=18)
ax[1].set_xlabel('Iteration', fontsize=18)
ax[0].set_title('Social Learning', fontsize=20)
ax[1].set_ylim(0.5, 3.5)
ax[1].yaxis.grid()
ax[1].set_axisbelow(True)
ax[1].tick_params(axis='x', which='major', labelsize=18)
ax[1].set_yticks([1, 2, 3])
ax[1].set_yticklabels(['S', 'C', 'R'], size=14)
fig.legend([r'Sunny', r'Cloudy', r'Rainy'], ncol=M, loc='center', bbox_to_anchor=(0.5, 0.05),
           fontsize=18, handlelength =1)
fig.tight_layout(rect=(0,0.05,1,1))
fig.savefig(FIG_PATH + '1.pdf', bbox_inches='tight')
#%%
fig, ax = plt.subplots(2, 1, figsize = (6,5), gridspec_kw={'height_ratios': [1, 1]})
plot_agent(MU_adapt, 0, ax[0])
ax[0].set_ylabel('Belief \nof Agent 1')
ax[0].set_xlabel('Iteration')
ax[1].scatter(np.argwhere(dec_asl == 0)[:, 0], np.ones(len(np.argwhere(dec_asl == 0)[:, 0])),
              s=20, marker='.', color='C0')
ax[1].scatter(np.argwhere(dec_asl == 1)[:, 0], 2*np.ones(len(np.argwhere(dec_asl == 1)[:, 0])),
              s=20, marker='.', color='C1')
ax[1].scatter(np.argwhere(dec_asl == 2)[:, 0], 3*np.ones(len(np.argwhere(dec_asl == 2)[:, 0])),
              s=20, marker='.', color='C2')
ax[1].set_xlim(0, N_ITER)
ax[1].set_ylabel('Opinion \nof Agent 1', fontsize=18)
ax[1].set_xlabel('Iteration', fontsize=18)
ax[0].set_title('Adaptive Social Learning', fontsize = 20)
ax[1].set_ylim(0.5, 3.5)
ax[0].set_ylim(0.2, 0.5)
ax[1].yaxis.grid()
ax[1].set_axisbelow(True)
ax[1].tick_params(axis='x', which='major', labelsize=18)
ax[1].set_yticks([1, 2, 3])
ax[1].set_yticklabels(['S', 'C', 'R'], size=14)
fig.legend([r'Sunny', r'Cloudy', r'Rainy'], ncol=M, loc='center', bbox_to_anchor=(0.5, 0.05),
           fontsize=18, handlelength=1)
fig.tight_layout(rect=(0, 0.05, 1,1))
fig.savefig(FIG_PATH + '2.pdf', bbox_inches='tight')


