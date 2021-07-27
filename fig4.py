import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from functions import *

#%%
mpl.style.use('seaborn-deep')
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['text.latex.preamble'] = r'\usepackage{bm}'


# %% Figure path
FIG_PATH = 'steadystate/figs/'
if not os.path.isdir(FIG_PATH):
    os.makedirs(FIG_PATH)

#%%
def make_lambdatilde_it(log_like_ratio, delta, A_dec, i):
    lambd = delta*sum([(1-delta)**m*np.linalg.matrix_power(A_dec.T, m+1).dot(np.array(log_like_ratio[m]))
                       for m in range(i + 1)])
    return lambd

def make_lambda_it(log_like_ratio, delta, A_dec, i):
    lambd = delta*sum([(1-delta)**m*np.linalg.matrix_power(A_dec.T, m+1).dot(np.array(log_like_ratio[i - m - 1]))
                       for m in range(i + 1)])
    return lambd

# %% Setup
N = 10
M = 3
N_ITER = 300
delta = 0.1

# %% Graph
np.random.seed(0)
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
theta = np.array([1., 2., 3.]) * 0.6
thetadec = decimal_array(theta)
var = 1
vardec = Decimal(var)

x = np.linspace(-10, 10, 1000)
x_dec = decimal_array(x)
dt = (max(x)-min(x))/len(x)
dtdec = (max(x_dec)-min(x_dec))/len(x_dec)

# %% Generate observations
np.random.seed(0)

csi = []
for l in range(0, N):
    csi.append(theta[0] + np.sqrt(var) * np.random.randn(N_ITER))
csi = np.array(csi)

# %% Generate random variables
like_ratio = [gaussian(csi[:,m], theta[0], var)/gaussian(csi[:,m], theta[2], var)
              for m in range(N_ITER)]
log_like_ratio = np.log(like_ratio)

lambda_2 = [make_lambda_it(log_like_ratio, delta, A, i)
            for i in range(N_ITER)]
lambdatilde_2 = [make_lambdatilde_it(log_like_ratio, delta, A, i)
                 for i in range(N_ITER)]

# %%
like_ratio = [gaussian(csi[:,m], theta[0], var)/gaussian(csi[:,m], theta[1], var)
              for m in range(N_ITER)]
log_like_ratio = np.log(like_ratio)

lambda_1 = [make_lambda_it(log_like_ratio, delta, A, i) for i in range(N_ITER)]
lambdatilde_1 = [make_lambdatilde_it(log_like_ratio, delta, A, i) for i in range(N_ITER)]

# %% Plot comparison
f, ax = plt.subplots(2,1, figsize = (6,6))

h1 = ax[0].plot(np.arange(1, N_ITER+1),[lambdatilde_1[j][0] for j in range(N_ITER)],
                linewidth=2, color = 'k', linestyle = 'dashed')
h2 = ax[0].plot(np.arange(1, N_ITER+1),[lambda_1[j][0] for j in range(N_ITER)],
                linewidth=2, color = 'C1')

ax[0].tick_params(axis='both', which='major', labelsize=18)
ax[0].set_xlim(0, N_ITER)
ax[0].set_xlabel(r'$i$', fontsize = 18)
ax[0].set_title(r'Log-Belief Ratio: $\theta=2$', fontsize=20)

ax[1].plot(np.arange(1, N_ITER+1),[lambdatilde_2[j][0] for j in range(N_ITER)],
           linewidth=2, color='k', linestyle='dashed')
h3 = ax[1].plot(np.arange(1, N_ITER+1),[lambda_2[j][0] for j in range(N_ITER)],
                linewidth=2, color='C2')

ax[1].tick_params(axis='both', which='major', labelsize=18)
ax[1].set_xlim(0, N_ITER)
ax[1].set_xlabel(r'$i$', fontsize=18)
ax[1].set_title(r'Log-Belief Ratio: $\theta=3$', fontsize=20)
ax[1].annotate(r"Solid line: $\widehat{\bm{\lambda}}^{(\delta)}_{k,i}(\theta)$ "
               r"\qquad \qquad Dashed line: $\widetilde{\bm{\lambda}}^{(\delta)}_{k,i}(\theta)$",
               xy=(0.0,-0.65), xycoords=('axes fraction', 'data'),
               color = 'k', fontsize=18,  bbox=dict(facecolor='None'))
f.tight_layout()
plt.savefig(FIG_PATH + '4.pdf', bbox_inches = 'tight')
#
