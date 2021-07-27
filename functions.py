import numpy as np
from decimal import *
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


def gaussian_dec(x, m, var):
    '''
    Computes the Gaussian pdf value (Decimal type) at x.
    x: value at which the pdf is computed (Decimal type)
    m: mean (Decimal type)
    var: variance
    '''
    p = np.exp(-(x-m)**Decimal(2)/(Decimal(2)*Decimal(var)))/(np.sqrt(Decimal(2)*Decimal(np.pi)*Decimal(var)))
    return p


def gaussian(x, m, var):
    '''
    Computes the Gaussian pdf value at x.
    x: value at which the pdf is computed (Decimal type)
    m: mean
    var: variance
    '''
    p = np.exp(-(x-m)**2/(2*var))/(np.sqrt(2*np.pi*var))
    return p


def laplace_dec(x, m, b):
    '''
    Computes the Laplace pdf value (Decimal type) at x.
    x: value at which the pdf is computed (Decimal type)
    m: mean
    b: scale parameter
    '''
    p = np.exp(-np.abs(x-m)/(Decimal(b)))/(Decimal(2)*Decimal(b))
    return p


def laplace(x, m, b):
    '''
    Computes the Laplace pdf value at x.
    x: value at which the pdf is computed
    m: mean
    b: scale parameter
    '''
    p = np.exp(-np.abs((x-m)/b))/(2*b)
    return p


def bayesian_update(L, mu):
    '''
    Computes the Bayesian update.
    L: likelihoods matrix
    mu: beliefs matrix
    '''
    aux = L*mu
    bu = aux/aux.sum(axis = 1)[:, None]
    return bu


def asl_bayesian_update(L, mu, delta):
    '''
    Computes the adaptive Bayesian update.
    L: likelihoods matrix
    mu: beliefs matrix
    delta: step size
    '''
    aux = L**(delta)*mu**(1-delta)
    bu = aux/aux.sum(axis = 1)[:, None]
    return bu


def DKL(m,n,dx):
    '''
    Computes the KL divergence between m and n.
    m: true distribution in vector form
    n: second distribution in vector form
    dx : sample size
    '''
    mn = m/n
    mnlog = np.log(mn)
    return np.sum(m*dx*mnlog)


def decimal_array(arr):
    '''
    Converts an array to an array of Decimal objects.
    arr: array to be converted
    '''
    if len(arr.shape) == 1:
        return np.array([Decimal(y) for y in arr])
    else:
        return np.array([[Decimal(x) for x in y] for y in arr])


def float_array(arr):
    '''
    Converts an array to an array of float objects.
    arr: array to be converted
    '''
    if len(arr.shape)==1:
        return np.array([float(y) for y in arr])
    else:
        return np.array([[float(x) for x in y] for y in arr])


def asl_markov(mu_0, csi, A, N_ITER, theta, b, delta = 0):
    '''
    Executes the adaptive social learning algorithm with Laplace likelihoods
    for the Markovian example
    mu_0: initial beliefs
    csi: observations
    A: list of combination matrices
    N_ITER: number of iterations
    theta: vector of means for the Laplace likelihoods
    b: scale parameter of the Laplace likelihoods
    delta: step size
    '''
    mu = mu_0.copy()
    N = len(A[0])
    MU = [mu]
    for i in range(N_ITER):
        L_i = np.array([laplace(csi[:,i], t, b) for t in theta]).T

        # unindentifiability
        L_i[:N//3, 1] = L_i[:N//3, 0]
        L_i[N//3:2*N//3, 1] = L_i[N//3:2*N//3, 2]
        L_i[2*N//3:, 2] = L_i[2*N//3:, 0]

        psi = asl_bayesian_update(L_i, mu, delta)
        decpsi = np.log(psi)

        mu = np.exp((A[i].T).dot(decpsi)) / np.sum(np.exp((A[i].T).dot(decpsi)), axis=1)[:, None]
        MU.append(mu)
    return MU


def sl_markov(mu_0, csi, A, N_ITER, thetadec, b):
    '''
    Executes the social learning algorithm with Laplace likelihoods
    for the Markovian example
    mu_0: initial beliefs
    csi: observations
    A: Combination matrix
    N_ITER: number of iterations
    thetadec: vector of means for the Laplace likelihoods (Decimal type)
    b: scale parameter of the Laplace likelihoods
    is_gaussian: flag indicating if the likelihoods are Gaussian
    '''
    mu = decimal_array(mu_0)
    N = len(A[0])
    MU = [mu]
    for i in range(N_ITER):
        L_i = np.array([laplace_dec(csi[:,i], t, b) for t in thetadec]).T

        # unindentifiability
        L_i[:N//3, 1] = L_i[:N//3, 0]
        L_i[N//3:2*N//3, 1] = L_i[N//3:2*N//3, 2]
        L_i[2*N//3:, 2] = L_i[2*N//3:, 0]

        psi = bayesian_update(L_i, mu)
        decpsi = np.array([[x.ln() for x in y] for y in psi])
        C = decimal_array(A[i])
        mu = np.exp((C.T).dot(decpsi)) / np.sum(np.exp((C.T).dot(decpsi)), axis=1)[:, None]
        MU.append(mu)
    return MU


def asl(mu_0, csi, A, N_ITER, theta, var, delta = 0, is_gaussian = False):
    '''
    Executes the adaptive social learning algorithm with Gaussian or Laplace likelihoods.
    mu_0: initial beliefs
    csi: observations
    A: Combination matrix
    N_ITER: number of iterations
    theta: vector of means for the likelihoods
    var: variance of Gaussian likelihoods/ scale parameter of the Laplace likelihoods
    delta: step size
    is_gaussian: flag indicating if the likelihoods are Gaussian
    '''
    mu = mu_0.copy()
    N = len(A)
    MU = [mu]
    for i in range(N_ITER):
        if is_gaussian:
            L_i = np.array([gaussian(csi[:, i], t, var) for t in theta]).T
        else:
            L_i = np.array([laplace(csi[:, i], t, var) for t in theta]).T

        # unindentifiability
        L_i[:N//3, 1] = L_i[:N//3, 0]
        L_i[N//3:2*N//3, 1] = L_i[N//3:2*N//3, 2]
        L_i[2*N//3:, 2] = L_i[2*N//3:, 0]

        psi = asl_bayesian_update(L_i, mu, delta)
        decpsi = np.log(psi)
        mu = np.exp((A.T).dot(decpsi)) / np.sum(np.exp((A.T).dot(decpsi)), axis=1)[:, None]
        MU.append(mu)
    return MU


def sl(mu_0, csi, A, N_ITER, thetadec, var, is_gaussian = False):
    '''
    Executes the social learning algorithm with Gaussian or Laplace likelihoods.
    mu_0: initial beliefs
    csi: observations
    A: Combination matrix
    N_ITER: number of iterations
    thetadec: vector of means for the likelihoods (Decimal type)
    var: variance of Gaussian likelihoods/ scale parameter of the Laplace likelihoods
    is_gaussian: flag indicating if the likelihoods are Gaussian
    '''
    mu = mu_0.copy()
    N = len(A)
    MU = [mu]
    for i in range(N_ITER):
        if is_gaussian:
            L_i = np.array([gaussian_dec(csi[:, i], t, var) for t in thetadec]).T
        else:
            L_i = np.array([laplace_dec(csi[:, i], t, var) for t in thetadec]).T

        # unindentifiability
        L_i[:N//3, 1] = L_i[:N//3, 0]
        L_i[N//3:2*N//3, 1] = L_i[N//3:2*N//3, 2]
        L_i[2*N//3:, 2] = L_i[2*N//3:, 0]

        psi = bayesian_update(L_i, mu)
        decpsi = np.array([[x.ln() for x in y] for y in psi])
        mu = np.exp((A.T).dot(decpsi)) / np.sum(np.exp((A.T).dot(decpsi)), axis =1)[:, None]
        MU.append(mu)
    return MU

def comb_matrix_from_p(p, G):
    '''
    Computes combination matrix from the Perron eigenvector and graph specification.
    G: adjacency matrix
    p: Perron eigenvector
    '''
    A = np.where(G > 0, (p.T * np.ones((10,1))).T, 0)
    np.fill_diagonal(A, 1 - np.sum(A - np.diag(np.diag(A)), axis=0))
    return A


def lmgf(thetazero, theta1, t):
    '''
    Computes the LMGF for the family of Laplace distributions, specified in section VIII. C.
    thetazero: true state of nature
    theta1: state different than true state
    t: vector of real values
    '''
    alpha = theta1 - thetazero
    if thetazero >= theta1:
        f = np.log(np.exp(alpha * (t + 1)) + np.exp(-alpha * t) - np.exp(alpha / 2)
                   * np.sinh(alpha*(t + 1/2))/(t + 1/2)) - np.log(2)
    elif thetazero<theta1:
        f = np.log(np.exp(-alpha * (t + 1)) + np.exp(alpha * t) + np.exp(-alpha / 2)
                   * np.sinh(alpha*(t + 1/2))/(t + 1/2)) - np.log(2)
    return f


def Cave(L, t1, t2, dt):
    '''
    Computes the covariance between the log likelihood ratios for two hypotheses.
    L: likelihood Functions
    t1: hypothesis 1
    t2: hypothesis 2
    dt: step size
    '''
    aux = L[0] / L[t1]
    auxlog = np.log(aux)
    aux1 = L[0] / L[t2]
    auxlog1 = np.log(aux1)
    return np.sum(L[0] * dt * auxlog1 * auxlog) - DKL(L[0], L[t1], dt) * DKL(L[0], L[t2], dt)


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    '''
    Create a plot of the covariance confidence ellipse of *x* and *y*. Inspired on the code in:
    https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html
    '''
    if x.size != y.size:
        raise ValueError("x and y must be the same size")
    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs)
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)
    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def gaussian_ellipse(C, Ave, ax, n_std=3.0, facecolor='none', **kwargs):
    '''
    Create a plot of the covariance confidence ellipse relative to the Covariance matrix C.
    C: covariance matrix
    Ave: average vector
    n_std: number of standard deviation ellipses
    '''
    cov = C
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width = ell_radius_x * 2,
        height = ell_radius_y * 2,
        facecolor = facecolor,
        **kwargs)
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = Ave[0]
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = Ave[1]
    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)
