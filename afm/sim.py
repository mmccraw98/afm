import numpy as np
from scipy.special import ellipk
import pandas as pd
import torch
import time as time_

def hertzian(inden, E, v, R):
    '''
    calculate hertzian (elastic) contact force
    :param inden: float indentation depth
    :param E: float shear modulus
    :param v: float poissons ratio
    :param R: float indenter radius
    :return: float force
    '''
    return 4 / 3 * np.sqrt(R) * E / (1 - v ** 2) * inden ** 1.5

def lee_radok(inden, v, R, Gg, Ge, Tau, t):
    '''
    calculate lee and radok (viscoelastic) contact force
    :param inden: float indentation depth
    :param v: float poissons ratio
    :param R: float indenter radius
    :param Gg: float instantaneous shear modulus
    :param Ge: float equilibrium shear modulus
    :param Tau: float relaxation time
    :param t: float time axis
    :return: float force
    '''
    dt = t[1] - t[0]
    G = Gg - Ge
    if G < 0:
        print('invalid moduli')
    exp = np.exp(-t / Tau)
    a = 4 * np.sqrt(R) / (3 * (1 - v ** 2))
    H = inden ** 1.5
    return a * (Gg * H - G / Tau * np.convolve(exp, H, 'full')[: t.size] * dt)

def In_L(r_n, dr):
    '''
    using: Interaction and Deformation of Elastic Bodies: Origin of Adhesion Hysteresis - Phil Attard
    calculate the left hand limit at the singularity in the complete elliptic integral of the first kind
    :param r_n: (float) singularity point
    :param dr: (float) discretization
    :return: (float) analytically evaluated left hand limit
    '''
    return ((dr / np.pi - dr ** 2 / (4 * np.pi * r_n)) *
            (1 + np.log(16 * r_n ** 2 / (r_n * dr - dr ** 2 / 4))))


def In_R(r_n, dr):
    '''
    using: Interaction and Deformation of Elastic Bodies: Origin of Adhesion Hysteresis - Phil Attard
    calculate the right hand limit at the singularity in the complete elliptic integral of the first kind
    :param r_n: (float) singularity point
    :param dr: (float) discretization
    :return: (float) analytically evaluated right hand limit
    '''
    return (dr / np.pi * np.log(16 * (r_n + dr / 2) ** 2 / (r_n * dr + dr ** 2 / 4)) +
            4 * r_n / np.pi * np.log((2 * r_n + dr) / (2 * r_n + dr / 2)))


def K(r, r_p, dr):
    '''
    using: Interaction and Deformation of Elastic Bodies: Origin of Adhesion Hysteresis - Phil Attard
    calculate the complete elliptic integral of the first kind for handling the radial integrating kernel
    :param r: (float) radius value
    :param r_p: (numpy vector (nx1)) dummy radius axis
    :param dr: (float) radius axis (including dummy axis) discretization
    :return: (numpy matrix (nxn)) analytically evaluated complete elliptic integral of the first kind
    '''
    values = np.zeros(r_p.shape)  # make matrix
    values[r_p < r] = ellipk(r_p[r_p < r] ** 2 / r ** 2) * 4 / (r * np.pi)  # handle up to singularity
    values[r_p > r] = ellipk(r ** 2 / r_p[r_p > r] ** 2) * 4 / (r_p[r_p > r] * np.pi)  # handle past singularity
    values[r_p == r] = (In_L(r, dr) + In_R(r, dr)) / (r * dr)  # analytically evaluate singularity
    return values


def f_sphere(r, R):
    '''
    calculate the radially parameterized position of the contacting indenter
    :param r: (numpy array) radius axis
    :param R: (float) radius of probe
    :return: (numpy array) position of the contacting indenter
    '''
    return r ** 2 / (2 * R)


def h_calc(h0, u, f_tip, r, *args):
    '''
    calculate the radially parameterized separation between the indenter and the surface
    :param h0: (float) 'nominal separation' (in the language of Attard) absolute position of indenter relative to
    the undeformed sample surface
    :param u: (numpy array) radially parameterized deformation of the surface
    :param f_tip: function of the form f_tip(r, *args) which parameterizes the probe position
    :param r: (numpy array) radial discretization
    :param args: (optional) arguments of the f_tip probe position calculation function
    :return: (numpy array) distance between the indenter and the surface
    '''
    return h0 + f_tip(r, *args) - u


def p_vdw(h, H1=1e-19, H2=1e-19, z0=1e-9):
    '''
    calculates the van der Waals pressure and derivative as a function of distance
    :param h: (float or numpy array) distance between bodies
    :param H1: (float) repulsive Hamaker constant
    :param H2: (float) attractive Hamaker constant
    :param z0: (float) equilibrium separation distance
    :return: (float or numpy array x2) pressure, derivative of pressure with respect to distance
    '''
    return (1 / (6 * np.pi * h ** 3) * (H1 * (z0 / h) ** 6 - H2),
            1 / (2 * np.pi * h ** 4) * (H2 - 3 * H1 * (z0 / h) ** 6))

def p_hw(h, H=1e-19, z0=1e-9, n=9):
    '''
    calculates the physical apprximation of the hard-wall repulsive pressure and derivative as a function of distance
    :param h: (float or numpy array) distance between bodies
    :param H: (float) repulsive Hamaker constant
    :param z0: (float) equilibrium separation distance
    :param n: (int) repulsion exponent
    :return: (float or numpy array x2) pressure, derivative of pressure with respect to distance
    '''
    return (H * z0 ** (n - 3) / (6 * np.pi * h ** n),
            - n * H * z0 ** (n - 3) / (6 * np.pi * h ** n))

def p_exp_approx(h, H_rep=1e7, k_inv=1/1e-9, h_off=0):
    '''
    calculates an approximation for an exponential repulsion potential
    :param h: float distance between bodies
    :param H_rep: float repulsive Hamaker constant
    :param k_inv: float debye length
    :param h_off: float offset shift to distance
    :return: float pressure and pressure derivative
    '''
    # P, Ph
    return (H_rep / (h_off + k_inv * h),
            - H_rep * k_inv / (h_off + k_inv * h) ** 2)

def p_exp(h, magnitude=1e7, decay_length=1e-9):
    '''
    calculates an exponential repulsive potential distribution (cannot generally be used on its own)
    :param h: float distance between bodies
    :param magnitude: float magnitude of repulsive force at 0 separation
    :param decay_length: float decay constant
    :return: float pressure and pressure derivative
    '''
    return (magnitude * np.exp(- h / decay_length),
            - magnitude / decay_length * np.exp(- h / decay_length))

def p_exp_TORCH(h, magnitude=1e7, decay_length=1e-9):
    '''
    calculates an exponential repulsive potential distribution (cannot generally be used on its own)
    FOR PYTORCH TENSORS
    :param h: float distance between bodies
    :param magnitude: float magnitude of repulsive force at 0 separation
    :param decay_length: float decay constant
    :return: float pressure and pressure derivative
    '''
    return (magnitude * torch.exp(- h / decay_length),
            - magnitude / decay_length * torch.exp(- h / decay_length))
def p_exp_and_rep(h, magnitude=1e7, decay_length=1e-9, H_rep=1e-19, z0=0.5e-9):
    '''
    superposition of exponential repulsion and short range, hard wall repulsion
    :param h: float distance between bodies
    :param magnitude: float magnitude of exponential repulsive force at 0 separation
    :param decay_length: float decay constant of exponential force
    :param H_rep: float hard wall repulsion hamaker constant
    :param z0: decay constant of hard wall repulsion
    :return: float pressure and pressure derivative
    '''
    p1, ph1 = p_exp(h, magnitude, decay_length)
    p2, ph2 = p_vdw(h, H2=0, H1=H_rep, z0=z0)
    return (p1 + p2,
            ph1 + ph2)

def p_exp_and_rep_TORCH(h, magnitude=1e7, decay_length=1e-9, H_rep=1e-19, z0=0.5e-9):
    '''
    superposition of exponential repulsion and short range, hard wall repulsion
    FOR PYTORCH TENSORS
    :param h: float distance between bodies
    :param magnitude: float magnitude of exponential repulsive force at 0 separation
    :param decay_length: float decay constant of exponential force
    :param H_rep: float hard wall repulsion hamaker constant
    :param z0: decay constant of hard wall repulsion
    :return: float pressure and pressure derivative
    '''
    p1, ph1 = p_exp_TORCH(h, magnitude, decay_length)
    p2, ph2 = p_vdw(h, H2=0, H1=H_rep, z0=z0)
    return (p1 + p2,
            ph1 + ph2)


def p_degennes(h, L0=2e-6, N=3e14, T=300, H_rep=1e-19, z0=0.5e-9):
    '''
    de-gennes grafted polymer repulsive pressure superposed with a hard-wall repulsion
    coefficients taken from sokolov (experiments on cells): https://doi.org/10.1063/1.2757104
    model derived using Surface and Interaction Forces 2ed (pp. 341)
    :param h: float separation between ungrafted surface and grafted surface
    :param L0: float polymer equilibrium length (sokolov reports 2 microns on average)
    :param N: float polymer grafting density (sokolov reports about 300 polymers per square micron)
    :param T: float temperature
    :param H_rep: float hard-wall repulsion magnitude
    :param z0: float hard-wall repulsion distance
    :return: (float, float) pressure and pressure derivative
    '''
    kb = 1.38e-23
    C = kb * T * L0 * N ** (3 / 2)

    P = 2 * C * (L0 ** (5 / 4) * h ** (-9 / 4) - L0 ** (-7 / 4) * h ** (3 / 4))
    Ph = - C / 2 * (3 * L0 ** (-7 / 4) * h ** (-1 / 4) + 9 * L0 ** (5 / 4) * h ** (-13 / 4))

    p_hw, ph_hw = p_vdw(h, H1=H_rep, H2=0, z0=z0)
    return (P + p_hw, Ph + ph_hw)

def rk4(state, dt, rhs_func, *args):
    '''
    Runge Kutta 4th order time integration update scheme
    :param state: (numpy array) variables to integrate
    :param dt: (float) time discretization
    :param rhs_func: function to calculate the time derivatives of the state of form rhs_func(state, *args)
    returns (time derivative of state vector, (extra stuff))
    :param args: (optional) arguments for the rhs_func
    :return: (numpy array) integrated state at t=t+dt, (extra stuff calculated at final step)
    '''
    k1, _ = rhs_func(state, *args)
    k2, _ = rhs_func(state + dt * k1 / 2, *args)
    k3, _ = rhs_func(state + dt * k2 / 2, *args)
    k4, extra = rhs_func(state + dt * k3, *args)
    return state + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6, extra

def rhs_N_1_rigid(state, r, dr, R, k_ij, I, v0, b1, b0, c1, c0, p_func, *args):
    '''
    calculates the derivative of the state vector (surface deformation and probe position) assuming that the
    probe position is prescribed by a time dependent velocity v0(t) (i.e. by a rigid cantilever)
    also assuming that the viscoelastic ODE is order N=1
    :param state: numpy array state vector (deformation, probe position)
    :param r: float radial spatial axis
    :param dr: float spatial discretization
    :param R: float spherical probe radius
    :param k_ij: numpy matrix (nr x nr) integrating kernel
    :param I: numpy matrix (nr x nr) identity matrix
    :param v0: float time dependent prescribed velocity of the probe
    :param b1: float viscoelastic coefficient b1
    :param b0: float viscoelastic coefficient b0
    :param c1: float viscoelastic coefficient c1
    :param c0: float viscoelastic coefficient c0
    :param p_func: function for calculating sphere-point pressure distribution p_func(h, *args)
    :param args: optional arguments for p_func
    :return: numpy array derivative of state vector (deformation rate, probe velocity)
    '''
    u, h0 = state  # unpack the state vector
    h = h_calc(h0, u, f_sphere, r, R)  # calculate the separation
    p, p_h = p_func(h, *args)  # calculate the pressure and derivative
    # calculate LHS G matrix
    G_ij = c1 * dr * k_ij * p_h * r - b1 * I
    # calculate RHS vectors, d_j
    d_j = c1 * v0 * dr * (p_h * r) @ k_ij + c0 * dr * (p * r) @ k_ij + b0 * u
    u_dot = np.linalg.solve(G_ij, d_j)  # solve the system of equations
    return np.array([u_dot, v0], dtype='object'), (p, h)

def rhs_N_1_rigid_cuda(state, r, dr, R, k_ij, I, v0, b1, b0, c1, c0, p_func, *args):
    '''
    calculates the derivative of the state vector (surface deformation and probe position) assuming that the
    probe position is prescribed by a time dependent velocity v0(t) (i.e. by a rigid cantilever)
    also assuming that the viscoelastic ODE is order N=1
    ASSUMES VARIABLES ARE TORCH TENSORS
    :param state: numpy array state vector (deformation, probe position)
    :param r: float radial spatial axis
    :param dr: float spatial discretization
    :param R: float spherical probe radius
    :param k_ij: numpy matrix (nr x nr) integrating kernel
    :param I: numpy matrix (nr x nr) identity matrix
    :param v0: float time dependent prescribed velocity of the probe
    :param b1: float viscoelastic coefficient b1
    :param b0: float viscoelastic coefficient b0
    :param c1: float viscoelastic coefficient c1
    :param c0: float viscoelastic coefficient c0
    :param p_func: function for calculating sphere-point pressure distribution p_func(h, *args)
    :param args: optional arguments for p_func
    :return: numpy array derivative of state vector (deformation rate, probe velocity)
    '''
    u = state[:, 0]  # unpack the state vector
    h0 = state[:, 1]  # unpack the state vector
    h = h_calc(h0, u, f_sphere, r, R)  # calculate the separation
    p, p_h = p_func(h, *args)  # calculate the pressure and derivative
    # calculate LHS G matrix
    G_ij = c1 * dr * k_ij * p_h * r - b1 * I
    # calculate RHS vectors, d_j
    d_j = c1 * v0 * dr * k_ij @ (p_h * r) + c0 * dr * k_ij @ (p * r) + b0 * u
    u_dot = torch.linalg.solve(G_ij, d_j)  # solve the system of equations
    return torch.cat((torch.reshape(u_dot, v0.shape), v0), 1), (p, h)


def get_coefs_rajabifar(Gg, Ge, Tau, v):
    '''
    get viscoelastic ODE coefficients in the style of Rajabifar and Attard
    :param Gg: float instantaneous shear modulus
    :param Ge: float equilibrium shear modulus
    :param Tau: float relaxation time
    :param v: float poisson's ratio
    :return: b1, b0, c1, c0
    '''
    return 1, 1 / Tau, (1 - v ** 2) / Gg, (1 - v ** 2) / (Ge * Tau)

def get_coefs_sls(Gg, Ge, Tau, v):
    '''
    get viscoelastic ODE coefficients for a standard linear solid (SLS) material
    :param Gg: flaot instantaneous shear modulus
    :param Ge: float equilibrium shear modulus
    :param Tau: float relaxation time
    :param v: float poisson's ratio
    :return: b1, b0, c1, c0
    '''
    return Gg * Tau, Ge, (1 - v ** 2) * Tau, (1 - v ** 2)

def get_coefs_elastic(G, v):
    '''
    get viscoelastic ODE coefficients for an elastic solid
    :param G: float shear modulus
    :param v: float poisson's ratio
    :return: b1, b0, c1, c0
    '''
    return G, 0, (1 - v ** 2), 0

def get_coefs_viscous(mu, v):
    '''
    get viscoelastic ODE coefficients for a viscous solid
    :param mu: float viscosity
    :param v: float poisson's ratio
    :return: b1, b0, c1, c0
    '''
    return mu, 0, 0, (1 - v ** 2)

def get_coefs_kvf(J, Tau, v):
    '''
    get viscoelastic ODE coefficients for a kelvin-voigt viscous solid (fluid)
    :param J: float shear compliance
    :param Tau: float retardation time
    :param v: float poisson's ratio
    :return: b1, b0, c1, c0
    '''
    return Tau, 1, 0, (1 - v ** 2) * J

def get_coefs_mf(G, Tau, v):
    '''
    get viscoelastic ODE coefficients for a maxwell viscous solid (fluid)
    :param G: float shear modulus
    :param Tau: float relaxation time
    :param v: float poisson's ratio
    :return: b1, b0, c1, c0
    '''
    return G * Tau, 0, (1 - v ** 2) * Tau, (1 - v ** 2)

def get_coefs_gkv(Jg, Je, Tau, v):
    '''
    get viscoelastic ODE coefficients for (generalized) kelvin-voigt solid
    :param Jg: float instantaneous shear compliance
    :param Je: float equilibrium shear compliance
    :param Tau: float retardation time
    :param v: float poisson's ratio
    :return: b1, b0, c1, c0
    '''
    return Tau, 1, (1 - v ** 2) * Jg, (1 - v ** 2) * Je


def get_explicit_sim_arguments(l, f, rho_c, R, G_star, N_R, H_R_target=-0.1, F_target=1):
    '''
    calculate the explicit arguments for simulate_rigid_N1 given the dimensionless quantities l and f
    using the dimension of R (probe radius), G_star (reduced shear modulus)
    :param l: float (1e-2 to 1e3, theoretically also to inf) dimensionless length scale l=(Rh')^1/2
    :param f: float (theoretically from 0, 1e-10 to 1e6) dimensionless pressure scale f=P0/G*
    :param rho_c: float (usually 1 to 10) dimensionless maximum radial distance in discrete domain
    :param R: float probe radius in m
    :param G_star: float reduced shear modulus in Pa (G*=G/(1-v^2))
    :param N_R: int number of points in discrete domain
    :param H_R_target: float (-inf, inf) target probe position as a ratio of probe radius
    :param F_target: float (0, inf) ratio of theoretical hertz force obtained at maximum indentation (usually >1)
    :return:
    '''
    r_c = rho_c * R / l
    d_r = r_c / N_R
    P_magnitude = f * G_star
    decay_constant = R / l ** 2
    pos_target = R * H_R_target
    # estimate the maximum force as a ratio of the maximum hertzian force
    force_target = abs(pos_target) ** 1.5 * G_star * R ** 0.5 * F_target
    return r_c, d_r, P_magnitude, decay_constant, pos_target, force_target



def simulate_rigid_N1(Gg, Ge, Tau, v, v0, h0, R, p_func, *args,
                      nr=int(1e3), dr=1.5e-9,
                      dt=1e-4, nt=int(1e6),
                      force_target=1e-6, pos_target=-1e-8, pct_log=0.0001, use_cuda=False,
                      remesh=False, u_tol=1e-9, remesh_factor=1.01, log_all=False):
    '''
    integrate the interaction of the probe (connected to a rigid cantilever) with a sample defined by a viscoelastic
    ODE of maximum order N=1, using RK4 time integration
    :param Gg: float instantaneous shear modulus
    :param Ge: float equilibrium shear modulus
    :param Tau: float relaxation time
    :param v: float poisson's ratio
    :param v0: float prescribed probe velocity
    :param h0: float initial position of the probe relative to the sample
    :param R: float radius of the spherical probe
    :param p_func: function describing the point-probe pressure distribution p_func(h, *args)
    :param args: optional arguments for p_func pressure distribution
    :param nr: int number of spatial discretization points
    :param dr: float spatial discretization
    :param dt: float temporal discretization
    :param nt: int number of time discretization points
    :param force_target: float maximum force in simulation
    :param pos_target: float target position of probe
    :param pct_log: float percentage of sim steps to log sim state to console
    :param use_cuda: bool whether to use CUDA cores for calculation acceleration (nearly 1 order of magnitude
    compared against numpy.linalg.solve)
    :param remesh: bool whether to remesh the domain
    :param u_tol: tolerance for the 'infinite' displacement value
    :param remesh_factor: float how large the domain will expand if remeshing
    :param log_all: bool whether to log surface distribution data, default is False
    :return: data dict containing sim dataframe and sim parameters
    '''
    saved_args = locals()  # save all function arguments for later

    # discretize domain
    r = np.linspace(1, nr, nr) * dr
    # calculate integrating kernels
    k_ij = np.array([K(r_, r, dr) for r_ in r])
    I = np.eye(r.size)  # make identity matrix
    u = np.zeros(r.shape)  # initialize deformation
    state = np.array([u, h0], dtype='object')  # make state vector

    if use_cuda and torch.cuda.is_available():
        print('Solving on GPU:')
        print(torch.cuda.get_device_name(0))
        torch.set_default_tensor_type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)
        r = torch.linspace(1, nr, nr, device='cuda') * dr
        # TODO: there is a lot of computational overhead in making the cuda version of k_ij, should be fixed easily
        k = torch.zeros([nr, nr], device='cuda')
        for i, row in enumerate(k_ij):
            for j, K_IJ in enumerate(row):
                k[i, j] += K_IJ
        k_ij = k
        I = torch.eye(nr, device='cuda')
        u = torch.zeros([nr, 1], device='cuda')
        h0 = torch.ones([nr, 1], device='cuda') * h0
        v0 = torch.ones([nr, 1], device='cuda') * v0
        state = torch.cat((u, h0), 1)

    # get generalized viscoelastic coefficients
    if Ge == 0 and Tau == 0:
        b1, b0, c1, c0 = get_coefs_elastic(Gg, v)
    else:
        b1, b0, c1, c0 = get_coefs_sls(Gg, Ge, Tau, v)

    print('b1 | b0 | c1 | c0')
    print(b1, b0, c1, c0)
    print('max. r: {}'.format(max(r)))

    # make loggers
    force = np.zeros(nt)
    separation = np.zeros(nt)
    tip_pos = np.zeros(nt)
    time = np.zeros(nt)
    u_inf = np.zeros(nt)

    if log_all:
        U_log = np.zeros((nt, u.size))
        P_log = np.zeros((nt, u.size))
        r_log = np.zeros((nt, u.size))

    # run simulation
    print(' % |  z_tip  |  z_target |  force  |  force_target  |  u(inf,t)')
    start = time_.time()
    for n in range(nt):
        # update the state according to rk4 integration
        if use_cuda:  # need to benchmark this
            state, (p, h) = rk4(state, dt, rhs_N_1_rigid_cuda, r, dr, R, k_ij, I, v0, b1, b0, c1, c0, p_func, *args)
            u = state[:, 0]#.cpu().numpy()
            h0 = state[:, 1][0]#.cpu().numpy()[0]
        else:
            state, (p, h) = rk4(state, dt, rhs_N_1_rigid, r, dr, R, k_ij, I, v0, b1, b0, c1, c0, p_func, *args)
            u, h0 = state
        force[n] = (2 * np.pi * p @ r * dr)  # calculate the force
        # log
        separation[n] = h[0]
        tip_pos[n] = h0
        time[n] = n * dt
        u_inf[n] = u[-1]
        if log_all:
            U_log[n] = u
            P_log[n] = p
            r_log[n] = r
        if n % (pct_log * nt) == 0:
            print('{} | {:.1f} nm | {:.1f} nm | {:.1f} nN | {:.1f} nN | {:.1f} nm'.format(
                n / nt, tip_pos[n] * 1e9, pos_target * 1e9, force[n] * 1e9,
                force_target * 1e9, u[-1] * 1e9))
        # early exit conditions
        if force[n] > force_target or tip_pos[n] < pos_target:
            print('force and target ', force[n], force_target)
            print('pos and target ', tip_pos[n], pos_target)
            force = force[:n + 1]
            separation = separation[:n + 1]
            tip_pos = tip_pos[:n + 1]
            time = time[:n + 1]
            u_inf = u_inf[:n + 1]
            if log_all:
                U_log = U_log[:n + 1]
                P_log = P_log[:n + 1]
                r_log = r_log[:n + 1]
            break
        if remesh and abs(u[-1]) > u_tol:
            # remake domain
            print(max(r))
            dr *= remesh_factor
            # discretize domain
            r = np.linspace(1, nr, nr) * dr
            print(max(r))
            # calculate integrating kernels
            k_ij = np.array([K(r_, r, dr) for r_ in r])
            I = np.eye(r.size)  # make identity matrix
            u = np.zeros(r.shape)  # initialize deformation
            state = np.array([u, h0], dtype='object')  # make state vector
            print('this is the issue, in the line above.  u needs to be properly defined using the previous values')
            if use_cuda and torch.cuda.is_available():
                torch.set_default_tensor_type(
                    torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)
                r = torch.linspace(1, nr, nr, device='cuda') * dr
                k = torch.zeros([nr, nr], device='cuda')
                for i, row in enumerate(k_ij):
                    for j, K_IJ in enumerate(row):
                        k[i, j] += K_IJ
                k_ij = k
                I = torch.eye(nr, device='cuda')
                u = torch.zeros([nr, 1], device='cuda')
                h0 = torch.ones([nr, 1], device='cuda') * h0
                v0 = torch.ones([nr, 1], device='cuda') * v0
                state = torch.cat((u, h0), 1)

    print('done in {:.0f}s'.format(time_.time() - start))
    # save sim data
    inden = np.zeros(time.size)
    contact_index = np.argmin(force)
    inden[contact_index:] = tip_pos[contact_index] - tip_pos[contact_index:]
    force_rep = np.zeros(time.size)
    force_rep[contact_index:] = force[contact_index:] - force[contact_index]
    df = pd.DataFrame({'time': time, 'separation': separation, 'tip_pos': tip_pos,
                       'force': force, 'inden': inden, 'force_rep': force_rep,
                       'deformation': separation - tip_pos, 'u_inf': u_inf})
    data = {'df': df, 'sim_arguments': saved_args, 'log_all': 0}
    if log_all:
        data['log_all'] = {'u': U_log, 'p': P_log, 'r': r_log}
    return data




def simulate_prescribed_N1(Gg, Ge, Tau, v, v_t, h0, R, p_func, *args,
                           nr=int(1e3), dr=1.5e-9, dt=1e-4, pct_log=0.01):
    '''
    integrate the interaction of the probe (connected to a rigid cantilever) with a sample defined by a viscoelastic
    ODE of maximum order N=1, using RK4 time integration with the probe position prescribed as a function of time
    :param Gg: float instantaneous shear modulus
    :param Ge: float equilibrium shear modulus
    :param Tau: float relaxation time
    :param v: float poisson's ratio
    :param v_t: np.array prescribed probe velocity as a function of time
    :param h0: float initial position of the probe relative to the sample
    :param R: float radius of the spherical probe
    :param p_func: function describing the point-probe pressure distribution p_func(h, *args)
    :param args: optional arguments for p_func pressure distribution
    :param nr: int number of spatial discretization points
    :param dr: float spatial discretization
    :param dt: float temporal discretization
    :param pct_log: float percentage of sim steps to log sim state to console
    :return: data dict containing sim dataframe and sim parameters
    '''
    saved_args = locals()  # save all function arguments for later

    # discretize domain
    r = np.linspace(1, nr, nr) * dr
    # calculate integrating kernels
    k_ij = np.array([K(r_, r, dr) for r_ in r])
    I = np.eye(r.size)  # make identity matrix
    u = np.zeros(r.shape)  # initialize deformation
    state = np.array([u, h0], dtype='object')  # make state vector

    # get generalized viscoelastic coefficients
    if Ge == 0 and Tau == 0:
        b1, b0, c1, c0 = get_coefs_elastic(Gg, v)
    else:
        b1, b0, c1, c0 = get_coefs_sls(Gg, Ge, Tau, v)

    print('b1 | b0 | c1 | c0')
    print(b1, b0, c1, c0)
    print('max. r: {}'.format(max(r)))

    # make loggers
    nt = v_t.size
    force = np.zeros(nt)
    separation = np.zeros(nt)
    tip_pos = np.zeros(nt)
    time = np.zeros(nt)
    u_inf = np.zeros(nt)

    # run simulation
    print(' % |  z_tip  |  force  |  u(inf,t) | t')
    start = time_.time()
    for n in range(nt):
        # update the state according to rk4 integration
        v0 = v_t[n]
        state, (p, h) = rk4(state, dt, rhs_N_1_rigid, r, dr, R, k_ij, I, v0, b1, b0, c1, c0, p_func, *args)
        u, h0 = state
        force[n] = (2 * np.pi * p @ r * dr)  # calculate the force
        # log
        separation[n] = h[0]
        tip_pos[n] = h0
        time[n] = n * dt
        u_inf[n] = u[-1]
        if n % (pct_log * nt) == 0:
            print('{} | {:.1f} nm | {:.1f} nN | {:.1f} nm | {:.1f} s'.format(
                n / nt, tip_pos[n] * 1e9, force[n] * 1e9, u[-1] * 1e9, n * dt))

    print('done in {:.0f}s'.format(time_.time() - start))
    # save sim data
    inden = np.zeros(time.size)
    contact_index = np.argmin(force)
    inden[contact_index:] = tip_pos[contact_index] - tip_pos[contact_index:]
    force_rep = np.zeros(time.size)
    force_rep[contact_index:] = force[contact_index:] - force[contact_index]
    df = pd.DataFrame({'time': time, 'separation': separation, 'tip_pos': tip_pos,
                       'force': force, 'inden': inden, 'force_rep': force_rep,
                       'deformation': separation - tip_pos, 'u_inf': u_inf})
    data = {'df': df, 'sim_arguments': saved_args, 'log_all': 0}
    return data


def rk4_mixed(state_vectors, state_scalars, dt, rhs_func_mixed, *args):
    '''
    Runge Kutta 4th order time integration update scheme
    :param state_vectors: (vector) vector variables to integrate
    :param state_scalars: (vector like) vector of scalar variables to integrate
    :param dt: (float) time discretization
    :param rhs_func_mixed: function to calculate the time derivatives of the state of form
    rhs_func(state_vectors, state_scalars, *args) returns (time derivative of state vector, (extra stuff))
    :param args: (optional) arguments for the rhs_func
    :return: (numpy array) integrated state at t=t+dt, (extra stuff calculated at final step)
    '''
    k1v, k1s, _, _ = rhs_func_mixed(state_vectors, state_scalars, *args)
    k2v, k2s, _, _ = rhs_func_mixed(state_vectors + dt * k1v / 2, state_scalars + dt * k1s,
                                    *args)
    k3v, k3s, _, _ = rhs_func_mixed(state_vectors + dt * k2v / 2, state_scalars + dt * k2s / 2,
                                    *args)
    k4v, k4s, extra_vectors, extra_scalars = rhs_func_mixed(state_vectors + dt * k3v,
                                                            state_scalars + dt * k3s, *args)
    return (state_vectors + (k1v + 2 * k2v + 2 * k3v + k4v) * dt / 6,
            state_scalars + (k1s + 2 * k2s + 2 * k3s + k4s) * dt / 6,
            extra_vectors, extra_scalars)

def rhs_mixed(state_vectors, state_scalars, t, b1, b0, c1, c0, r, dr, R, k_ij, I, use_cuda, f_exc_func, vb_func,
              m, k_eff, Q0, w0, p_func, *args):
    '''
    calculates the derivative of the state_vectors and state_scalars (surface deformation and probe position)
    assuming that the probe position is governed by a simple harmonic oscillator with a prescribed base velocity
    and base excitation - also assuming that the viscoelastic ODE is order N=1
    :param state_vectors: array state vector (deformation)
    :param state_scalars: array state scalars (zbase, ztip, vtip)
    :param t: float time
    :param b1: float viscoelastic coefficient b1
    :param b0: float viscoelastic coefficient b0
    :param c1: float viscoelastic coefficient c1
    :param c0: float viscoelastic coefficient c0
    :param r: float radial spatial axis
    :param dr: float spatial discretization
    :param R: float spherical probe radius
    :param k_ij: matrix (nr x nr) integrating kernel
    :param I: numpy matrix (nr x nr) identity matrix
    :param use_cuda: bool True to use cuda
    :param f_exc_func: function to calculate the excitation force on the cantilever, uses args: (t, zt, zb, f_ts, k_eff)
    :param vb_func: function to calculate the velocity of the cantilever base, uses args: (t, zt, zb, f_ts, k_eff)
    :param m: effective mass of first cantilever mode
    :param k_eff: effective spring constant of first cantilever mode
    :param Q0: Q factor of first cantilever mode
    :param w0: resonant frequency of first cantilever mode
    :param p_func: function for calculating sphere-point pressure distribution p_func(h, *args)
    :param args: optional arguments for p_func
    :return: numpy array derivative of state vector (deformation rate, probe velocity)
    '''
    u = state_vectors
    zb, zt, vt = state_scalars
    # calculate state derivatives
    # first calculate the derivatives for the surface
    h_calc(zt, u, f_sphere, r, R)
    h = h_calc(zt, u, f_sphere, r, R)  # calculate the separation
    p, p_h = p_func(h, *args)  # calculate the pressure and derivative
    f_ts = 2 * np.pi * p @ r * dr  # calculate the tip-sample force
    # calculate LHS G matrix
    G_ij = c1 * dr * k_ij * p_h * r - b1 * I
    d_j = c1 * vt * dr * k_ij @ (p_h * r) + c0 * dr * k_ij @ (p * r) + b0 * u
    # solve the system of equations to calculate u_dot
    if use_cuda:
        u_dot = torch.linalg.solve(G_ij, d_j)
    else:
        u_dot = np.linalg.solve(G_ij, d_j)
    # then calculate the derivatives of the cantilever
    f_exc = f_exc_func(t, zt, zb, f_ts, k_eff)
    zb_dot = vb_func(t, zt, zb, f_ts, k_eff)
    zt_dot = vt
    vt_dot = 1 / m * (f_exc + f_ts + k_eff * (zb - zt) - m * Q0 / w0 * vt)
    # returns: ((state vectors, state scalars), (extra vectors, extra scalars) (won't be used for integration))
    if use_cuda:
        return (u_dot, torch.tensor([zb_dot, zt_dot, vt_dot]), (p, h), torch.tensor([f_ts, f_exc, vt_dot]))
    else:
        return (u_dot, np.array([zb_dot, zt_dot, vt_dot]), (p, h), np.array([f_ts, f_exc, vt_dot]))

def simulate_cantilever_N1(Gg, Ge, Tau, v, zt, zb, R, f_exc_func, vb_func, p_func, *args,
                           nr=int(1e3), dr=1.5e-9, dt=1e-4, nt=int(1e6), w0=10e3, Q0=100, k_eff=1,
                           force_target=1e-6, pos_target=-1e-8, pct_log=0.0001, use_cuda=False, log_all=False):
    '''
    integrate the interaction of the probe (connected to a SHO cantilever) with a sample defined by a viscoelastic
    ODE of maximum order N=1, using RK4 time integration
    :param Gg: float instantaneous shear modulus
    :param Ge: float equilibrium shear modulus
    :param Tau: float relaxation time
    :param v: float poisson's ratio
    :param zt: float initial tip position
    :param zb: float initial probe position
    :param R: float radius of the spherical probe
    :param f_exc_func: function to calculate the excitation force on the cantilever, uses args: (t, zt, zb, f_ts, k_eff)
    :param vb_func: function to calculate the velocity of the cantilever base, uses args: (t, zt, zb, f_ts, k_eff)
    :param p_func: function describing the point-probe pressure distribution p_func(h, *args)
    :param args: optional arguments for p_func pressure distribution
    :param nr: int number of spatial discretization points
    :param dr: float spatial discretization
    :param dt: float temporal discretization
    :param nt: int number of time discretization points
    :param force_target: float maximum force in simulation
    :param pos_target: float target position of probe
    :param pct_log: float percentage of sim steps to log sim state to console
    :param use_cuda: bool whether to use CUDA cores for calculation acceleration (nearly 1 order of magnitude
    compared against numpy.linalg.solve)
    :param log_all: bool whether to log surface distribution data, default is False
    :return: data dict containing sim dataframe and sim parameters
    '''
    saved_args = locals()  # save all function arguments for later
    # discretize domain
    r = np.linspace(1, nr, nr) * dr
    # calculate integrating kernels
    k_ij = np.array([K(r_, r, dr) for r_ in r])
    I = np.eye(r.size)  # make identity matrix
    u = np.zeros(r.shape)  # initialize deformation

    # get generalized viscoelastic coefficients
    if Ge == 0 and Tau == 0:
        b1, b0, c1, c0 = get_coefs_elastic(Gg, v)
    else:
        b1, b0, c1, c0 = get_coefs_sls(Gg, Ge, Tau, v)

    print('b1 | b0 | c1 | c0')
    print(b1, b0, c1, c0)
    print('max. r: {}'.format(max(r)))

    # forces
    f_ts = 0
    f_exc = 0

    # probe variables
    m = k_eff / w0 ** 2
    vb = vb_func(0, zt, zb, f_ts, k_eff)
    vt = vb

    # form state
    state_vectors = u
    state_scalars = np.array([zb, zt, vt])

    # make loggers
    central_deformation = np.zeros(nt)
    zb_zt_vt = np.zeros((nt, state_scalars.size))
    f_ts_f_exc_at = np.zeros((nt, 3))
    time = np.zeros(nt)
    u_inf = np.zeros(nt)

    log_all = False
    if log_all:
        U_log = np.zeros((nt, u.size))
        P_log = np.zeros((nt, u.size))
        H_log = np.zeros((nt, u.size))
        r_log = np.zeros((nt, u.size))

    if use_cuda and torch.cuda.is_available():
        print('Solving on GPU:')
        print(torch.cuda.get_device_name(0))
        torch.set_default_tensor_type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)
        r = torch.linspace(1, nr, nr, device='cuda') * dr
        # TODO: there is a lot of computational overhead in making the cuda version of k_ij, should be fixed easily
        k = torch.zeros([nr, nr], device='cuda')
        for i, row in enumerate(k_ij):
            for j, K_IJ in enumerate(row):
                k[i, j] += K_IJ
        k_ij = k
        I = torch.eye(nr, device='cuda')
        u = torch.zeros(nr, device='cuda')

        # forces
        f_ts = torch.tensor(f_ts, device='cuda')
        f_exc = torch.tensor(f_exc, device='cuda')
        # probe variables
        zt = torch.tensor(zt, device='cuda')
        zb = torch.tensor(zb, device='cuda')
        vb = torch.tensor(vb, device='cuda')
        vt = torch.tensor(vt, device='cuda')

        # form state
        state_vectors = u
        state_scalars = torch.tensor([zb, zt, vt], device='cuda')

        # make loggers
        central_deformation = torch.zeros(nt, device='cuda')
        zb_zt_vt = torch.zeros((nt, state_scalars.size()[0]), device='cuda')
        f_ts_f_exc_at = torch.zeros((nt, state_scalars.size()[0]), device='cuda')
        time = torch.zeros(nt, device='cuda')
        u_inf = torch.zeros(nt, device='cuda')

        # cantilever, probe, and sample variables
        R = torch.tensor(R, device='cuda')
        b1, b0 = torch.tensor(b1, device='cuda'), torch.tensor(b0, device='cuda')
        c1, c0 = torch.tensor(c1, device='cuda'), torch.tensor(c0, device='cuda')
        w0 = torch.tensor(w0, device='cuda')
        Q0 = torch.tensor(Q0, device='cuda')
        k_eff = torch.tensor(k_eff, device='cuda')
        m = torch.tensor(m, device='cuda')

        if log_all:
            U_log = torch.zeros((nt, u.size()[0]), device='cuda')
            P_log = torch.zeros((nt, u.size()[0]), device='cuda')
            H_log = torch.zeros((nt, u.size()[0]), device='cuda')
            r_log = torch.zeros((nt, u.size()[0]), device='cuda')

    # solve
    print(' % |  z_tip  |  z_target |  force  |  force_target  |  u(inf,t)')
    start = time_.time()
    for n in range(nt):
        t = n * dt  # advance time
        state_vectors, state_scalars, extra_vectors, extra_scalars = rk4_mixed(state_vectors, state_scalars, dt,
                                                                               rhs_mixed, t, b1, b0, c1, c0, r, dr,
                                                                               R, k_ij, I, use_cuda, f_exc_func,
                                                                               vb_func, m, k_eff, Q0, w0,
                                                                               p_vdw, 1e-19, 0, 1e-9)
        # log
        zb_zt_vt[n] = state_scalars
        f_ts_f_exc_at[n] = extra_scalars
        central_deformation[n] = state_vectors[0]
        time[n] = t
        u_inf[n] = state_vectors[-1]

        if n % (pct_log * nt) == 0:
            print('{} | {:.1f} nm | {:.1f} nN | {:.1f} nm | {:.1f} s'.format(
                n / nt, zb_zt_vt[n, 0] * 1e9, f_ts_f_exc_at[n, 0] * 1e9, state_vectors[-1] * 1e9, nt * dt))
        if log_all:
            U_log[n] = state_vectors[0]
            P_log[n] = extra_vectors[0]
            H_log[n] = extra_vectors[1]
            r_log[n] = r
    if use_cuda:
        zb_zt_vt = zb_zt_vt.cpu().numpy()
        f_ts_f_exc_at = f_ts_f_exc_at.cpu().numpy()
        central_deformation = central_deformation.cpu().numpy()
        time = time.cpu().numpy()
        k_eff = k_eff.cpu().numpy()
        u_inf = u_inf.cpu().numpy()
        if log_all:
            U_log = U_log.cpu().numpy()
            P_log = P_log.cpu().numpy()
            H_log = H_log.cpu().numpy()
            r_log = r_log.cpu().numpy()
    zb, zt, vt = zb_zt_vt.T
    f_ts, f_exc, at = f_ts_f_exc_at.T
    defl = zt - zb
    inden = - (zb + defl)  # this must be reversed as compared to the true relationship as zbase increases in AFM

    print('done in {:.0f}s'.format(time_.time() - start))
    # save sim data
    separation = zt - central_deformation
    df = pd.DataFrame({'time': time, 'separation_true': separation, 'tip_pos': zt, 'tip_vel': vt, 'tip_accel': at,
                       'base_pos': zb, 'force_ts_true': f_ts, 'force_ts_afm': defl * k_eff, 'defl': defl,
                       'inden_afm': inden, 'sample_pos': central_deformation, 'u_inf': u_inf})
    data = {'df': df, 'sim_args': saved_args, 'log_all': 0}
    if log_all:
        data['log_all'] = {'U_log': U_log, 'P_log': P_log, 'H_log': H_log, 'r_log': r_log}
    return data