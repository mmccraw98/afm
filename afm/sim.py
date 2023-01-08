import numpy as np
from scipy.special import ellipk, jv
from afm.data_io import next_path
import pandas as pd
import pickle


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
    :param H: (float) Hamaker constant
    :param z0: (float) equilibrium separation distance
    :return: (float or numpy array x2) pressure, derivative of pressure with respect to distance
    '''
    return (1 / (6 * np.pi * h ** 3) * (H1 * (z0 / h) ** 6 - H2),
            1 / (2 * np.pi * h ** 4) * (H2 - 3 * H1 * (z0 / h) ** 6))


def rk4(state, dt, rhs_func, *args):
    '''
    Runge Kutta 4th order time integration update scheme
    :param state: (numpy array) variables to integrate
    :param dt: (float) time discretization
    :param rhs_func: function to calculate the time derivatives of the state of form rhs_func(state, *args)
    :param args: (optional) arguments for the rhs_func
    :return: (numpy array) integrated state at t=t+dt
    '''
    k1 = rhs_func(state, *args)
    k2 = rhs_func(state + dt * k1 / 2, *args)
    k3 = rhs_func(state + dt * k2 / 2, *args)
    k4 = rhs_func(state + dt * k3, *args)
    return state + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6


class sim_rajabifar_1():
    def __init__(self, v0, h0, R, nr=1000, dr=1.5e-9, dt=1e-4, nt=int(1e6)):
        '''
        create an instance of Rajabifar's 'exact' approach to solving the Attard equation as outlined here:
        Dynamic AFM on Viscoelastic Polymer Samples with Surface Forces - Bahram Rajabifar
        :param v0: (float) directional velocity of the indenter (indenter follows linear path in time) (m/s)
        :param h0: (float) initial position of indenter (m)
        :param R: (float) radius of indenter (m)
        :param nr: (int) number of points in the radial discretization
        :param dr: (float) radial discretization (m)
        :param dt: (float) temporal discretization (s)
        :param nt: (int) number of points in the temporal discretization
        '''
        self.v0 = v0
        self.h0 = h0
        self.R = R
        self.nr = nr
        self.dr = dr
        self.dt = dt
        self.nt = nt
        # initialize
        self.r = np.linspace(1, self.nr, self.nr) * self.dr
        self.k_ij = np.array([K(r_, self.r, self.dr) for r_ in self.r])  # calculate integrating kernel
        self.I = np.eye(self.r.size)  # identity matrix
        self.force = None
        self.separation = None
        self.tip_pos = None
        self.time = None
        self.inden = None
        self.force_rep = None
        self.viscoelastic = None
        self.E0 = None
        self.Einf = None
        self.Tau = None
        self.H1 = None
        self.H2 = None
        self.z0 = None

    def rhs_vdw_sls(self, state, E0, Einf, Tau, viscoelastic):
        '''
        calculate the time derivative of the state vector (u, h0) assuming that h0(t)=v0*t
        works via inversion of J_ij * u_dot_i = b_i
        :param state: (numpy array) state vector (u sample deformation, h0 indenter position)
        :return: (numpy array) derivative of state vector (u_dot, v0)
        '''
        u, h0 = state  # unpack state vector
        h = h_calc(h0, u, f_sphere, self.r, self.R)  # calculate the separation for the spherical indenter
        p, p_h = p_vdw(h, H1=self.H1, H2=self.H2, z0=self.z0)  # calculate the vdW pressure
        J_ij = 1 / E0 * self.k_ij * self.dr * p_h * self.r - self.I  # calculate the LHS matrix J_ij
        b_i = 1 / E0 * self.dr * self.v0 * (p_h * self.r) @ self.k_ij  # calculate the RHS vector b_i
        if viscoelastic:  # introduce viscoelastic relaxation of SLS material
            u_inf = - 1 / Einf * self.dr * (p * self.r) @ self.k_ij  # skip if elastic
            b_i += 1 / Tau * (u - u_inf)  # skip if elastic
        u_dot = np.linalg.solve(J_ij, b_i)  # invert governing matrix eqn to find u_dot_i directly
        return np.array([u_dot, self.v0], dtype='object')

    def simulate(self, E0, Einf, Tau, viscoelastic, force_target, depth_ratio_target=0.1, H1=1e-19, H2=1e-19, z0=1e-9):
        '''
        :param E0: (float) normalized instantaneous modulus (Pa) E0 / (1 - v^2)
        :param Einf: (float) normalized equilibrium modulus (Pa) Einf / (1 - v^2)
        :param Tau: (float) relaxation time (s)
        :param viscoelastic: (bool) whether sample is viscoelastic (SLS) or elastic (w/ modulus E0)
        :param force_target: (float) target force (N)
        :param depth_ratio_target: (float) ratio of indenter radius to indent into sample
        :return:
        '''
        pos_target = - depth_ratio_target * self.R  # calculate position target for iteration

        # set loggers
        self.force = np.zeros(self.nt)
        self.separation = np.zeros(self.nt)
        self.tip_pos = np.zeros(self.nt)
        self.time = np.zeros(self.nt)
        self.viscoelastic = viscoelastic
        self.E0 = E0
        self.Einf = Einf
        self.Tau = Tau
        self.H1 = H1
        self.H2 = H2
        self.z0 = z0

        h0 = self.h0
        u = np.zeros(self.r.shape)
        state = np.array([u, h0], dtype='object')
        print(' % |  z_tip  |  z_target |  force  |  force_target')
        for n in range(self.nt):
            state = rk4(state, self.dt, self.rhs_vdw_sls, E0, Einf, Tau, viscoelastic)
            u, h0 = state
            h = h_calc(h0, u, f_sphere, self.r, self.R)
            p, p_h = p_vdw(h, H1=self.H1, H2=self.H2, z0=self.z0)
            self.force[n] = (2 * np.pi * p @ self.r * self.dr)
            self.separation[n] = h[0]
            self.tip_pos[n] = h0
            self.time[n] = n * self.dt
            if n % (0.01 * self.nt) == 0:
                print('{} | {:.1f} nm | {:.1f} nm | {:.1f} nN | {:.1f} nN'.format(n / self.nt,
                                                                                  self.tip_pos[n] * 1e9, pos_target * 1e9,
                                                                                  self.force[n] * 1e9, force_target * 1e9))
            if self.force[n] > force_target or self.tip_pos[n] < pos_target:
                self.force = self.force[:n]
                self.separation = self.separation[:n]
                self.tip_pos = self.tip_pos[:n]
                self.time = self.time[:n]
                break
        print('done!')
        self.inden = np.zeros(self.time.size)
        contact_index = np.argmin(self.force)
        self.inden[contact_index:] = self.tip_pos[contact_index] - self.tip_pos[contact_index:]
        self.force_rep = np.zeros(self.time.size)
        self.force_rep[contact_index:] = self.force[contact_index:] - self.force[contact_index]

    def save(self, savename):
        if self.force is None:
            print('nothing has been integrated, saving will be empty!')
        saveroot = '/home/mmccraw/Desktop/gwu/projects/repulsive contact/sim data/'
        df = pd.DataFrame({'time': self.time, 'separation': self.separation, 'tip_pos': self.tip_pos,
                           'inden': self.inden, 'force_rep': self.force_rep})
        data = {'df': df, 'v0': self.v0, 'viscoelastic': self.viscoelastic, 'R': self.R, 'dr': self.dr,
                'nr': self.nr, 'E0': self.E0, 'Einf': self.Einf, 'Tau': self.Tau}
        with open(next_path(saveroot + savename + '-%s.sim'), 'wb') as f:
            pickle.dump(data, f)