import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quadrature, quad, RK45
from scipy.special import ellipk, ellipkm1, jv
from scipy.optimize import curve_fit
from afm.data_io import get_files
import sympy as sp
import pandas as pd
import pickle
from afm.visco import mdft, get_r


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


def p_vdw(h, H=1e-19, z0=1e-9):
    '''
    calculates the van der Waals pressure and derivative as a function of distance
    :param h: (float or numpy array) distance between bodies
    :param H: (float) Hamaker constant
    :param z0: (float) equilibrium separation distance
    :return: (float or numpy array x2) pressure, derivative of pressure with respect to distance
    '''
    return (H / (6 * np.pi * h ** 3) * ((z0 / h) ** 6 - 1),
            H / (2 * np.pi * h ** 4) * (1 - 3 * (z0 / h) ** 6))


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
    def __init__(self, E0, viscoelastic, Einf, Tau, v0, h0, R, force_target, depth_ratio_target, nr=1000,
                 dr=1.5e-9, dt=1e-4, nt=1e6):
        '''
        create an instance of Rajabifar's 'exact' approach to solving the Attard equation as outlined here:
        Dynamic AFM on Viscoelastic Polymer Samples with Surface Forces - Bahram Rajabifar
        :param E0: (float) normalized instantaneous modulus (Pa) E0 / (1 - v^2)
        :param viscoelastic: (bool) whether sample is viscoelastic (SLS) or elastic (w/ modulus E0)
        :param Einf: (float) normalized equilibrium modulus (Pa) Einf / (1 - v^2)
        :param Tau: (float) relaxation time (s)
        :param v0: (float) directional velocity of the indenter (indenter follows linear path in time) (m/s)
        :param h0: (float) initial position of indenter (m)
        :param R: (float) radius of indenter (m)
        :param force_target: (float) target force (N)
        :param depth_ratio_target: (float) ratio of indenter radius to indent into sample
        :param nr: (int) number of points in the radial discretization
        :param dr: (float) radial discretization (m)
        :param dt: (float) temporal discretization (s)
        :param nt: (int) number of points in the temporal discretization
        '''
        self.E0 = E0
        self.viscoelastic = viscoelastic
        self.Einf = Einf
        self.Tau = Tau
        self.v0 = v0
        self.h0 = h0
        self.R = R
        self.force_target = force_target
        self.pos_target = - depth_ratio_target * R
        self.nr = nr
        self.dr = dr
        self.dt = dt
        self.nt = nt
        # initialize
        self.r = np.linspace(1, self.nr, self.nr) * self.dr
        self.k_ij = np.array([K(r_, self.r, self.dr) for r_ in self.r])  # calculate integrating kernel
        self.I = np.eye(self.r.size)  # identity matrix
        # set loggers
        self.force = np.zeros(nt)
        self.separation = np.zeros(nt)
        self.tip_pos = np.zeros(nt)
        self.time = np.zeros(nt)

    def rhs(self, state):
        '''
        calculate the time derivative of the state vector (u, h0) assuming that h0(t)=v0*t
        :param state: (numpy array) state vector (u sample deformation, h0 indenter position)
        :return: (numpy array) derivative of state vector (u_dot, v0)
        '''
        u, h0 = state  # unpack state vector
        h = h_calc(h0, u, f_sphere, self.r, self.R)  # calculate the separation for the spherical indenter
        p, p_h = p_vdw(h)  # calculate the vdW pressure
        J_ij = 1 / self.E0 * self.k_ij * self.dr * p_h * self.r - self.I
        b_i = 1 / self.E0 * self.dr * self.v0 * (p_h * self.r) @ self.k_ij
        if self.viscoelastic:
            u_inf = - 1 / self.Einf * self.dr * (p * self.r) @ self.k_ij  # skip if elastic
            b_i += 1 / self.Tau * (u - u_inf)  # skip if elastic
        u_dot = np.linalg.solve(J_ij, b_i)
        return np.array([u_dot, self.v0], dtype='object')