# -*- coding: utf-8 -*-
"""
Routines for performing Galerkin expansions of the data.
@author: Erik
"""

import numpy as np
import scipy.linalg as spl
import data_manipulation as dm

def get_generator(basis,traj_edges,delay=1,dt_eff=1.):
    """
    Constructs an approximation of the generator.
    """

    N = len(basis)
    # Get Starting indices and stopping
    t_0_indices, t_lag_indices = dm.start_stop_indices(traj_edges,delay)
    M = len(t_0_indices)
    basis_t_lag = basis[t_lag_indices]
    basis_t_0 = basis[t_0_indices]
    du = 1.*(basis_t_lag - basis_t_0)/(dt_eff * delay)
    L = np.dot(np.transpose(basis_t_0),du)/M
    return L

def get_stiffness_mat(basis,traj_edges,delay=1):
    N = len(basis)
    # Get Starting indices and stopping
    t_0_indices, t_lag_indices = dm.start_stop_indices(traj_edges,delay)
    basis_t_0 = basis[t_0_indices]
    M = len(t_0_indices)
    S = np.dot(np.transpose(basis_t_0),basis_t_0)/M
    return S

def get_transop(basis,traj_edges,delay=1):
    """
    Constructs an approximation of the generator.
    """
    N = len(basis)
    # Get Starting indices and stopping
    t_0_indices, t_lag_indices = dm.start_stop_indices(traj_edges,delay)
    M = len(t_0_indices)
    basis_t_lag = basis[t_lag_indices]
    basis_t_0 = basis[t_0_indices]
    T = np.dot(np.transpose(basis_t_0),basis_t_lag)/M
    return T

def get_beta(fxn_vals,basis,traj_edges,delay=1):
    """

    """
    N = len(basis)
    t_0_indices, t_lag_indices = dm.start_stop_indices(traj_edges,delay)
    M = len(t_0_indices)
    basis_t_0 = basis[t_0_indices]
    fxn_vals_t_0 = fxn_vals[t_0_indices]
    return np.dot(fxn_vals_t_0,basis_t_0)/M


def get_ht(basis,stateA,traj_edges,delay=1,dt_eff=1.):
    """
    Calculates the hitting time using a galerkin method.
    """
    # Check if any of the basis functions are nonzero on target state.
    A_locs = np.where(stateA)[0]

    if np.any(basis[A_locs]):
        raise RuntimeWarning("Some of the basis vectors are nonzero in state A.")

    L = get_generator(basis,traj_edges,delay=delay,dt_eff=dt_eff)
    beta = get_beta(stateA-1.,basis,traj_edges,delay=delay)
    coeffs = spl.solve(L,beta)
    ht = np.dot(basis,coeffs)
    return ht,coeffs

def get_committor(basis,g_guess,stateA,stateB,traj_edges,delay=1,expand_guess=False):
    """
    Calculates the committor for hitting B using a galerkin method.
    """
    # Check if any of the basis functions are nonzero on target state.
    N = len(basis) # Number of datapoints
    print 'N', N
    A_locs = np.where(stateA)[0]
    B_locs = np.where(stateB)[0]
    if np.any(basis[A_locs]):
        raise RuntimeWarning("Some of the basis vectors are nonzero in state A.")
    if np.any(basis[B_locs]):
        raise RuntimeWarning("Some of the basis vectors are nonzero in state B.")


    L = get_generator(basis,traj_edges,delay=delay,dt_eff=1)
    if expand_guess:
        guess_coeffs = get_beta(g_guess,basis,traj_edges,delay=delay)
        L_guess = np.dot(L,guess_coeffs)
    else:
        g_diff_full = np.zeros(N)
        g_diff = (g_guess[delay:]-g_guess[:-delay])/delay
        print np.shape(g_diff_full), np.shape(g_diff)
        g_diff_full[:-delay] = g_diff
        L_guess = get_beta(g_diff,basis,traj_edges,delay=delay)
    coeffs = spl.solve(L,-L_guess)
    delta_g = np.dot(basis,coeffs)
    return g_guess + delta_g

def get_esystem(basis,traj_edges,delay=1):
    """

    """
    # Calculate Generator, Stiffness matrix
    L = get_generator(basis,traj_edges,delay=delay,dt_eff=1)
    S = get_stiffness_mat(basis,traj_edges,delay=delay)
    print np.shape(L), np.shape(S)
    # Calculate, sort eigensystem
    evals, evecs_l, evecs_r = spl.eig(L,b=S,left=True,right=True)
    idx = evals.argsort()[::-1]
    evals = evals[idx]
    evecs_l = evecs_l[:,idx]
    evecs_r = evecs_r[:,idx]
    # Expand eigenvectors into real space.
    expanded_evecs_l = np.dot(basis,evecs_l)
    expanded_evecs_r = np.dot(basis,evecs_r)
    return evals, expanded_evecs_l, expanded_evecs_r

