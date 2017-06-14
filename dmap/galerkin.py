# -*- coding: utf-8 -*-
"""
Routines for performing Galerkin expansions of the data.
@author: Erik
"""

import numpy as np
import scipy.linalg as spl
import linalg as LA
import data_manipulation as dm

def get_generator(basis,traj_edges,delay=1,dt_eff=1.):
    """
    Constructs an approximation of the generator.
    """

    N = len(basis)
    # Get Starting indices and stopping
    t_0_indices, t_lag_indices = dm.start_stop_indices(traj_edges,delay)
    M = len(t_0_indices)
    du = 1.*(basis_t_lag - basis_t_0)/(dt_eff * delay)
    L = np.dot(np.transpose(basis_t_0),du)/M
    return L

def get_beta(fxn_vals,basis,traj_edges,delay=1):
    """

    """
    N = len(basis)
    t_0_indices, t_lag_indices = dm.start_stop_indices(traj_edges,delay)
    basis_t_0 = basis[t_0_indices]
    fxn_vals_t_0 = fxn_vals[t_0_indices]
    np.dot(fxn_vals_t_0,basis_t_0)/M


def get_ht(basis,stateA,traj_edges,delay=1,dt_eff=1.,on_tol=1E-4,normalize=True):
    """
    Calculates the hitting time using a galerkin method.
    """
    # Check if any of the basis functions are nonzero on target state.
    A_locs = np.where(stateA == 0)[0]

    if np.any(basis[A_locs] != 0.):
        raise RuntimeWarning("Some of the basis vectors are nonzero in state A.")

    L = get_generator(basis,traj_edges,delay=delay,dt_eff=dt_eff)
    beta = get_beta(stateA,basis,traj_edges,delay=delay)
    coeffs = spl.solve(L,beta)
    ht = np.dot(basis,coeffs)
    return ht

def get_committor(basis,g_guess,stateA,traj_edges,delay=1,expand_guess=False):
    """
    Calculates the committor for hitting B using a galerkin method.
    """
    # Check if any of the basis functions are nonzero on target state.
    N = len(basis) # Number of datapoints
    A_locs = np.where(stateA == 0)[0]
    B_locs = np.where(stateB == 0)[B]

    if np.any(basis[A_locs] != 0.):
        raise RuntimeWarning("Some of the basis vectors are nonzero in state A.")
    if np.any(basis[B_locs] != 0.):
        raise RuntimeWarning("Some of the basis vectors are nonzero in state B.")


    L = get_generator(basis,traj_edges,delay=delay,dt_eff=1,on_tol=on_tol,normalize=normalize)
    if expand_guess:
        guess_coeffs = get_beta(g_guess,basis,traj_edges,delay=delay)
        L_guess = np.dot(L,guess_coeffs)
    else:
        g_diff_full = np.zeros(N)
        g_diff = (g_guess[delay:]-g_guess[:-delay])/delay
        g_diff_full[:delay] = g_diff
        L_guess = get_beta(g_diff,basis,traj_edges,delay=delay)

    coeffs = spl.solve(L,-L_guess)
    delta_g = np.dot(basis,coeffs)
    return g_guess + delta_g

def clean_basis(basis,traj_edges,delay,on_tol=1E-4,backwards=False):
    # Normalize the eigenvectors
    N,k = np.shape(basis)
    t_0_indices, t_lag_indices = dm.start_stop_indices(traj_edges,delay)
    evec_norm = np.linalg.norm(basis,axis=0)
    basis *= np.sqrt(N)/evec_norm

    # Calculate orthogonal coefficients
    basis_t_0 = basis[t_0_indices]
    Q,R = spl.qr(basis_t_0)
    R_sub = R[:k,:k]
    basis_orthogonalized = np.dot(basis,R_sub)
    return basis_orthogonalized

    # basis_t_0 = basis[t_0_indices]
    # basis_t_lag = basis[t_lag_indices]
    #
    # # Check if orthonormal according to start_basis.
    # M = len(basis_t_0)
    # on_chck = np.dot(basis_t_0.T,basis_t_0)/M
    # np.fill_diagonal(on_chck,0)
    # if (np.abs(on_chck)>on_tol).any():
    #     raise RuntimeWarning('The dot product of the basis vectors restricted to the initial points does not pass the orthogonality test.')
    #
    # return basis
    #

# def get_reverse_generator(evecs,delay=1,dt_eff=1.,normalize=False):
#     """
#     Constructs an approximation of the generator.
#     """
#     N = len(basis)
#     # Get Starting indices and stopping
#     t_0_indices, t_lag_indices = dm.start_stop_indices(traj_edges,delay)
#     # Normalize the eigenvectors
#     if normalize:
#         evec_norm = np.linalg.norm(basis,axis=0)
#         basis *= np.sqrt(N)/evec_norm
#     basis_t_0 = basis[t_0_indices]
#     basis_t_lag = basis[t_lag_indices]
#
#     # Check if orthonormal according to start_basis.
#     M = len(basis_t_0)
#     on_chck = np.dot(basis_t_0.T,basis_t_0)/M
#     np.fill_diagonal(on_chck,0)
#
#     if (np.abs(on_chck)>on_tol).any():
#         raise RuntimeWarning('The dot product of the basis vectors restricted to the initial points does not pass the orthogonality test.')
#
#     du = 1.*(basis_t_lag - basis_t_0)/(dt_eff * delay)
#     A = np.dot(np.transpose(basis_t_0),du)/(M)
#     return beta

# def get_committor_dense(evecs,state_A,state_B,complement=None,dt_eff=1.,normalize=False):
#     # Normalize eigenvectors appropriately.
#     evecs = np.array(evecs).astype('float')
#     N = len(evecs)
#     if normalize:
#         evec_norm = np.linalg.norm(evecs,axis=0)
#         evecs*= np.sqrt(N)/evec_norm
#     if complement is None:
#         complement = 1-state_A-state_B # Set of data points not in A or B.
#
#     # Get Propagating Term, E[u_j 1_(AUC)^c L u_i]
#     du = np.diff(evecs,axis=0)/dt_eff
#     u_dot_ind = np.transpose(evecs[:-1]) * complement[:-1]
#     L_prop = np.dot(u_dot_ind,du)/(N-1)
#
#     # Get boundary terms
#     L_A = np.dot(np.transpose(evecs)*state_A,evecs)/N
#     L_B = np.dot(np.transpose(evecs)*state_B,evecs)/N
#     mod_Gen = L_prop+L_A+L_B
#
#     b =  np.dot(np.transpose(evecs),state_B)/N
#     try:
#         x = spl.solve(mod_Gen,b)
#     except:
#         print mod_Gen
#         print L_A
#         print L_B
#         raise
#     g = np.dot(evecs,x)
#     return g

# def get_ht(evecs,state_A,dt_eff=1.,normalize=False):
#     # Normalize eigenvectors appropriately.
#     evecs = np.array(evecs).astype('float')
#     N = len(evecs)
#     if normalize:
#         evec_norm = np.linalg.norm(evecs,axis=0)
#         evecs*= np.sqrt(N)/evec_norm
#     complement = 1-state_A # Set of data points not in A or B.
#
#     # Get Propagating Term, E[u_j 1_(AUC)^c L u_i]
#     du = np.diff(evecs,axis=0)/dt_eff
#     u_dot_ind = np.transpose(evecs[:-1]) * complement[:-1]
#     L_prop = np.dot(u_dot_ind,du)/(N-1)
#
#     # Get boundary terms
#     L_A = np.dot(np.transpose(evecs)*state_A,evecs)/N
#     mod_Gen = L_prop+L_A
#
#     b =  np.dot(np.transpose(evecs),(state_A-1))/N
#     try:
#         np.save('L_prop',L_prop)
#         np.save('L_A',L_A)
#         x = spl.solve(mod_Gen,b)
#     except:
#         print mod_Gen
#         print b
#         raise
#     tau = np.dot(evecs,x)
#     return tau,x

#
# def get_tau(data,epsilon,nevecs,alpha=None,beta=None,weights=None,D=1,period=None,nneighb=200,normalize=False):
#     dm_evals, evecs = get_Dmap_evecs(data,epsilon,nevecs,weights=weights,D=D,alpha=alpha,beta=beta,period=period,nneighb=nneighb)
#     if normalize:
#         evec_norm = np.linalg.norm(evecs,axis=0)
#         evecs*= np.sqrt(N)/evec_norm
#
#     nevecs = len(evecs[0])
#     A = get_generator(evecs,dt_eff)
#     beta = get_beta(evecs,f)
#     Ainv = LA.groupInverse(A)
#     gksoln = np.dot(LA.groupInverse(A),beta)
#     top = np.dot(evecs,gksoln)
#     fntop = np.dot(f,top)
#     tau = -2.*fntop/np.dot(f,f)
#     return tau
