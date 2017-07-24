# -*- coding: utf-8 -*-
"""
Routines for constructing a model on the full dataset.
@author: Erik
"""

import numpy as np
import scipy.linalg as spl
import scipy.sparse.linalg as spsl
import scipy.sparse as sps
import linalg as LA
import data_manipulation as dm
import galerkin as gkn

_stat_eval_tol = 10**-5

def get_data_model(basis,traj_edges,delay=1,dt_eff=1,clean_basis=True):
    """

    """
    # Orthogonalize basis vectors appropriately
    N,k = np.shape(basis)
    if clean_basis:
        bbasis = clean_basis(basis,traj_edges,delay=delay,orthogonalize=True)
    T_op = gkn.get_transop(basis,traj_edges,delay=delay)
    L = np.dot(np.dot(basis,T_op),basis.T)
    L -= np.eye(N)
    L /= dt_eff*delay
    return L

def get_ht(L,stateA):
    """

    """
    d_locs = np.where(stateA<1.0)[0]
    d_locs = d_locs.astype(int)
    L_r = L[d_locs]
    L_r = L_r[:,d_locs]
    if isinstance(L_r,sps.spmatrix):
        tau_small = spsl.spsolve(L_r,-np.ones(len(d_locs)))
    else:
        tau_small = spl.solve(L_r,-np.ones(len(d_locs)))
    tau = np.zeros(stateA.shape,dtype='float')
    tau[d_locs] = tau_small
    return tau

def get_committor(L,stateA,stateB):
    complement = np.where(stateA+stateB<1.)[0]
    Lr = L[complement,:]
    LB = Lr.dot(stateB)
    if isinstance(Lr,sps.spmatrix):
        Lr = Lr.tocsc()[:,complement].tocsr()
        g_small = spsl.spsolve(Lr,-LB)
    else:
        Lr = Lr[:,complement]
        g_small = spl.solve(Lr,-LB)
    g = np.copy(stateB)
    g[complement] = g_small
    return g

def get_stationary_distrib(L):
    """

    """
    # FIX FOR SPARSE MATRICES!!
    evals, evecs = spl.eig(L,left=True,right=False)
    evals, evecs = LA._sort_esystem(evals,evecs)
    stat = evecs[:,0]
    l1 = evals[0]
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print evals[:10], 'evals'
    print np.min(evals), np.max(evals)
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    if np.abs(l1) > _stat_eval_tol:
        raise RuntimeWarning('Eigenvalue corresponding to stationary distribution is %.4e, which has magnitude greater than the tolerance.'%l1)
    return stat
