# -*- coding: utf-8 -*-
"""
Routines for performing Galerkin expansions of the data.

@author: Erik
"""

import numpy as np
from scipy.sparse.linalg import eigsh
import scipy.sparse as sps
import scipy.linalg as spl
import linalg as LA
#from diffusion_map import diffusion_map

def get_generator(evecs,delay=1,dt_eff=1.,normalize=False):
    """
    Constructs an approximation of the generator.
    """
    N = len(evecs)
    if normalize:
        evec_norm = np.linalg.norm(evecs,axis=0)
        evecs*= np.sqrt(N)/evec_norm
    du = evecs[delay:] - evecs[:-1*delay]
    du /= dt_eff * delay
    A = np.dot(np.transpose(evecs[:-1]),du)/(N-1)
    return A

def get_beta(evecs,f):
    N = len(f)
    beta = np.dot(f[:-1],evecs[:-1])/(N-1)
    return beta

def get_committor_dense(evecs,state_A,state_B,complement=None,dt_eff=1.,normalize=False):
    # Normalize eigenvectors appropriately.
    evecs = np.array(evecs).astype('float')
    N = len(evecs)
    if normalize:
        evec_norm = np.linalg.norm(evecs,axis=0)
        evecs*= np.sqrt(N)/evec_norm
    if complement is None:
        complement = 1-state_A-state_B # Set of data points not in A or B.

    # Get Propagating Term, E[u_j 1_(AUC)^c L u_i]
    du = np.diff(evecs,axis=0)/dt_eff
    u_dot_ind = np.transpose(evecs[:-1]) * complement[:-1]
    L_prop = np.dot(u_dot_ind,du)/(N-1)

    # Get boundary terms
    print 'state A check',np.sum(evecs[:,0]*state_A)
    print 'state B check',np.sum(evecs[:,-1]*state_B)
    L_A = np.dot(np.transpose(evecs)*state_A,evecs)/N
    L_B = np.dot(np.transpose(evecs)*state_B,evecs)/N
    mod_Gen = L_prop+L_A+L_B

    b =  np.dot(np.transpose(evecs),state_B)/N
    try:
        x = spl.solve(mod_Gen,b)
    except:
        print mod_Gen
        print L_A
        print L_B
        raise
    g = np.dot(evecs,x)
    return g
    
    
    
def get_tau(data,epsilon,nevecs,alpha=None,beta=None,weights=None,D=1,period=None,nneighb=200,normalize=False):
    dm_evals, evecs = get_Dmap_evecs(data,epsilon,nevecs,weights=weights,D=D,alpha=alpha,beta=beta,period=period,nneighb=nneighb)
    if normalize:
        evec_norm = np.linalg.norm(evecs,axis=0)
        evecs*= np.sqrt(N)/evec_norm

    nevecs = len(evecs[0])
    A = get_generator(evecs,dt_eff)
    beta = get_beta(evecs,f)
    Ainv = LA.groupInverse(A)
    gksoln = np.dot(LA.groupInverse(A),beta)
    top = np.dot(evecs,gksoln)
    fntop = np.dot(f,top)
    tau = -2.*fntop/np.dot(f,f)
    return tau
    
if __name__=='__main__':
	main()
