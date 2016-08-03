# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 11:07:50 2016

@author: erikthiede
"""
import numpy as np
from scipy.spatial.distance import cdist
import scipy.sparse as sps
from time import time


def diffusion_map(data,epsilon,weights=None,alpha=0.5,D=1.0,period=None):
    # Initialize variables
    if len(np.shape(data)) == 1:
        ndim = 1
    elif len(np.shape(data)) == 2:
        ndim = len(data[0]) 
    else:
        raise ValueError('Too many dimensions in Input')
    npnts = len(data)

    if period is not None: # Periodicity provided.
        if not hasattr(period,'__getitem__'): # Check if period is scalar
            period = [period]
    else:
        period = [None]*ndim
    data = np.array([dat for dat in data]) # Clean the data.
    if len(np.shape(data)) == 1: # If data is 1D, make it 2D so indices work
        data = np.array([data])
        data = np.transpose(data)
    if weights is None: # datapoints are unweighted.
        weights = np.ones(npnts)
    
    distsq = np.zeros((npnts,npnts))
    for x in xrange(ndim):
        trajt = np.transpose(np.array([data[:,x]]))
        dist_x = cdist(trajt,trajt)
        p = period[x]
        if p is not None:
            dist_x -= p*np.rint(dist_x/p)
        distsq += dist_x**2
    # Implement automatic epsilons?
    ks = np.exp(-distsq/(4.*D*epsilon))
    q_eps_i = np.sum(ks/weights,axis=1)
    q_sqrt = q_eps_i**alpha
    ks/=q_sqrt*weights
    d = np.sum(ks,axis=1,keepdims=True)
    P = ks/d
    d = d.flatten()
    d /= np.sum(d)
    return P,d

def kernel(distsq,epsilon,D,kfxn='quartic'):
    distsq  = np.array(distsq)
    if kfxn == 'gaussian':
        ks = np.exp(-distsq/(4.*D*epsilon))
    elif kfxn == 'quartic':
        uscaled = distsq/(14.*D*epsilon)
        df = 1.-uscaled
        df *= df> 0. 
        ks = df*df
    elif kfxn == 'triweight':
        print 'triweight!'
        uscaled = distsq/(2.*9.*D*epsilon)
        df = 1.-uscaled
        df *= df> 0. 
        ks = df*df*df
    return ks

def sparse_diff_map(data,epsilon,weights=None,alpha=0.5,D=1.0,period=None,kfxn='quartic'):
    # Initialize variables
    if len(np.shape(data)) == 1:
        ndim = 1
    elif len(np.shape(data)) == 2:
        ndim = len(data[0]) 
    else:
        raise ValueError('Too many dimensions in Input')
    npnts = len(data)

    if period is not None: # Periodicity provided.
        if not hasattr(period,'__getitem__'): # Check if period is scalar
            period = [period]
    else:
        period = [None]*ndim
    data = np.array([dat for dat in data]) # Clean the data.
    if len(np.shape(data)) == 1: # If data is 1D, make it 2D so indices work
        data = np.array([data])
        data = np.transpose(data)
    if weights is None: # datapoints are unweighted.
        weights = np.ones(npnts)

    print np.shape(data), np.shape(weights), 'shapes'
    print data[0]
    
    # Construct matrix of k values
    rows = [] 
    columns = []
    values = []
    for n, pnt in enumerate(data):
        dat2 = data[n:]
        vec_pnt = pnt-dat2
        # Enforce periodic boundary conditions:
        for dim in xrange(len(pnt)):
            p = period[dim]
            if p is not None:
                vec_pnt[:,dim] -= p*np.rint(vec_pnt/p)
        distsq_pnt = np.sum(vec_pnt**2,axis=1)
#        print distsq_pnt.sort()[:5]
        kvec = kernel(distsq_pnt,epsilon,D,kfxn=kfxn) # Values of the kernel
#        print np.shape(kvec)
        k_locs = np.nonzero(kvec) # Location of neighboring, nonadjacent windows
#        print np.shape(k_locs)
#        print n, len(k_locs), len(kvec), 'k loc length'
        kvec /= weights[n]*weights[n:]
        other_coord = list(np.ones(k_locs[0].shape)*n)
        nz_ks = list(kvec[k_locs]) # nonzero kernel values
        kl = list(k_locs[0]+n)
        rows += kl
        columns += other_coord
        values += nz_ks
        rows += other_coord[1:]
        columns += kl[1:]
        values += nz_ks[1:]
    ks = sps.coo_matrix((values,(rows,columns)))
    q_eps_i = np.array(ks.sum(axis=1)).flatten()
    q_sqrt = 1./(q_eps_i**alpha)
    q_dg = sps.dia_matrix((q_sqrt,[0]),shape=(npnts,npnts))
    ks = ks.dot(q_dg)
    d = np.array(ks.sum(axis=1)).flatten()
    pi = d / np.sum(d)
    P_dg = sps.dia_matrix((1/d,[0]),shape=(npnts,npnts))
    ks = P_dg.dot(ks)
    return ks,pi
        
def minimage_traj(rv,period):
    """Calculates the minimum trajectory

    Parameters
    ----------
    rv : 1 or 2D array-like 
        Minimum image trajectory
    period : array-like or scalar
        Periodicity in each dimension.

    Returns
    -------
    minimage : array-like
        minimum image trajectory
    """
    rvmin = np.array(np.copy(rv))
    if len(np.shape(rv)) == 1: # 1D trajectory array provided
        if period is not None:
            p = period[0]
            if (p is not None) and (p != 0): 
                rvmin -= p*np.rint(rvmin/p)

    elif len(np.shape(rv)) == 2: # 2D trajectory array provided
        ndim = len(rv[0])
        if period is not None:
            for d in xrange(ndim):
                p = period[d]
                if (p is not None) and (p != 0): 
                    rvmin[:,d]-= p*np.rint(rvmin[:,d]/p)
    else: # User provided something weird...
        raise ValueError("Trajectory provided has wrong dimensionality %d, "+ \
            "dimension should be 1 or 2."%len(np.shape(rv)))
    return rvmin

