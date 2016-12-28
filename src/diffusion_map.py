# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 11:07:50 2016

@author: erikthiede
"""
import numpy as np
from scipy.spatial.distance import cdist
import scipy.sparse as sps
import itertools

def diffusion_map(data,epsilon,alpha=0.5,weights=None,D=1.0,period=None):
    """
    Implementation of the Diffusion Map algorithm as described by Coifman and Laffon.

    Parameters
    ----------
    data : ndarray
        Data to create the diffusion map on.  Can either be a one-dimensional time series, or a timeseries of Nxd, where N is the number of data points and d is the dimensionality of data.
    epsilon : float
        Diffusion map lengthscale parameter
    alpha : float
        Diffusion map parameter, the exponent on the inverse of the density estimate.
    weights : ndarray
        Importance sampling ratio for each point in the trajectory 
    D : float
        Diffusion Constant for the system.
    period : 1D array-like or float
        Period of the collective variable e.g. 360 for an angle. If None, all collective variables are taken to be aperiodic.  If scalar, assumed to be period of each collective variable. If 1D iterable with each value a scalar or None, each cv has periodicity of that size.
    
    Returns
    -------
    L : ndarray
        Diffusion map generator
    d : ndarray
        Stationary distribution of the generator.
    """
    if len(np.shape(data)) == 1: # If data is 1D, make it 2D so indices work
        data = np.array([data])
        data = np.transpose(data)
    # Initialize variables
    ndim = len(data[0]) 
    npnts = len(data)
    
    if period is not None: # Periodicity provided.
        if not hasattr(period,'__getitem__'): # Check if period is scalar
            period = [period]*ndim
    else:
        period = [None]*ndim
    data = np.array([dat for dat in data]) # Clean the data.
    if weights is None: # datapoints are unweighted.
        weights = np.ones(npnts)
    
    distsq = np.zeros((npnts,npnts))
    for x in xrange(ndim):
        trajt = np.transpose(np.array([data[:,x]]))
        dist_x = cdist(trajt,trajt)
        p = period[x]
        if p is not None:
            print p, 'periodic'
            dist_x -= p*np.rint(dist_x/p)
        distsq += dist_x**2
#        print x
    # Implement automatic epsilons?
    distsq = -distsq/(4.*D*epsilon)
    ks = np.exp(distsq)
    q_eps_i = np.sum(ks*weights,axis=1)
    q_sqrt = q_eps_i**alpha
    ks *=weights/q_sqrt
    ks /= np.transpose([q_sqrt])
    d = np.sum(ks,axis=1,keepdims=True)
    P = ks/d
    d = d.flatten()
    d /= np.sum(d)
    return (P-np.eye(len(P)))/epsilon,d

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

def sparse_diff_map(data,epsilon,weights=None,alpha=0.5,D=1.0,period=None,kfxn='gaussian'):
    # Initialize variables
    if len(np.shape(data)) == 1:
        data = np.transpose([data])
    ndim = len(data[0]) 
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

#    print np.shape(data), np.shape(weights), 'shapes'
#    print data[0]
    
    # Construct matrix of k values
    rows = [] 
    columns = []
    values = []
    for n, pnt in enumerate(data):
        print np.shape(data), np.shape(pnt)
        dat2 = data[n:]
        vec_pnt = pnt-dat2
        # Enforce periodic boundary conditions:
        for dim in xrange(len(pnt)):
            p = period[dim]
            if p is not None:
                vec_pnt[:,dim] -= p*np.rint(vec_pnt[:,dim]/p)
        distsq_pnt = np.sum(vec_pnt**2,axis=1)
        kvec = kernel(distsq_pnt,epsilon,D,kfxn=kfxn) # Values of the kernel
        k_locs = np.nonzero(kvec) # Location of neighboring, nonadjacent windows
        kvec *= weights[n]*weights[n:]
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


def scaled_diffusion_map(data,epsilon,density=None,weights=None,D=1.0,alpha=None,beta=None,d=None,period=None,nneighb=64,return_q=False):
    """
    Code implementing the Variable Bandwidth Diffusion Map algorithm by Berry and Harlim (see section 3). 
    """
    if len(np.shape(data)) == 1:
        data = np.transpose([data])
    if d is None:
        d = len(data[0]) # Number of dimensions
    N = len(data) # Number of datapoints
    if density is not None:
        if len(density) != N:
            raise Exception
    if alpha is None:
        alpha = -1.*d/4.
#        alpha = 1./2.
    if beta is None:
        beta = -0.5
#        beta = 0.0

    if period is not None: # Periodicity provided.
        if not hasattr(period,'__getitem__'): # Check if period is scalar
            period = [period]
    else:
        period = [None]*len(data[0])

    data = np.array([dat for dat in data]) # Clean the data.
    nn_indices, nn_distsq = get_nns(data,period,nneighb)

    ##### If no density provided, calculate initial estimate of the probability density
    if density is None:
        rho0= np.sqrt(np.sum(nn_distsq[:,:8],axis=1)/(8-1)) # Initial bandwidth for KDE
        q0 = np.copy(nn_distsq)
        for i, row in enumerate(q0):
            row /= 2.*rho0[i]*rho0[nn_indices[i]]
        q0 = np.sum(np.exp(-q0),axis=1)
        q0 /= (rho0**d) 
        q0 *= (2.*np.pi)**(-d/2.) /len(q0)
        rho = q0**beta # Bandwidth for the Diffusion Map.
        # Delete unused datastructures
        del q0
        del rho0
    else:
        rho = density**beta

    ##### Calculate the new Kernel.
    # Create the Sparse Kernel Matrix
    K = np.copy(nn_distsq)
    for i, row in enumerate(K):
        row /= 4.*epsilon*D*rho[i]*rho[nn_indices[i]]
    K = np.exp(-K) # Value of the Kernel fxn for Dmaps
    # We first convert to sparse matrix format.
    rows = np.outer(np.arange(N),np.ones(nneighb))
    Kmat_coo = sps.coo_matrix((K.flatten(),(rows.flatten(),nn_indices.flatten())),shape=(N,N))
    Kmat = Kmat_coo.tocsr() # Convert to 
    # We symmetrize K.
    dKmat = (Kmat -Kmat.transpose())
    dKmat = dKmat.multiply(dKmat.sign())
    Kmat = (Kmat +  Kmat.transpose())/2. + dKmat/2.
    
    if density is None:
        q = np.array(Kmat.sum(axis=1)).flatten()
        q /= (rho**d)
    else:
        q = density
    diagq = sps.dia_matrix((1./(q**alpha),[0]),shape=(N,N))
    Kmat = diagq * Kmat * diagq
    q_alpha = np.array(Kmat.sum(axis=1)).flatten()
    diagq_alpha = sps.dia_matrix((1./(q_alpha),[0]),shape=(N,N))
    L = diagq_alpha * Kmat # Normalize each of the rows of the matrix
    L = L - sps.eye(N)
    diag_norm = sps.dia_matrix((1./(rho**2*epsilon),0),shape=(N,N))
    L = diag_norm * L
    if return_q:
        return L, Kmat, rho, q_alpha, q
    else:
        return L, Kmat, rho, q_alpha

def get_nns(data,period=None,nneighb=64):
    """
    get the indices of the nneighb nearest neighbors, and calculate the distance to them

    Parameters
    ----------
    data : 2D array-like
        The location of every data point in the space
    period: array-like or scalar
        Periodicity of the space in each dimension.
    nneighb : int
        Number of nearest neighbors to calculate.

    Returns
    -------
    indices : 2D array
        indices of the nearest neighbers.  Element i,j is the j'th nearest neighbor of the i'th data point.  
    distsq : 2D array
        Squared distance between points in the neighborlist.
        
    """
    npnts = len(data)
    indices = np.zeros((npnts,nneighb),dtype=np.int)
    distsq = np.zeros((npnts,nneighb))
    for i,pnt in enumerate(data):
        dx = pnt - data
        # Enforce periodic boundary conditions.
        for dim in xrange(len(pnt)):
            p = period[dim]
            if p is not None:
                dx[:,dim] -= p*np.rint(dx[:,dim]/p)
        dsq_i = np.sum(dx**2,axis=1) # distance squared values for all points
        # Find nneighb largest elements
        ui_i = np.argpartition(dsq_i,nneighb-1)[:nneighb] #unsorted indices
        ud_i = dsq_i[ui_i] # unsorted distances
        sorter = ud_i.argsort()
        indices[i] = ui_i[sorter]
        distsq[i] = ud_i[sorter]
    return indices, distsq
        

    
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

