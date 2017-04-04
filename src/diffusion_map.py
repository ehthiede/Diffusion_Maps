# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 11:07:50 2016

@author: erikthiede
"""
import numpy as np
import scipy.sparse as sps
import scipy.linalg as spl
import gc

def scaled_diffusion_map(data,epsilon,D=1.0,alpha=None,beta=None,period=None,nneighb=None,density=None,weights=None,d=None,return_full=False):
    """
    Code implementing the Variable Bandwidth Diffusion Map algorithm by Berry and Harlim (see section 3).

    Parameters
    ----------
    data : ndarray
        Data to create the diffusion map on.  Can either be a one-dimensional time series, or a timeseries of Nxk, where N is the number of data points and k is the dimensionality of data.
    epsilon : float
        Diffusion map lengthscale parameter
    D : scalar or 2D array-like, optional
        Diffusion tensor for the system.  If scalar, this is taken to be a diagonal matrix with the scaler on the diagonal.  If array, the array must be square..  Default value is 1.0.
    alpha : float, optional
        Diffusion map parameter, the exponent on the inverse of the density estimate in front of the Kernel: see Berry and Harlim for more details.  Default value is 0.5
    beta : float, optional
        Diffusion map parameter, the exponent of the density estimate in the Bandwidth scaling the distances.  Default value is 0.0 (No Bandwidth scaling)
    period : 1D array-like or float, optional
        Period of the collective variable e.g. 360 for an angle. If None, all collective variables are taken to be aperiodic.  If scalar, assumed to be period of each collective variable. If 1D iterable with each value a scalar or None, each cv has periodicity of that size.
    nneighb : int, optional
        Approximate number of nearest neighbors to keep for each state in the kernel matrix.  This introduces a truncation error into the diffusion map.  However, due to exponential decay of the Kernel, this is generally negligible.  Default is None, i.e. to keep all neighbors.
    density : 1d array-like, optional
        The stationary probability density of the process, given up to a constant multiple.  If provided, this is used to estimate the bandwidth in the numerator of the diffusion map.  If not provided, the density is instead estimated through Kernel density estimation.
    weights : 1d array-like, optional
        The ratio between the desired probability distribution and the sampled probability distribution.  Unless you are doing importance sampling or umbrella sampling, this should probably be one for each sample point (the default).
    d : int, optional
        Dimensionality of the system.  Default is the number of coordinates in each datapoint.  However, for some systems (e.g. a circlular orbit in 2D), this may be higher than the true dimensionality of the system.
    return_full : bool, optional
        If True, returns the expanded set of matrices and vectors.  
    
    Returns
    -------
    L : scipy sparse matrix
        Diffusion map generator.  This is in a sparse matrix representation, a dense representation can be achieved by running L.toarray().
    pi : ndarray
        Stationary distribution of the generator.
    Kmat : scipy sparse matrix, if return_full==True
        Symmetric Kernel matrix used to construct L.
    rho : ndarray, if return_full==True
        Bandwidth used to construct K and L.

    """
    if len(np.shape(data)) == 1: # If data is 1D, make it 2D so indices work
        data = np.array([data])
        data = np.transpose(data)
    # Initialize variables
    ndim = len(data[0]) 
    npnts = len(data)
    

    if len(np.shape(data)) == 1:
        data = np.transpose([data])
    if d is None:
        d = len(data[0]) # Number of dimensions
    N = len(data) # Number of datapoints
    if nneighb is None:
        nneighb = N # Use full data set.
    if density is not None:
        if len(density) != N:
            raise Exception
    if alpha is None:
        alpha = -1.*d/4.
    if beta is None:
        beta = -0.5

    if period is not None: # Periodicity provided.
        if not hasattr(period,'__getitem__'): # Check if period is scalar
            period = [period]
    else:
        period = [None]*len(data[0])

    data = np.array([dat for dat in data]) # Clean the data.
    nn_indices, nn_distsq = get_nns(data,period,nneighb)

    ##### If no density provided, calculate initial estimate of the probability density
    if beta == 0:
        rho = np.ones(len(data))
    elif density is None:
#    if True:
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
    # Normalize values to make epsilon choices commensurate....
    rho /= np.median(rho)

    ##### Calculate the new Kernel.
    try:
        Dinv = spl.inv(D)
    except:
        Dinv = 1./D
    print 'Dinv is ', Dinv
    nn_indices, nn_distsq = get_nns(data,period,nneighb,Dinv=Dinv)
    # Create the Sparse Kernel Matrix
    K = np.copy(nn_distsq)
    for i, row in enumerate(K):
        row /= 4.*epsilon*rho[i]*rho[nn_indices[i]]
    K = np.exp(-K) # Value of the Kernel fxn for Dmaps
    # We first convert to sparse matrix format.
    rows = np.outer(np.arange(N),np.ones(nneighb))
    Kmat = sps.csr_matrix((K.flatten(),(rows.flatten(),nn_indices.flatten())),shape=(N,N))
    gc.collect()
    # We symmetrize K.
    Ktrans = Kmat.transpose()
    dKmat = (Kmat -Ktrans)
    dKmat = dKmat.multiply(dKmat.sign())
    dKmat /= 2.
    Kmat = Kmat + Ktrans
    Kmat /= 2.
    Kmat = Kmat + dKmat
    
    if density is None:
        q = np.array(Kmat.sum(axis=1)).flatten()
        q /= (rho**d)
    else:
        q = density
    diagq = sps.dia_matrix((1./(q**alpha),[0]),shape=(N,N))
#    Kmat = diagq * Kmat * diagq
    Kmat = diagq * Kmat 
    Kmat = Kmat * diagq
    del diagq
    if weights is not None:
        print 'found weight vector'
        diag_wt = sps.dia_matrix((weights**0.5,[0]),shape=(N,N))
#        diag_wt = sps.dia_matrix((weights,[0]),shape=(N,N))
        Kmat = diag_wt * Kmat
        Kmat = Kmat * diag_wt
        del diag_wt
    q_alpha = np.array(Kmat.sum(axis=1)).flatten()
    diagq_alpha = sps.dia_matrix((1./(q_alpha),[0]),shape=(N,N))
    L = diagq_alpha * Kmat # Normalize each of the rows of the matrix
    diag = L.diagonal()-1.
    L.setdiag(diag)
    diag_norm = sps.dia_matrix((1./(rho**2*epsilon),0),shape=(N,N))
    L = diag_norm * L
    pi = rho**2* q_alpha
    pi /= np.sum(pi)
    if return_full:
        return L,pi, Kmat, rho, q
    else:
        return L,pi 

def get_nns(data,period=None,nneighb=64,Dinv=1):
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
                
        dsq_i = np.sum(dx*np.dot(dx,Dinv),axis=1)  
        # Find nneighb largest elements
        ui_i = np.argpartition(dsq_i,nneighb-1)[:nneighb] #unsorted indices
        ud_i = dsq_i[ui_i] # unsorted distances
        sorter = ud_i.argsort()
        indices[i] = ui_i[sorter]
        distsq[i] = ud_i[sorter]
    return indices, distsq

    
def _minimage_traj(rv,period):
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

