# -*- coding: utf-8 -*-
"""
Code for calculating Kernel Density approximation of the density, as described in "Variable Bandwidth Diffusion Kernels" by Berry and Harlim.

@author: erikthiede
"""
import numpy as np
import numbers

def kde(data,rho=None,period=None,nneighb=None,d=None,nn_rho=8,epses=2.**np.arange(-40,41),verbosity=0):
    """Code implementing Kernel Density estimatation.  Algorithm is heavily based on that presented in Berry, Giannakis, and Harlim, Phys. Rev. E. 91, 032915 (2015). 

    Parameters
    ----------
    data : ndarray
        Data to create the diffusion map on.  Can either be a one-dimensional time series, or a timeseries of Nxk, where N is the number of data points and k is the dimensionality of data.
    rho : ndarray or None, optional
        Bandwidth function rho(x) to use in the kernel, evaluated at each data point.  The kernel used is exp(-||x-y||^2/(rho(x)rho(y))).  If None is given (default), uses the automatic bandwidth procedure defined using nearest neighbors, as given by BGH.
    period : 1D array-like or float, optional
        Period of the coordinate, e.g. 360 for an angle in degrees. If None, all collective variables are taken to be aperiodic.  If scalar, assumed to be period of each collective variable. If 1D array-like with each value a scalar or None, each cv has periodicity of that size.
    nneighb : int, optional
        Approximate number of nearest neighbors to keep for each state in the kernel matrix.  This introduces a truncation error into the diffusion map.  However, due to exponential decay of the Kernel, this is generally negligible.  Default is None, i.e. to keep all neighbors.
    d : int, optional
        Dimensionality of the data.  If not given, detected automatically as part of the automatic bandwidth detection procedure.  Note that automatic bandwidth detection will only give good results if the values of epsilon provided include the optimal region descriped by Coifman and Lafon and BGH.  
    nn_rho : int, optional
        Number of nearest neighbors to use when constructing the automatic bandwidth function.  Default is 8.  If rho is provided by the user, this does nothing.
    epses : array-like, optional
        Epsilon values to be used for automatic bandwidth detection.  Requires at least three values.  Default is powers of 2 between 2^40 and 2^-40.  Note that the optimal epsilon value used will actually be *between* these values, due to it being estimated using a central difference of a function of the epsilon values. 
    verbosity : int, optional
        Whether to print verbose output.  If 0 (default), no updates are printed.  If 1, prints results of automated bandwidth and dimensionality routines.  If 2, prints program status updates.  CURRENTLY DOESN'T DO ANYTHING, TODO: ADD VERBOSE MESSAGES!

    Returns
    -------
    q : 1d array
        Estimated value of the Density.
    d_est : int
        Estimated dimensionality of the system.  This is not necessarily the same as the dimensionality used in the calculation if the user provides a value of d.
    eps : float
        Optimal bandwidth parameter for the system.
    """
    # Default Parameter Selection and Type Cleaning
    N = len(data)
    if nneighb is None:
        nneighb = N # If no neighbor no. provided, use full data set.
    if len(np.shape(data)) == 1: # If data is 1D structure, make it 2D 
        data = np.array([data])
        data = np.transpose(data)
    data = np.array([dat for dat in data]) 

    # Get nearest neighbors
    nn_indices, nn_distsq = get_nns(data,period,nneighb) 

    # Construct a bandwidth function if none is provided by the user. 
    if rho is None:
        rho_indices = np.zeros((N,nn_rho),dtype=np.int)
        rho_distsq = np.zeros((N,nn_rho))
        for i,row in enumerate(nn_distsq):
            # Get nearest nn_rho points to point i
            row_indices = np.argpartition(row,nn_rho-1)[:nn_rho]
            row_d2 = row[row_indices]
            rho_indices[i] = row_indices
            rho_distsq[i] = row_d2
        rho = np.sqrt(np.sum(rho_distsq,axis=1)/(nn_rho-1)) 

    # Perform automatic bandwidth selection. 
    scaled_distsq = np.copy(nn_distsq)
    for i, row in enumerate(scaled_distsq):
        row /= 2.*rho[i]*rho[nn_indices[i]]

    if isinstance(epses,numbers.Number):
        epsilon = epses
    else:
        eps_opt, d_est = get_optimal_bandwidth(scaled_distsq,epses=epses)
        if d is None: # If dimensionality is not provided, use estimated value.
            d = d_est

    # Estimated density.
    q0 = np.sum(np.exp(-scaled_distsq/eps_opt),axis=1)
    if np.any(rho-1.):
        if d is None:
            raise ValueError('Dimensionality needed to normalize the density estimate , but no dimensionality information found or estimated.'%param)
    q0 /= (rho**d) 
    q0 *= (2.*np.pi)**(-d/2.) /len(q0)
    return q0, d_est, eps_opt

def get_optimal_bandwidth(scaled_distsq,epses=2.**np.arange(-40,41)):
    """Calculates the optimal bandwidth for kernel density estimation, according to the algorithm of Berry and Harlim.

    Parameters
    ----------
    scaled_distsq : 1D array-like
        Value of the distances squared, scaled by the bandwidth function.  For instance, this could be ||x-y||^2 / (\rho(x) \rho(y)) evaluated at each pair of points.
    epses : 1D array-like, optional
        Possible values of the bandwidth constant.  The optimal value is selected by estimating the derivative in Giannakis, Berry, and Harlim using a forward difference.  Note: it is explicitely assumed that the the values are either monotonically increasing or decreasing.  Default is all powers of two from 2^-40 to 2^40.
    """
    # Calculate double sum.
    N = len(scaled_distsq)
    log_T = []
    log_eps = []
    for eps in epses:
#        kernel = np.exp(-scaled_distsq/float(eps))
        kernel = np.exp(-scaled_distsq/float(eps))
        log_T.append(np.log(np.average(kernel)))
        log_eps.append(np.log(eps))

    #### DEBUG ####
    np.save('log_T.npy',log_T)
    ###############
    
    # Find max of derivative of d(log(T))/d(log(epsilon)), get optimal eps, d
    log_deriv = np.diff(log_T)/np.diff(log_eps)
    max_loc = np.argmax(log_deriv)
    eps_opt = np.max([np.exp(log_eps[max_loc]),np.exp(log_eps[max_loc+1])])
    d = np.round(2.*log_deriv[max_loc])
    return eps_opt,d

def get_nns(data,period=None,nneighb=None,M=1,sort=False):
    """Get the indices of the nneighb nearest neighbors, and calculate the distance to them.

    Parameters
    ----------
    data : 2D array-like
        The location of every data point in the space
    period : 1D array-like or float, optional
        Period of the coordinate, e.g. 360 for an angle in degrees. If None, all collective variables are taken to be aperiodic.  If scalar, assumed to be period of each collective variable. If 1D array-like with each value a scalar or None, each cv has periodicity of that size.
    nneighb : int or None, optional
        Number of nearest neighbors to calculate.  If None, calculates all nearest neighbor distances.
    M : int or 2D array-like, optional
        Optional matrix defining the metric used to calculate distance.  If provided, calculates distance using the norm x^T M x.  Note: M should be positive definite.  If a scalar is provided, M is taken to be diagonal with that value on the diagonal.
    sort : bool, optional
        If True, returns the nearest neighbor distances sorted by distance to each point

    Returns
    -------
    indices : 2D array
        indices of the nearest neighbers.  Element i,j is the j'th nearest neighbor of the i'th data point.  
    distsq : 2D array
        Squared distance between points in the neighborlist.  If D is provided, this matrix is weighted.
        
    """
    if period is not None: # Periodicity provided.
        if not hasattr(period,'__getitem__'): # Check if period is scalar
            period = [period]
    else:
        period = [None]*len(data[0])
    npnts = len(data)
    if nneighb == None:
        nneighb = npnts
    indices = np.zeros((npnts,nneighb),dtype=np.int)
    distsq = np.zeros((npnts,nneighb))
    for i,pnt in enumerate(data):
        dx = pnt - data
        # Enforce periodic boundary conditions.
        for dim in xrange(len(pnt)):
            p = period[dim]
            if p is not None:
                dx[:,dim] -= p*np.rint(dx[:,dim]/p)
                
        dsq_i = np.sum(dx*np.dot(dx,M),axis=1)  
        # Find nneighb largest elements
        ui_i = np.argpartition(dsq_i,nneighb-1)[:nneighb] #unsorted indices
        ud_i = dsq_i[ui_i] # unsorted distances
        if sort: 
            sorter = ud_i.argsort()
            indices[i] = ui_i[sorter]
            distsq[i] = ud_i[sorter]
        else:
            indices[i] = ui_i
            distsq[i] = ud_i
    return indices, distsq
