# -*- coding: utf-8 -*-
"""
Code that constructs a diffusion map.

@author: erikthiede
"""
import numpy as np
import scipy.sparse as sps
import scipy.linalg as spl
import kde 
import numbers
from _NumericStringParser import _NumericStringParser

def diffusion_map(data,rho=None,period=None,nneighb=None,D=1.0,weights=None,d=None,alpha='0',beta='-1/(d+2)',epses=2.**np.arange(-40,41),rho_norm=True,return_full=False,verbosity=0):
    """Routine that creates a diffusion map.  A diffusion map is a transition rate matrix that accounts for the local structure of the data.

    Parameters
    ----------
    data : 2D array-like
        Two-dimensional dataset used to create the diffusion map.
    rho : 1d array-like or None, optional
        Bandwidth function to be used in the variable bandwdith kernel.  If None, the code estimates the density of the data q using a kernel density estimate, and sets the bandwdith to q_\epsilon^beta
    period : 1D array-like or float, optional
        Period of the coordinate, e.g. 360 for an angle in degrees. If None, all coordinates are taken to be aperiodic.  If scalar, assumed to be period of each coordinate. If 1D array-like with each value a scalar or None, each coordinate has periodicity of that size.
    nneighb : int or None, optional
        Number of neighbors to include in constructing the diffusion map.  Default is None, which corresponds to using all neighbors.
    D : float or 2D square array-like, optional
        The diffusion tensor to be used in the system.  If float, is taken to be isotropic diffusion with that float on the diagonal.  If an array, is taken to be a positive definite diffusion matrix.  The distance used to construct the diffusion map will be (x-y)^T D^{-1} (x-y).  Default value is 1., or the identity.
    weights : 1D array-like or None, optional
        Importance sampling weights for each datapoint, w(x). 
    d : int or None, optional
        Dimension of the system. If None, dimension is estimated using the kernel density estimate, if a kernel density estimate is performed.
    alpha : float or string, optional
        Parameter for left normalization of the Diffusion map.  Either a float, or a string that cane be interpreted as a mathematical expression.  The variable "d" stands for dimension, so "1/d" sets the alpha to the system dimension.  Default is 0.5
    beta : float or string, optional
        Parameter for constructing the bandwidth function for the Diffusion map.  If rho is None, it will be set to q_\epsilon^beta, where q_\epsilon is an estimate of the density.  If rho is provided, this parameter is unused.  As with alpha, this will interpret strings that are evaluatable expressions.  Default is 0.0
    epses: float or 1d array, optional
        Bandwidth constant to use.  If float, uses that value for the bandwidth.  If array, performs automatic bandwidth detection according to the algorithm given by Berry and Giannakis and Harlim.  Default is all powers of 2 from 2^-40 to 2^40.
    rho_norm : bool, optional
        Whether or not to normalize q and L by rho(x)^2.  Default is True (perform normalization)
    return_full : bool, optional
        Whether or not to return expanded output.  Default is False.
    verbosity : int, optional
        Whether to print verbose output.  If 0 (default), no updates are printed.  If 1, prints results of automated bandwidth and dimensionality routines.  If 2, prints program status updates.

    Returns
    -------
    L : scipy sparse matrix
        The diffusion map operator.  This is a transition rate matrix whose rows sum to zero.
    pi : 1D numpy array
        The stationary distribution of the chain. pi.L should be the vector of all zeros.
    K : scipy sparse matrix, optional
        The symmetric kernel matrix used to construct the diffusion map.  Returned if return_full is set to True.
    rho : 1D numpy array, optional
        The bandwidth function used.  Returned if return_full is set to True.
    q_alpha : 1D numpy array, optional
        Row sum used to normalize the transition matrix in the Diffusion map.  Returned if return_full is set to True.
    epsilon : float, optional
        Bandwidth constant used in the calculation.  Returned if return_full is set to True.

    """
    ## Default Parameter Selection and Type Cleaning
    if len(np.shape(data)) == 1: # If data is 1D, make it 2D so indices work
        data = np.array([data])
        data = np.transpose(data)
    N = len(data)
    d_kde = None # Density estimate from kde
    if rho is None: # If no bandwidth fxn given, get one from KDE.
        rho,d_kde = get_bandwidth_fxn(data,period,nneighb,epses=epses,beta=beta,d=d)
        if verbosity >= 1 : 
            if d_kde is None:
                print "No Diffusion Map Bandwidth given.  Bandwidth constructed using a KDE.  No dimensionality info detected."
            else:
                print "No Diffusion Map Bandwidth given.  Bandwidth constructed using a KDE.  KDE dimension is %d"%d_kde
    if nneighb is None:
        nneighb = N

    # Evaluate scaled distances
    try:
        Dinv = spl.inv(D)
    except:
        Dinv = 1./D
    nn_indices, nn_distsq = kde.get_nns(data,period,nneighb,M=Dinv)
    for i, row in enumerate(nn_distsq):
        row /= rho[i]*rho[nn_indices[i]]
    old_distsq_sc = np.copy(nn_distsq)

    # Calculate optimal bandwidth
    if isinstance(epses,numbers.Number):
        epsilon = epses
        if verbosity >= 1 : print "Epsilon provided by the User: %f"%epsilon
    else:
        epsilon, d_est = kde.get_optimal_bandwidth(nn_distsq,epses=epses)
        if verbosity >= 1 : print "Epsilon automatically detected to be : %f"%epsilon
        if d is None: # If dimensionality is not provided, use estimated value.
            d = d_est
            if verbosity >= 1 : print "Dimensionality estimated to be %d."% d

    # Construct sparse kernel matrix.
    nn_distsq /= epsilon
    nn_distsq = np.exp(-nn_distsq) # Value of the Kernel fxn for Dmaps
    rows = np.outer(np.arange(N),np.ones(nneighb))
    K = sps.csr_matrix((nn_distsq.flatten(),
                       (rows.flatten(),nn_indices.flatten())),shape=(N,N))
    if verbosity >= 2 : print "Evaluated Kernel"

    # Symmetrize K using 'or' operator.
    Ktrans = K.transpose()
    dK = abs(K -Ktrans)
    K = K + Ktrans
    K = K + dK
    K *= 0.5

    # Apply q^alpha normalization.
    q = np.array(K.sum(axis=1)).flatten()
    if rho_norm:
        if np.any(rho-1.): # Check if bandwidth function is nonuniform. 
            if d is None:
                if d_kde is None:
                    raise ValueError('Dimensionality needed to normalize the density estimate , but no dimensionality information found or estimated.')
                else:
                    d = d_kde
            q /= (rho**d)
    alpha = _eval_param(alpha,d)
    diagq = sps.dia_matrix((1./(q**alpha),[0]),shape=(N,N))
    K = diagq * K 
    K = K * diagq
    if verbosity >= 2 : print r"Applied q**\alpha normalization."

    # Apply importance sampling weights if provided.
    if weights is not None:
        diag_wt = sps.dia_matrix((weights**0.5,[0]),shape=(N,N))
        K = diag_wt * K
        K = K * diag_wt

    # Normalize to Transition Rate Matrix
    q_alpha = np.array(K.sum(axis=1)).flatten()
    diagq_alpha = sps.dia_matrix((1./(q_alpha),[0]),shape=(N,N))
    L = diagq_alpha * K # Normalize row sum to one.
    diag = L.diagonal()-1.
    L.setdiag(diag) #  subtract identity.
    if verbosity >= 2 : print r"Applied q**\alpha normalization."

    # Normalize matrix by epsilon, and (if specified) by bandwidth fxn. 
    if rho_norm:
        diag_norm = sps.dia_matrix((1./(rho**2*epsilon),0),shape=(N,N))
    else:
        diag_norm = sps.eye(N)*(1./epsilon)
        pi = q_alpha
    L = diag_norm * L
    if verbosity >= 2 : print "Normalized matrix to transition rate matrix."

    # Calculate stationary density.
    if rho_norm:
        pi = rho**2* q_alpha
    else:
        pi = q_alpha
    pi /= np.sum(pi)
    if verbosity >= 2 : print "Estimated Stationary Distribution."

    # Return calculated quantities.
    if return_full:
        if verbosity >= 2 : print "Returning Expanded Output"
        return L,pi, K, rho, q_alpha, epsilon
    else:
        return L,pi


def get_bandwidth_fxn(data,period=None,nneighb=None,epses=2.**np.arange(-40,41),beta='-1/d',d=None):
    """
    Constructs a bandwidth function for a given dataset.  Performs a kernel density estimate q_\epsilon, and sets the bandwidth to q_epsilon^beta. 
    
    Parameters
    ----------
    data : 2D array-like
        Two-dimensional dataset used to create the diffusion map.
    period : 1D array-like or float, optional
        Period of the coordinate, e.g. 360 for an angle in degrees. If None, all coordinates are taken to be aperiodic.  If scalar, assumed to be period of each coordinate. If 1D array-like with each value a scalar or None, each coordinate has periodicity of that size.
    nneighb : int or None, optional
        Number of neighbors to include in constructing the diffusion map.  Default is None, which corresponds to using all neighbors.
    beta : float or string, optional
        Parameter for constructing the bandwidth function for the Diffusion map.  If rho is None, it will be set to q_\epsilon^beta, where q_\epsilon is an estimate of the density.  If rho is provided, this parameter is unused.  As with alpha, this will interpret strings that are evaluatable expressions.  Default is 0.0
    d : int or None, optional
        Dimension of the system. If None, dimension is estimated using the kde.

    Returns
    -------
    rho : 1d array
        The estimated bandwidth function.
    """
    N = len(data)
    if ((beta == 0) or (beta == '0')):
        return np.ones(N),None  # Handle uniform bandwidth case.
    else:
        # Use q^beta as bandwidth, where q is an estimate of the density.
        print epses, 'epses', d, 'd', beta, 'beta'
        q,d_est,eps_opt = kde.kde(data,epses=epses,period=period,nneighb=nneighb,d=d)
        if d is None:
            d = d_est
        
        # If beta parameter is an expression, evaluate it and convert to float
        beta = _eval_param(beta,d)
        return q**beta,d


def _eval_param(param,d):
    """
    Evaluates the alpha or beta parameters.  For instance, if the user passes "1/d", this must be converted to a float.

    Parameters
    ----------
    param : float or string
        The parameter to be evaluated.  Either a float or an evaluatable string, where "d" stands in for the system dimensionality.
    d : int
        Dimensionality of the system.

    Returns
    -------
    eval_param : float
        The value of the parameter, evaluated.
    """
    nsp = _NumericStringParser()
    if type(param) is str:
        if 'd' in param:
            if d is None:
                raise ValueError('Dimensionality needed in evaluating %s, but no dimensionality information found or estimated.'%param)
            param = param.replace('d',str(d))
        return nsp.eval(param)
    else:
        return float(param)

