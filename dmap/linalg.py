# -*- coding: utf-8 -*-
"""
Collection of linear algebra routines used in the EMUS algorithm and
associated error analysis.
"""
from scipy.linalg import qr
from scipy.linalg import inv
from scipy.linalg import solve
import numpy as np


def stationary_distrib(F,residtol = 1.E-10,max_iter=100):
    """
    Calculates the eigenvector of the matrix F with eigenvalue 1 (if it exists).
    
    Parameters
    ----------
    F : ndarray
        A matrix known to have a single left eigenvector with 
        eigenvalue 1.
    
    residtol : float or scalar
        To improve the accuracy of the computation, the algorithm will
        "polish" the final result using several iterations of the power
        method, z^T F = z^T.  Residtol gives the tolerance for the 
        associated relative residual to determine convergence.

    maxiter : int
        Maximum number of iterations to use the power method to reduce
        the residual.  In practice, should never be reached.
    
    Returns
    -------
    z : ndarray
        The eigenvector of the matrix F with eigenvalue 1.  For a Markov
        chain stationary distribution, this is the stationary distribution.
        Normalization is chosen s.t. entries sum to one.
    
    """

    L = len(F) # Number of States
    M = np.eye(L)-F
    q,r=qr(M)
    z=q[:,-1] # Stationary dist. is last column of QR fact
    z/=np.sum(z) # Normalize Trajectory
    # Polish solution using power method.
    for itr in xrange(max_iter):
        znew = np.dot(z,F)
        maxresid = np.max(np.abs(znew - z)/z) # Convergence Criterion 
        if maxresid < residtol:
            break
        else:
            z = znew

    return z/np.sum(z) # Return normalized (by convention)
   
def _sort_esystem(evals,evecs):
    """
    Sorts an eigenvector system in order of descending eigenvalue.

    Parameters
    ----------
    evals : 1D array-like
        1D structure containing the eigenvalues.  Must be sortable.
    evecs : 2D array-like
        Data structure containing the eigenvectors.  Column i the eigenvector corresponding to evals[i].

    Returns
    -------
    evals : 1D array-like
        Sorted eigenvalues.
    evals : 1D array-like
        Sorted eigenvetors.
    """
    idx = evals.argsort()[::-1]
    evals = evals[idx]
    evecs = evecs[:,idx]
    return evals, evecs


def _submatrix(F,i):
    """
    Calculates the submatrix of F with the i'th row and column removed.
    
    Parameters
    ----------
    F : ndarray
        A matrix with at least i rows and columns
    i : int
        The row or column to delete
        
    Returns
    -------
    submatrix: ndarray
        The ensuing submatrix with the i'th row and column deleted.
    """
    submat = np.copy(F)
    submat = np.delete(submat,i,axis=1)
    submat = np.delete(submat,i,axis=0)
    return submat
 
def groupInverse(M):
    """
    Computes the group inverse of stochastic matrix using the algorithm
    given by Golub and Meyer in:
    G. H. Golub and C. D. Meyer, Jr, SIAM J. Alg. Disc. Meth. 7, 273-
    281 (1986)

    Parameters
    ----------
        M : ndarray
            A square matrix with index 1.
    
    Returns
    -------
        grpInvM : ndarray
            The group inverse of M.
    """
    L=np.shape(M)[1]
    q,r=qr(M)
    piDist=q[:,L-1]
    piDist=(1/np.sum(piDist))*piDist
    specProjector=np.identity(L)-np.outer(np.ones(L),piDist)
    u=r[0:(L-1),0:(L-1)]#remember 0:(L-1) actually means 0 to L-2!
    uInv= inv(u)#REPLACE W. lapack, invert triangular matrix ROUTINE
    uInv = np.real(uInv)
    grpInvM=np.zeros((L,L))
    grpInvM[0:(L-1),0:(L-1)]=uInv
    grpInvM=np.dot(specProjector,np.dot(grpInvM,np.dot(q.transpose(),specProjector)))
    return grpInvM
