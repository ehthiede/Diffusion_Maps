
# -*- coding: utf-8 -*-
"""
@author: erikthiede
"""
import numpy as np

def ud_diffmap(positions, velocities, potentials, eps, weights=None,gamma=1,kT=1,alpha=0.5,period=None,uniform=False):
    # Initialize variables and defaults
    if len(np.shape(positions)) == 1:
        ndim = 1
    elif len(np.shape(positions)) == 2:
        ndim = len(positions[0]) 
    else:
        raise ValueError('Too many dimensions in Input')
    npnts = len(positions)
    
    if np.shape(positions) != np.shape(velocities):
        print np.shape(positions), np.shape(velocities)
        raise ValueError('Position and Velocity arrays are different sizes')
    if npnts != len(potentials):
        print npnts, len(potentials), np.shape(potentials)
        raise ValueError('Different number of data points and potential energies.')
    if period is not None: # Periodicity provided.
        if not hasattr(period,'__getitem__'): # Check if period is scalar
            period = [period]*ndim
    else:
        period = [None]*ndim

    if len(np.shape(positions)) == 1: # If data is 1D, make it 2D so indices work
        positions = np.array([positions])
        positions = np.transpose(positions)
        velocities = np.array([velocities])
        velocities = np.transpose(velocities)
    if weights is None: # datapoints are unweighted.
        weights = np.ones(npnts)

    kernel = np.zeros((npnts,npnts)) # argument in the kernel.
    sigmasq = 2.*gamma*kT
    for i in xrange(npnts):
        for j in xrange(npnts):
            Ux = potentials[i]
            Uy = potentials[j]
            dxy = 0
            for dim in xrange(ndim):
                p = period[dim]
                x = positions[i,dim] 
                vx = velocities[i,dim]
                y = positions[j,dim] 
                vy = velocities[j,dim]
                dist_x = x-y
                if p is not None:
                    dist_x -= p*np.rint(dist_x/p)
                dxy += 12.*dist_x*(dist_x/(eps*eps*eps)+(vx+vy)/(eps*eps))
                dxy += (20.*(vx*vy+vx*vx+vy*vy)+6*gamma*gamma*dist_x*dist_x)/(5*eps)
                dxy += gamma*(vx+vy)*(vy-vx+gamma*dist_x/5.)
            dxy -= 4.*(Ux-Uy)/eps
            dxy /= sigmasq
            kernel[i,j] = np.exp(-dxy/2.)* weights[j]
            if uniform:
#                print 'moose'
                kernel[i,j] *= np.exp((-alpha*Uy/kT))
#                kernel[i,j] *= np.exp((-0.5*np.dot(velocities[j],velocities[j]))/kT)
    d = np.sum(kernel,axis=1,keepdims=True)
    kernel /= d
    return kernel
