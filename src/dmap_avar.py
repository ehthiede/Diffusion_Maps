"""
Estimates the asymptotic variance of the EMUS estimated free energy from a weighted data set.

"""

import numpy as np
import usutils as uu
import emus
import linalg
from diffusion_map import diffusion_map
import matplotlib.pyplot as plt

def dmap_avar(cntrs,fks,x,f1,kT,eps,f2=None,wt=None,frac_samples=None,taus=None,period=None,density_cutoff=1E-4,npnts_dmap=5E2,log_avg=True,eps_max=None,tau_max=1.E10):
    """
    Estimates the asymptotic variance an observable as calculated using an 
    umbrella sampling calculation with harmonic windows from an existing data 
    set.

    Parameters
    ----------
        cntrs : ndarray
            Center of each harmonic window.  Shape is No. of windows by number
            of dimensions in cv space.
        fks : ndarray
            Force Constants of each harmonic window.  Shape is like cntrs.
        x : ndarray
            2D array of data points.  No. data points by no. dimensions.
        f1 : ndarray
            Value of numerator function at each point.  1D, Same length as x.
        f2 : ndarray, optional
            Value of denominator function.  Same shape as f1.  If None, taken
            to be 1.
        wt : ndarray, optional
            1D array of weights.  Same shape as f1.  If None, assumed to be 
            one for each point (points sampled from unbiased density)
        log_avg : bool, optional
            If True (default), calculates asymptotic variance of ln(f1/f2).
            If False, calculates a.v. for f1/f2 instead.


    FINISH!

    Returns
    -------
        avar : float
            Estimate of the asymptotic variance
        imp : ndarray
            Array of window importances.
    """
    #### Calculate properties of provided data
    L = len(cntrs) # Number of windows
    N = len(x)
    ## Calculate default values
    if wt is None:
        wt = np.ones(N)
    wt /= np.sum(wt)
    if f2 is None:
        f2 = np.ones(N)
    if frac_samples is None:
        frac_samples = np.ones(L)/L
    compute_taus = False
    if taus is None:
        compute_taus = True

    psis = uu.calc_harmonic_psis(x,cntrs,fks,kT,period) # \psi_i for all points
    denom = 1./np.sum(psis,axis=1) # (sum_i \psi_i(x))^-1
#    for i in xrange(L):
#        plt.plot(x,psis[:,i]*wt)
#    plt.show()
    z = np.dot(wt,psis) # z_i for each window
    F = np.dot(np.transpose(psis),psis*np.transpose([wt*denom]))*np.transpose([1./z])
    f1savg =  np.dot(f1*wt*denom,psis/z)
    f2savg =  np.dot(f2*wt*denom,psis/z)

    #### Estimate partial of observable with respect to parameters.
    gInv = linalg.groupInverse(np.eye(L)-F)
    if log_avg:
        dBdF = np.dot(gInv,f1savg)/np.dot(z,f1savg)
        dBdF -= np.dot(gInv,f2savg)/np.dot(z,f2savg)
        dBdF = kT*np.outer(z,dBdF)
        dBdf1 = kT*z/np.dot(z,f1savg)
        dBdf2 = -kT*z/np.dot(z,f2savg)
    else:
        raise NotImplementedError('CODE THIS UP, ERIK!')

    imp = []
    for i in xrange(L):
        ## Calculate variance for IID.
        errtraj = np.dot((psis*np.transpose([denom])-F[i,:]),dBdF[i,:])
#        plt.plot(x,errtraj,label=str(1))
        errtraj += dBdf1[i]*(f1*denom - f1savg[i])
#        plt.plot(x,errtraj,label=str(2))
        errtraj += dBdf2[i]*(f2*denom - f2savg[i])
#        plt.plot(x,errtraj,label=str(i))
#        plt.legend()
#        plt.show()
#        raise Exception
        errtrajsq = errtraj*errtraj
        imp_i = np.sum(psis[:,i]*wt*errtrajsq/z[i])
        ## Calculate Autocorrelation Time
        if compute_taus:
            # Extract relevant points
            subpnts = np.where(wt*psis[:,i]/z[i] > density_cutoff)[0]
            if len(subpnts) > npnts_dmap: # Trim no. points for DMAP scaling
                indices = np.linspace(0,len(subpnts)-1,npnts_dmap).astype('int')
                subpnts = subpnts[indices]
            sub_wt = wt[subpnts]*psis[:,i][subpnts]
            sub_errtraj = errtraj[subpnts]
            sub_errtraj -= np.dot(sub_wt,sub_errtraj)/np.sum(sub_wt)
            sub_x = x[subpnts]
            # Get DMAP tau
#            print np.max(sub_errtraj), np.min(sub_errtraj), np.average(sub_errtraj), np.std(sub_errtraj), 'suberrtraj'
            tau = dmap_tau(sub_x,sub_errtraj,sub_wt,eps,eps_max,tau_max,period=period)
#            fprime = x[subpnts]
#            fprime -= np.dot(sub_wt,fprime)/np.sum(sub_wt)
#            print sub_x, fprime, sub_wt,eps, len(sub_x), len(fprime)
#            np.save('sub_x',sub_x)
#            np.save('fprime',fprime)
#            np.save('sub_wt',sub_wt)
#            fetst = -np.log(sub_wt)
#            fetst-= np.min(fetst)
#            plt.plot(sub_x,fetst)
#            plt.plot(sub_x,(sub_x-cntrs[i])**2*fks[i]/2)
#            plt.show()
#            tau2 = dmap_tau(sub_x,fprime,sub_wt,eps,eps_max,tau_max,period=period)
#            print i, tau2, tau, 'taus'
            print i, tau, 'taus'
        else:
            tau = taus[i]
        imp.append(imp_i*2*tau*frac_samples[i])
    imp = np.array(imp)
    avar = np.sum(imp)
#    plt.show()
    return avar,imp, taus

def dmap_tau(x,ftraj,wt,eps,eps_max=None,tau_max=1.E10,period=None):
    if eps_max is None:
        eps_max = eps*1024
    P,pi = diffusion_map(x,eps,weights=wt,period=period)
#    print eps
#    print P
    try:
        Gen = (np.eye(len(ftraj))-P)/eps
#        Gen = np.eye(len(ftraj))-P
        Ginv = linalg.groupInverse(Gen)
        tau = np.dot(ftraj*wt,np.dot(Ginv,ftraj))/np.dot(ftraj*wt,ftraj)
    except IOError:
        print 'moose'
#        if tau > tau_max:
#            raise ValueError
#    except ValueError:
#        if 2*eps >= eps_max:
#            raise ValueError('Iteration Failed: epsilon too large')
#        else:
#            tau = dmap_tau(x,ftraj,wt,2.*eps,eps_max,tau_max)
    return 2*tau

    
    

    
