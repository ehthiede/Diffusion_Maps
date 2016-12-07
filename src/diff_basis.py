import numpy as np
from diffusion_map import scaled_diffusion_map
from numpy.linalg import eig
import scipy.sparse as sps
from scipy.sparse.linalg import eigs, eigsh
from time import time
import linalg as LA

def _sort_esystem(evals,evecs):
    idx = evals.argsort()[::-1]
    evals = evals[idx]
    evecs = evecs[:,idx]
    return evals, evecs
    

def basis_tau(data,f,epsilon,weights=None,nevecs=10,D=1.0,alpha=None,beta=None,period=None):
#    P, d = diffusion_map(data,epsilon,weights=weights,alpha=alpha,D=D,period=period)
    L, K, rho, q = scaled_diffusion_map(data,epsilon,weights=weights,D=D,alpha=alpha,beta=beta,period=period)
    f = np.copy(f)
    N= len(f)
#    sttime = time()
#    evals, evecs = eigs(L,which='LR',k=nevecs)
#    idx = evals.argsort()[::-1]
#    evals = evals[idx]
#    evecs = evecs[:,idx]
#    evecs = evecs[:,:nevecs]
#    print 'evals'
#    print evals
#    print 'evals'
#    print time()-sttime
    Sinvdiag = 1./(rho*np.sqrt(q))
    Sinv = sps.dia_matrix((Sinvdiag,[0]),shape=(N,N))
    Pinv2 = sps.dia_matrix((1./rho**2,[0]),shape=(N,N))
    Lhat = Sinv * K * Sinv
    Lhat= (Lhat - Pinv2)
    Lhat = Lhat.multiply(1./epsilon)
    sttime = time()
    evals, evecs = eigsh(Lhat,which='LA',k=nevecs)
    evals, evecs = _sort_esystem(evals,evecs)
    print time()-sttime, 'largest albebraic time'
    sttime = time()
    evals2, evecs2 = eigsh(Lhat,which='SM',k=nevecs)
    evals2, evecs2 = _sort_esystem(evals2,evecs2)
    print time()-sttime, 'small mag time'
    sttime = time()
    mxval = max(np.abs(Lhat.max()),np.abs(Lhat.min()))
    P = sps.eye(N) + Lhat.multiply(.01/mxval)
    evals3, evecs3 = eigsh(P,which='LM',k=nevecs)
    evals3, evecs3 = _sort_esystem(evals3,evecs3)
    print time()-sttime, 'shift time'
    print np.max(np.abs(evecs-evecs2))
    print np.max(np.abs(evecs-evecs3))
#    idx = evals.argsort()[::-1]
#    evals = evals[idx]
#    evecs = evecs[:,idx]
#    evecs *= np.transpose([Sinvdiag])
#    evnorms = np.linalg.norm(evecs,axis=0)
#    evecs /= evnorms
    import matplotlib.pyplot as plt
    for i in xrange(7):
        plt.scatter(data, evecs[:,i],color='b')
        plt.scatter(data, evecs2[:,i],color='g')
        plt.scatter(data, evecs3[:,i],color='r')
        plt.title('evec %d'%i)
        plt.show()
#        evec = evecs[:,i]
#        print i
    return
#    P = LA.groupInverse(P)
#    for k in xrange(len(evecs[0])):
#        evecs[:,k] /= np.average(np.abs(evecs[:,k]))
#        evecs[:,k] *= 0.000001
#        print k
    du = np.diff(evecs,axis=0)
#    import matplotlib.pyplot as plt
#    for k in xrange(len(evecs[0])):
#        for j in xrange(len(evecs[0])):
#            fig, (ax1,ax2,ax3) = plt.subplots(3)
#            ax1.plot(evecs[:-1,k])
#            ax2.plot(du[:,j])
#            ax3.plot(evecs[:-1,k]*du[:,j])
#            ax3.set_xlabel('timepoint')
#            ax1.set_title('u %d, du %d'%(k,j))
#            plt.show()
#    np.savetxt('du.txt',du)
#    print np.shape(du), np.shape(evecs[:-1])
#    print np.min(evecs[:-1]), np.max(evecs[:-1])
    
    A = np.dot(np.transpose(evecs[:-1]),du)/(N-1)
    print np.shape(A), 'shape A'
    nevecs = len(evecs[0])
#    for row in A:   
#        print row, 'row'
    print evals[:3], 'evals'
    print [A[i,i]+1 for i in xrange(3)]
    beta = np.dot(f[:-1],evecs[:-1])
    Af = np.dot(A,beta)
    Afx = np.dot(evecs, Af)
#    print np.min(Afx), np.max(Afx)
#    print np.shape(Af),np.shape(Afx),'af shape'
#    import matplotlib.pyplot as plt
#    fig, (ax1,ax2,ax3) = plt.subplots(3)
#    ax1.scatter(data,np.dot(evecs,beta))
#    ax1.set_ylabel('f approx')
#    print np.shape(data), np.shape(Af)
#    ax2.scatter(data,Afx)
#    ax2.scatter(data,Af)
#    ax2.set_ylabel('Af approx (Galerkin)')
#    ax3.scatter(data,np.dot((P-np.eye(len(P)))/epsilon,f))
#    ax3.set_ylabel('Af approx (Standard)')
#    plt.show()
    print beta, 'beta 1'
    print np.dot(np.transpose(evecs[:-1]),f[:-1])
#    for row in A:
#        print row, 'row'
    Ainv = LA.groupInverse(A)
#    for row in Ainv:   
#        print row, 'row'
    gksoln = np.dot(LA.groupInverse(A),beta)
    top = np.dot(evecs,gksoln)
    fntop = np.dot(f,top)
    print np.shape(A), np.shape(gksoln), np.shape(top)
    print fntop, 'moose'
    print np.dot(f,f)
    return 0


