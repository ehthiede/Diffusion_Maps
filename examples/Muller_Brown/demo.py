import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as spsl
from dmap.diffusion_map import diffusion_map 

# Load in the trajectory
traj = np.loadtxt('mb_traj.txt')

# For slow computers, you can speed the calculation a bit by subsampling the 
# trajectory.  To do so, uncomment the line below.
#traj = traj[::10]

# Construct a diffusion map.  We will use a variable-bandwidth diffusion map
# that converges to the Brownian generator, with 400 neighbors for each point.
L, pi = diffusion_map(traj,alpha=0.0,beta='-1/(d+2)',nneighb=400)

# We now calculate the top 10 eigenvectors and eigenvacues of the diffusion map
# using scipy's sparse linalg library.
k = 10
evals, evecs = spsl.eigs(L,k=k,which='LR')
print np.shape(evals), np.shape(evecs)
idx = evals.argsort()[::-1]
evals = np.real(evals[idx])
evecs = evecs[:,idx]

# We plot the eigenvalues.  The first eigenvalue is zero up to numerical error 
# because the matrix is a transition rate matrix.  Notice the large gap between
# the second and the third eigenvalues, indicating that mode 2 is the slowest 
# mode in the system.
print "Eigenvalues are : ",evals
plt.scatter(np.arange(k),evals)
plt.xlabel('Eigenvalue No.')
plt.ylabel('Eigenvalue')
plt.show()


# We plot the top four eigenvectors.
fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(6.5,5))
SC = ax1.scatter(traj[:,0],traj[:,1],c=evecs[:,0],s=4,vmin=-0.02,vmax=0.02)
plt.colorbar(SC,ax=ax1)
ax1.set_title(r'evec 1, $\lambda=$%.3e'%evals[0])
SC = ax2.scatter(traj[:,0],traj[:,1],c=evecs[:,1],s=4)
plt.colorbar(SC,ax=ax2)
ax2.set_title(r'evec 2, $\lambda=$%.3e'%evals[1])
SC = ax3.scatter(traj[:,0],traj[:,1],c=evecs[:,2],s=4)
plt.colorbar(SC,ax=ax3)
ax3.set_title(r'evec 3, $\lambda=$%.3e'%evals[2])
SC = ax4.scatter(traj[:,0],traj[:,1],c=evecs[:,3],s=4)
plt.colorbar(SC,ax=ax4)
ax4.set_title(r'evec 4, $\lambda=$%.3e'%evals[3])
plt.tight_layout()
plt.show()
