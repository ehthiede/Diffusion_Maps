import numpy as np
import scipy.linalg as spl

def tlist2flat(trajs):
    """Flattens a list of two dimensional trajectories into a single two dimensional datastructure, and returns it along with a list of tuples giving the locations of each trajectory.

    Parameters
    ----------
    trajs : list of array-likes
        List where each element n is a array-like object of shape N_n x d, where N_n is the number of data points in that trajectory and d is the number of coordinates for each datapoint.

    Returns
    -------
    traj2D : 2D numpy array
        Numpy array containing the flattened trajectory information.
    traj_edges : 1D numpy array
        Numpy array where each element is the start of each trajectory: the n'th trajectory runs from traj_edges[n] to traj_edges[n+1]

    """
    # Get dimensions of 2D traj object
    d = len(trajs[0][0])
    # Populate the large trajectory.
    traj_2d = []
    traj_edges = [0]
    len_traj_2d = 0
    for i,traj in enumerate(trajs):
        # Check that trajectory is of right format.
        if len(np.shape(traj)) != 2:
            raise ValueError('Trajectory %d is not two dimensional!'%i)
        d2  = np.shape(traj)[1]
        if d2 != d:
            raise ValueError('Trajectories are of incompatible dimension.  The first trajectory has dimension %d and trajectory %d has dimension %d'%(d,i,d2))
        traj_2d += list(traj)
        len_traj_2d += len(traj)
        traj_edges.append(len_traj_2d)
    return np.array(traj_2d), traj_edges


def flat2tlist(traj_2d,traj_edges):
    """Takes a flattened trajectory with stop and start points and reformats it into a list of separate trajectories.

    Parameters
    ----------
    traj2D : 2D numpy array
        Numpy array containing the flattened trajectory information.
    traj_edges : 1D numpy array
        Numpy array where each element is the start of each trajectory: the n'th trajectory runs from traj_edges[n] to traj_edges[n+1]

    Returns
    -------
    trajs : list of array-likes
        List where each element n is a array-like object of shape N_n x d, where N_n is the number of data points in that trajectory and d is the number of coordinates for each datapoint.

    """
    trajs = []
    ntraj = len(traj_edges)-1
    for i in xrange(ntraj):
        start = traj_edges[i] ; stop = traj_edges[i+1]
        trajs.append(traj_2d[start:stop])
    return trajs

def start_stop_indices(traj_edges,delay):
    """
    Returns the indices of all the initial and final points of the trajectory over the lag time.
    """
    ntraj = len(traj_edges)-1
    t_0_indices = []
    t_lag_indices = []
    for i in xrange(ntraj):
        t_start = traj_edges[i]
        t_stop = traj_edges[i+1]
        if t_stop - t_start > delay:
            t_0_indices += range(t_start,t_stop-delay)
            t_lag_indices += range(t_start+delay,t_stop)
    return t_0_indices, t_lag_indices

def clean_basis(basis,traj_edges,delay=1,orthogonalize=False,make_positive=False):

    # Calculate orthogonal coefficients
    if orthogonalize:
        basis_t_0 = basis[t_0_indices]
        Q,R = spl.qr(basis_t_0)
        R_sub = R[:k,:k]
        basis = np.dot(basis,R_sub)

    # Split into positive, negative components.
    if make_positive:
        print basis
        print "-------------"
        basis_shape = np.shape(basis)
        pos_basis = basis*(basis>0)
        neg_basis = basis*(basis<=0)
        print pos_basis, neg_basis
        new_basis = np.zeros((basis_shape[0],basis_shape[1]*2))
        new_basis[:,::2] = pos_basis
        new_basis[:,1::2] = -1*neg_basis
        basis = new_basis[:,np.any(new_basis,axis=0)]
        print basis

    # Normalize the eigenvectors
    N,k = np.shape(basis)
    t_0_indices, t_lag_indices = start_stop_indices(traj_edges,delay)
    evec_norm = np.linalg.norm(basis,axis=0)
    basis *= np.sqrt(N)/evec_norm
    return basis

def delay_embed(data,traj_edges,nembed=1,difference=False):
    N,d = np.shape(data)
    new_traj_list = []
    ntraj = len(traj_edges)-1
    for i in xrange(ntraj):
        start = traj_edges[i] ; stop = traj_edges[i+1]
        traj = data[start:stop]
        N_i = len(traj)
        if N_i > nembed+1: # Check if trajectory is long enough to embed
            new_traj = np.empty((len(traj)-nembed,(nembed+1)*d))
            for n in xrange(nembed+1):
                dtraj = traj[nembed-n:N_i-n]
                new_traj[:,n*d:(n+1)*d] = dtraj
            if difference == True:
                traj_diff = -1*np.diff(new_traj,axis=1) 
                new_traj[:,d:] = traj_diff
            new_traj_list.append(new_traj)
    new_t2d,new_edges = tlist2flat(new_traj_list)
    return new_t2d, new_edges
