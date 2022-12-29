import numpy as np 
from scipy.linalg import lstsq

def local_planes_fitting(x   :np.ndarray,
                         y   :np.ndarray,
                         ts  :np.ndarray,   ):
    """
    find a plane that best approximates a point 
    cloud in space by minimising the squared error 
    between the points and the plane. This algorithm 
    can be used to model a surface locally, i.e. in a 
    restricted area around each point of the point cloud.
    
    @ parameters:
    -------------
        x, y          : variable contains an np.arrayay of data of type float, 
                        which represents the x-coordinate of each point 
                        in a two-dimensional scatter plot.
                        
        t             : This variable contains an np.arrayay of float data, 
                        which represents the time associated with each 
                        point in the scatterplot. This variable can be 
                        used to represent the time each point was 
                        captured using an event-driven camera.
    
    @ return:
    ---------
        Plane: the parameters (a, b, c) of the plane 
        y = ax + by + t + c for each point of the scatter plot.
        
        neighborhood : spatiotemporal neighborhood  (5x5) of (x, y, t).
        
    """
    
        # 1. Define a spatiotemporal neighborhood, centered on 'e' of  #down left
    # spatial dimensions L×L and duration [t + dt, t - dt].
    index_right = np.searchsorted(ts, ts[e] + dt, side='right')
    index_left = np.searchsorted(ts, ts[e] - dt, side='left')
    
    S = np.arange(index_left, index_right, 1)

    # Search for neighbours in the S.
    neighborhood = S[np.where((x[e] - N <= x[S]) & 
                                        (x[S] <= x[e] + N))[0]]
    
    neighborhood = neighborhood[np.where((y[e] - N <= y[neighborhood]) &
                                         (y[neighborhood] <= y[e] + N))[0]]
    
        # 2. apply a least square minimization to estimate the plane
    #  fitting all events ei(pi,ti)
    # Solve the linear matrix equation
    Plane_0, _, _, _ = lstsq(x[neighborhood], y[neighborhood], ts[neighborhood])
    #Plane_0 = compute_lmsq(x[neighborhood], y[neighborhood], ts[neighborhood])
    
    if Plane_0 is None:
        return None
    
    # declaration of constants thresholds and 
    # set eps to some arbitrarly high value (∼ 10e6).
    th1 = 1e-5 
    th2 = 0.05
    eps = 10e6

    while eps > th1:
        A, B, _ = Plane_0
        # Reject the event ei if the event is too far from the plane
        indices_to_delete = np.argwhere( A*x[neighborhood] + 
                                         B*y[neighborhood] + 
                                         C - ts[neighborhood] > th2 )[:, 0]
        # Update neighborhood list
        new_neighborhood = np.delete(neighborhood, indices_to_delete)
        # if no event has been deleted, so Plane_0 is directly considered as the solution plane
        if len(new_neighborhood) == len(neighborhood):
            break
        # apply lmsq to estimate the new Plane with the non rejected events e˜i
        Plane = compute_lmsq(x[new_neighborhood], y[new_neighborhood], ts[new_neighborhood], show_plane)
        if Plane is not None:
            # calculate the error and overwriting the old plan
            eps = np.linalg.norm(np.array(Plane) - np.array(Plane_0))
            Plane_0 = Plane
        else:
            break
    return Plane_0, new_neighborhood




def correct_flow(x, y, t, e, Un_tab, tpast=500):

    index_right = np.searchsorted(ts, ts[e] + tpast, side='right')
    index_left = np.searchsorted(ts, ts[e] - tpast, side='left')
    
    S = np.arange(index_left, index_right, 1)

    U_means = []
    tetas_means = []
    scale_indices = []
    
    for r in range(10, 100, 10):
        indices = S[np.where((x[e] - r <= x[S]) & (x[S] <= x[e] + r))[0]]
        indices = indices[np.where((y[e] - r <= y[indices]) & (y[indices] <= y[e] + r))[0]]
        scale_indices.append(indices)
        

        sum_un = 0
        sum_teta = 0
        for i in indices:
            sum_un += Un_tab[i,0]
            sum_teta += Un_tab[i,1]

        if len(indices) != 0:
            U_means.append(sum_un/len(indices))
            tetas_means.append(sum_teta/len(indices))
        else:
            U_means.append(0)
            tetas_means.append(0)
        
    
    sig_max = np.argmax(U_means)
    return U_means[sig_max], tetas_means[sig_max], scale_indices[sig_max]