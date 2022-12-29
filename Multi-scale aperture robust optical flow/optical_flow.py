import numpy as np 
import math
import tqdm
from utils import local_planes_fitting

def multi_scale_aperture_robust_optical_flow(x:np.ndarray, 
                                             y:np.ndarray, 
                                             ts:np.ndarray,
                                             N:int=3, 
                                             tpast:int=500):
    """
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
        N :
        tpast :
    
    @ return:
    ---------
        a :
        b :
        c :
    """

    assert x.shape == y.shape  == ts.shape
    corrected_flow = np.zeros((len(x), 2))
    flow_local = np.zeros((len(x), 2))
    ts_sec = ts*10**(-6)
    
    for event, (x_, y_, _) in tqdm.tqdm(enumerate(zip(x, y, ts))):
        # 1. COMPUTE LOCAL FLOW (EDL):
        """
        Apply the plane fitting [8] to estimate the plane parameters 
        [a, b, c] within a neighborhood (5x5) of (x, y, t)
        """
        P, neighborhood = local_planes_fitting(x, y, ts_sec, event)
        if P is None:
            continue
        (a, b, _) = P
        U_hat = np.linalg.norm(a - b)
        inliers_count = 0
        z_hat = np.sqrt(a**2 + b**2)

        for neighbor in neighborhood:
            t_hat = (a*(x[neighbor] - x_)) + (b*(y[neighbor] - y_))
            if np.abs((ts_sec[neighbor] - ts_sec[event]) - t_hat) < z_hat/2:
                inliers_count += 1
        
        if inliers_count >= (0.5 * N**2):
            theta = np.arctan2(a, b)
            Un = np.array([U_hat, theta]).T
        else:
            Un = np.zeros(2).T
        
        # 2. MULTI-SPATIAL SCALE MAX-POOLING:
        """
        Define S = {σk}, the set of neighborhoods, centered
        on (x, y, t), σk with increasing radius and δ t(σk) ≤ tpast
        (tpast is temporal cut-off delta)
        """
        right_id = np.searchsorted(ts, ts[event] + tpast, side='right')
        left_id = np.searchsorted(ts, ts[event] - tpast, side='left')
        time_window = np.arange(left_id, right_id, 1)
        
        U_means = []
        tetas_means = []
        flow_local[event] = Un
        
        S = range(10, 100, 10)
        
        if not np.array_equal(Un, np.zeros(2).T):
            for sigma_k in S:
                indices = time_window[np.where((x[event] - sigma_k <= x[time_window]) & (x[time_window] <= x[event] + sigma_k))[0]]
                indices = indices[np.where((y[event] - sigma_k <= y[indices]) & (y[indices] <= y[event] + sigma_k))[0]]
                
                sum_un = 0
                sum_teta = 0
                for i in indices:
                    sum_un += flow_local[i,0]
                    sum_teta += flow_local[i,1]

                if len(indices) != 0:
                    U_means.append(sum_un/len(indices))
                    tetas_means.append(sum_teta/len(indices))
                else:
                    U_means.append(0)
                    tetas_means.append(0)
                    
                sig_max = np.argmax(U_means)
                
            # 3. UPDATE FLOW:
            bestU, bestTeta = U_means[sig_max], tetas_means[sig_max]
            corrected_flow[event,:] = np.array([bestU, bestTeta])
            
            
    return flow_local, corrected_flow