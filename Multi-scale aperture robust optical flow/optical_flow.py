import numpy as np 
import math
import tqdm
from utils import local_planes_fitting






def multi_scale_aperture_robust_optical_flow(x:np.ndarray, 
                                             y:np.ndarray, 
                                             ts:np.ndarray,
                                             N:int=5, 
                                             tpast:int=0.5):
    """
    @ parameters:
    -------------
        x : 
        y : 
        t :
        N :
        tpast :
    
    @ return:
    ---------
        a :
        b :
        c :
    """

    assert x.shape == y.shape  == ts.shape
    corrected_flow = []
    flow_local = np.zeros((len(x), 2))
    
    for event, (x_, y_, _) in tqdm.tqdm(enumerate(zip(x, y, ts))):
        # 1. COMPUTE LOCAL FLOW (EDL):
        """
        Apply the plane fitting [8] to estimate the plane parameters 
        [a, b, c] within a neighborhood (5x5) of (x, y, t)
        """
        (a, b, _), neighborhood = local_planes_fitting(x, y, ts, event)
        U_hat = np.norm(a - b)
        inliers_count = 0
        z_hat = np.sqrt(a**2 + b**2)

        for neighbor in neighborhood:
            t_hat = (a*x[neighbor] - x_) + (b*y[neighbor] - y_)
            if np.abs(ts[neighbor] - t_hat) < z_hat/2:
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
        
        S = range(0, 100, 10)
        
        if not np.array_equal(Un, np.zeros(2).T):
            for sigma_k in S:
                indices = time_window[np.argwhere((x[e] - sigma_k <= x[time_window]) & (x[time_window] <= x[e] + sigma_k))]
                indices = indices[np.argwhere((y[e] - sigma_k <= y[indices]) & (y[indices] <= y[e] + sigma_k))]
                
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
            corrected_flow.append(np.array(bestU, bestTeta))
            
    return flow_local, corrected_flow

        





