import numpy as np 
import math
import tqdm
from utils import plane_fitting






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

    for event, (x_, y_, _) in tqdm.tqdm(enumerate(zip(x, y, ts))):
        # 1. COMPUTE LOCAL FLOW (EDL):
        """
        Apply the plane fitting [8] to estimate the plane parameters 
        [a, b, c] within a neighborhood (5x5) of (x, y, t)
        """
        (a, b, _), neighborhood = plane_fitting(x, y, ts, event)
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
        S = np.arange(left_id, right_id, 1)
        
        if not np.array_equal(Un, np.zeros(2).T):
            for k in range(0, 100, 10):
                U_mean = np.array([U_hat[k].mean(), theta[k].mean()]).T
            sigma_max = np.argmax(U_hat[k].mean())
        
        # 3. UPDATE FLOW:
        """
        Flow (x,y) = mean j∈σmax(Uˆj cos θj, 
                                 Uˆj sin θj)
        """

        





