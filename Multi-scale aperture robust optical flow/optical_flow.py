import numpy as np 
import math
import tqdm
from utils import plane_fitting






def multi_scale_aperture_robust_optical_flow(x:np.ndarray, 
                                             y:np.ndarray, 
                                             t:np.ndarray,
                                             N:int=5):
    """
    @ parameters:
    -------------
        x : 
        y : 
        t :
        N :
    
    @ return:
    ---------
        a :
        b :
        c :
    """

    assert x.shape == y.shape  == t.shape

    for (x_, y_, t_) in tqdm.tqdm(zip(x, y, t)):
        # 1. COMPUTE LOCAL FLOW (EDL):
        """
        Apply the plane fitting [8] to estimate the plane parameters 
        [a, b, c] within a neighborhood (5x5) of (x, y, t)
        """
        (a, b, _), neighborhood = plane_fitting(x_, y_, t_)
        U_hat = np.norm(a - b)
        inliers_count = 0
        z_hat = np.sqrt(a**2 + b**2)

        for neighbor in neighborhood:
            t_hat = (a*x[neighbor] - x_) + (b*y[neighbor] - y_)
            if np.abs(t[neighbor] - t_hat) < z_hat/2:
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

        if np.array_equal(Un, np.zeros(2).T):
            for k, sigma in enumerate(S):
                U_mean = np.array([U_hat[k].mean(), theta[k].mean()]).T
            sigma_max = np.argmax(U_hat[k].mean())
        
        # 3. UPDATE FLOW:
        """
        Flow (x,y) = mean j∈σmax(Uˆj cos θj, 
                                 Uˆj sin θj)
        """

        





