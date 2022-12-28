import numpy as np 







def multi_scale_aperture_robust_optical_flow(   x:np.ndarray, 
                                                y:np.ndarray, 
                                                t:np.ndarray, 
                                                neighborhood:tuple=(5, 5)   ):
    """
    @ parameters:
    -------------
        x            : 
        y            : 
        t            :
        neighborhood :
    
    @ return:
    ---------
        a :
        b :
        c :
        

    """
    # 1. COMPUTE LOCAL FLOW (EDL):

    for i, (x_, y_, t_) in enumerate(zip(x, y, t)):
    pass