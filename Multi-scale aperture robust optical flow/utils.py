import cv2 as cv
import numpy as np 
import matplotlib.pyplot as plt

def local_planes_fitting(x     :np.ndarray,
                         y     :np.ndarray,
                         ts    :np.ndarray,
                         event :int       ,
                         N     :int = 3   ,  
                         dt    :int = 1000  ):
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
    index_right = np.searchsorted(ts*10**(6), (ts*10**(6))[event] + dt, side='right')
    index_left = np.searchsorted(ts*10**(6), (ts*10**(6))[event] - dt, side='left')
    
    time_window = np.arange(index_left, index_right, 1)

    # Search for neighbours in the S.
    neighborhood = time_window[np.where((x[event] - N <= x[time_window]) & 
                                        (x[time_window] <= x[event] + N))[0]]
    
    neighborhood = neighborhood[np.where((y[event] - N <= y[neighborhood]) &
                                         (y[neighborhood] <= y[event] + N))[0]]
    
        # 2. apply a least square minimization to estimate the plane
    #  fitting all events ei(pi,ti)
    # Solve the linear matrix equation
    A = np.c_[x[neighborhood], y[neighborhood], np.ones(len(x[neighborhood]))]
    if len(x[neighborhood]) >= 4:
        Plane_0, _, _, _ = np.linalg.lstsq(A, ts[neighborhood])    # coefficients
    else :
        Plane_0 = None
    
    if Plane_0 is None:
        return Plane_0, neighborhood
    
    # declaration of constants thresholds and 
    # set eps to some arbitrarly high value (∼ 10e6).
    th1 = 1e-5 
    th2 = 0.05
    eps = 10e6

    while eps > th1:
        A, B, C = Plane_0
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
        A = np.c_[x[new_neighborhood], y[new_neighborhood], np.ones(len(x[new_neighborhood]))]
        if len(x[neighborhood]) >= 4:
            Plane, _, _, _ = np.linalg.lstsq(A, ts[new_neighborhood])    # coefficients
        else:
            Plane = None
        if Plane is not None:
            # calculate the error and overwriting the old plan
            eps = np.linalg.norm(np.array(Plane) - np.array(Plane_0))
            Plane_0 = Plane
        else:
            break
    return Plane_0, new_neighborhood


def visu_flow(x         :np.ndarray,
              y         :np.ndarray,
              ts        :np.ndarray,
              EDL       :np.ndarray,          
              ARMS      :np.ndarray, 
              time_delay:int       , 
              step_size :int       ):
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
                        
        ts            : This variable contains an np.arrayay of float data, 
                        which represents the time associated with each 
                        point in the scatterplot. This variable can be 
                        used to represent the time each point was 
                        captured using an event-driven camera.
        EDL           : Flux local
        ARMS          : 
        time_delay    :
        step_size     :
        
    @ return:
    ---------
        Plane: the parameters (a, b, c) of the plane 
        y = ax + by + t + c for each point of the scatter plot.
        
        neighborhood : spatiotemporal neighborhood  (5x5) of (x, y, t).
        
    """
    ind_min = 0
    ind_max = 0
    h = max(y) + 1
    w = max(x) + 1
    step = 0
    while (ind_max < len(x)):
        
        EDL_im = np.zeros((w,h,3), dtype = np.uint8)
        ARMS_im= np.zeros((w,h,3), dtype = np.uint8)
        
        ind_max += step_size
        indices = np.arange(ind_min, ind_max)

        H_EDL = np.uint16(EDL[indices,1] * 255/(2*np.pi))
        H_EDL = H_EDL[:, np.newaxis]
        H_ARMS = np.uint16(ARMS[indices,1] * 255/(2*np.pi))
        H_ARMS= H_ARMS[:, np.newaxis]
        
        fill = 255* np.ones_like(H_EDL)
        fill_edl = fill.copy()
        fill_edl[np.where(H_EDL == 0)] = 0

        fill_arms= fill.copy()
        fill_arms[np.where(H_ARMS == 0)] = 0

        EDL_im[x[indices], y[indices]] = np.concatenate((H_EDL,fill_edl,fill_edl), axis = 1)
        ARMS_im[x[indices], y[indices]] = np.concatenate((H_ARMS,fill_arms,fill_arms), axis = 1)

        ind_min = int (ind_max - ind_max/1.4)

        EDL_im = cv.cvtColor(EDL_im, cv.COLOR_HSV2BGR)
        ARMS_im= cv.cvtColor(ARMS_im, cv.COLOR_HSV2BGR)

        # nonzeros = np.array(np.where(EDL[indices,1] != 0))
        # nonzeros = nonzeros.flatten()
        # # Set up the polar plot
        # plt.subplot(111, projection='polar')
        # # Plot the histogram
        # plt.hist(EDL[nonzeros,1], bins=7, range=(0, np.pi))
        # plt.show(block=False)
        # plt.pause(time_delay/1000)
        
        cv.imshow("ARMS flow", ARMS_im)
        cv.imshow("EDL flow", EDL_im)
        cv.waitKey(time_delay)
        # Set up the polar plot
