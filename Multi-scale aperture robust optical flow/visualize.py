import cv2 as cv
import numpy as np 
import matplotlib.pyplot as plt


def visu_flow(x,y,ts, EDL, ARMS, time_delay, step_size):

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

        ind_min = int (ind_max - ind_max/1.5)

        EDL_im = cv.cvtColor(EDL_im, cv.COLOR_HSV2BGR)
        ARMS_im= cv.cvtColor(ARMS_im, cv.COLOR_HSV2BGR)
        
        # # Plot the histogram
        # plt.subplot(111, projection='polar')
        # plt.hist(EDL[indices,1], bins=20, range=(0, 360))
        # plt.show(block=False)
        # plt.pause(time_delay/1000)
        
        cv.imshow("ARMS flow", ARMS_im)
        cv.imshow("EDL flow", EDL_im)
        cv.waitKey(time_delay)
        # Set up the polar plot
        
