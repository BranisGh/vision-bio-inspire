# vision-bio-inspire

In this folder you will find an implementation of the 
multi_scale_aperture_robust_optical_flow in python. 

- main.py : To execute the algorithm you will need to open this file and set the variable 'compute' :
                    if True the program will caluclte the local and the corrected flow and save it to a npy file
                    if False the program will visualise the flow like the video attached in this folder

- optical_flow.py : This function is called if 'compute' is set to True and calculate the flow

- utils.py        : This file contains two functions. 
                    - local_planes_fitting : Apply the plane fitting to estimate the plane
                                             parameters [a, b, c] within a neighborhood
                    - visu_flow : Visulize the flow