import numpy as np
from scipy.io import loadmat
import os
from optical_flow import *

# Retrieving the current path
current_path = os.path.dirname(os.path.abspath(__file__))
name_data_file = 'multipattern1.mat'
path_file_data = os.path.join(full_path_data,'data', name_data_file)
# Loading the .mat file
data = scipy.io.loadmat(path_file_data)

# Access to the data in the .mat file
ts = data['ts']
x = data['x']
y = data['y']


flow_local, corrected_flow = multi_scale_aperture_robust_optical_flow(x, y, ts)


