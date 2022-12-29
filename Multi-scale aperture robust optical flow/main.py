import numpy as np
from scipy.io import loadmat
import os
from optical_flow import multi_scale_aperture_robust_optical_flow
from utils import visu_flow

# Retrieving the current path
current_path = os.path.dirname(os.path.abspath(__file__))
# Retrieving the data file path
name_data_file = 'multipattern1.mat'
path_file_data = os.path.join(current_path,'data', name_data_file)
# Loading the .mat file
data = loadmat(path_file_data)
N_event_to_load = 200000
# Access to the data in the .mat file
ts = data['ts'].reshape(-1)#[:N_event_to_load]
x  = data['x'] .reshape(-1)#[:N_event_to_load]
y  = data['y'] .reshape(-1)#[:N_event_to_load]

# Choice of visualisation or flow calculation
compute = False
nb_images_to_show = 100

if compute:
    # estimate the local and the corrected flow
    flow_local, corrected_flow = multi_scale_aperture_robust_optical_flow(x, y, ts)
    # Save data into data folder
    np.save(os.path.join(current_path,'data', 'flow_local_out.npy'    ), flow_local    )
    np.save(os.path.join(current_path,'data', 'corrected_flow_out.npy'), corrected_flow)
    
else:
    # Load data from data folder
    flow_local     = np.load(os.path.join(current_path,'data', 'flow_local_out.npy'    ))
    corrected_flow = np.load(os.path.join(current_path,'data', 'corrected_flow_out.npy'))
    # Visualize the local and the corrected flow
    visu_flow(x, y, ts, flow_local, corrected_flow, time_delay= 100, step_size = int(N_event_to_load/nb_images_to_show))
print('ok')


