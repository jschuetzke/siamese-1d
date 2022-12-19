import numpy as np
from numpy.random import default_rng

def simulate_noise(input_scan, seed=None):
    # input scan either [datapoints,] or [1,datapoints]
    dim = len(input_scan.shape)
    if dim == 1: # [datapoints,]
        scan = input_scan.copy() # copy to avoid pointer confusion
    elif dim == 2: # [1,datapoints]
        scan = input_scan[0].copy()
    elif dim == 3: # [1,datapoints,1] but should not be the case
        scan = input_scan[0,:,0].copy()
    else:
        raise ValueError('Dimensions unknown, check input scan!')
    
    rng = default_rng(seed)

    # min of base scans ~ 0
    log_max = np.max(np.log10(scan+10))
    
    scaling_max_diff = log_max - 1.
    scaling_min_diff = .9 
    
    # new diff between mix and max between 0.9 and 2.1 on log10 scale
    new_diff = rng.uniform(scaling_min_diff, scaling_max_diff)
    new_min = min(log_max - new_diff, 3)
    scan = np.power(10, (new_min)) + scan
    
    pure_noise = 1/3 * np.clip(rng.normal(0, 1, scan.shape), -3, 3)
    
    noise_scale = rng.uniform(0.05, 0.10) * new_min
    noise = pure_noise * (np.power(10,(new_min+noise_scale)) - np.power(10,(new_min))) # * noise_level
    scan += noise
    
    if dim == 2:
        return scan[np.newaxis, :]
    if dim == 3:
        return scan[np.newaxis, :, np.newaxis]
    else:
        return scan