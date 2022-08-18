import pandas as pd
import numpy as np
import os

import sys
sys.path.insert(0, '../utils')
import constants
import visualization as vis


def main():

    """
    Averages the values of the GEs obtained from DKTA_2_attacks.py, in order to
    generalize the results.
    Settings parameters (provided in order via command line):
        - n_devs: Number of train devices
        - tuning_method: HP searching method (Random Search (rs) or Genetic Algorithm (ga))
        - target: Target of the attack (SBOX_IN or SBOX_OUT)
        
    The result is a PNG file containing the average GE.
    """
    
    _, n_devs, tuning_method, target = sys.argv
    
    n_devs = int(n_devs) # Number of training devices (to access the right results folder)
    target = target.upper()
    
    res_path = f'{constants.RESULTS_PATH}/DKTA/{target}/{n_devs}d'
    
    all_ges = []
    for filename in os.listdir(res_path):
        if f'{tuning_method}.npy' in filename:
            if not 'avg' in filename: # Avoid to consider avg_ge NPY file if already present
                npy_file = f'{res_path}/{filename}'
                ges = np.load(npy_file)
                all_ges.append(ges)
    
    n_train_configs = len(constants.KEYS) - 1 # There is exactly 1 train-config per key
                                              # Do not consider K0
    
    avg_ges = []
    for i in range(n_train_configs):
        ges_per_train_config = np.array([ges[i] for ges in all_ges])
        avg_ge = np.mean(ges_per_train_config, axis=0)
        avg_ges.append(avg_ge)
        
    avg_ges = np.array(avg_ges)
    np.save(f'{res_path}/avg_ge__{tuning_method}.npy', avg_ges)
    
    
    # Plot Avg GEs
    vis.plot_avg_ges(
        avg_ges, 
        n_devs, 
        f'{res_path}/avg_ge__{tuning_method}.png'
    )
    

if __name__ == '__main__':
    main()
