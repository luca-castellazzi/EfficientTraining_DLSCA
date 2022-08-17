import pandas as pd
import numpy as np
import os

import sys
sys.path.insert(0, '../utils')
import constants
import visualization as vis


def main():

    """
    This script averages the values of the GEs obtained at the end of the DKTA process 
    in order to generalize the results.
    
    The number of train-devices used during the DKTA  and the used hyperparameter 
    tuning method must be provided via command line in this order.
    
    The average GE is plotted.
    """
    
    n_train_devs = int(sys.argv[1]) # Number of training devices (to access the right results folder)
    used_tuning_method = sys.argv[2]
    
    res_folder_path = f'{constants.RESULTS_PATH}/DKTA/{n_train_devs}d'
    
    all_ges = []
    for filename in os.listdir(res_folder_path):
        if f'{used_tuning_method}.npy' in filename:
            if not 'avg' in filename: # Avoid to consider avg_ge NPY file if already present
                npy_file = f'{res_folder_path}/{filename}'
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
    np.save(f'{res_folder_path}/avg_ge__{used_tuning_method}.npy', avg_ges)
    
    
    # Plot Avg GEs
    vis.plot_avg_ges(
        avg_ges, 
        n_train_devs, 
        f'{res_folder_path}/avg_ge__{used_tuning_method}.png'
    )
    

if __name__ == '__main__':
    main()
