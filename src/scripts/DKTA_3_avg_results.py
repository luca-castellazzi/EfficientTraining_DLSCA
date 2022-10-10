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
        - target: Target of the attack (SBOX_IN or SBOX_OUT)
        - b: Byte to be retrieved (from 0 to 15)

    The result is a PNG file containing the average GE.
    """
    
    _, n_devs, target, b = sys.argv
    
    n_devs = int(n_devs) # Number of training devices (to access the right results folder)
    target = target.upper()
    b = int(b)
    
    RES_ROOT = f'{constants.RESULTS_PATH}/DKTA/{target}/byte{b}/{n_devs}d'
    AVG_GES_PATH = RES_ROOT + f'/avg_ges.npy'
    AVG_GES_PLOT_PATH = RES_ROOT + f'/avg_ges_plot.png'


    all_ges = []
    for filename in os.listdir(RES_ROOT): 
        # Consider all .NPY files except avg_ges, if already present
        if ('.npy' in filename) and (not 'avg' in filename):
            NPY_FILE = RES_ROOT + f'/{filename}'
            ges = np.load(NPY_FILE)
            all_ges.append(ges)
    all_ges = np.array(all_ges) # n_npy_files x n_keys x n_traces
    
    avg_ges = np.mean(all_ges, axis=0)
    np.save(AVG_GES_PATH, avg_ges)
    
    
    # Plot Avg GEs
    vis.plot_avg_ges(
        avg_ges[:, :10], 
        n_devs, 
        b,
        AVG_GES_PLOT_PATH 
    )
    

if __name__ == '__main__':
    main()
