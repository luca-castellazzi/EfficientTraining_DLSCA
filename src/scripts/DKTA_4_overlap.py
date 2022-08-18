import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import os

import sys
sys.path.insert(0, '../utils')
import constants
import visualization as vis

def main():

    """
    Overlaps the average GEs obtained from DKTA_2_attacks.py and DKTA_3_avg_results.py
    in order to show dissimilarities between training phases with different 
    number of devices.
    Settings parameters (provided in order via command line):
        - tuning_method: HP searching method (Random Search (rs) or Genetic Algorithm (ga))
        - target: Target of the attack (SBOX_IN or SBOX_OUT)
        
    The result is a PNG file containing all the specified average GEs, plotted
    with different colors.
    """

    _, tuning_method, target = sys.argv
    
    res_path = f'{constants.RESULTS_PATH}/DKTA/{target}'
        
    dirs = [f'{filename}' 
            for filename in os.listdir(res_path)
            if os.path.isdir(f'{res_path}/{filename}')]   

    dirs.sort(key=lambda x: int(x[0])) # Sort the directories w.r.t. the number of devices
    
    avg_ges = [np.load(f'{res_path}/{d}/avg_ge__{tuning_method}.npy')
               for d in dirs]
    
    output_path = f'{res_path}/avg_ges_comparison.png'
    
    vis.plot_overlap(avg_ges, output_path)
    
    
if __name__ == '__main__':
    main()