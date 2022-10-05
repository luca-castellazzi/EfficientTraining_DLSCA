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
        - to_compare: Comma-separated list of bytes whose results will be compared (overlapped)
        - n_devs: Number of devices used to generate the results to overlap
        - target: Target of the attack (SBOX_IN or SBOX_OUT)
        
    The result is a PNG file containing all the specified average GEs, plotted
    with different colors.
    """

    _, to_compare, n_devs, target = sys.argv
    to_compare = [int(tc) for tc in to_compare.split(',')]
    n_devs = int(n_devs)
    target = target.upper()

    RES_ROOT = f'{constants.RESULTS_PATH}/DKTA/{target}'
    OUTPUT_PATH = RES_ROOT + f'/comparison_{n_devs}d_{"-".join([f"b{tc}" for tc in to_compare])}'

    ges_files = [RES_ROOT + f'/byte{tc}/{n_devs}d/avg_ges.npy'
                 for tc in to_compare]

    ges = [np.load(gf) for gf in ges_files]
    
    vis.plot_overlap(
        ges, 
        to_compare, 
        f'{" vs ".join([f"Byte{tc}" for tc in to_compare])} | Train-Devices: {n_devs}',     
        OUTPUT_PATH
    )
    
    
if __name__ == '__main__':
    main()