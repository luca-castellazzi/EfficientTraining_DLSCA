# Basics
import numpy as np

# Custom
import sys
sys.path.insert(0, '../utils')
import helpers
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
    COMPARISON_FILE = RES_ROOT + f'/comparison_{n_devs}d_{"-".join([f"b{tc}" for tc in to_compare])}.csv'
    COMPARISON_PLOT = RES_ROOT + f'/comparison_{n_devs}d_{"-".join([f"b{tc}" for tc in to_compare])}.svg'

    MAX_TRACES = 10

    ges_files = [RES_ROOT + f'/byte{tc}/{n_devs}d/avg_ges.npy'
                 for tc in to_compare]

    ges = [np.load(gf)[:, :MAX_TRACES] for gf in ges_files]

    ges_data = np.vstack(ges)
    csv_ges_data = np.vstack(
        (
            np.arange(ges_data.shape[1])+1, # The values of the x-axis in the plot
            ges_data # The values of the y-axis in the plot
        )
    ).T
    
    n_ges_per_byte = int(ges_data.shape[0] / len(to_compare)) # ges_data is a (n_keys * n_bytes) x n_traces matrix
    helpers.save_csv(
        data=csv_ges_data,
        columns=['NTraces']+[f'Byte{b}_{i}'
                             for b in to_compare
                             for i in range(n_ges_per_byte)],
        output_path=COMPARISON_FILE
    )
    
    vis.plot_overlap(
        ges, 
        to_compare, 
        f'{" vs ".join([f"Byte{tc}" for tc in to_compare])} | Train-Devices: {n_devs}',     
        COMPARISON_PLOT
    )
    
    
if __name__ == '__main__':
    main()