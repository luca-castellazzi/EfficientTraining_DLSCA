# Basics
import os
import numpy as np

# Custom
import sys
sys.path.insert(0, '../utils')
import helpers
import constants
import visualization as vis

TARGET = 'SBOX_OUT'


def main():

    """
    Averages the values of the GEs obtained from DKTA_2_attacks.py, in order to
    generalize the results.
    Settings parameters (provided in order via command line):
        - n_devs: Number of train devices
        - b: Byte to be retrieved (from 0 to 15)

    The result is a PNG file containing the average GE.
    """
    
    _, n_devs, b = sys.argv
    
    n_devs = int(n_devs) # Number of training devices (to access the right results folder)
    b = int(b)
    
    RES_ROOT = f'{constants.RESULTS_PATH}/DKTA/{TARGET}/byte{b}/{n_devs}d'
    AVG_GES_FILE = RES_ROOT + f'/avg_ges.csv'
    AVG_GES_FILE_NPY = RES_ROOT + f'/avg_ges.npy'
    AVG_GES_PLOT = RES_ROOT + f'/avg_ges_plot.svg'


    all_ges = []
    for filename in os.listdir(RES_ROOT): 
        # Consider all .NPY files except avg_ges, if already present
        if ('.npy' in filename) and (not 'avg' in filename):
            NPY_FILE = RES_ROOT + f'/{filename}'
            ges = np.load(NPY_FILE)
            all_ges.append(ges)
    all_ges = np.array(all_ges) # n_npy_files x n_keys x n_traces
    
    avg_ges = np.mean(all_ges, axis=0)

    # Save average GEs files 
    # In .CSV for future use
    csv_ges_data = np.vstack(
        (
            np.arange(avg_ges.shape[1])+1, # The values of the x-axis in the plot
            avg_ges # The values of the y-axis in the plot
        )
    ).T

    helpers.save_csv(
        data=csv_ges_data,
        columns=['NTraces']+[f'NKeys_{nk+1}' for nk in range(avg_ges.shape[0])],
        output_path=AVG_GES_FILE
    )
    # In .NPY for direct use in DKTA_4_overlap.py
    np.save(AVG_GES_FILE_NPY, avg_ges)
    
    # Plot Avg GEs
    labels = []
    for i, _ in enumerate(avg_ges):
        l = f'{i+1} key'
        if i != 0:
            l += 's' # Plural
        labels.append(l)

    vis.plot_ges(
        ges=avg_ges[:, :10],
        labels=labels,
        title=f'Byte: {b}  |  Train-Devices: {n_devs}',
        ylim_max=50,
        output_path=AVG_GES_PLOT 
    )
    

if __name__ == '__main__':
    main()
