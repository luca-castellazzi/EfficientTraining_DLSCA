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
BYTE = 5


def main():

    """
    Averages the values of the GEs obtained from DKTA_2_attacks.py, in order to
    generalize the results.
    Settings parameters (provided in order via command line):
        - n_devs: Number of train devices
        - b: Byte to be retrieved (from 0 to 15)

    The result is a .SVG file containing the average GE.
    """
    
    _, n_devs = sys.argv
    n_devs = int(n_devs)

    RES_ROOT = f'{constants.RESULTS_PATH}/DKTA/{TARGET}/byte{BYTE}/msk_{n_devs}d'
    AVG_GES_FILE = RES_ROOT + '/avg_ges.csv'
    AVG_GES_PLOT = RES_ROOT + '/avg_ges.svg'
    SINGLE_GES_FILES = [RES_ROOT + f'/ges_{"".join(train_devs)}vs{test_dev}.csv' 
                           for train_devs, test_dev in constants.PERMUTATIONS[n_devs]]
    SINGLE_GES_PLOTS = [RES_ROOT + f'/ges_{"".join(train_devs)}vs{test_dev}.svg' 
                           for train_devs, test_dev in constants.PERMUTATIONS[n_devs]]

    # Get avg GEs and Tot Traces
    ges = []
    k_labels = []
    for el in os.listdir(RES_ROOT):

        el_path = f'{RES_ROOT}/{el}'
        
        if os.path.isdir(el_path):
            k_labels.append(int(el.split('k')[0]))
        else:   
            if '.npy' in el: # If the element is a folder
                ge = np.load(el_path)
                ges.append(ge)
    
    k_labels.sort()
    ges = np.array(ges) # 3 dims 
    avg_ges = np.mean(ges, axis=0)

    # Save Avg GEs
    avg_ges_data = np.vstack(
        (
            np.arange(avg_ges.shape[1])+1, # The values of the x-axis in the plot
            avg_ges # The values of the y-axis in the plot
        )
    ).T
    helpers.save_csv(
        data=avg_ges_data,
        columns=['NTraces']+[f'NKeys_{k_l}' for k_l in k_labels],
        output_path=AVG_GES_FILE
    )
    # Plot GEs
    vis.plot_ges(
        ges=avg_ges, 
        labels=[f'{k_l} key' if k_l == 1 else f'{k_l} keys' for k_l in k_labels],
        title=f'DKTA - Avg GEs  |  Byte: {BYTE}  |  Train-Devices: {n_devs}',
        ylim_max=150,
        output_path=AVG_GES_PLOT 
    )

    # Save and Plot single GEs
    i = 0
    for (train_devs, test_dev), ges in zip(constants.PERMUTATIONS[n_devs], ges):
        ges_data = np.vstack(
            (
                np.arange(ges.shape[1])+1, # The values of the x-axis in the plot
                ges # The values of the y-axis in the plot
            )
        ).T
        helpers.save_csv(
            data=ges_data,
            columns=['NTraces']+[f'NKeys_{k_l}' for k_l in k_labels],
            output_path=SINGLE_GES_FILES[i]
        )
        # Plot GEs
        vis.plot_ges(
            ges=ges, 
            labels=[f'{k_l} key' if k_l == 1 else f'{k_l} keys' for k_l in k_labels],
            title=f'DKTA - {"".join(train_devs)}vs{test_dev}  |  Byte: {BYTE}',
            ylim_max=150,
            output_path=SINGLE_GES_PLOTS[i] 
        )
        i += 1


















    
    
    # RES_ROOT = f'{constants.RESULTS_PATH}/DKTA/{TARGET}/byte{BYTE}/msk_{n_devs}d'
    # AVG_GES_FILE = RES_ROOT + f'/avg_ges.csv'
    # AVG_GES_FILE_NPY = RES_ROOT + f'/avg_ges.npy'
    # AVG_GES_PLOT = RES_ROOT + f'/avg_ges_plot.svg'


    # all_ges = []
    # for filename in os.listdir(RES_ROOT): 
    #     # Consider all .NPY files except avg_ges, if already present
    #     if ('.npy' in filename) and (not 'avg' in filename):
    #         NPY_FILE = RES_ROOT + f'/{filename}'
    #         ges = np.load(NPY_FILE)
    #         all_ges.append(ges)
    # all_ges = np.array(all_ges) # n_npy_files x n_keys x n_traces
    
    # avg_ges = np.mean(all_ges, axis=0)

    # # Save average GEs files 
    # # In .CSV for future use
    # csv_ges_data = np.vstack(
    #     (
    #         np.arange(avg_ges.shape[1])+1, # The values of the x-axis in the plot
    #         avg_ges # The values of the y-axis in the plot
    #     )
    # ).T

    # helpers.save_csv(
    #     data=csv_ges_data,
    #     columns=['NTraces']+[f'NKeys_{nk+1}' for nk in range(avg_ges.shape[0])],
    #     output_path=AVG_GES_FILE
    # )
    # # In .NPY for direct use in DKTA_4_overlap.py
    # np.save(AVG_GES_FILE_NPY, avg_ges)
    
    # # Plot Avg GEs
    # labels = []
    # for i, _ in enumerate(avg_ges):
    #     l = f'{i+1} key'
    #     if i != 0:
    #         l += 's' # Plural
    #     labels.append(l)

    # vis.plot_ges(
    #     ges=avg_ges,
    #     labels=labels,
    #     title=f'DKTA  |  Byte: {BYTE}  |  Train-Devices: {n_devs}',
    #     ylim_max=150,
    #     output_path=AVG_GES_PLOT,
    #     grid=False
    # )
    

if __name__ == '__main__':
    main()
