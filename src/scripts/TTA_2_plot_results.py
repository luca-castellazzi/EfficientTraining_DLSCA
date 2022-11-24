# Basics
import os
import numpy as np

# Custom
import sys
sys.path.insert(0, '../utils')
import constants
import results
import helpers
import visualization as vis


BYTE = 5
THRESHOLD = 0.5


def main():

    """
    Plots GEs resulting from TTA_1_attacks.py.
    Settings parameters (provided in order via command line):
        - n_devs: Number of train devices

    The result is a .SVG file containing the average GE.
    """
    
    _, n_devs = sys.argv
    n_devs = int(n_devs)
    
    RES_ROOT = f'{constants.RESULTS_PATH}/TTA/{n_devs}d'
    TT_GES_FILE = RES_ROOT + '/ges.csv'
    TT_GES_PLOT = RES_ROOT + '/ges.svg'
    MIN_ATT_TR_FILE = RES_ROOT + '/min_att_tr.csv'
    MIN_ATT_TR_PLOT = RES_ROOT + '/min_att_tr.svg'


    tt_labels = []
    ges_files = []

    # Retrieve total number of train traces (as labels) and GEs files
    for filename in os.listdir(RES_ROOT): 

        # 3 possible types of files in RES_ROOT:
        # * Directory (relative to a given amount of tot traces)
        # * .npy (containing GEs data)
        # * .csv (containing GEs data for plots)
        # * .svg (plots)
        #
        # Directories allow to have the amount of traces
        # .npy files give GEs data

        filepath = RES_ROOT + f'/{filename}'

        if os.path.isdir(filepath):
            tt_labels.append(filename.split('_')[0])
        else:
            if '.npy' in filename:
                ges_files.append(filename)

    # Labels and GE files can be appended in the wrong order: need to sort
    tt_labels.sort(key=lambda x: int(x.split('k')[0])) # N k_traces --> int(N) is needed to sort
    ges_files.sort(key=lambda x: int(x.split('_')[-1].split('k.')[0])) # avg_ge_ N k.npy --> int(N) is needed to sort


    # Get actual GE data
    ges = []
    for ge_file in ges_files:
        NPY_FILE = RES_ROOT + f'/{ge_file}'
        ge = np.load(NPY_FILE)
        ges.append(ge)
    ges = np.vstack(ges)


    # Compute min number of attack traces per train-set size
    min_att_traces = [results.min_att_tr(ge, THRESHOLD) for ge in ges]


##### Save Results #############################################################

    # Save GEs
    ges_data = np.vstack(
        (
            np.arange(ges.shape[1])+1, # The values of the x-axis in the plot
            ges # The values of the y-axis in the plot
        )
    ).T
    helpers.save_csv(
        data=ges_data,
        columns=['AttTraces']+[f'TotTraces_{tot}' for tot in tt_labels],
        output_path=TT_GES_FILE
    )
    # Plot GEs
    vis.plot_ges(
        ges=ges[:, :10], 
        labels=[f'{tt_l} traces' for tt_l in tt_labels],
        title=f'Tot Traces Analysis  |  Byte: {BYTE}, Train-Devices: {n_devs}',
        ylim_max=50,
        output_path=TT_GES_PLOT 
    )

    # Save min number of attack traces
    min_att_tr_data = np.vstack(
        (
            tt_labels, # The values of the x-axis in the plot
            min_att_traces # The values of the y-axis in the plot
        )
    ).T
    helpers.save_csv(
        data=min_att_tr_data,
        columns=['TotTraces', 'MinAttTraces'],
        output_path=MIN_ATT_TR_FILE
    )
    # Plot min number of attack traces
    vis.plot_min_att_tr(
        min_att_tr=min_att_traces,
        xlabels=tt_labels,
        ylim_max=9,
        title=f'Min Number of Attack Traces  |  Byte: {BYTE}  |  Train-Devices: {n_devs}',
        output_path=MIN_ATT_TR_PLOT 
    )


if __name__ == '__main__':
    main()