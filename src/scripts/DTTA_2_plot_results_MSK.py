import os
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '../utils')
import constants
import results
import helpers
import visualization as vis

THRESHOLD = 0.5
BYTE = 5


def main():

    _, n_devs = sys.argv
    n_devs = int(n_devs)

    RES_ROOT = f'{constants.RESULTS_PATH}/DTTA/msk/{n_devs}d'
    TT_AVG_GES_FILE = RES_ROOT + '/avg_ges.csv'
    TT_AVG_GES_PLOT = RES_ROOT + '/avg_ges.svg'
    MIN_ATT_TR_FILE = RES_ROOT + '/min_att_tr.csv'
    MIN_ATT_TR_PLOT = RES_ROOT + '/min_att_tr.svg'
    TT_SINGLE_GES_FILES = [RES_ROOT + f'/ges_{"".join(train_devs)}vs{test_dev}.csv' 
                           for train_devs, test_dev in constants.PERMUTATIONS[n_devs]]
    TT_SINGLE_GES_PLOTS = [RES_ROOT + f'/ges_{"".join(train_devs)}vs{test_dev}.svg' 
                           for train_devs, test_dev in constants.PERMUTATIONS[n_devs]]

    # Get avg GEs and Tot Traces
    avg_ges = []
    tt_labels = []
    for el in os.listdir(RES_ROOT):
        
        if '.npy' in el: # If the element is a folder
            avg_ge = np.load(RES_ROOT + f'/{el}')
            avg_ges.append(avg_ge)
        elif 'traces' in el:
            tt_labels.append(el.split('_')[0])
    
    avg_ges = np.vstack(avg_ges)

    min_att_traces = [results.min_att_tr(avg_ge, THRESHOLD) for avg_ge in avg_ges]

    # Gen single GEs
    single_ges = []
    for train_devs, test_dev in constants.PERMUTATIONS[n_devs]:
        
        p_res = []
        
        for el in os.listdir(RES_ROOT):
            
            if 'traces' in el:
                ge = np.load(RES_ROOT + f'/{el}/ge_{"".join(train_devs)}vs{test_dev}.npy')
                p_res.append(ge)
                
        p_res = np.vstack(p_res)
        single_ges.append(p_res) 


    # Save Avg GEs
    avg_ges_data = np.vstack(
        (
            np.arange(avg_ges.shape[1])+1, # The values of the x-axis in the plot
            avg_ges # The values of the y-axis in the plot
        )
    ).T
    helpers.save_csv(
        data=avg_ges_data,
        columns=['AttTraces']+[f'TotTraces_{tot}' for tot in tt_labels],
        output_path=TT_AVG_GES_FILE
    )
    # Plot GEs
    vis.plot_ges(
        ges=avg_ges, 
        labels=[f'{tt_l} traces' for tt_l in tt_labels],
        title=f'DTTA - Avg GEs  |  Byte: {BYTE}  |  Train-Devices: {n_devs}',
        ylim_max=150,
        output_path=TT_AVG_GES_PLOT 
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
        threshold=THRESHOLD,
        xlabels=tt_labels,
        ylim_max=500,
        title=f'Min Number of Attack Traces - Avg GEs  |  Byte: {BYTE}  |  Train-Devices: {n_devs}',
        output_path=MIN_ATT_TR_PLOT 
    )

    # Save and Plot single GEs
    i = 0
    for (train_devs, test_dev), ges in zip(constants.PERMUTATIONS[n_devs], single_ges):
        ges_data = np.vstack(
            (
                np.arange(ges.shape[1])+1, # The values of the x-axis in the plot
                ges # The values of the y-axis in the plot
            )
        ).T
        helpers.save_csv(
            data=ges_data,
            columns=['AttTraces']+[f'TotTraces_{tot}' for tot in tt_labels],
            output_path=TT_SINGLE_GES_FILES[i]
        )
        # Plot GEs
        vis.plot_ges(
            ges=ges, 
            labels=[f'{tt_l} traces' for tt_l in tt_labels],
            title=f'DTTA - {"".join(train_devs)}vs{test_dev}  |  Byte: {BYTE}',
            ylim_max=150,
            output_path=TT_SINGLE_GES_PLOTS[i] 
        )
        i += 1


if __name__ == '__main__':
    main()