# Basic
import polars as pl
import numpy as np

# Custom
import sys
sys.path.insert(0, '../utils')
import constants
import helpers
import visualization as vis


BYTE = 5


def main():

    _, n_devs = sys.argv
    n_devs = int(n_devs)

    RES_ROOT = f'{constants.RESULTS_PATH}/TTA'
    OVERLAP_FILE = RES_ROOT + f'/min_att_tr_overlap_{n_devs}d.csv'
    OVERLAP_PLOT = RES_ROOT + f'/min_att_tr_overlap_{n_devs}d.svg'

    # SoA
    soa_csv = RES_ROOT + f'/soa/{n_devs}d/min_att_tr.csv'
    soa_df = pl.read_csv(soa_csv)
    soa_min_att_tr = soa_df['MinAttTraces'].to_numpy()
    labels = list(soa_df['TotTraces'])

    custom_csv = RES_ROOT + f'/custom/{n_devs}d/min_att_tr.csv'
    custom_df = pl.read_csv(custom_csv)
    custom_min_att_tr = custom_df['MinAttTraces'].to_numpy()
    
    # Save Results
    overlap_data = np.vstack(
        (
            labels, # The values of the x-axis in the plot
            soa_min_att_tr, # The values of the y-axis in the plot
            custom_min_att_tr, # The values of the y-axis in the plot
        )
    ).T
    helpers.save_csv(
        data=overlap_data,
        columns=['TotTraces', 'SoA', 'Custom'],
        output_path=OVERLAP_FILE
    )

    # Plot Results
    vis.plot_overlap_min_att_tr(
        soa_data=soa_min_att_tr,
        custom_data=custom_min_att_tr,
        xlabels=labels,
        ylim_max=15,
        title=f'Min Number of Attack Traces - SoA vs Custom  |  Byte: {BYTE}  |  Train-Devices: {n_devs}',
        output_path=OVERLAP_PLOT 
    )

if __name__ == '__main__':
    main()

