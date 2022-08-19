# Basics
import numpy as np
import os
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

# Custom
import sys 
sys.path.insert(0, '../utils')
from data_loader import DataLoader
import constants
from nicv import nicv
import visualization as vis


def main():

    """
    Computes NICV index with the given settings.
    Settings parameters (provided in order via command line):
        - tr_type: Type of traces to use (CURR for Current, EM for ElectroMagnetic)
        
    NICVs are computed for fixed device-key configurations, but all scenarios 
    are covered: same config, same key but different device, same device but 
    different key, different device and different key.
    
    The results are NPY files containing all NICV indices for each scenario and 
    PNG files containing their plots.
    """
    
    _, tr_type = sys.argv
    
    tr_type = tr_type.upper()
    
    
    configs = ['D3-K3', 'D3-K3_nicv',
               'D1-K1', 'D2-K1', 'D3-K1',
               'D1-K0', 'D1-K5', 'D1-K9',
               'D2-K0', 'D2-K5', 'D2-K9',
               'D3-K0', 'D3-K5', 'D3-K9']
    
    # Compute NICVs for all scenarios and save them
    nicv_dict = {}
    for c in tqdm(configs, desc='Computing NICVs: '):
        dl = DataLoader(
            [c],
            n_tot_traces=constants.TRACE_NUM,
            target='SBOX_OUT' # Not important in NICV computation
        )
        
        trs, _, pltxts, _ = dl.load()

        nicv_dict[c] = np.array([nicv(trs, pltxts, b) for b in range(16)])


    # Plot NICVs for all scenarios
    scenarios = {
        'same-config': ['D3-K3', 'D3-K3_nicv'],
        'same-key-K1': ['D1-K1', 'D2-K1', 'D3-K1'],
        'same-dev-D1': ['D1-K0', 'D1-K1', 'D1-K5', 'D1-K9'],
        'same-dev-D2': ['D2-K0', 'D2-K1', 'D2-K5', 'D2-K9'],
        'same-dev-D3': ['D3-K0', 'D3-K1', 'D3-K5', 'D3-K9'],
        'diff':        ['D1-K1', 'D2-K5', 'D3-K9']
    }
    
    for s_name, s_configs in tqdm(scenarios.items(), desc='Plotting NICVs: '):
    
        output = f'{constants.RESULTS_PATH}/NICV/nicv_{s_name}'
        
        nicvs = np.array([nicv_dict[c] for c in s_configs])
        
        np.save(f'{output}.npy', nicvs)
        
        vis.plot_nicv(
            nicvs, 
            s_configs,
            f'{output}.png'
        )
        
   
    
    
    
    
    # print()
    # print('Plotting NICV for "same config"...')
    # output = f'{constants.RESULTS_PATH}/NICV/nicv_same-config-D3-K3'
    # nicvs = np.array([nicv_dict[c] for c in same_config])
    # np.save(f'{output}.npy', nicvs)
    # vis.plot_nicv(
        # nicvs, 
        # same_config,
        # f'{output}.png'
    # )


    # print()
    # print('Plotting NICV for "same key K1"...')
    # output = f'{constants.RESULTS_PATH}/NICV/nicv_same-key-K1'
    # nicvs = [nicv_dict[c] for c in same_key_k1]
    # np.save(f'{output}.npy', nicvs)
    # vis.plot_nicv(
        # nicvs, 
        # same_key_k1,
        # f'{output}.png'
    # )

 
    # print()
    # print('Plotting NICV for "same dev D1"...')
    # nicvs = [nicv_dict[c] for c in same_dev_d1]
    # np.save(f'{output}.npy', nicvs)
    # vis.plot_nicv(
        # nicvs, 
        # same_dev_d1,
        # f'{constants.RESULTS_PATH}/NICV/nicv_same-dev-D1.png'
    # )
    
    
    # print()
    # print('Plotting NICV for "same dev D2"...')
    # nicvs = [nicv_dict[c] for c in same_dev_d2]
    # vis.plot_nicv(
        # nicvs, 
        # same_dev_d2,
        # f'{constants.RESULTS_PATH}/NICV/nicv_same-dev-D2.png'
    # )
        
        
    # print()
    # print('Plotting NICV for "same dev D3"...')
    # nicvs = [nicv_dict[c] for c in same_dev_d3]
    # vis.plot_nicv(
        # nicvs, 
        # same_dev_d3,
        # f'{constants.RESULTS_PATH}/NICV/nicv_same-dev-D3.png'
    # )
        
        
    # print()
    # print('Plotting NICV for "diff"...')
    # nicvs = [nicv_dict[c] for c in diff]
    # vis.plot_nicv(
        # nicvs, 
        # diff,
        # f'{constants.RESULTS_PATH}/NICV/nicv_diff.png'
    # )


if __name__ == '__main__':
    main()
