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
    
    tr_type = sys.argv[1].upper() # CURR or EM
    target = sys.argv[2].upper() # SBOX_OUT or HW or KEY

    if tr_type == 'CURR':
        dset_path = constants.CURR_DATASETS_PATH + f'/{target}'
    else:
        dset_path = constants.EM_DATASETS_PATH + f'/{target}'
    
    
    configs = ['D3-K3', 'D3-K3_nicv',
               'D1-K1', 'D2-K1', 'D3-K1',
               'D1-K0', 'D1-K5', 'D1-K9',
               'D2-K0', 'D2-K5', 'D2-K9',
               'D3-K0', 'D3-K5', 'D3-K9']
    
    same_config = ['D3-K3', 'D3-K3_nicv']
    same_key_k1 = ['D1-K1', 'D2-K1', 'D3-K1']
    same_dev_d1 = ['D1-K1', 'D1-K0', 'D1-K5', 'D1-K9']
    same_dev_d2 = ['D2-K1', 'D2-K0', 'D2-K5', 'D2-K9']
    same_dev_d3 = ['D3-K1', 'D3-K0', 'D3-K5', 'D3-K9']
    
    # Compute NICVs for all scenarios and save them
    nicv_dict = {}
    for config in tqdm(configs, desc='Computing NICVs: '):
        path = dset_path + f'/{config}_train.json'
        dl = DataLoader([path])
        trs, _, pltxts, _ = dl.load_data()

        nicv_dict[config] = np.array([nicv(trs, pltxts, b) for b in range(16)])
    
    cmap = plt.cm.Set1

    # Plot NICVs for all scenarios
    print()
    print('Plotting NICV for "same config"...')
    nicvs = [nicv_dict[c] for c in same_config]
    vis.plot_nicv(
        nicvs, 
        same_config, 
        ('same-config-D3-K3', cmap))


    print()
    print('Plotting NICV for "same key K1"...')
    nicvs = [nicv_dict[c] for c in same_key_k1]
    vis.plot_nicv(
        nicvs, 
        same_key_k1, 
        (f'same-key-K1', cmap))

 
    print()
    print('Plotting NICV for "same dev D1"...')
    nicvs = [nicv_dict[c] for c in same_dev_d1]
    vis.plot_nicv(
        nicvs, 
        same_dev_d1, 
        (f'same-dev-D1', cmap))
    
    
    print()
    print('Plotting NICV for "same dev D2"...')
    nicvs = [nicv_dict[c] for c in same_dev_d2]
    vis.plot_nicv(
        nicvs, 
        same_dev_d2, 
        (f'same-dev-D2', cmap))
        
        
    print()
    print('Plotting NICV for "same dev D3"...')
    nicvs = [nicv_dict[c] for c in same_dev_d3]
    vis.plot_nicv(
        nicvs, 
        same_dev_d3, 
        (f'same-dev-D3', cmap))


if __name__ == '__main__':
    main()
