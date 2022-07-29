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
    

    
    # Compute NICVs for all scenarios and save them
    nicv_dict = {}
    for d in constants.DEVICES:
        for k in tqdm(constants.KEYS, desc=f'Dev {d}: '):
            config = f'{d}-{k}'
            path = dset_path + f'/{config}_train.json'
            dl = DataLoader([path])
            trs, _, pltxts, _ = dl.load_data()

            nicv_dict[config] = np.array([nicv(trs, pltxts, b) for b in range(16)])
        
            #np.savetxt(constants.RESULTS_PATH + f'/nicv/nicv_files/nicv_{config}.csv',
            #           nicv_dict[config],
            #           delimiter=',')
    
    # Compute NICV for particular case "same config"
    config = f'D3-K3_nicv'
    path = dset_path + f'/{config}_train.json'
    dl = DataLoader([path])
    trs, _, pltxts, _ = dl.load_data()

    nicv_dict[config] = np.array([nicv(trs, pltxts, b) for b in range(16)])
        
    #np.savetxt(constants.RESULTS_PATH + f'/nicv/nicv_files/nicv_{config}.csv',
    #           nicv_dict[config],
    #           delimiter=',')
    

    # Plot NICVs for all scenarios
    print()
    print('Computing NICV for "same config"...')
    same_config = ['D3-K3', 'D3-K3_nicv']
    same_config_cmap = plt.cm.Set1
    nicvs = [nicv_dict[c] for c in same_config]
    vis.plot_nicv(nicvs, 
                  same_config, 
                  ('same-config', same_config_cmap))


    print()
    print('Computing NICV for "same device"...')
    same_device = {d: [f'{d}-{k}' for k in constants.KEYS] 
                   for d in constants.DEVICES}
    same_device_cmap = plt.cm.Dark2
    for d, configs in same_device.items():
        nicvs = [nicv_dict[c] for c in configs]
        vis.plot_nicv(nicvs, 
                      configs, 
                      (f'same-device-{d}', same_device_cmap))

 
    print()
    print('Computing NICV for "same key"...')
    same_key = {k: [f'{d}-{k}' for d in constants.DEVICES] 
                for k in constants.KEYS}
    same_key_cmap = plt.cm.tab10
    for k, configs in same_key.items():
        nicvs = [nicv_dict[c] for c in configs] 
        vis.plot_nicv(nicvs, 
                      configs, 
                      (f'same-key-{k}', same_key_cmap))
    
    
    print()
    print('Computing NICV for "all configs"...')
    all_config = [f'{d}-{k}' 
                  for d in constants.DEVICES 
                  for k in constants.KEYS]
    all_config_cmap = plt.cm.Set1    
    nicvs = [nicv_dict[c] for c in all_config] 
    vis.plot_nicv(nicvs, 
                  all_config, 
                  ('all-config', all_config_cmap))


if __name__ == '__main__':
    main()
