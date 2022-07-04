# Basics
import numpy as np
import os
import matplotlib
matplotlib.use('Agg') # Avoid interactive mode (and use .PNG as default)
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

# Custom
import sys 
sys.path.insert(0, '../utils')
from data_loader import DataLoader
import constants
from nicv import nicv
sys.path.insert(0, '../modeling')
import visual as vis


def main():
    
    tr_type = sys.argv[1].upper() # CURR or EM
    target = sys.argv[2].upper() # SBOX_OUT or HW or KEY

    if tr_type == 'CURR':
        dset_path = constants.CURR_DATASETS_PATH + f'/{target}'
    else:
        dset_path = constants.EM_DATASETS_PATH + f'/{target}'

    
    # Compute NICVs for all scenarios and save them
    nicv_dict = {}
    for jfile in os.listdir(dset_path):
        
        dl = DataLoader(dset_path + f'/{jfile}', train_perc=1.0)
        trs, _, pltxts = dl.gen_set()

        config = jfile.split('.')[0]
        nicv_dict[config] = np.array([nicv(trs, pltxts, b) for b in range(16)])
        
        np.savetxt(constants.RESULTS_PATH + f'/nicv/nicv_files/nicv_{config}.csv',
                   nicv_dict[config],
                   delimiter=',')


    date = datetime.now().strftime("%m%d%Y-%I%M%p") 
    

    # Plot NICVs for all scenarios
    print()
    print('Computing NICV for "same config"...')
    same_config = ['D1-K1', 'D1-K1_NICV']
    same_config_cmap = plt.cm.Set1
    nicvs = [nicv_dict[c] for c in same_config]
    vis.plot_nicv(nicvs, 
                  same_config, 
                  ('same-config', same_config_cmap, date))


    print()
    print('Computing NICV for "same device"...')
    same_device = {d: [f'{d}-{k}' for k in constants.KEYS] 
                   for d in constants.DEVICES}
    same_device_cmap = plt.cm.Dark2
    for d, configs in same_device.items():
        nicvs = [nicv_dict[c] for c in configs]
        vis.plot_nicv(nicvs, 
                      configs, 
                      (f'same-device-{d}', same_device_cmap, date))

 
    print()
    print('Computing NICV for "same key"...')
    same_key = {k: [f'{d}-{k}' for d in constants.DEVICES] 
                for k in constants.KEYS}
    same_key_cmap = plt.cm.tab10
    for k, configs in same_key.items():
        nicvs = [nicv_dict[c] for c in configs] 
        vis.plot_nicv(nicvs, 
                      configs, 
                      (f'same-key-{k}', same_key_cmap, date))
    
    
    print()
    print('Computing NICV for "all configs"...')
    all_config = [f'{d}-{k}' 
                  for d in constants.DEVICES 
                  for k in constants.KEYS]
    all_config_cmap = plt.cm.Set1    
    nicvs = [nicv_dict[c] for c in all_config] 
    vis.plot_nicv(nicvs, 
                  all_config, 
                  ('all-config', all_config_cmap, date))

    return



if __name__ == '__main__':
    main()
