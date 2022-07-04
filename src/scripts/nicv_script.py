# Basics
import numpy as np
import os
import matplotlib
matplotlib.use('pdf') # Avoid interactive mode
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

# Custom
import sys 
sys.path.insert(0, '../utils')
from data_loader import DataLoader
import constants
sys.path.insert(0, '../modeling')
import visual as vis


def main():
    
    tr_type = sys.argv[1].upper() # CURR or EM
    target = sys.argv[2].upper() # SBOX_OUT or HW or KEY

    if tr_type == 'CURR':
        dset_path = constants.CURR_DATASETS_PATH + f'/{target}'
    else:
        dset_path = constants.EM_DATASETS_PATH + f'/{target}'

    # Define a DataLoader for each traceset file
    data_loaders = {}
    for dset_name in os.listdir(dset_path):

        c = dset_name[:5] # Di-Kj
        if 'NICV' in dset_name:
            c = dset_name[:10] # D1-K1_NICV

        data_loaders[c] = DataLoader(dset_path + f'/{c}.json', train_perc=1.0)
    

    # Define all possible groups of DataLoaders to compute NICVs
    dl_groups = []

    # Same Device, Same Key at different times
    same_config = ['D1-K1', 'D1-K1_NICV']
    same_config_cmap = plt.cm.Set1
    dl_groups.append(({c: data_loaders[c] for c in same_config},
                      same_config_cmap, 
                      'same_config'))

    # Same Device, Different Key
    same_device = {d: [f'{d}-{k}' for k in constants.KEYS] 
                   for d in constants.DEVICES}
    same_device_cmap = plt.cm.Dark2
    for d, config in same_device.items():
        dl_groups.append(({c: data_loaders[c] for c in config},
                          same_device_cmap,
                          f'same_device-{d}'))

    # Different Device, Same Key
    same_key = {k: [f'{d}-{k}' for d in constants.DEVICES] 
                for k in constants.KEYS}
    same_key_cmap = plt.cm.tab10
    for k, config in same_key.items():
        dl_groups.append(({c: data_loaders[c] for c in config},
                          same_key_cmap,
                          f'same_key-{k}'))

    # All possible Device-Key configurations
    all_config = [f'{d}-{k}' for d in constants.DEVICES 
                  for k in constants.KEYS]
    all_config_cmap = plt.cm.Set1    
    dl_groups.append(({c: data_loaders[c] for c in all_config},
                      all_config_cmap,
                      'all_config'))
    
    date = datetime.now().strftime("%m%d%Y-%I%M%p") 
    
    # Plot NICVs
    nicvs = []
    for dl_group in tqdm(dl_groups, desc='Computing NICVs: '):
        nicvs.append(vis.plot_nicv(dl_group, date))

    nicvs = np.array(nicvs)
    np.savetxt(constants.RESULTS_PATH + f'/nicv/nicv_{data}.csv', 
               nicvs, 
               delimiter=',') 


if __name__ == '__main__':
    main()
