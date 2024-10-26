# Basics
import numpy as np
from tqdm import tqdm

# Custom
import sys 
sys.path.insert(0, '../utils')
from data_loader import DataLoader
import constants
from nicvNEW import nicv
import visualization as vis

TARGET = 'SBOX_OUT'
BYTE = 5


def main():

    """
    Computes NICV index.
        
    NICVs are computed for fixed device-key configurations, but all scenarios 
    are covered: same config, same key but different device, same device but 
    different key, different device and different key.
    
    The results are NPY files containing all NICV indices for each scenario and 
    PNG files containing their plots.
    """
    
    same_config = ['D3-K3', 'D3-K3_nicv']
    same_dev = [f'D1-{k}' for k in list(constants.KEYS)[1:]]
    same_key = [f'{d}-K1' for d in constants.DEVICES]
    diff = [f'{d}-{k}' for d in constants.DEVICES for k in list(constants.KEYS)[1:5]]

    configs = same_config + same_dev + same_key + diff
    
    # Compute NICVs for all scenarios and save them
    nicv_dict = {}
    for c in tqdm(configs, desc='Computing NICVs: '):
        dl = DataLoader(
            [f'{constants.PC_TRACES_PATH}/{c}_500MHz + Resampled.trs'],
            tot_traces=50000,
            target=TARGET,
            byte_idx=BYTE
        )
        
        trs, _, pltxts, _ = dl.load()

        nicv_dict[c] = np.array(nicv(trs, pltxts))


    # Plot NICVs for all scenarios
    scenarios = {
        'same-config': same_config,
        'same_dev':    same_dev,
        'same_key':    same_key,
        'diff':        diff
    }
    
    for s_name, s_configs in tqdm(scenarios.items(), desc='Plotting NICVs: '):
    
        output = f'{constants.RESULTS_PATH}/NICV/nicv_{s_name}'
        
        nicvs = np.array([nicv_dict[c] for c in s_configs])
        
        np.save(f'{output}.npy', nicvs)
        
        vis.plot_nicv(
            nicvs, 
            s_configs,
            f'{output}.svg'
        )


if __name__ == '__main__':
    main()
