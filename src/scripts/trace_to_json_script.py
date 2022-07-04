# Generation of a JSON file containing all trace-values and trace-metadata

# JSON structure:
# {
#     'traces': [    
#		    [
#		        'samples': [...] (1 x 1183)
#			'pltxt':   [...] (1 x 16)
#			'labels':  [...] (1 x 16)
#  		    ]    
#                   ...
#               ]
# 
#     'key':    [...] (1 x 16)
# }

# Basics
import numpy as np
import trsfile
import json
from tqdm import tqdm
import os

# Custom
import sys
sys.path.insert(0, '../utils')
import aes
import constants


def main():
    
    tr_type = sys.argv[1].upper() # CURR or EM
    target = sys.argv[2].upper() # SBOX_OUT or HW or KEY

    if tr_type == 'CURR':
        tr_path = constants.CURR_TRACES_PATH
        dset_path = constants.CURR_DATASETS_PATH + f'/{target}'
    else:
        tr_path = constants.EM_TRACES_PATH
        dset_path = constants.EM_DATASETS_PATH + f'/{target}'


    for tr_name in os.listdir(tr_path):
        
        if 'Resampled' in tr_name: # Do NOT consider 500MHz traces

            if 'NICV' in tr_name:
                config = tr_name[:10] # D1-K1_NICV
            else:    
                config = tr_name[:5] # Di-Kj
            
            trace_path = tr_path + '/' + tr_name
            json_path = dset_path + f'/{config}.json'

            traces = []
            plaintexts = []
            labels = []
            with trsfile.open(trace_path, 'r') as tr_set:

                for i, tr in enumerate(tqdm(tr_set, desc=f'Labeling {config}: ')):
                    key = np.array(tr.get_key()) # int format by default

                    samples = np.array(tr.samples)
                    pltxt = np.array(tr.get_input()) # int format by default
                    all_labels = aes.labels_from_key(pltxt, key, target) # Compute the set of 16 labels

                    traces.append(samples.tolist())
                    plaintexts.append(pltxt.tolist())
                    labels.append(all_labels.tolist())

            traces_dict = [{'samples': spl[0],
                            'pltxt':   spl[1],
                            'labels':  spl[2]}
                           for spl in zip(traces, plaintexts, labels)]

            dataset_dict = {'traces': traces_dict, 
                            'key':    key.tolist()}
            
            print(f'Generating {config} JSON file...')

            with open(json_path, 'w') as j_file:
                json.dump(dataset_dict, j_file)
        
            print(f'{config} JSON file successfully created.')
            print()

if __name__ == '__main__':
    main()
