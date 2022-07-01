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

import numpy as np
import trsfile
import json
from tqdm import tqdm

import sys
sys.path.insert(0, '../utils')
import aes
import constants


def main():
    
    target = 'SBOX_OUT'

    for d in constants.DEVICES:
        for k in constants.KEYS.keys():

            print(f'----- {d}-{k} -----')

            trace_path = constants.CURR_TRACES_PATH + f'/{d}-{k}_50k_500MHz + Resampled at 168MHz.trs'
            json_path = constants.CURR_DATASES_PATH + f'/{target}/{d}-{k}.json'

            traces = []
            plaintexts = []
            labels = []

            with trsfile.open(trace_path, 'r') as tr_set:

                for i, tr in enumerate(tqdm(tr_set, desc='Labeling traces: ')):
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
            
            print(f'Generating {d}-{k} JSON file...')

            with open(json_path, 'w') as j_file:
                json.dump(dataset_dict, j_file)
        
            print(f'{d}-{k} JSON file successfully created.')
            print()

if __name__ == '__main__':
    main()
