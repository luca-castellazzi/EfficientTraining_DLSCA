# Basics
import numpy as np
import pandas as pd
import trsfile
import json
from tqdm import tqdm
import os

# Custom
import sys
sys.path.insert(0, '../utils')
import aes
import constants

TRAIN_PERC = 0.8

def main():
    
    tr_type = sys.argv[1].upper() # CURR or EM
    target = sys.argv[2].upper() # SBOX_OUT or HW or KEY

    if tr_type == 'CURR':
        tr_path = constants.CURR_TRACES_PATH
        dset_path = constants.CURR_DATASETS_PATH + f'/{target}'
    else:
        tr_path = constants.EM_TRACES_PATH
        dset_path = constants.EM_DATASETS_PATH + f'/{target}'

    train_size = int(TRAIN_PERC * constants.TRACE_NUM)


    # Get all the names of the traces whose key has been specified
    new_traces = []
    for tr_name in os.listdir(tr_path):
        if 'Resampled' in tr_name:
            tmp_substr = tr_name.replace('_500MHz + Resampled.trs', '')
            if not any([tmp_substr in dset for dset in os.listdir(dset_path)]):
                new_traces.append(tr_name)


    # Save all info as JSON file
    columns = ['samples', 'pltxt', 'labels', 'key']
    
    for tr_name in new_traces:
    
        config = tr_name.replace('_500MHz + Resampled.trs', '') # Di-Kj or D3-K0_NICV
            
        df = pd.DataFrame(
            data=np.zeros((constants.TRACE_NUM, len(columns)), dtype=object),
            columns=columns
        )
            
        with trsfile.open(f'{tr_path}/{tr_name}', 'r') as tr_set:

            for i, tr in enumerate(tqdm(tr_set, desc=f'Creating {config} DataFrame: ')):
                key = np.array(tr.get_key()) # int format by default
                samples = np.array(tr.samples)
                pltxt = np.array(tr.get_input()) # int format by default
                labels = aes.labels_from_key(pltxt, key, target) # Compute the set of 16 labels
                    
                df.at[i, 'samples'] = samples 
                df.at[i, 'pltxt'] = pltxt
                df.at[i, 'labels'] = labels
                df.at[i, 'key'] = key
            
        # Shuffle the Dataframe and set its index to the default one (from 0)
        df = df.sample(frac=1, random_state=24).reset_index(drop=True)
            
        # Split the DataFrame into train and test
        train_df = df.iloc[:train_size].reset_index(drop=True)
        test_df = df.iloc[train_size:].reset_index(drop=True)

        print(f'Creating TRAIN {config} JSON file...')
        train_df.to_json(f'{dset_path}/{config}_train.json')

        print(f'Creating TEST {config} JSON file...')
        test_df.to_json(f'{dset_path}/{config}_test.json')
            
        print(f'{config} JSON files successfully created.')
        print()


if __name__ == '__main__':
    main()
