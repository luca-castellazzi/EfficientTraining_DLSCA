# Generation of a CSV file containing all trace data and metadata
#
# CSV structure:
# 
#  'samples'         |  'pltxt'         |  'labels'        |  'key'
# ---------------------------------------------------------------------------
#  [...] (1 x 1183)  |  [...] (1 x 16)  |  [...] (1 x 16)  |  [...] (1 x 16)
#  [...] (1 x 1183)  |  [...] (1 x 16)  |  [...] (1 x 16)  |  [...] (1 x 16)
#   ...
#  [...] (1 x 1183)  |  [...] (1 x 16)  |  [...] (1 x 16)  |  [...] (1 x 16)


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


def main():
    
    tr_type = sys.argv[1].upper() # CURR or EM
    target = sys.argv[2].upper() # SBOX_OUT or HW or KEY
    train_perc = float(sys.argv[3])

    if tr_type == 'CURR':
        tr_path = constants.CURR_TRACES_PATH
        dset_path = constants.CURR_DATASETS_PATH + f'/{target}'
    else:
        tr_path = constants.EM_TRACES_PATH
        dset_path = constants.EM_DATASETS_PATH + f'/{target}'

    columns = ['samples', 'pltxt', 'labels', 'key']

    train_size = int(train_perc * constants.TRACE_NUM)


    for tr_name in os.listdir(tr_path):    
            
        if 'Resampled' in tr_name: # Do NOT consider 500MHz traces

            if 'NICV' in tr_name:
                config = tr_name[:10] # D1-K1_NICV
            else:    
                config = tr_name[:5] # Di-Kj
            
            df = pd.DataFrame(data=np.zeros((constants.TRACE_NUM, len(columns)), dtype=object),
                              columns=columns)
            
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

            print(f'Saving TEST {config} JSON file...')
            test_df.to_json(f'{dset_path}/{config}_test.json')
            
            print(f'{config} CSV file successfully created.')
            print()


if __name__ == '__main__':
    main()
