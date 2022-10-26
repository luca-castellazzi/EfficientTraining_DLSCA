# Basics
from tqdm import tqdm
import numpy as np
import json
from src.scripts.DKTA_1_hp_tuning import TOT_TRACES
from tensorflow.keras.backend import clear_session

# Custom
import sys
sys.path.insert(0, '../src/utils')
from data_loader import DataLoader, SplitDataLoader
import constants
import helpers
import results
import visualization as vis
sys.path.insert(0, '../src/modeling')
from network import Network

# Suppress TensorFlow messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # 1 for INFO, 2 for INFO & WARNINGs, 3 for INFO & WARNINGs & ERRORs


EPOCHS = 100
TARGET = 'SBOX_OUT'

PERMUTATIONS = [('D3', 'D1'), ('D3', 'D2')]

N_KEYS = [1, 50, 100]
TOT_TRACES = [1000, 3000, 4000, 5000, 10000]


def main():

    _, b = sys.argv()
    b = int(b) # Byte to attack

    RES_ROOT = f'{constants.RESULTS_PATH}/DKTA/SBOX_OUT/byte{b}/mk_1d'
    HP_PATH = RES_ROOT + '/hp.json'


    # Get hyperparameters
    with open(HP_PATH, 'r') as jfile:
        hp = json.load(jfile)


    # Attack with different configurations of device, key and number of traces
    ges_per_traces = []

    for tot_traces in TOT_TRACES:

        GES_FILE = RES_ROOT + f'ges_{tot_traces}.csv'
        GES_FILE_NPY = RES_ROOT + f'ges_{tot_traces}.npy'
        GES_PLOT = RES_ROOT + f'ges_{tot_traces}.svg'

        print(f'*** Number of Traces: {tot_traces}')

        ges_per_keys = []

        for n_keys in N_KEYS:

            ges = []

            for train_dev, test_dev in tqdm(PERMUTATIONS, desc=f'Using {n_keys} Keys: '):

                N_KEYS_FOLDER = RES_ROOT + f'/{n_keys}k'
                if not os.path.exists(N_KEYS_FOLDER):
                    os.mkdir(N_KEYS_FOLDER)
                SAVED_MODEL_PATH = N_KEYS_FOLDER + f'/model_{train_dev}vs{test_dev}.h5'

                test_files = [f'{constants.PC_TRACES_PATH}/{test_dev}-K0_500MHz + Resampled.trs'] # list is needed in DataLoader      

                train_files = [f'/prj/side_channel/Pinata/PC/swAES/MultiKeySplits/{train_dev}-MK{k}.trs'
                               for k in range(n_keys)]

                train_dl = SplitDataLoader(
                    train_files, 
                    tot_traces=tot_traces,
                    train_size=0.9,
                    target=TARGET,
                    byte_idx=b
                )
                train_data, val_data = train_dl.load()
                x_train, y_train, _, _ = train_data 
                x_val, y_val, _, _ = val_data # Val data is kept to consider callbacks 

                clear_session() # Start with a new Keras session every time            
                
                attack_net = Network('MLP', hp)
                attack_net.build_model()
                attack_net.add_checkpoint_callback(SAVED_MODEL_PATH)
                
                #Training (with Validation)
                attack_net.model.fit(
                    x_train, 
                    y_train, 
                    validation_data=(x_val, y_val),
                    epochs=EPOCHS,
                    batch_size=attack_net.hp['batch_size'],
                    callbacks=attack_net.callbacks,
                    verbose=0
                )
                
                # Testing (every time consider random test traces from the same set)
                test_dl = DataLoader(
                    test_files, 
                    n_tr_per_config=2000,
                    target=TARGET,
                    byte_idx=b
                )
                x_test, _, pbs_test, tkb_test = test_dl.load()
                
                preds = attack_net.model.predict(x_test)            
                
                # Compute GE
                ge = results.ge(
                    preds=preds, 
                    pltxt_bytes=pbs_test, 
                    true_key_byte=tkb_test, 
                    n_exp=100, 
                    target=TARGET,
                    n_traces=500 # Default: 500
                )
                ges.append(ge)

            ges = np.array(ges)
            ges = np.mean(ges, axis=0) # Average the results of the "cycle" over the permutations

            ges_per_keys.append(ges)

        ges_per_keys = np.array(ges_per_keys)
        # ges_per_traces.append(ges_per_keys)


        # Save the results for the current total number of traces
        csv_ges_data = np.vstack(
            (
                np.arange(ges_per_keys.shape[1])+1, # The values of the x-axis in the plot
                ges_per_keys # The values of the y-axis in the plot
            )
        ).T

        helpers.save_csv(
            data=csv_ges_data,
            columns=['NTraces']+[f'NKeys_{nk+1}' for nk in N_KEYS],
            output_path=GES_FILE
        )
        # In .NPY for direct use in DKTA_4_overlap.py
        np.save(GES_FILE_NPY, ges_per_keys)
        
        # Plot Avg GEs
        vis.plot_multikey(
            ges_per_keys, 
            N_KEYS,
            f'Total Traces: {tot_traces}',
            GES_PLOT            
        )

        print()

    # ges_per_traces = np.array(ges_per_traces)




if __name__ == '__main__':
    main()