# Basics
from tqdm import tqdm
import numpy as np
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.backend import clear_session

# Custom
import sys
sys.path.insert(0, '../utils')
from data_loader import DataLoader, RandomSplitDataLoader
import constants
import helpers
import results
import visualization as vis
sys.path.insert(0, '../modeling')
from network import Network

# Suppress TensorFlow messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # 1 for INFO, 2 for INFO & WARNINGs, 3 for INFO & WARNINGs & ERRORs


EPOCHS = 100
TARGET = 'SBOX_OUT'

TRAIN_DEV = 'D3'
ATTACKED_DEV = 'D1'

N_KEYS = [1, 50, 100]

# Different experiments:
# - Test different sizes for train-set
# - Compute average of experiments, where smaller train-sets need more repetitions
EXPS = [(1000, 10), (3000, 10), (4000, 10) (5000, 2), (7000, 2)] # (tot_traces, n_exps)


def main():

    _, b = sys.argv
    b = int(b) # Byte to attack

    RES_ROOT = f'{constants.RESULTS_PATH}/DKTA/SBOX_OUT/byte{b}/mk_1d'
    HP_PATH = RES_ROOT + '/hp.json'


    # Get hyperparameters
    with open(HP_PATH, 'r') as jfile:
        hp = json.load(jfile)


    # Attack with different configurations of device, key and number of traces
    for n_keys in N_KEYS:
        GES_FILE = RES_ROOT + f'/ges_{n_keys}k.csv'
        GES_PLOT = RES_ROOT + f'/ges_{n_keys}k.svg'

        print(f'*** Number of Keys: {n_keys} ***')

        ges_per_traces = []

        for tot_traces, n_exps in EXPS:

            avg_ges = []

            for _ in tqdm(range(n_exps), desc=f'{tot_traces} Traces: '):

                N_KEYS_FOLDER = RES_ROOT + f'/{n_keys}k'
                if not os.path.exists(N_KEYS_FOLDER):
                    os.mkdir(N_KEYS_FOLDER)
                SAVED_MODEL_PATH = N_KEYS_FOLDER + f'/model_D3vsD1_{tot_traces}t.h5'

                test_files = [f'{constants.PC_TRACES_PATH}/D1-K0_500MHz + Resampled.trs'] # list is needed in DataLoader      

                train_files = [f'/prj/side_channel/Pinata/PC/swAES/MultiKeySplits/D3-MK{k}.trs'
                               for k in range(n_keys)]

                train_dl = RandomSplitDataLoader(
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
                    tot_traces=2000,
                    target=TARGET,
                    byte_idx=b
                )
                x_test, _, pbs_test, tkb_test = test_dl.load()
                
                test_model = load_model(SAVED_MODEL_PATH)
                preds = test_model.predict(x_test)            
                
                # Compute GE
                ge = results.ge(
                    preds=preds, 
                    pltxt_bytes=pbs_test, 
                    true_key_byte=tkb_test, 
                    n_exp=100, 
                    target=TARGET,
                    n_traces=500 # Default: 500
                )
                avg_ges.append(ge)

            avg_ges = np.array(avg_ges)
            avg_ges = np.mean(avg_ges, axis=0) # Average the results over the experiments

            ges_per_traces.append(avg_ges)

        ges_per_traces = np.array(ges_per_traces)


        # Save the results for the current total number of traces
        # In .CSV for future use
        csv_ges_data = np.vstack(
            (
                np.arange(ges_per_traces.shape[1])+1, # The values of the x-axis in the plot
                ges_per_traces # The values of the y-axis in the plot
            )
        ).T

        helpers.save_csv(
            data=csv_ges_data,
            columns=['AttackTraces']+[f'TrainTraces_{t}' for t, _ in EXPS],
            output_path=GES_FILE
        )

        # Plot Avg GEs
        vis.plot_multikey(
            ges_per_traces, 
            [tot_tr for tot_tr, _ in EXPS],
            f'Number of Keys: {n_keys}',
            GES_PLOT            
        )

        print()


if __name__ == '__main__':
    main()