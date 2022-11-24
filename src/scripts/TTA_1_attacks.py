# Basics
import json
import time
import numpy as np
from tqdm import tqdm
from tensorflow.keras.models import load_model
from tensorflow.keras.backend import clear_session
from sklearn.preprocessing import StandardScaler

# Custom
import sys
sys.path.insert(0, '../utils')
import constants
import results
import helpers
import visualization as vis
from data_loader import DataLoader, SplitDataLoader
sys.path.insert(0, '../modeling')
from network import Network

# Suppress TensorFlow messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # 1 for INFO, 2 for INFO & WARNINGs, 3 for INFO & WARNINGs & ERRORs

BYTE = 5
TARGET = 'SBOX_OUT'
EPOCHS = 100
TOT_TRAIN = range(50000, 550000, 50000) # 50k, 100k, 150k, 200k, 250k, 300k, 350k, 400k, 450k, 500k
VAL_SIZE = 5000
TOT_TEST = 50000


def main():

    _, n_devs = sys.argv
    n_devs = int(n_devs)

    # Get HPs
    HP_PATH = f'{constants.RESULTS_PATH}/DKTA/{TARGET}/byte{BYTE}/{n_devs}d/hp.json'
    with open(HP_PATH, 'r') as jfile:
        hp = json.load(jfile)

    # Definition of constant root path 
    RES_ROOT = f'{constants.RESULTS_PATH}/TTA/{n_devs}d'

    for tot_train in TOT_TRAIN:

        # Fix the size of validation set
        # This ensures to increase only the size of the actual train set
        train_perc = 1 - (VAL_SIZE / tot_train)

        ges = []

        AVG_GE_FILE = RES_ROOT + f'/avg_ge_{int(tot_train / 1000)}k'

        # Tot Traces Analysis (TTA)
        for train_devs, test_dev in tqdm(constants.PERMUTATIONS[n_devs], desc=f'{int(tot_train / 1000)}k Traces: '):

            train_files = [f'{constants.PC_TRACES_PATH}/{dev}-{k}_500MHz + Resampled.trs' 
                        for k in list(constants.KEYS)[1:]
                        for dev in train_devs]
            test_files = [f'{constants.PC_TRACES_PATH}/{test_dev}-K0_500MHz + Resampled.trs']

            TRACES_FOLDER = RES_ROOT + f'/{int(tot_train / 1000)}k_traces'
            if not os.path.exists(TRACES_FOLDER):
                os.mkdir(TRACES_FOLDER)
            SAVED_MODEL_PATH = TRACES_FOLDER + f'/model_{"".join(train_devs)}vs{test_dev}.h5'

            train_dl = SplitDataLoader(
                train_files, 
                tot_traces=tot_train,
                train_size=train_perc,
                target=TARGET,
                byte_idx=BYTE
            )
            train_data, val_data = train_dl.load()
            x_train, y_train, _, _ = train_data 
            x_val, y_val, _, _ = val_data

            test_dl = DataLoader(
                test_files, 
                tot_traces=TOT_TEST,
                target=TARGET,
                byte_idx=BYTE
            )
            x_test, _, pbs_test, tkb_test = test_dl.load()

            # Scale data to 0-mean and 1-variance
            scaler = StandardScaler()
            scaler.fit(x_train)
            x_train = scaler.transform(x_train) 
            x_val = scaler.transform(x_val)
            x_test = scaler.transform(x_test)


            clear_session() # Start with a new Keras session every time    

            # Train and Attack
            attack_net = Network('MLP', hp)
            attack_net.build_model()
            attack_net.add_checkpoint_callback(SAVED_MODEL_PATH)

            attack_net.model.fit(
                x_train, 
                y_train, 
                validation_data=(x_val, y_val),
                epochs=EPOCHS,
                batch_size=attack_net.hp['batch_size'],
                callbacks=attack_net.callbacks,
                verbose=0
            )

            test_model = load_model(SAVED_MODEL_PATH)          
            
            # Compute GE
            ge = results.ge(
                model=test_model,
                x_test=x_test,
                pltxt_bytes=pbs_test, 
                true_key_byte=tkb_test, 
                n_exp=100, 
                target=TARGET
            )
            ges.append(ge)

        ges = np.vstack(ges)
        avg_ge = np.mean(ges, axis=0)

        np.save(AVG_GE_FILE, avg_ge)


if __name__ == '__main__':
    start = time.time()
    main()
    print(f'Elapsed time: {((time.time() - start) / 3600):.2f} h')