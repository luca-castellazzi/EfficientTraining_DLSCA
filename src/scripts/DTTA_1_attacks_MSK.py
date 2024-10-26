# Basics
import time
import numpy as np
from tqdm import tqdm
from tensorflow.keras.models import load_model
from tensorflow.keras.backend import clear_session
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Custom
import sys
sys.path.insert(0, '../utils')
import constants
import results
from data_loader import DataLoader, SplitDataLoader
sys.path.insert(0, '../modeling')
from models import msk_mlp

# Suppress TensorFlow messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # 1 for INFO, 2 for INFO & WARNINGs, 3 for INFO & WARNINGs & ERRORs

BYTE = 11
TARGET = 'SBOX_OUT'
EPOCHS = 100
ADD_TRACES = 50000
TOT = range(50000, 400000, ADD_TRACES) # 50k, 100k, 150k, 200k, 250k, 300k, 350k
VAL_TOT = 5000

TR_ORIGINAL_LEN = 8736
STOP_SAMPLE = 7700


def main():

    # To proper run this script "as is", copy-paste the .h5 model files from DKTA to this script's 50k_traces folder
    # They will be used directly without further training
    # And they will be improved for additional number of traces 

    _, n_devs = sys.argv
    n_devs = int(n_devs)

    # Set the number of keys to use (given by DKTA)
    if n_devs == 1:
        N_KEYS = 9 # 9 for both byte 5 and 11
    else:
        N_KEYS = 9 # 7 for byte 5, 9 for byte 11

    # Definition of constant root path 
    RES_ROOT = f'{constants.RESULTS_PATH}/DTTA/msk/byte{BYTE}/{n_devs}d'

    for i, tot in enumerate(TOT):

        ges = []

        AVG_GE_FILE = RES_ROOT + f'/avg_ge_{int(tot / 1000)}k.npy'

        for train_devs, test_dev in tqdm(constants.PERMUTATIONS[n_devs], desc=f'{int(tot / 1000)}k Traces: '):

            train_files = [f'{constants.MSK_PC_TRACES_PATH}/second_order/{dev}-{k} + Resampled.trs' 
                           for k in list(constants.KEYS)[1:N_KEYS+1]
                           for dev in train_devs]
            test_files = [f'{constants.MSK_PC_TRACES_PATH}/second_order/{test_dev}-K0 + Resampled.trs']

            TRACES_FOLDER = RES_ROOT + f'/{int(tot / 1000)}k_traces'
            if not os.path.exists(TRACES_FOLDER):
                os.mkdir(TRACES_FOLDER)
            NEW_MODEL_PATH = TRACES_FOLDER + f'/model_{"".join(train_devs)}vs{test_dev}.h5'

            train_dl = SplitDataLoader(
                train_files, 
                tot_traces=ADD_TRACES,
                train_size=1-(VAL_TOT/ADD_TRACES),
                target=TARGET,
                start_tr_idx=i,
                byte_idx=BYTE,
                stop_sample=STOP_SAMPLE
            )
            train_data, val_data = train_dl.load()
            x_train, y_train, _, _ = train_data 
            x_val, y_val, _, _ = val_data # Val data is kept to consider callbacks

            # Scale data to 0-mean and 1-variance
            scaler = StandardScaler()
            scaler.fit(x_train)
            x_train = scaler.transform(x_train) 
            x_val = scaler.transform(x_val)   

            clear_session() # Start with a new Keras session every time    

            # Continue the training only if the considered number of traces exceeds 50k
            if tot > ADD_TRACES:

                # Load the previous model to continue training
                PREV_MODEL_PATH = RES_ROOT + f'/{int(TOT[i-1] / 1000)}k_traces/model_{"".join(train_devs)}vs{test_dev}.h5'
                model = load_model(PREV_MODEL_PATH)

                callbacks = [
                    EarlyStopping(
                        monitor='val_loss', 
                        patience=15
                    ),
                    ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.2,
                        patience=7,
                        min_lr=1e-7),
                    ModelCheckpoint(
                        NEW_MODEL_PATH,
                        monitor='val_loss',
                        save_best_only=True
                    )
                ]
                
                model.fit(
                    x_train,
                    y_train,
                    validation_data=(x_val, y_val),
                    epochs=EPOCHS,
                    callbacks=callbacks,
                    verbose=0
                )

            attack_model = load_model(NEW_MODEL_PATH)  

            # Testing (every time consider random test traces from the same set)
            test_dl = DataLoader(
                test_files, 
                tot_traces=ADD_TRACES,
                target=TARGET,
                byte_idx=BYTE,
                start_sample=0,
                stop_sample=STOP_SAMPLE
            )
            x_test, _, pbs_test, tkb_test = test_dl.load()

            # Scale test data to 0-mean and 1-variance w.r.t. train data
            x_test = scaler.transform(x_test)        
            
            # Compute GE
            ge = results.ge(
                model=attack_model,
                x_test=x_test,
                pltxt_bytes=pbs_test, 
                true_key_byte=tkb_test, 
                n_exp=100, 
                target=TARGET
            )

            SINGLE_GE_FILE = TRACES_FOLDER + f'/ge_{"".join(train_devs)}vs{test_dev}.npy'
            np.save(SINGLE_GE_FILE, ge)

            ges.append(ge)

        ges = np.vstack(ges)
        avg_ge = np.mean(ges, axis=0)

        np.save(AVG_GE_FILE, avg_ge)


if __name__ == '__main__':
    start = time.time()
    main()
    print(f'Elapsed time: {((time.time() - start) / 3600):.2f} h')