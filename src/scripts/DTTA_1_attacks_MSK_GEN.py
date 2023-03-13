# Basics
import json
import time
import numpy as np
from tqdm import tqdm
from tensorflow.keras.models import load_model
from tensorflow.keras.backend import clear_session
from tensorflow.keras.metrics import TopKCategoricalAccuracy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Custom
import sys
sys.path.insert(0, '../utils')
import constants
import results
from batch_scalers import BatchStandardScaler
from data_generator import DataGenerator
from data_loader import DataLoader
sys.path.insert(0, '../modeling')
from models import msk_mlp

# Suppress TensorFlow messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # 1 for INFO, 2 for INFO & WARNINGs, 3 for INFO & WARNINGs & ERRORs

BYTE = 5
TARGET = 'SBOX_OUT'
EPOCHS = 100
TOT = range(50000, 400000, 50000) # 50k, 100k, 150k, 200k, 250k, 300k, 350k
VAL_TOT = 5000
TOT_TEST = 50000

TR_ORIGINAL_LEN = 8736
STOP_SAMPLE = 7700


def main():

    _, n_devs = sys.argv
    n_devs = int(n_devs)

    # Set the number of keys to use (given by DKTA)
    if n_devs == 1:
        N_KEYS = 9
    else:
        N_KEYS = 7 # This fixes the max amount of traces available (50k * 7 = 350k)

    # Get HPs
    HP_PATH = f'{constants.RESULTS_PATH}/DKTA/{TARGET}/byte{BYTE}/msk_{n_devs}d/hp.json'
    with open(HP_PATH, 'r') as jfile:
        hp = json.load(jfile)

    # Definition of constant root path 
    RES_ROOT = f'{constants.RESULTS_PATH}/DTTA/msk/{n_devs}d'

    for tot in TOT:

        train_tot = tot - VAL_TOT

        ges = []

        AVG_GE_FILE = RES_ROOT + f'/avg_ge_{int(tot / 1000)}k'

        for train_devs, test_dev in tqdm(constants.PERMUTATIONS[n_devs], desc=f'{int(tot / 1000)}k Traces: '):

            train_files = [f'{constants.MSK_PC_TRACES_PATH}/second_order/{dev}-{k} + Resampled.trs' 
                           for k in list(constants.KEYS)[1:N_KEYS+1]
                           for dev in train_devs]
            test_files = [f'{constants.MSK_PC_TRACES_PATH}/second_order/{test_dev}-K0 + Resampled.trs']

            TRACES_FOLDER = RES_ROOT + f'/{int(tot / 1000)}k_traces'
            if not os.path.exists(TRACES_FOLDER):
                os.mkdir(TRACES_FOLDER)
            SAVED_MODEL_PATH = TRACES_FOLDER + f'/model_{"".join(train_devs)}vs{test_dev}.h5'

            batch_scaler = BatchStandardScaler(
                tr_files=train_files,
                tr_tot=train_tot,
                tr_original_len=TR_ORIGINAL_LEN,
                batch_size=hp['batch_size'],
                stop_sample=STOP_SAMPLE
            )
            batch_scaler.fit()

            train_indices = range(train_tot)
            val_indices = range(train_tot, train_tot+VAL_TOT)

            train_gen = DataGenerator(
                tr_files=train_files,
                tr_indices=train_indices,
                target=TARGET,
                byte_idx=BYTE,
                scaler=batch_scaler,
                batch_size=hp['batch_size'],
                stop_sample=STOP_SAMPLE
            )

            val_gen = DataGenerator(
                tr_files=train_files,
                tr_indices=val_indices,
                target=TARGET,
                byte_idx=BYTE,
                scaler=batch_scaler,
                batch_size=hp['batch_size'],
                stop_sample=STOP_SAMPLE
            )


            clear_session() # Start with a new Keras session every time    

            # Train and Attack
            metrics = [
                'accuracy',
                TopKCategoricalAccuracy(k=10, name='topK')
            ] 

            model = msk_mlp(
                hp=hp, 
                input_len=STOP_SAMPLE, 
                n_classes=constants.N_CLASSES[TARGET],
                metrics=metrics
            )

            #Training (with Validation)
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
                    SAVED_MODEL_PATH,
                    monitor='val_loss',
                    save_best_only=True
                )
            ]
            
            model.fit(
                train_gen,
                validation_data=val_gen,
                epochs=EPOCHS,
                callbacks=callbacks,
                verbose=0
            )

            attack_model = load_model(SAVED_MODEL_PATH)  

            # Testing (every time consider random test traces from the same set)
            test_dl = DataLoader(
                test_files, 
                tot_traces=tot,
                target=TARGET,
                byte_idx=BYTE,
                start_sample=0,
                stop_sample=STOP_SAMPLE
            )
            x_test, _, pbs_test, tkb_test = test_dl.load()

            # Scale test data to 0-mean and 1-variance w.r.t. train data
            x_test = batch_scaler.transform(x_test)        
            
            # Compute GE
            ge = results.ge(
                model=attack_model,
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