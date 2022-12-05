# Basic
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
from data_loader import DataLoader, SplitDataLoader
import constants
import results
import helpers
import visualization as vis
import soa_reproduction as soa
sys.path.insert(0, '../modeling')
from network import Network

# Suppress TensorFlow messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # 1 for INFO, 2 for INFO & WARNINGs, 3 for INFO & WARNINGs & ERRORs


# Constants
BYTE = 5
TARGET = 'SBOX_OUT'
GE_EXP = 100
TOT_TRAIN = [100000, 200000]
VAL_SIZE = 5000
TOT_TEST = 50000
ATT_TRACES = 15 # Number of attack-traces to consider when plotting GEs


def main():

    _, n_devs = sys.argv
    n_devs = int(n_devs)

    if n_devs == 1:
        train_files = [f'{constants.PC_TRACES_PATH}/D1-{k}_500MHz + Resampled.trs'
                       for k in list(constants.KEYS)[1:]]
    else:
        train_files = [f'{constants.PC_TRACES_PATH}/{d}-{k}_500MHz + Resampled.trs'
                       for k in list(constants.KEYS)[1:]
                       for d in ['D1', 'D2']]

    test_file = [f'{constants.PC_TRACES_PATH}/D3-K0_500MHz + Resampled.trs']


    # Definition of constant paths
    RES_ROOT = f'{constants.RESULTS_PATH}/HPComparison/BiggerDatasets/{n_devs}d'
    SOA_HP = f'{constants.RESULTS_PATH}/HPComparison/TuningApproach/{n_devs}d/soa_hp.json'
    CSTM_HP = f'{constants.RESULTS_PATH}/DKTA/{TARGET}/byte{BYTE}/{n_devs}d/hp.json'


    # Retrieve HPs
    # SoA
    with open(SOA_HP, 'r') as jfile:
        soa_hp = json.load(jfile)
    # Custom
    with open(CSTM_HP, 'r') as jfile:
        cstm_hp = json.load(jfile)


    # Train and Test with different trainset-sizes
    soa_ges = []
    cstm_ges = []
    soa_times = []
    cstm_times = []
    
    for tot_train in tqdm(TOT_TRAIN, desc='Attacking: '):

        # Fix the size of validation set
        # This ensures to increase only the size of the actual train set
        train_perc = 1 - (VAL_SIZE / tot_train)


        # Get Data
        # Data-loading needs to be done inside the loop in order to ensure the 
        # even distribution of traces accross devices and keys
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
            test_file,
            tot_traces=TOT_TEST,
            target=TARGET,
            byte_idx=BYTE
        )
        x_test, _, pbs_test, tkb_test = test_dl.load()

        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_val = scaler.transform(x_val)
        x_test = scaler.transform(x_test)
        

        clear_session() # Start with a new Keras session every time

        # SoA
        soa_model = soa.build_model(soa_hp['layers'], soa_hp['neurons'])
        soa_start = time.time()
        _ = soa.fit(soa_model, x_train, y_train, x_val, y_val)
        soa_end = time.time()
        soa_ge = results.ge(
            model=soa_model,
            x_test=x_test,
            pltxt_bytes=pbs_test, 
            true_key_byte=tkb_test, 
            n_exp=GE_EXP, 
            target=TARGET
        )
        soa_ges.append(soa_ge)

        # Custom
        cstm_net = Network('MLP', cstm_hp)
        cstm_net.build_model()
        CSTM_MODEL = RES_ROOT + f'/model_{tot_train}.h5'
        cstm_net.add_checkpoint_callback(CSTM_MODEL)
        cstm_start = time.time()
        _ = cstm_net.model.fit(
            x_train, 
            y_train, 
            validation_data=(x_val, y_val),
            epochs=100,
            batch_size=cstm_net.hp['batch_size'],
            callbacks=cstm_net.callbacks,
            verbose=0
        )
        cstm_end = time.time()
        cstm_model = load_model(CSTM_MODEL)
        cstm_ge = results.ge(
            model=cstm_model,
            x_test=x_test,
            pltxt_bytes=pbs_test, 
            true_key_byte=tkb_test, 
            n_exp=GE_EXP, 
            target=TARGET
        )
        cstm_ges.append(cstm_ge)

        # Get train times
        soa_train_time = (soa_end - soa_start) / 60
        cstm_train_time = (cstm_end - cstm_start) / 60
        soa_times.append(soa_train_time)
        cstm_times.append(cstm_train_time)

    
    # Compare the results
    for tot_train, soa_ge, cstm_ge, soa_t, cstm_t in zip(TOT_TRAIN, soa_ges, cstm_ges, soa_times, cstm_times):

        COMP_FILE = RES_ROOT + f'/comparison_{tot_train}.csv'
        COMP_PLOT = RES_ROOT + f'/comparison_{tot_train}.svg'
        
        # Save Results
        plottable_soa_ge = soa_ge[:ATT_TRACES]
        plottable_cstm_ge = cstm_ge[:ATT_TRACES]
        comp_data = np.vstack(
            (
                np.arange(ATT_TRACES)+1, # X-axis values
                plottable_soa_ge, # Y-axis values for SoA
                plottable_cstm_ge # Y-axis values for Custom
            )
        ).T
        helpers.save_csv(
            data=comp_data, 
            columns=['AttackTraces', 'SoA', 'GenAlg'],
            output_path=COMP_FILE
        )

        # Plot GE Comparison
        vis.plot_soa_vs_custom(
            soa_ge=plottable_soa_ge,
            custom_ge=plottable_cstm_ge,
            threshold=0.5,
            title=f'HP Tuning Approach: State-of-the-Art vs Genetic Algorithm  | Train-Devices: {n_devs}  |  Tot Traces: {int(tot_train/1000)}k',
            ylim_max=50,
            output_path=COMP_PLOT
        )

        # Print train times
        print(f'SoA train-time with {int(tot_train/1000)}k traces:    {soa_t:.2f} min')
        print(f'Genetic Algorithm train-time with {int(tot_train/1000)}k traces: {cstm_t:.2f} min')
        print()


if __name__ == '__main__':
    main()