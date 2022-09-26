# Basics
import numpy as np
import random
import json
from tensorflow.keras.backend import clear_session
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Custom
import sys
sys.path.insert(0, '../../utils')
from data_loader import DataLoader, SplitDataLoader
import constants
import results
sys.path.insert(0, '../../modeling')
from hp_tuner import HPTuner
from network import Network

# Suppress TensorFlow messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # 1 for INFO, 2 for INFO & WARNINGs, 3 for INFO & WARNINGs & ERRORs

MAX_TRACES = 50000
EPOCHS = 100
n_devs = 2
model_type = 'MLP'
target = 'KEY'
b = 0


def main():
    n_tot_traces = n_devs * MAX_TRACES
    dev_permutations = constants.PERMUTATIONS[n_devs]

    RES_ROOT = f'{constants.RESULTS_PATH}/DKTA/{target}/{n_devs}d'
#    HP_PATH = RES_ROOT + '/hp.json'
#
#
#    with open(HP_PATH, 'r') as jfile:
#        hp = json.load(jfile)
#
#
#    for train_devs, test_dev in dev_permutations[:1]:
#
#        GES_FILE_PATH = RES_ROOT + f'/ges_{"".join(train_devs)}vs{test_dev}.npy'
#
#        test_config = f'{test_dev}-K0'
#
#        ges = []
#
#        for n_keys in range(10, len(constants.KEYS)):
#
#            N_KEYS_FOLDER = RES_ROOT + f'/{n_keys}k'
#            if not os.path.exists(N_KEYS_FOLDER):
#                os.mkdir(N_KEYS_FOLDER)
#            SAVED_MODEL_PATH = N_KEYS_FOLDER + f'/model_{"".join(train_devs)}vs{test_dev}.h5'
#            CONF_MATRIX_PATH = N_KEYS_FOLDER + f'/conf_matrix_{"".join(train_devs)}vs{test_dev}.png'
#
#
#            print(f'===== {"".join(train_devs)}vs{test_dev} | Number of keys: {n_keys}  =====')
#
#            train_configs = [f'{dev}-{k}' for k in list(constants.KEYS)[1:n_keys+1]
#                             for dev in train_devs]
#            train_dl = SplitDataLoader(
#                train_configs,
#                n_tot_traces=n_tot_traces,
#                train_size=0.9,
#                target=target,
#                byte_idx=b
#            )
#            train_data, val_data = train_dl.load()
#            x_train, y_train, _, _ = train_data
#            x_val, y_val, _, _ = val_data # Val data is kept to consider callbacks
#
#            attack_net = Network(model_type, hp)
#            attack_net.build_model()
#            attack_net.add_checkpoint_callback(SAVED_MODEL_PATH)
#            train_model = attack_net.model
#
#            #Training (with Validation)
#            train_model.fit(
#                x_train,
#                y_train,
#                validation_data=(x_val, y_val),
#                epochs=EPOCHS,
#                batch_size=attack_net.hp['batch_size'],
#                callbacks=attack_net.callbacks,
#                verbose=0
#            )
#
#            # Testing (every time consider random test traces from the same set)
#            test_dl = DataLoader(
#                [test_config],
#                n_tot_traces=5000,
#                target=target,
#                byte_idx=b
#            )
#            x_test, y_test, pbs_test, tkb_test = test_dl.load()
#
#            test_model = load_model(SAVED_MODEL_PATH)
#            preds = test_model.predict(x_test)
#
#            # Compute GE
#            ge = results.ge(
#                preds=preds,
#                pltxt_bytes=pbs_test,
#                true_key_byte=tkb_test,
#                n_exp=100,
#                target=target,
#                n_traces=100 # Default: 500
#            )
#            ges.append(ge)
#
#
#            clear_session()
#
#
#        ges = np.array(ges)
#        np.save(ges_file_path, ges)


    SAVED_MODEL_PATH = RES_ROOT + '/10k/model_D1D2vsD3.h5'
    GE_PLOT_PATH = RES_ROOT + '/10k/ge_plot.png'

    # Testing (every time consider random test traces from the same set)
    test_dl = DataLoader(
        ['D3-K0'],
        n_tot_traces=5000,
        target=target,
        byte_idx=b
    )
    x_test, y_test, pbs_test, tkb_test = test_dl.load()

    test_model = load_model(SAVED_MODEL_PATH)
    preds = test_model.predict(x_test)

    prods = np.cumprod(preds, axis=0)
    logsum = np.cumsum(np.log10(preds + 1e-22), axis=0)
    
    res_p = np.argmax(prods, axis=1)
    res_ls = np.argmax(logsum, axis=1)
    print('PROD')
    print(res_p)
    print('LOG_SUM')
    print(res_ls)
    print('---')
    print(tkb_test)
#    # Compute GE
#    ge = results.ge(
#        preds=preds,
#        pltxt_bytes=pbs_test,
#        true_key_byte=tkb_test,
#        n_exp=100,
#        target=target,
#        n_traces=100 # Default: 500
#    )
#
#    plt.plot(ge)
#    plt.show()

    plt.plot(res_p)
    plt.plot(res_ls)
    plt.show()


if __name__ == '__main__':
    main()
