# Basics
import numpy as np
from tqdm import tqdm
import random
import time
from datetime import datetime
import json

# Tensorflow/Keras
from tensorflow.keras.backend import clear_session
from tensorflow.keras.utils import to_categorical

# sklearn
from sklearn.model_selection import KFold

# Custom
import sys
sys.path.insert(0, '../utils')
from lazy_data_loader import DataLoader
import constants
import aes
sys.path.insert(0, '../modeling')
from network import Network
import evaluation as ev
import visual as vis

# Suppress TensorFlow messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # 1 for INFO, 2 for INFO & WARNINGs, 3 for INFO & WARNINGs & ERRORs


# Global variables (constants)
BYTE_IDX = 0
N_FOLDS = 10
N_MODELS = 10
TRAIN_EPOCHS = 100
TEST_EPOCHS = 100
GE_VAL_TR = 500
HP_CHOICES = {#'kernel_initializer': ['random_normal', 'random_uniform', 
              #                       'truncated_normal', 
              #                       'zeros', 'ones', 
              #                       'glorot_normal', 'glorot_uniform',
              #                       'he_normal', 'he_uniform',
              #                       'identity', 'orthogonal', 'constant', 'variance_scaling'],
              'kernel_initializer': ['random_normal', 'random_uniform', 'truncated_normal', 'he_normal', 'he_uniform'],  
              'activation':         ['relu', 'tanh'],
              'hidden_layers':      [1, 2, 3, 4, 5, 6],
              'hidden_neurons':     [100, 200, 300, 400, 500, 600],
              'dropout_rate':       [0.0, 0.2, 0.4],
              'optimizer':          ['sgd', 'adam', 'rmsprop'],
              'learning_rate':      [1e-3, 1e-4, 1e-5, 1e-6],
              'batch_size':         [50, 100, 200, 500, 1000]}


def main():
    
    train_dk = sys.argv[1].upper() # Di-Kj
    model_type = sys.argv[2].upper() # ['MLP', 'CNN']
    val_metric = sys.argv[3] # ['acc', 'ge', 'geiter']

    # Create the train DataLoader
    path = constants.CURR_DATASETS_PATH + f'/SBOX_OUT/{train_dk}_train.json'
    train_dl = DataLoader(path, BYTE_IDX)
    
    # Create the Networks
    networks = [Network(model_type) for _ in range(N_MODELS)]
    
    # Create KFold
    kf = KFold(n_splits=N_FOLDS)

    # Build a "metadata package" for training
    train_data = [HP_CHOICES, TRAIN_EPOCHS]

    if val_metric == 'acc':
        best_hp = ev.xval_acc(kf, networks, train_dl, train_data)
    else:
        train_data.append(GE_VAL_TR)
        key_bytes_train = np.array([aes.key_from_labels(pb, 'SBOX_OUT') for pb in pltxt_train])
        train_data.append(key_bytes_train)
        if val_metric == 'ge':
            best_hp = ev.xval_ge(kf, networks, train_dl, train_data) ################## to modify
        elif val_metric == 'geiter':
            best_hp = ev.xval_ge_iter(kf, networks, train_dl, train_data) ################# to modify
        else:
            print('ERROR: wrong val metric value')
            return


    # Attack with the best hyperparameters
    ge_per_testset = []
    
    x_train, y_train, _, _ = train_dl.load_data()

    attack_net = Network(model_type)
    attack_net.set_hp(best_hp)
    attack_net.build_model()
            
    print()
    print('Training the attack model...')
    attack_net.model.fit(x_train,
                         y_train,
                         epochs=TEST_EPOCHS,
                         verbose=0)
    print('Training completed.')

    for d in constants.DEVICES:
        for k in constants.KEYS:
            print()        
            print(f'----- Attack set {d}-{k} -----')
        
            # Define the current test DataLoader
            path = constants.CURR_DATASETS_PATH + f'/SBOX_OUT/{d}-{k}_test.json'
            test_dl = DataLoader(path, BYTE_IDX)
            
            ge = ev.guessing_entropy(attack_net.model, 
                                     test_dl=test_dl,
                                     n_exp=10,
                                     n_traces=2000)
            ge_per_testset.append(ge)

    ge_per_testset = np.array(ge_per_testset)
    scores = np.array([ev.ge_score(ge, n_zeros=10) 
                       for ge in ge_per_testset])


    date = datetime.now().strftime("%m%d%Y-%I%M%p")

    # Save best hyperparameters
    with open(constants.RESULTS_PATH + f'/models/hp_{date}.json', 'w') as j_file:
        json.dump(best_hp, j_file)
    
    # Save attack model 
    attack_net.save_model(constants.RESULTS_PATH + f'/models/model_{date}')

    # Save GE values
    np.savetxt(constants.RESULTS_PATH + f'/ge/ge_files/ge_{date}.csv', ge_per_testset, delimiter=',')

    # Save GE scores
    np.savetxt(constants.RESULTS_PATH + f'/scores/score_files/scores_{date}.csv', scores, delimiter=',')


    # Plot the results
    filename_data = (val_metric, train_dk, date)
    vis.plot_ges(ge_per_testset, n=30, metadata=filename_data)
    vis.plot_ges(ge_per_testset, n=30, metadata=filename_data, subplots=True)
    vis.plot_ges(ge_per_testset, n=500, metadata=filename_data)
    vis.plot_ges(ge_per_testset, n=len(ge_per_testset[0]), metadata=filename_data)



if __name__ == '__main__':
    
    start = time.time()
    main()
    end = time.time()

    print()
    print(f'Script completed in {((end-start)/3600):.2f} hours')
