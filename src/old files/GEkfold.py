# Basics
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from tqdm import tqdm
import random

# Tensorflow/Keras
from tensorflow.keras.optimizers import SGD, Adam, RMSprop

# sklearn
from sklearn.model_selection import KFold

# Custom
import sys
sys.path.insert(0, '../utils')
from data_loader import DataLoader
import constants
import aes
sys.path.insert(0, '../modeling')
from network import Network
from evaluator import Evaluator
from ge import compute_key_probs

# Suppress TensorFlow messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # 1 for INFO, 2 for INFO & WARNINGs, 3 for INFO & WARNINGs & ERRORs


def ge_score(ge, n=10):

    iszero = ge==0
    
    if len(iszero) == 0:
        return len(ge)

    tmp = np.concatenate(([0], iszero, [0])) # generate a vector with 1s where ge is 0 (all other elements are 0)
    start_end_zeros = np.abs(np.diff(tmp)) # generate a vector with 1s only at start_idx, end_idx of a 0-sequence in ge
    
    zero_slices = np.where(start_end_zeros==1)[0].reshape(-1, 2)
    for el in zero_slices:
        diff = el[1] - el[0]
        if diff >= n:
            return el[0]
        
    return len(ge)


def plot_ge_all(ges_single, ges_diff, n=30):
    f, ax = plt.subplots(2, 1, figsize=(20,15))

    device = 1
    for i in range(len(ges_single)):
    
        key = i % 3
    
        ax[0].plot(ges_single[i][:n], marker='o', label=f'D{device}-K{key+1}')
        ax[1].plot(ges_diff[i][:n], marker='o', label=f'D{device}-K{key+1}')
    
        if key == 2:
            device += 1

    ax[0].plot(np.zeros(n), color='r', ls='--')
    ax[0].set_title('GE SINGLE MODEL: D1-K1 for training')
    ax[0].set_xlabel('Number of Traces')
    ax[0].set_ylabel('Guessing Entropy')
    ax[0].set_xticks(range(n))
    ax[0].grid()
    ax[0].legend()

    ax[1].plot(np.zeros(n), color='r', ls='--')
    ax[1].set_title('GE INDEPENDENT MODELS: D1-K1 for training')
    ax[1].set_xlabel('Number of Traces')
    ax[1].set_ylabel('Guessing Entropy')
    ax[1].set_xticks(range(n))
    ax[1].grid()
    ax[1].legend()

    f.savefig(f'../../MDM32/notebooks/images/GE_ALL_{n}.png', bbox_inches='tight', dpi=600)


def plot_ge_subplots(ges, name):
    f, ax = plt.subplots(len(constants.DEVICES), len(constants.KEYS), figsize=(30,15))
    
    row = 0
    for i in range(len(ges)):
    
        col = i % 3
    
        ax[row, col].plot(ges[i][:30], marker='o', color=list(colors.TABLEAU_COLORS.keys())[i])
        ax[row, col].plot(np.zeros(30), color='r', ls='--')
        ax[row, col].set_title(f'GE SINGLE MODEL: D{row+1}-K{col+1}')
        ax[row, col].set_xlabel('Number of Traces')
        ax[row, col].set_ylabel('Guessing Entropy')
        ax[row, col].grid()
        ax[row, col].set_xticks(range(30))
    
        if col == 2:
            row += 1

    f.savefig(f'../../MDM32/notebooks/images/GE_SUBPLOTS_{name}.png', bbox_inches='tight', dpi=600)


def main():
    BYTE_IDX = 0
    N_MODELS = 30
    N_EXP = 10
    EPOCHS = 300
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
                  'optimizer':          [SGD, Adam, RMSprop],
                  'learning_rate':      [1e-3, 1e-4, 1e-5, 1e-6],
                  'batch_size':         [50, 100, 200, 500, 1000]}

    # Load train data
    train_dl = DataLoader('/prj/side_channel/PinataTraces/datasets/SBOX_OUT/D1-K1.json', BYTE_IDX)
    x_train, y_train, pltxt_train = train_dl.gen_train() # Default 80% train (40,000 train traces)
    key_bytes_train = np.array([aes.key_from_labels(pb, 'SBOX_OUT') for pb in pltxt_train])
    true_kb_train = train_dl.get_true_key_byte()


    # K-Fold CrossValidation with GE as metric
    kf = KFold(n_splits=N_EXP)
    
    hps = []
    results = []
    for i in range(N_MODELS):
        print(f'------------------------------ Model {i+1}/{N_MODELS} ------------------------------')
    
        random_hp = {k: random.choice(HP_CHOICES[k]) for k in HP_CHOICES}
        hps.append(random_hp)
    
        net = Network('MLP')
        net.set_hp(random_hp)
    
        ranks_per_exp = []
        for e, (train_indices, val_indices) in tqdm(enumerate(kf.split(x_train))):
        
            net.build_model()
        
            x_t = x_train[train_indices]
            y_t = y_train[train_indices]
        
            x_v = x_train[val_indices]
            y_v = y_train[val_indices]
            kb_v = key_bytes_train[val_indices]
        
            net.train_model(x_t, y_t, epochs=EPOCHS)
            probs = net.predict(x_v)
        
            key_probs = compute_key_probs(probs, kb_v)
            log_probs = np.log10(key_probs + 1e-22)
            cum_tot_probs = np.cumsum(log_probs, axis=0)
        
            indexed_cum_tot_probs = [list(zip(range(256), tot_probs)) 
                                     for tot_probs in cum_tot_probs]
            sorted_cum_tot_probs = [sorted(el, key=lambda x: x[1], reverse=True) 
                                    for el in indexed_cum_tot_probs]
            sorted_key_bytes = [[el[0] for el in tot_probs] 
                                for tot_probs in sorted_cum_tot_probs]

            true_key_byte_ranks = [skb.index(true_kb_train) 
                                   for skb in sorted_key_bytes]
            true_key_byte_ranks = np.array(true_key_byte_ranks)
            ranks_per_exp.append(true_key_byte_ranks)
        
            net.reset_model()

        ranks_per_exp = np.array(ranks_per_exp)
        ge = np.mean(ranks_per_exp, axis=0)
    
        score = ge_score(ge)
        results.append((i, score))
    
        print(f'Score: {score}')
        print()

    
    # Sort the hyperparameters w.r.t. the score they achieved
    results.sort(key=lambda x: x[1])
    print(f'K-Fold Crossvalidation Results: {[(idx, metric) for idx, metric in results]}')

    # Select the best hp configuration
    best_hp = hps[results[0][0]]


    ges_single = []
    ges_diff = []

    for i, device in enumerate(constants.DEVICES):
        for j, key in enumerate(constants.KEYS):
        
            print(f'----- {device}-{key} -----')
        
            if device == 'D1' and key == 'K1':
                path = '/prj/side_channel/PinataTraces/datasets/SBOX_OUT/D1-K1.json'
            else:
                path = f'/prj/side_channel/PinataTraces/datasets/SBOX_OUT/{device}-{key}.json'

            test_dl = DataLoader(path, BYTE_IDX)
            x_test, y_test, pltxt_test = test_dl.gen_test()
            true_key_byte = test_dl.get_true_key_byte()
        
            evaluator = Evaluator(x_test, pltxt_test, true_key_byte, 'MLP')

            ge_single_model = evaluator.guessing_entropy(n_exp=10,
                                                         hp=best_hp,
                                                         x_train=x_train,
                                                         y_train=y_train,
                                                         epochs=200,
                                                         single_model=True)
            ges_single.append(ge_single_model)
        
            print()
        
            ge_diff_models = evaluator.guessing_entropy(n_exp=10,
                                                        hp=best_hp,
                                                        x_train=x_train,
                                                        y_train=y_train,
                                                        epochs=200,
                                                        single_model=False)
            ges_diff.append(ge_diff_models)
       
            print()

    ges_single = np.array(ges_single)
    ges_diff = np.array(ges_diff)


    # Plot
    plot_ge_all(ges_single, ges_diff, n=30)
    plot_ge_subplots(ges_single, name='onlyTest')
    plot_ge_subplots(ges_diff, name='trainTest')
    plot_ge_all(ges_single, ges_diff, n=2000)


if __name__ == '__main__':
    main()
