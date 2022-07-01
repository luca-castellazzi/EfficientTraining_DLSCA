# Basics
import numpy as np
import matplotlib
matplotlib.use('pdf') # Avoid interactive mode
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from tqdm import tqdm
import random

# Tensorflow/Keras
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.backend import clear_session

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


# Global variables (constants)
BYTE_IDX = 0
N_FOLDS = 10
N_MODELS = 30
EPOCHS = 300
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
              'optimizer':          [SGD, Adam, RMSprop],
              'learning_rate':      [1e-3, 1e-4, 1e-5, 1e-6],
              'batch_size':         [50, 100, 200, 500, 1000]}


# Functions
def load_train():

    train_dl = DataLoader('/prj/side_channel/PinataTraces/datasets/SBOX_OUT/D1-K1.json', BYTE_IDX)
    x_train, y_train, pltxt_train = train_dl.gen_set(train=True) # Default 80% train (40,000 train traces)
    key_bytes_train = np.array([aes.key_from_labels(pb, 'SBOX_OUT') for pb in pltxt_train])
    true_kb_train = train_dl.get_true_key_byte()

    return x_train, y_train, key_bytes_train, true_kb_train


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
        
    return len(ge) - 1


# KFold XVal with GE as metric
# GE is computed for N exp for every val set
def kfold_crossval_ge(x_train, y_train, key_bytes_train, true_kb_train):
    
    kf = KFold(n_splits=N_FOLDS)
    
    nets = [Network('MLP') for _ in range(N_MODELS)]
    results = []
    for i, net in enumerate(nets):
        
        print()
        print(f'HP Config {i+1}/{N_MODELS}...')
        
        random_hp = {k: random.choice(HP_CHOICES[k]) for k in HP_CHOICES}
        net.set_hp(random_hp)
        
        scores = []
        for train_indices, val_indices in tqdm(kf.split(x_train)):
            
            net.build_model()
        
            x_t = x_train[train_indices]
            y_t = y_train[train_indices]
        
            x_v = x_train[val_indices]
            y_v = y_train[val_indices]
            kb_v = key_bytes_train[val_indices]
            
            x_kb_v = list(zip(x_v, kb_v))

            net.train_model(x_t, 
                            y_t, 
                            epochs=EPOCHS,
                            cb={'es': True,'reduceLR': True},
                            validate=True,
                            x_val=x_v,
                            y_val=y_v)
            
            ranks_per_exp = []
            for _ in range(10):
                random.shuffle(x_kb_v)
                shuffled_x_v, shuffled_kb_v = list(zip(*x_kb_v))

                probs = net.predict(np.array(shuffled_x_v))
        
                key_probs = compute_key_probs(probs[:GE_VAL_TR], shuffled_kb_v[:GE_VAL_TR])
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
        
            ranks_per_exp = np.array(ranks_per_exp)
            ge = np.mean(ranks_per_exp, axis=0)
            
            scores.append(ge_score(ge))

            net.reset()
            clear_session()

    
        avg_score = np.mean(np.array(scores))
        print(f'Score: {avg_score}')
        results.append((i, avg_score))
    
    # Sort the hyperparameters w.r.t. the score they achieved
    results.sort(key=lambda x: x[1])
    print()
    print(f'K-Fold Crossvalidation Results: {[(idx, metric) for idx, metric in results]}')

    # Select the best hp configuration
    best_net = nets[results[0][0]]
    
    return best_net.get_hp()


# KFold XVal with GE as metric
# GE is computed exploiting the iterations of KFold XVal
def kfold_crossval_ge_iter(x_train, y_train, key_bytes_train, true_kb_train):
    
    kf = KFold(n_splits=N_FOLDS)
    
    nets = [Network('MLP') for _ in range(N_MODELS)]
    results = []
    for i, net in enumerate(nets):
        
        print()
        print(f'HP Config {i+1}/{N_MODELS}...')
        
        random_hp = {k: random.choice(HP_CHOICES[k]) for k in HP_CHOICES}
        net.set_hp(random_hp)
        
        ranks_per_exp = []
        for train_indices, val_indices in tqdm(kf.split(x_train)):
            
            net.build_model()
        
            x_t = x_train[train_indices]
            y_t = y_train[train_indices]
        
            x_v = x_train[val_indices]
            y_v = y_train[val_indices]
            kb_v = key_bytes_train[val_indices]
        
            net.train_model(x_t, 
                            y_t, 
                            epochs=EPOCHS,
                            cb={'es': True,'reduceLR': True},
                            validate=True,
                            x_val=x_v,
                            y_val=y_v)
            probs = net.predict(x_v)
        
            key_probs = compute_key_probs(probs[:GE_VAL_TR], kb_v[:GE_VAL_TR])
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
        
            net.reset()
            clear_session()

        ranks_per_exp = np.array(ranks_per_exp)
        ge = np.mean(ranks_per_exp, axis=0)
    
        score = ge_score(ge)
        print(f'Score: {score}')
        results.append((i, score))
    
    # Sort the hyperparameters w.r.t. the score they achieved
    results.sort(key=lambda x: x[1])
    print()
    print(f'K-Fold Crossvalidation Results: {[(idx, metric) for idx, metric in results]}')

    # Select the best hp configuration
    best_net = nets[results[0][0]]
    
    return best_net.get_hp()


def test(x_train, y_train, best_hp):

    ges = []

    for i, device in enumerate(constants.DEVICES):
        for j, key in enumerate(constants.KEYS):
            
            print()        
            print(f'----- {device}-{key} -----')
        
            if device == 'D1' and key == 'K1':
                path = '/prj/side_channel/PinataTraces/datasets/SBOX_OUT/D1-K1.json'
            else:
                path = f'/prj/side_channel/PinataTraces/datasets/SBOX_OUT/{device}-{key}.json'

            test_dl = DataLoader(path, BYTE_IDX)
            x_test, y_test, pltxt_test = test_dl.gen_set(train=False)
            true_key_byte = test_dl.get_true_key_byte()
        
            evaluator = Evaluator(x_test, pltxt_test, true_key_byte, 'MLP')

            ge = evaluator.guessing_entropy(n_exp=10,
                                            hp=best_hp,
                                            x_train=x_train,
                                            y_train=y_train,
                                            epochs=EPOCHS,
                                            single_model=True)
            ges.append(ge)
    
    return np.array(ges)


def final_scores(ges):
    print()
    print('Computing the scores...')
    scores = [ge_score(ge) + 1 for ge in ges]
    print(f'Scores per Device-Key Config: {scores}') 


def plot_all_config(ges, n=30):
    f, ax = plt.subplots(figsize=(10,5))

    device = 1
    for i in range(len(ges)):
    
        key = i % 3
    
        if n <= 100:
            ax.plot(ges[i][:n], marker='o', label=f'D{device}-K{key+1}')
        else:
            ax.plot(ges[i][:n], label=f'D{device}-K{key+1}')

        if key == 2:
            device += 1

    ax.plot(np.zeros(n), color='r', ls='--')
    ax.set_title('GE FOR ALL CONFIGS (D1-K1 for training)')
    ax.set_xlabel('Number of Traces')
    ax.set_ylabel('Guessing Entropy')
    if n <= 100:
        ax.set_xticks(ticks=range(n), labels=range(1, n+1))
        ax.grid()
    ax.legend()

    f.savefig(f'/home/lcastellazzi/MDM32/notebooks/images/GE_ALL_CONFIG_{n}.png', bbox_inches='tight', dpi=600)


def plot_subplots(ges):
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
        ax[row, col].set_xticks(range(30), labels=range(1, 31))
    
        if col == 2:
            row += 1

    f.savefig(f'/home/lcastellazzi/MDM32/notebooks/images/GE_SUBPLOTS.png', bbox_inches='tight', dpi=600)



def main():

    # Data loading
    x_train, y_train, key_bytes_train, true_kb_train = load_train()
    
    # K-Fold CrossValidation for hp tuning
    best_hp = kfold_crossval_ge(x_train, y_train, key_bytes_train, true_kb_train)
    #best_hp = kfold_crossval_ge_iter(x_train, y_train, key_bytes_train, true_kb_train)

    # Testing
    ges = test(x_train, y_train, best_hp)

    # Results visualization
    final_scores(ges)
    plot_all_config(ges, n=30)
    plot_all_config(ges, n=500)
    plot_all_config(ges, n=2000)
    plot_subplots(ges)


if __name__ == '__main__':
    main()
