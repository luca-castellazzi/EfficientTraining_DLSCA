# Basics
import json
import time
import statistics
from tqdm import tqdm
from sklearn.model_selection import KFold

# Custom
import sys
sys.path.insert(0, '../utils')
from data_loader import DataLoader
import constants
# import visualization as vis
sys.path.insert(0, '../modeling')
from hp_tuner import HPTuner

# Suppress TensorFlow messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # 1 for INFO, 2 for INFO & WARNINGs, 3 for INFO & WARNINGs & ERRORs


N_MODELS = 15
N_TOT_TRACES = 100000
EPOCHS = 100
HP = {
    'hidden_layers':  [1, 2, 3, 4, 5],
    'hidden_neurons': [100, 200, 300, 400, 500],
    'dropout_rate':   [0.0, 0.1, 0.2,  0.3, 0.4, 0.5],
    'l2':             [0.0, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4],
    'optimizer':      ['adam', 'rmsprop', 'sgd'],
    'learning_rate':  [5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
    'batch_size':     [128, 256, 512, 1024]
}


def main():

    """
    Performs hyperparameter tuning with the specified settings.
    Settings parameters (provided in order via command line):
        - train_devs: Devices to use during training, provided as comma-separated string without spaces
        - model_type: Type of model to consider (MLP or CNN)
        - target: Target of the attack (SBOX_IN or SBOX_OUT)
        - byte_list: Bytes to be retrieved, provided as comma-separated string without spaces
    
    HP tuning is performed considering all the keys.
    
    The result is a JSON file containing the best hyperparameters.
    """
    
    _, train_devs, model_type, target, b = sys.argv
    
    train_devs = train_devs.upper().split(',')
    n_devs = len(train_devs)
    model_type = model_type.upper()
    target = target.upper()
    b = int(b)

    RES_ROOT = f'{constants.RESULTS_PATH}/DKTA/MultiKey/{target}/byte{b}/{n_devs}d'
    # HISTORY_PATH = RES_ROOT + f'/hp_tuning_history.png' 
    HP_PATH = RES_ROOT + f'/hp.json'
    
    configs = [f'{dev}-MK{k}' for k in range(100)
               for dev in train_devs]


    # Get dataset
    dl = DataLoader(
        configs,
        n_tot_traces=N_TOT_TRACES,
        target=target,
        byte_idx=b,
        mk_traces=True
    )
    x, y, _, _ = dl.load()

    
    # Initialize Tuner
    hp_tuner = HPTuner(
        model_type=model_type, 
        hp_space=HP, 
        n_models=N_MODELS, 
        n_epochs=EPOCHS
    )


    # Perform HP Tuning with Genetic Algorithm and 10-fold Crossvalidation
    kf = KFold(n_splits=10)
    
    xval_hps = []

    for train_idx, val_idx in tqdm(kf.split(x), desc='10-Fold Crossvalidation: '):

        print() # To separate tqdm from print of Genetic Algorithm's generations
        
        x_train = x[train_idx]
        y_train = y[train_idx]

        x_val = x[val_idx]
        y_val = y[val_idx]
    
        hp = hp_tuner.genetic_algorithm(
            n_gen=10,
            selection_perc=0.3,
            second_chance_prob=0.2,
            mutation_prob=0.2,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val
        )

        xval_hps.append(hp)
        
    xval_hp_space = {k: [el[k] for el in xval_hps]
                     for k in HP.keys()}

    best_hp = {k: statistics.mode(it) for k, it in xval_hp_space.items()}

    # vis.plot_history(hp_tuner.best_history, HISTORY_PATH)
    
    with open(HP_PATH, 'w') as jfile:
        json.dump(best_hp, jfile)

    print()


if __name__ == '__main__':
    start = time.time()
    main()
    print(f'Elapsed time: {((time.time() - start) / 3600):.2f} h')
    print()
