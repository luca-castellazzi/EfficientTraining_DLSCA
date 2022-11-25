# Basics
import json
import numpy as np
from tqdm import tqdm
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.backend import clear_session


# Custom
import sys
sys.path.insert(0, '../utils')
import results
import constants
from data_loader import DataLoader, SplitDataLoader
sys.path.insert(0, '../modeling')
from network import Network

# Suppress TensorFlow messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # 1 for INFO, 2 for INFO & WARNINGs, 3 for INFO & WARNINGs & ERRORs

TOT_TRACES = 50000
EPOCHS = 100

TARGET = 'SBOX_OUT'


def main():

    """
    Performs a complete Device-Key Trade-off Analysis (DKTA) with the specified 
    settings.
    Settings parameters (provided in order via command line):
        - n_devs: Number of train devices
        - b: Byte to be retrieved (from 0 to 15)
    
    All possible permutations of devices are automatically considered.
    Each DKTA consists in model-building, model-training, attack: all models 
    have the same hyperparameters, but each time an increasing number of keys 
    is considered.
    
    The results are multiple NPY files containing each average GE.
    In addition, the confusion matrices relative to each attack are considered
    alongside plots about attack-loss (both as PNG files).
    """
    
    _, n_devs, b = sys.argv
    n_devs = int(n_devs)
    b = int(b)
    
    RES_ROOT = f'{constants.RESULTS_PATH}/DKTA/{TARGET}/byte{b}/{n_devs}d' 
    HP_PATH = RES_ROOT + '/hp.json'


    with open(HP_PATH, 'r') as jfile:
        hp = json.load(jfile)
    
    
    for train_devs, test_dev in constants.PERMUTATIONS[n_devs]:
      
        GES_FILE = RES_ROOT + f'/ges_{"".join(train_devs)}vs{test_dev}.npy'
        
        test_files = [f'{constants.PC_TRACES_PATH}/{test_dev}-K0_500MHz + Resampled.trs'] # list is needed in DataLoader
            
        ges = []

        for n_keys in tqdm(range(1, len(constants.KEYS)), desc=f'{"".join(train_devs)}vs{test_dev}: '): 
        
            N_KEYS_FOLDER = RES_ROOT + f'/{n_keys}k'
            if not os.path.exists(N_KEYS_FOLDER):
                os.mkdir(N_KEYS_FOLDER)
            SAVED_MODEL_PATH = N_KEYS_FOLDER + f'/model_{"".join(train_devs)}vs{test_dev}.h5'
            
            train_files = [f'{constants.PC_TRACES_PATH}/{dev}-{k}_500MHz + Resampled.trs' 
                           for k in list(constants.KEYS)[1:n_keys+1]
                           for dev in train_devs]

            train_dl = SplitDataLoader(
                train_files, 
                tot_traces=TOT_TRACES,
                train_size=0.9,
                target=TARGET,
                byte_idx=b
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
                tot_traces=TOT_TRACES,
                target=TARGET,
                byte_idx=b
            )
            x_test, _, pbs_test, tkb_test = test_dl.load()

            # Scale test data to 0-mean and 1-variance w.r.t. train data
            x_test = scaler.transform(x_test)        
            
            # Compute GE
            test_model = load_model(SAVED_MODEL_PATH) 
            ge = results.ge(
                model=test_model,
                x_test=x_test,
                pltxt_bytes=pbs_test, 
                true_key_byte=tkb_test, 
                n_exp=100, 
                target=TARGET
            )
            ges.append(ge)
        
        ges = np.array(ges)
        np.save(GES_FILE, ges) # .NPY file because no direct plot
        
        print()
        
        
if __name__ == '__main__':
    main()
