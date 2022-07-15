import pandas as pd
import numpy as np
import os

import sys
sys.path.insert(0, '../utils')
import constants
import visualization as vis


# RESULTS_TO_CSV
def ges_to_csv(ges, labels, csv_path):
    
    ges = np.array(ges)
    ge_dict = {f'{i+1}traces': ges[:, i] for i in range(ges.shape[1])}
    ge_dict['train_config'] = labels

    ge_df = pd.DataFrame(ge_dict)
    ge_df.to_csv(csv_path, index=False)


def main():
    
    n_train_dev = int(sys.argv[1]) # Number of training devices (to access the right results folder)
    
    res_folder_path = constants.RESULTS_PATH + f'/{n_train_dev}d'
    
    all_ges = []
    for filename in os.listdir(res_folder_path):
        if '.csv' in filename:
            csv_file = res_folder_path + f'/{filename}'
            df = pd.read_csv(csv_file)
            ges = df.iloc[:, :-1].values
            all_ges.append(ges)
    
    train_configs = list(df['train_config'])
    labels = [c.split('_')[1] for c in train_configs]
    
    avg_ges = []
    for i in range(len(train_configs)):
        ges_per_train_config = np.array([ges[i] for ges in all_ges])
        avg_ge = np.mean(ges_per_train_config, axis=0)
        avg_ges.append(avg_ge)
        
    avg_ges_file_path = constants.RESULTS_PATH + f'/{n_train_dev}d/avg_ge-{n_train_dev}d.csv'
    ges_to_csv(avg_ges, train_configs, avg_ges_file_path)
   
    # Plot Avg GEs
    title = f'Avg GE - {n_train_dev} train devices'
    path = constants.RESULTS_PATH + f'/{n_train_dev}d/avg_ge-{n_train_dev}d.png'
    vis.plot_ges(avg_ges, len(avg_ges[0]), labels, title, path)
    

if __name__ == '__main__':
    main()
