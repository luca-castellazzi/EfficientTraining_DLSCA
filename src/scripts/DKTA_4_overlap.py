import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import os

import sys
sys.path.insert(0, '../utils')
import constants
import visualization as vis

def main():

    used_tuning_method = sys.argv[1]
    
    main_dir = f'{constants.RESULTS_PATH}/DKTA'
        
    dirs = [f'{filename}' 
            for filename in os.listdir(main_dir)
            if os.path.isdir(f'{main_dir}/{filename}')]   

    dirs.sort(key=lambda x: int(x[0])) # Sort the directories w.r.t. the number of devices
    
    avg_ges = [np.load(f'{main_dir}/{d}/avg_ge__{used_tuning_method}.npy')
               for d in dirs]
    
    output_path = f'{main_dir}/avg_ges_comparison.png'
    
    vis.plot_overlap(avg_ges, output_path)
    
    
    
    
    
    
    
    # avg_file_names = [fname for fname in os.listdir('./') if '.npy' in fname]
    # # colors = ['r', 'b', 'g'] # Not reversed !!!
    
    # avg_file_names.reverse()
    # colors = ['b', 'r', 'g'] # Reversed
    
    
    # f, ax = plt.subplots(figsize=(10,5))
    
    
    
    # for i, filename in enumerate(avg_file_names):
        # avg_ges = np.load(f'./{filename}')
        
        # for j, ge in enumerate(avg_ges):
        
            # n_devs = int(filename[0])
         
            # if n_devs == 1:
            
                # if j == len(avg_ges)-1:
                    # label = '1 device, 10 keys'
                    # ax.plot(ge, label=label, marker='o', color='r', alpha=1)#, linewidth=1, markersize=5)   
                # else:
                    # ax.plot(ge, marker='o', color='r', alpha=0.1)#, linewidth=1, markersize=5) 
            
            # else:
                # ax.plot(ge, marker='o', color='b', alpha=0.1)#, linewidth=1, markersize=5)
                    
        
        #for j, ge in enumerate(avg_ges):
        #    
        #    if j == len(avg_ges) - 1:
        #        
        #        n_devs = int(filename[0])
        #        if n_devs == 1:
        #            label = '1 device'
        #        else:
        #            label =  f'{n_devs} devices'
        #        
        #        ax.plot(ge, label=label, marker='o', color=colors[i], alpha=1)#, linewidth=1, markersize=5)
        #    
        #    else:
        #        
        #        ax.plot(ge, marker='o', color=colors[i], alpha=1)#, linewidth=1, markersize=5)
                
            
    # ax.set_title(f'Avg GEs - Comparison')
    # ax.set_xticks(range(len(ge)), labels=range(1, len(ge)+1))
    # ax.set_ylim([-3, 55])
    # ax.set_xlabel('Number of traces')
    # ax.set_ylabel('Avg GE')
    # ax.legend()
    # ax.grid()
    
    # f.savefig(
        # './res_overlapped.png', 
        # bbox_inches='tight', 
        # dpi=600
    # )
    
if __name__ == '__main__':
    main()