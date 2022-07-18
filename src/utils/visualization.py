# Basics
import numpy as np
import matplotlib
matplotlib.use('agg') # Avoid interactive mode (and save files as .PNG as default)
import matplotlib.pyplot as plt
import seaborn as sns

# Custom
import constants
from nicv import nicv

def plot_nicv(nicvs, configs, metadata):

    scenario, cmap = metadata
    colors = cmap(range(len(configs)))
 
    f, ax = plt.subplots(4, 4, figsize=(25,25))
    
    for i, c in enumerate(configs):
        row = 0
        for b in range(16):
            col = b % 4

            ax[row, col].plot(nicvs[i][b], label=c, color=colors[i])
            ax[row, col].legend()
            ax[row, col].set_title(f'Byte {b}')
            ax[row, col].set_xlabel('Samples')
            ax[row, col].set_ylabel('NICV')

            if col == 3:
                row += 1

    f.savefig(constants.RESULTS_PATH + f'/nicv/nicv_plots/nicv_{scenario}.png', 
              bbox_inches='tight', 
              dpi=600)

def plot_history(history, output_path):

    f, ax = plt.subplots(2, 1, figsize=(10,18))
    
    ax[0].plot(history['loss'], label='train_loss')
    ax[0].plot(history['val_loss'], label='val_loss')
    ax[0].set_title('Train and Val Loss')
    ax[0].set_ylabel('Loss')
    ax[0].set_xlabel('Epochs')
    ax[0].legend()
    ax[0].grid()
    
    ax[1].plot(history['accuracy'], label='train_acc')
    ax[1].plot(history['val_accuracy'], label='val_acc')
    ax[1].set_title('Train and Val Acc')
    ax[1].set_ylabel('Acc')
    ax[1].set_xlabel('Epochs')
    ax[1].legend()
    ax[1].grid()
    
    f.savefig(
        output_path, 
        bbox_inches='tight', 
        dpi=600
    )
    
    
def plot_ges(ges, n_traces, labels, title, output_path):

    # Get the colors
    cmap = plt.cm.jet # Google Turbo
    colors = cmap(range(0, cmap.N, int(cmap.N/len(ges))))
    
    # Plot
    f, ax = plt.subplots(figsize=(10,5))
    ax.plot(np.zeros(n_traces), color='r', ls='--', linewidth=0.5)

    for i, ge in enumerate(ges):
        if n_traces <= 100:
            ax.plot(ge[:n_traces], label=labels[i], color=colors[i], marker='o')
            ax.set_xticks(range(n_traces), labels=range(1, n_traces+1))
            ax.grid()
        else:
            ax.plot(ge[:n_traces], label=labels[i], color=colors[i])
        ax.legend()
        ax.set_title(title)
        ax.set_xlabel('Number of traces')
        ax.set_ylabel('GE')
        ax.grid()
    
    f.savefig(
        output_path, 
        bbox_inches='tight', 
        dpi=600
    )
    

def plot_scores(scores_dict, title, output_path):

    f, ax = plt.subplots(figsize=(10,5))
    ax.bar(scores_dict.keys(), scores_dict.values())
    ax.set_title(title)
    
    f.savefig(
        output_path, 
        bbox_inches='tight', 
        dpi=600
    )
    