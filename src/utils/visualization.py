# Basics
import numpy as np
import matplotlib
matplotlib.use('agg') # Avoid interactive mode (and save files as .PNG as default)
import matplotlib.pyplot as plt
import seaborn as sns

# Custom
import constants
from nicv import nicv


def plot_nicv(nicvs, configs, output_path):

    """
    Plots NICV values and saves the result in a PNG file.
    
    Parameters:
        - nicvs (np.ndarray):
            NICV values to plot.
        - configs (str list):
            Device-key configurations that generate the traces used to compute
            NICV values.
        - output_path (str):
            Absolute path to the PNG file containing the plot.
    """

    cmap = plt.cm.Set1
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

    f.savefig(
        output_path,
        bbox_inches='tight', 
        dpi=600
    )
    
    plt.close(f)
    

def plot_history(history, output_path):

    """
    Plots a train history and saves the result in a PNG file.
    
    Parameters:
        - history (dict):
            Train history to plot.
        - output_path (str):
            Absolute path to the PNG file containing the plot.
    """

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
    
    plt.close(f)
    
    
# def plot_ges(ges, n_traces, labels, title, output_path):

    # Get the colors
    # cmap = plt.cm.jet # Google Turbo
    # colors = cmap(range(0, cmap.N, int(cmap.N/len(ges))))
    
    # Plot
    # f, ax = plt.subplots(figsize=(10,5))
    # ax.plot(np.zeros(n_traces), color='r', ls='--', linewidth=0.5)

    # for i, ge in enumerate(ges):
        # if n_traces <= 100:
            # ax.plot(ge[:n_traces], label=labels[i], color=colors[i], marker='o')
            # ax.set_xticks(range(n_traces), labels=range(1, n_traces+1))
            # ax.set_yticks(range(0, 50, 5))
            # ax.grid()
        # else:
            # ax.plot(ge[:n_traces], label=labels[i], color=colors[i])
        # ax.legend()
        # ax.set_title(title)
        # ax.set_xlabel('Number of traces')
        # ax.set_ylabel('GE')
    
    # f.savefig(
        # output_path, 
        # bbox_inches='tight', 
        # dpi=600
    # )
    
    
def plot_conf_matrix(conf_matrix, output_path):

    """
    Plots a confusion matrix and saves the result in a PNG file.
    
    Parameters:
        - conf_matrix (np.ndarray):
            Confusion matrix to plot.
        - output_path (str):
            Absolute path to the PNG file containing the plot.
    """
    
    cmap = plt.cm.Blues
    
    f = plt.figure(figsize=(10,8))
    plt.imshow(conf_matrix, cmap=cmap)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    
    plt.colorbar()
    
    f.savefig(
        output_path,
        bbox_inches='tight',
        dpi=600
    )
    
    plt.close(f)
    

def plot_attack_losses(losses, output_path):

    """
    Plots attack losses and saves the result in a PNG file.
    
    Parameters:
        - losses (np.array):
            Attack losses to plot.
        - output_path (str):
            Absolute path to the PNG file containing the plot.
    """

    f = plt.figure(figsize=(10,5))
    plt.plot(losses, marker='o')
    plt.title('Attack Loss')
    plt.xlabel('Number of keys')
    plt.ylabel('Loss')
    
    plt.xticks(range(len(losses)), labels=range(1, len(losses)+1))
    plt.ylim(2, 5)
    
    plt.grid()
    
    f.savefig(
        output_path, 
        bbox_inches='tight', 
        dpi=600
    )
    
    plt.close(f)


def plot_avg_ges(ges, n_devs, output_path):

    """
    Plots the average GEs resulting from a DKTA experiment and saves the result 
    in a PNG file.
    
    Parameters:
        - ges (np.ndarray):
            Average GEs to plot.
        - n_devs (int):
            Number of train-devices used during the computation of GEs.
        - output_path (str):
            Absolute path to the PNG file containing the plot.
    """
    
    # Set the color palette
    cmap = plt.cm.jet # Google Turbo
    colors = cmap(range(0, cmap.N, int(cmap.N/len(ges))))
    
    # Plot the GEs
    f, ax = plt.subplots(figsize=(10,5))
    for i, ge in enumerate(ges):
            
        label = f'{i+1} key'
        if i != 0:
            label += 's' # Plural
            
        ax.plot(ge, label=label, marker='o', color=colors[i])
        
    ax.set_title(f'Number of Train-Devices: {n_devs}')
    ax.set_xticks(range(len(ge)), labels=range(1, len(ge)+1))
    ax.set_ylim([-3, 30]) 
    ax.set_xlabel('Number of traces')
    ax.set_ylabel('Avg GE')
    ax.legend()
    ax.grid()

    f.savefig(
        output_path, 
        bbox_inches='tight', 
        dpi=600
    )
    
    plt.close(f)


def plot_overlap(all_ges, output_path):

    """
    Plots GEs resulting from different DKTA experiments in a single plane and 
    saves the result in a PNG file.
    
    Parameters:
        - all_ges (np.ndarray):
            GEs to plot.
        - output_path (str):
            Absolute path to the PNG file containing the plot.
    """
    
    colors = ['r', 'b', 'g']
    
    f, ax = plt.subplots(figsize=(10,5))
    
    for i, ges in enumerate(all_ges): # i used for color and num devices
        
        for j, ge in enumerate(ges): # j used for label
            
            if j == len(ges) - 1: # Label only the last element of each group
                label = f'{i+1} device'
                if i != 0:
                    label += 's' # Plural
                ax.plot(ge, color=colors[i], marker='o', label=label)
            else:
                ax.plot(ge, color=colors[i], marker='o')
                
                
    ax.set_title(f'Avg GEs - Comparison')
    ax.set_xticks(range(len(ge)), labels=range(1, len(ge)+1)) # Consider the last ge, but all have same length
    ax.set_ylim([-3, 30])
    ax.set_xlabel('Number of traces')
    ax.set_ylabel('Avg GE')
    ax.legend()
    ax.grid()
    
    f.savefig(
        output_path, 
        bbox_inches='tight', 
        dpi=600
    )
    
    plt.close(f)
    
    
# def plot_scores(scores_dict, title, output_path):

    # f, ax = plt.subplots(figsize=(10,5))
    # ax.bar(scores_dict.keys(), scores_dict.values())
    # ax.set_title(title)
    
    # f.savefig(
        # output_path, 
        # bbox_inches='tight', 
        # dpi=600
    # )
    