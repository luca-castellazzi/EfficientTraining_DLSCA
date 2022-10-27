# Basics
from turtle import showturtle
import matplotlib
matplotlib.use('svg') # Avoid interactive mode (and save files as .SVG as default)
import matplotlib.pyplot as plt
import matplotlib.colors as clrs

# Custom


def plot_nicv(nicvs, configs, output_path):

    """
    Plots NICV values and saves the result in a SVG file.
    
    Parameters:
        - nicvs (np.ndarray):
            NICV values to plot.
        - configs (str list):
            Device-key configurations that generate the traces used to compute
            NICV values.
        - output_path (str):
            Absolute path to the SVG file containing the plot.
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
    Plots a train history and saves the result in a SVG file.
    
    Parameters:
        - history (dict):
            Train history to plot.
        - output_path (str):
            Absolute path to the SVG file containing the plot.
    """

    f, ax = plt.subplots(1, 2, figsize=(18,8))
    
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
    
    
def plot_conf_matrix(conf_matrix, output_path):

    """
    Plots a confusion matrix and saves the result in a SVG file.
    
    Parameters:
        - conf_matrix (np.ndarray):
            Confusion matrix to plot.
        - output_path (str):
            Absolute path to the SVG file containing the plot.
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


def plot_avg_ges(ges, title, output_path):

    """
    Plots the average GEs resulting from a DKTA experiment and saves the result 
    in a SVG file.
    
    Parameters:
        - ges (np.ndarray):
            Average GEs to plot.
        - title (str):
            Title of the plot.
        - output_path (str):
            Absolute path to the SVG file containing the plot.
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
        
    # ax.set_title(f'Byte: {b}  |  Train-Devices: {n_devs}')
    ax.set_title(title)
    ax.set_xticks(range(len(ge)), labels=range(1, len(ge)+1))
    ax.set_ylim([-3, 45]) 
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


def plot_overlap(all_ges, to_compare, title, output_path):

    """
    Plots GEs resulting from 2 different DKTA experiments in a single plane and 
    saves the result in a SVG file.
    
    Parameters:
        - all_ges (np.ndarray):
            GEs to plot.
        - to_compare (int list):
            Bytes whose results are compared.
        - output_path (str):
            Absolute path to the SVG file containing the plot.
    """
    
    colors = list(clrs.TABLEAU_COLORS)
    
    f, ax = plt.subplots(figsize=(10,5))
    
    for i, ges in enumerate(all_ges): # i used for indexing the compared bytes
        
        for j, ge in enumerate(ges): # j used for labeling
            
            ge = ge[:10]

            if j == len(ges) - 1: # Label only the last element of each group
                label = f'Byte {to_compare[i]}'
                ax.plot(ge, color=colors[i], marker='o', label=label, alpha=0.5)
            else:
                ax.plot(ge, color=colors[i], marker='o', alpha=0.5)

    ax.set_title(title)
    ax.set_xticks(range(len(ge)), labels=range(1, len(ge)+1)) # Consider the last ge, but all have same length
    ax.set_ylim([-3, 45])
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


def plot_multikey(ges, traces, title, output_path):

    """
    Plots the average GEs resulting from a MultiKey experiment and saves the result 
    in a SVG file.
    
    Parameters:
        - ges (np.ndarray):
            Average GEs to plot.
        - title (str):
            Title of the plot.
        - output_path (str):
            Absolute path to the SVG file containing the plot.
    """
    
    # Plot the GEs (default matplotlib colors)
    f, ax = plt.subplots(figsize=(10,5))
    for i, ge in enumerate(ges):
        ax.plot(ge, label=f'{traces[i]} Traces', linewidth=3)
        
    ax.set_title(title)
    ax.set_ylim([-3, 150])
    ax.set_xlabel('Number of Attack Traces')
    ax.set_ylabel('Avg GE')
    ax.legend()
    ax.grid()

    f.savefig(
        output_path, 
        bbox_inches='tight', 
        dpi=600
    )
    
    plt.close(f)