import numpy as np
import matplotlib
matplotlib.use('Agg') # Avoid interactive mode (and save files as .PNG as default)
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns

import sys
sys.path.insert(0, '../utils')
import constants
from nicv import nicv


def plot_ges(ges, n, metadata, subplots=False):
    
    metric, train_scenario, date = metadata
    
    if not subplots:
        f, ax = plt.subplots(figsize=(10,5))
        ax.plot(np.zeros(n), color='r', ls='--', linewidth=0.8)

        d = 1
        for i in range(len(ges)):

            k = i % 3

            if n <= 100:
                ax.plot(ges[i][:n], marker='o', label=f'D{d}-K{k+1}')
            else:
                ax.plot(ges[i][:n], label=f'D{d}-K{k+1}')

            if k == 2:
                d += 1

        ax.set_title(f'Training scenario: {train_scenario}')
        ax.set_xlabel('Number of Traces')
        ax.set_ylabel('Guessing Entropy')
        if n <= 100:
            ax.set_xticks(ticks=range(n), labels=range(1, n+1))
            ax.grid()
        ax.legend()

        filename = f'{metric}TOT{n}_{date}.png'
    else:
        f, ax = plt.subplots(len(constants.DEVICES), len(constants.KEYS), figsize=(30,15))

        row = 0
        for i in range(len(ges)):

            col = i % 3

            if n <= 100:
                ax[row, col].plot(ges[i][:n], marker='o', color=list(colors.TABLEAU_COLORS.keys())[i])
            else:
                ax[row, col].plot(ges[i][:n], color=list(colors.TABLEAU_COLORS.keys())[i])
            ax[row, col].plot(np.zeros(n), color='r', ls='--')
            ax[row, col].set_title(f'D{row+1}-K{col+1} (Training scenario {train_scenario})')
            ax[row, col].set_xlabel('Number of Traces')
            ax[row, col].set_ylabel('Guessing Entropy')
            if n <= 100:
                ax[row, col].set_xticks(range(n), labels=range(1, n+1))
                ax[row, col].grid()

            if col == 2:
                row += 1

        filename = f'{metric}SUB{n}_{train_scenario}_{date}.png'
    
    f.savefig(constants.RESULTS_PATH + f'/ge/ge_plots/{filename}', 
              bbox_inches='tight', 
              dpi=600)


def plot_nicv(nicvs, configs, metadata):

    scenario, cmap, date = metadata
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

    f.savefig(constants.RESULTS_PATH + f'/nicv/nicv_plots/nicv_{scenario}_{date}.png', 
              bbox_inches='tight', 
              dpi=600)


def plot_ge_per_trainConfig(ges, scores, train_configs, test_config, n=30):

    # Sort the GEs w.r.t. their scores from the highest to the lowest
    # (to have the right color more easily)
    ges = ges.tolist()
    ges_scores = list(zip(ges, scores))
    ges_scores.sort(key=lambda x: x[1], reverse=True)

    ges, _ = list(zip(*ges_scores))
        
    # Get the colorset
    cmap = plt.cm.jet # Google Turbo
    colors = cmap(range(0, cmap.N, int(cmap.N/len(ges))))
    
    # Plot
    f, ax = plt.subplots(figsize=(10,5))
    ax.plot(np.zeros(n), color='r', ls='--', linewidth=0.5)

    for i, ge in enumerate(ges):
        if n<= 100:
            ax.plot(ge[:n], label=train_configs[i], color=colors[i], marker='o')
            ax.set_xticks(range(n), labels=range(1, n+1))
            ax.grid()
        else:
            ax.plot(ge[:n], label=train_configs[i], color=colors[i])
        ax.legend()
        ax.set_title(f'Attack config: {test_config}')
        ax.set_xlabel('Number of traces')
        ax.set_ylabel('GE')

    f.savefig(constants.RESULTS_PATH + f'/ge/ge_plots/ge_{train_configs[0][:2]}only_{n}.png', 
              bbox_inches='tight', 
              dpi=600)
