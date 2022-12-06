# Basics
import trsfile
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Custom
import sys
sys.path.insert(0, '../utils')
import aes
import helpers
import constants


COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']


def main():

    _, to_plot = sys.argv
    to_plot = int(to_plot)

    TR_FILE = f'{constants.PC_TRACES_PATH}/D2-K3_500MHz + Resampled.trs'

    RES_ROOT = f'{constants.RESULTS_PATH}/SPA_unmasked/'
    JITTER_FILE = RES_ROOT + 'jitter.csv'
    JITTER_PLOT = RES_ROOT + 'jitter.svg'
    STD_FILE = RES_ROOT + 'std.csv'
    STD_PLOT = RES_ROOT + 'std.svg'
    STD_LOOPS_PLOT = RES_ROOT + 'std_loops.svg'
    CORR_FILE = RES_ROOT + 'corr.csv'
    CORR_PLOT = RES_ROOT + 'corr.svg'

    with trsfile.open(TR_FILE, 'r') as traces:
        samples = [tr.samples
                   for tr in tqdm(traces, desc='Traces: ')]
        labels = [aes.labels_from_key(
                    np.array(tr.get_input()), 
                    np.array(tr.get_key()), 
                    'SBOX_OUT'
                  )
                  for tr in tqdm(traces, desc='Labels')]

    samples = np.vstack(samples)
    labels = np.vstack(labels)

    avg_tr = np.mean(samples, axis=0)
    std_tr = np.std(samples, axis=0)

    corr = [[pearsonr(samples[:, i], labels[:, b])[0] 
            for i in range(samples.shape[1])]
            for b in tqdm(range(16), desc='Corr: ')]
    corr = np.vstack(corr)


    # Jitter Visualization ###################################################
    # Save Data
    jitter_data = np.vstack(
        [
            np.arange(samples.shape[1]), # x-axis values
            samples[:to_plot] # y-axis values
        ]
    ).T
    helpers.save_csv(
        data=jitter_data,
        columns=['Sample']+[f'Trace{t+1}' for t in range(to_plot)],
        output_path=JITTER_FILE
    )
    # Plot Data
    f = plt.figure(figsize=(10,5))
    for i, el in enumerate(samples[:to_plot]):
        if i == to_plot - 1:
            plt.plot(el, color='darkgrey', label='D2-K3 Traces', linewidth=0.5)
        else:
            plt.plot(el, color='darkgrey', linewidth=0.5)
    plt.plot(avg_tr, color='dodgerblue', label='Avg')
    plt.legend()
    plt.xlabel('Samples')
    plt.ylabel('mV')
    plt.title('Jitter Visualization')
    f.savefig(
        JITTER_PLOT, 
        bbox_inches='tight', 
        dpi=600
    )
    plt.close(f) 


    # Loops visualization from std #############################################
    # Save Data
    std_data = np.vstack(
        [
            np.arange(len(std_tr)), # x-axis values
            std_tr # y-axis values
        ]
    ).T
    helpers.save_csv(
        data=std_data,
        columns=['Sample', 'Std'],
        output_path=STD_FILE
    )
    # Plot Data
    f = plt.figure(figsize=(10,5))
    plt.plot(std_tr, color='dodgerblue')
    plt.xlabel('Samples')
    plt.ylabel('Std')
    plt.title("Samples' Standard Deviation")
    f.savefig(
        STD_PLOT, 
        bbox_inches='tight', 
        dpi=600
    )
    plt.close(f) 
    # Plot Data (rows)
    f = plt.figure(figsize=(10,5))
    plt.plot(range(350), std_tr[:350], color=COLORS[0], label='SubBytes Loop 1')
    plt.plot(range(350, 605), std_tr[350:605], color=COLORS[1], label='SubBytes Loop 2')
    plt.plot(range(605, 860), std_tr[605:860], color=COLORS[2], label='SubBytes Loop 3')
    plt.plot(range(860, len(std_tr)), std_tr[860:], color=COLORS[3], label='SubBytes Loop 4')
    plt.legend()
    plt.xlabel('Samples')
    plt.ylabel('Std')
    plt.title("Samples' Standard Deviation")
    f.savefig(
        STD_LOOPS_PLOT, 
        bbox_inches='tight', 
        dpi=600
    )
    plt.close(f)   


    # Correlation ##############################################################
    # Save Data
    corr_data = np.vstack(
        [
            np.arange(corr.shape[1]), # x-axis values
            corr # y-axis values
        ]
    ).T
    helpers.save_csv(
        data=corr_data,
        columns=['Sample']+[f'CorrByte{b+1}' for b in range(corr.shape[0])],
        output_path=CORR_FILE
    )
    # Plot Data per AES-state Row
    f = plt.figure(figsize=(10,5))
    for i, b_corr in enumerate(corr):
        c = COLORS[i % 4]
        if i in [12, 13, 14, 15]:
            plt.plot(b_corr, color=c, label=f'AES-State Row {i%4}')
        else:
            plt.plot(b_corr, color=c)
        plt.text(s=f'b{i}', x=np.argmin(b_corr)-10, y=np.min(b_corr)-0.04, color=c)
    plt.legend()
    plt.ylim([-0.5, 0.5])
    plt.xlabel('Sample Index')
    plt.ylabel('Correlation')
    plt.title('Pearson Correlation Between Traces and SBox-Output')
    f.savefig(
        CORR_PLOT, 
        bbox_inches='tight', 
        dpi=600
    )
    plt.close(f)


if __name__ == '__main__':
    main()