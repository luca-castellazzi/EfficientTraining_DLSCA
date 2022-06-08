################################################################################
#                                                                              # 
# This code is based on the NICV implementation from AISyLab.                  #
#                                                                              # 
# The reference project is                                                     # 
# https://github.com/AISyLab/side-channel-analysis-toolbox,                    # 
# which is licensed under MIT License.                                         #
#                                                                              #
################################################################################


import numpy as np


def nicv(traces, plaintexts, byte_idx):

    """
    Compute the Normalized Inter-Class Variance (NICV) index for the provided
    set of traces and byte.
    
    NICV = Var(E[T|X]) / Var(T), where T is a trace and X is a known parameter, 
    such as the plaintext.   
    In this case X takes the values of the SBox Output 


    Parameters:
        - traces (float np.ndarray): 
            set of traces to consider during the computation.
        - plaintexts (int np.ndarray):
            plaintexts that have been encrypted.
        - byte_idx (int): 
            index of the byte to consider during the computation (range 0-15).

    Returns:
        - nicv (float np.array): 
            NICV values, one for each sample of the given traces.
    """

    # Define all fixed values
    num_traces = len(traces) # Number of traces
    tr_len = len(traces[0]) # Number of samples per trace
    possible_plaintext_values = range(256) # Possible values that a single plaintext-byte can assume 

    # Define a structure to handle the mean values during the computation
    mean_t_given_x = []

    # Partition the traces w.r.t. X and compute the corresponding mean value
    for x in possible_plaintext_values:
        idx_tr_with_x = np.where(plaintexts[:, byte_idx] == x) # Isolate traces by index
        val_tr_with_x = traces[idx_tr_with_x] # Get the actual values of the isolated traces
        
        if len(val_tr_with_x) != 0:
            # If there is at least one isolated trace, 
            # Then compute the mean value w.r.t. axis=0 
            # (for each sample, sum the value of each trace and divide by the number of traces)
            mean_t_given_x.append(np.mean(val_tr_with_x, axis=0))
        else:
            # Otherwise add a "fictitious mean" of 0.0
            mean_t_given_x.append(0.0)

    mean_t_given_x = np.array(mean_t_given_x, dtype=object)

    # Compute NICV
    nicv = np.var(mean_t_given_x, axis=0) / np.var(traces, axis=0)

    return nicv
