##################################################################################
#                                                                                # 
#  This NICV implementation is based on the work by AISyLab                      #
#  (https://github.com/AISyLab) available at                                     #
#  https://github.com/AISyLab/side-channel-analysis-toolbox, licensed under MIT  #
#  License.                                                                      #
#                                                                                #
##################################################################################


import numpy as np


def nicv(traces, pltxt_bytes):

    """
    Computes the Normalized Inter-Class Variance (NICV) index for the provided
    traces, plaintexts and byte index.
    
    In general, NICV = Var(E[T|X]) / Var(T), where T is the trace-set and X is 
    a known parameter, such as the plaintext.

    Parameters:
        - traces (np.ndarray): 
            Values of the traces.
        - pltxt_bytes (np.ndarray):
            Integer-version of the plaintexts (single byte).

    Returns:
        - nicv (np.array): 
            NICV values.
    """

    # Define all fixed values
    possible_plaintext_values = range(256) # Possible values that a single plaintext-byte can assume 

    # Define a structure to handle the mean values during the computation
    mean_t_given_x = []

    # Partition the traces w.r.t. X and compute the corresponding mean value
    for x in possible_plaintext_values:
        idx_tr_with_x = np.where(pltxt_bytes == x) # Isolate traces by index
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