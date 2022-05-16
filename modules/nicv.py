import numpy as np

def nicv(traces: np.ndarray, labels: np.ndarray, byte_idx: int) -> np.array:

    """
    Compute the Normalized Inter-Class Variance (NICV) index for the provided
    set of traces and byte.
    
    NICV = Var(E[T|X]) / Var(T), where T is a trace and X is a known parameter.
    In this case X takes the values of the SBox Output 


    Parameters:
        - traces: set of traces to consider during the computation
        - labels: labels related to the given traces
        - byte_idx: index of the byte to consider during the computation (range 0-15)

    Returns:
        - nicv: NICV values, one for each sample of the given traces
    """

    # Define all fixed values
    num_traces = len(traces) # Number of traces
    tr_len = len(traces[0]) # Number of samples per trace
    num_possible_labels = 256 # Number of possible labels

    # Define a structure to handle the mean values during the computation
    mean_t_given_x = []

    # Partition the traces w.r.t. X and compute the corresponding mean value
    for x in range(num_possible_labels):
        idx_tr_with_x = np.where(labels[:, byte_idx] == x) # Isolate traces by index
        val_tr_with_x = traces[idx_tr_with_x] # Get the actual values of the isolated traces
        # Compute the mean value of the isolated traces w.r.t. axis=0 
        #(for each sample, sum the value of each trace and divide by the number of traces)
        mean_t_given_x.append(np.mean(val_tr_with_x, axis=0)) 

    mean_t_given_x = np.array(mean_t_given_x)

    # Compute NICV
    nicv = np.var(mean_t_given_x, axis=0) / np.var(traces, axis=0)

    return nicv
