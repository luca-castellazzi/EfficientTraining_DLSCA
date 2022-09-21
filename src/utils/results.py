# Basics
import numpy as np
import random

# Custom
import aes


def compute_key_preds(preds, key_bytes):
    
    """
    Converts target-predictions into key-predictions (key-byte).

    Parameters:
        - preds (np.ndarray):
            Target-predictions (probabilities for each possible target value).
        - key_bytes (np.array):
            Key-bytes relative to each possible value in the target-predictions.
            Target-predictions cover all possible values (1 to 255), and
            each value leads to a differnt key-byte.
    
    Returns:
        - key_preds (np.array):
           Key-predictions (probabilities for each possible key-byte value).
           Sorted from key-byte=0 to key-byte=255.
    """
    
    # Associate each sbox-out prediction with its relative key-byte
    association = list(zip(key_bytes, preds))
    
    # Sort the association w.r.t. key-bytes (0 to 255, for alignment within all traces)
    association.sort(key=lambda x: x[0])
    
    # Consider the sorted sbox-out predictons as key-byte predictons
    key_preds = list(zip(*association))[1]
    
    return key_preds
    
    
def compute_final_rankings(preds, pltxt_bytes, target):

    """
    Generates the ranking of the key-bytes starting from key-predictions.
    
    Parameters:
        - preds (np.ndarray):
            Predictions relative to the target.
        - pltxt_bytes (np.array):
            True plaintext bytes.
        - target (str):
            Target of the attack.

    Returns:
        - final_rankings (list):
            Ranking of the possible key-bytes (from the most probable to the 
            least probable) for increasing number of traces.
    """

    if target == 'KEY':
        # If the target is 'KEY', then key_preds is directly sampled_preds
        # because sampled_preds contains predictions related to each key-byte
        # already in order
        key_preds = preds
    else: 
        # SBOX-IN, SBOX-OUT, ... need further computations
        key_bytes = [aes.key_from_labels(pb, target) for pb in pltxt_bytes]
    
        key_preds = np.array([compute_key_preds(ps, kbs) 
                              for ps, kbs in zip(preds, key_bytes)])

    log_probs = np.log10(key_preds + 1e-22) # n_traces x 256
    
    cum_tot_probs = np.cumsum(log_probs, axis=0) # n_traces x 256
    
    indexed_cum_tot_probs = [list(zip(range(256), tot_probs))
                             for tot_probs in cum_tot_probs] # n_traces x 256 x 2
    
    sorted_cum_tot_probs = [sorted(el, key=lambda x: x[1], reverse=True)
                            for el in indexed_cum_tot_probs] # n_traces x 256 x 2
                            
    final_rankings = [[el[0] for el in tot_probs]
                      for tot_probs in sorted_cum_tot_probs] # n_traces x 256
                  
    return final_rankings


def ge(preds, pltxt_bytes, true_key_byte, n_exp, target, n_traces=500):

    """
    Computes the Guessing Entropy of an attack as the average rank of the 
    correct key-byte among the predictions.
    
    Parameters: 
        - preds (np.ndarray):
            Target predictions.
        - pltxt_bytes (np.array):
            Plaintext used during the encryption (single byte).
        - true_key_byte (int):
            Actual key used during the encryption (single byte).
        - n_exp (int):
            Number of experiment to compute the average value of GE.
        - n_traces (int):
            Number of traces to consider during GE computation.
        - target (str):
            Target of the attack.
            
    Returns:
        - ge (np.array):
            Guessing Entropy of the attack.
    """
    
    # Consider all couples predictions-plaintext
    all_preds_pltxt = list(zip(preds, pltxt_bytes))
    
    ranks_per_exp = []
    for _ in range(n_exp):
        
        # During each experiment:
        #   * Consider only a fixed number of target-predictions
        #   * Retrieve the corresponding key-predictions
        #   * Compute the final key-predictions
        #   * Rank the final key-predictions
        #   * Retrieve the rank of the correct key (key-byte)
        # 
        # The whole process considers incrementing number of traces (not only 1)
        
        sampled = random.sample(all_preds_pltxt, n_traces)
        sampled_preds, sampled_pltxt_bytes = list(zip(*sampled))
        
        # Compute the final rankings (for increasing number of traces)
        final_rankings = compute_final_rankings(sampled_preds, sampled_pltxt_bytes, target)
        
        # Retrieve the rank of the true key-byte (for increasing number of traces)
        true_kb_ranks = np.array([kbs.index(true_key_byte)
                                  for kbs in final_rankings]) # 1 x n_traces 

        ranks_per_exp.append(true_kb_ranks)
        
    # After the experiments, average the ranks
    ranks_per_exp = np.array(ranks_per_exp) # n_exp x n_traces
    ge = np.mean(ranks_per_exp, axis=0) # 1 x n_traces
    
    return ge


def retrieve_key_byte(preds, pltxt_bytes, target, n_traces=500):
    
    # Consider all couples predictions-plaintext
    all_preds_pltxt = list(zip(preds, pltxt_bytes))

    # Sample randomly the predictions that will generate the final result
    sampled = random.sample(all_preds_pltxt, n_traces)
    sampled_preds, sampled_pltxt_bytes = list(zip(*sampled))
    
    # Compute the final rankings (for increasing number of traces)
    final_rankings = compute_final_rankings(sampled_preds, sampled_pltxt_bytes, target)

    resulting_key_byte = np.array([ranking[0] for ranking in final_rankings])

    return resulting_key_byte