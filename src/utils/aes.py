import numpy as np

from helpers import int_to_hex, to_coords
import constants


def labels_from_key(plaintext, key, target='SBOX_OUT'):
    
    """ 
    Generates the labels associated to the encryption of the given plaintext
    with the given key, w.r.t. the given target.

    Parameters: 
        - plaintext (int np.array):
            plaintext considered as list of int values, each one relative to 
            a single byte of the hex version.
        - key (int np.array): 
            encryption key considered as list of int values, each one relative
            to a single byte of the hex version.
        - target (str, default: 'SBOX_OUT'):
            target of the attack ('SBOX_OUTPUT' for SBox Output).
            More targets in future (e.g. Hamming Weights, Key, ...).

    Returns:
        int np.array containing the 16 labels (one per byte of plaintext/key)
        relative to the given target.
        In case of 'SBOX_OUT' target, the output is the result of the SBox-lookup.
    """
    
    # AddRoundKey
    sbox_in = plaintext ^ key

    # SubBytes
    rows, cols = to_coords(sbox_in)
    sbox_out = constants.SBOX_DEC[rows, cols]

    # Labeling
    if target == 'SBOX_OUT':
        return sbox_out
    else:
        pass # Future implementations (target = HW, target = KEY, ...)


def key_from_labels(plaintext, labels, target='SBOX_OUT'):

    """ 
    Generates the key-bytes that produced the given labels during the encryption
    of the given plaintext, w.r.t. the given target.

    Parameters: 
        - plaintext (int np.array):
            plaintext considered as list of int values, each one relative to 
            a single byte of the hex version.
        - labels (int np.array): 
            leakage (target) values produced during the encryption of the given
            plaintext.
        - target (str, default: 'SBOX_OUT'):
            target of the attack ('SBOX_OUTPUT' for SBox Output).
            More targets in future (e.g. Hamming Weights, Key, ...).

    Returns:
        int np.ndarray containing the key-bytes that produced the given labels
        as leakage during the encryption of the given plaintext.
        The np.ndarray contains NUM_LABELS array for each byte of the given 
        plaintext.
    """

    if target == 'SBOX_OUT':
        
        # The provided labels are sbox_out values

        # Inverse-SubBytes
        rows, cols = to_coords(labels)
        sbox_in = constants.INV_SBOX_DEC[rows, cols]

        # Inverse-AddRoundKey
        key_bytes = [sbox_in ^ b for b in plaintext]

        return np.array(key_bytes)
    else:
        pass # Future implementations (target = HW, target = KEY, ...)
