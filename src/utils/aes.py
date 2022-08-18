import numpy as np

from helpers import int_to_hex, to_coords
import constants


def labels_from_key(plaintext, key, target):
    
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
    if target == 'SBOX_IN':
        return sbox_in
    elif target == 'SBOX_OUT':
        return sbox_out
    else:
        pass # Future implementations (target = HW, target = KEY, ...)


def key_from_labels(pltxt_byte, target):

    """ 
    Generates the key-bytes relative to each possible value of the sbox-output
    w.r.t. the plaintext byte and the given target.

    Parameters: 
        - pltxt_byte (int):
            plaintext byte considered as int value.
        - target (str):
            target of the attack ('SBOX_OUT' for SBox Output).
            More targets in future (e.g. Hamming Weights, Key, ...).

    Returns:
        int np.ndarray containing the key-bytes relative to each possible value 
        of the sbox-output w.r.t. the given plaintext byte.
    """ 

    possible_values = range(256)
    
    if target == 'SBOX_IN': # Directly sbox-in values
        sbox_in = np.array(possible_values)
    elif target == 'SBOX_OUT': # Sbox-out values: inverse-SubBytes needed
        rows, cols = to_coords(possible_values)
        sbox_in = constants.INV_SBOX_DEC[rows, cols]
    else:
        pass # Future implementations (target = HW, target = KEY, ...)

    # Inverse-AddRoundKey
    key_bytes = sbox_in ^ pltxt_byte # Shape: (1 x 256)

    return np.array(key_bytes)
