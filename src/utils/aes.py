import numpy as np

from helpers import int_to_hex, to_coords
import constants


def labels_from_key(plaintext, key, target):
    
    """ 
    Emulates AES-128 in order to generate target labels relative to the given 
    plaintext and key.

    Parameters: 
        - plaintext (np.array):
            Integer-version of the plaintext used during the encryption.
        - key (np.array): 
            Integer-version of the key used during the encryption.
        - target (str):
            Target of the attack.

    Returns:
        - labels (np.array):
            Integer-version of the target labels.
    """
    
    # AddRoundKey
    sbox_in = plaintext ^ key

    # Labeling
    if target == 'SBOX_IN':
        labels = sbox_in
    elif target == 'SBOX_OUT':
        # SubBytes
        rows, cols = to_coords(sbox_in)
        sbox_out = constants.SBOX_DEC[rows, cols]
        labels = sbox_out
    elif target == 'KEY':
        labels = key
    else:
        pass # Future implementations (target = HW, target = KEY, ...)

    return labels
    

def key_from_labels(pltxt_byte, target):

    """ 
    Recovers the key relative to each possible value of the attack target, 
    given a plaintext byte.
    
    Parameters:
        - pltxt_byte (int):
            Single plaintext byte used during the encryption.
        - target (str):
            Target of the attack.
            
    Returns:
        - key_bytes (np.array):
            Key-bytes relative to each possible value of the attack target
    """ 

    possible_values = range(256)
    
    if target == 'SBOX_IN': # Directly sbox-in values
        sbox_in = np.array(possible_values)
    elif target == 'SBOX_OUT': # Sbox-out values: inverse-SubBytes needed
        rows, cols = to_coords(possible_values)
        sbox_in = constants.INV_SBOX_DEC[rows, cols]
    elif target == 'KEY':
        pass # This function is bypassed if the target is 'KEY'
    else:
        pass # Future implementations (target = HW, target = KEY, ...)

    # Inverse-AddRoundKey
    key_bytes = np.array(sbox_in ^ pltxt_byte) # Shape: (1 x 256)

    return key_bytes
