import numpy as np

from helpers import int_to_hex
from targets import TargetEnum, hamming_weights
import constants

def compute_labels(plaintext, key, target='SBO'):
    
    """ 
    Generates the labels associated to each byte of the given plaintext/key 
    w.r.t. the given target.

    Parameters: 
        - plaintext (int np.array):
            plaintext considered as list of int values, each one relative to 
            a single byte of the hex version.
        - key (int np.array): 
            encryption key considered as list of int values, each one relative
            to a single byte of the hex version.
        - target (str):
            target of the attack (either 'SBO', for SBox Output, or 'HW', for 
            Hamming Weight of the SBox Output).
            Default 'SBO'.

    Returns:
        int np.array containing the 16 labels (one per byte of plaintext/key)
        relative to the given target.
        In case of 'SBO' target, the output is the result of the SBox-lookup.
        In case of 'HW' target, the output is the Hamming Weight of the result 
        of the SBox-lookup.
    """
    
    sbox_out = compute_sbox_out(plaintext, key)

    if target == TargetEnum.SBO:
        return sbox_out
    elif target == TargetEnum.HW:
        return hamming_weights(sbox_out)


def compute_sbox_out(plaintext, key):

    """ 
    Computes the result of the SBox-lookup performed with the given plaintext and key.

    Parameters:
        - plaintext (int np.array):
            plaintext considered as list of int values, each one relative to 
            a single byte of the hex version.
        - key (int np.array): 
            encryption key considered as list of int values, each one relative
            to a single byte of the hex version.
    
    Returns: 
        - sbox_out (int np.array):
            SBox-lookup outputs relative to the given plaintext and key.
    """

    sbox_in = add_round_key(plaintext, key) # AES AddRoundKey
    sbox_out = sub_bytes(sbox_in) # AES SubBytes

    return sbox_out
        
        
def add_round_key(plaintext, key):

    """ 
    Implements AES AddRoundKey (plaintext XOR key).

    Parameters:
        - plaintext (int np.array):
            plaintext considered as list of int values, each one relative to 
            a single byte of the hex version.
        - key (int np.array): 
            encryption key considered as list of int values, each one relative
            to a single byte of the hex version.

    Returns: 
        int np.array containing each byte of the XOR between the given plaintext
        and key. 
    """

    return plaintext ^ key
    
    
def sub_bytes(sbox_in):

    """ 
    Implements AES SubBytes (SBox-lookup).
    
    Input:
        - sbox_in (int np.array): 
            list of each byte of the XOR between a plaintext and a key. 

    Returns: 
        int np.array containing the result of the SBox-lookup.
    """

    sbox_in_hex = int_to_hex(sbox_in) # Convert the SBox input to well-formatted hex (each byte independently)

    rows = [int(byte[0], 16) for byte in sbox_in_hex] # The first 4 bits (of each byte) are the row index 
    cols = [int(byte[1], 16) for byte in sbox_in_hex] # The remaining 4 bits (of each byte) are the col index 

    return constants.SBOX_DEC[rows, cols]
