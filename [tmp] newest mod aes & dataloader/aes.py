# Basics
import numpy as np

# Custom
from helpers import to_coords
import constants


def key(plaintext, key):
    return key

def sbox_in(plaintext, key):
    # AddRoundKey
    sbox_in = plaintext ^ key
    return sbox_in

def sbox_out(plaintext, key):
    # AddRoundKey
    sbox_in = plaintext ^ key
    # SubBytes
    rows, cols = to_coords(sbox_in)
    sbox_out = constants.SBOX_DEC[rows, cols]
    return sbox_out

def hw_sbox_out(plaintext, key):
    # AddRoundKey
    sbox_in = plaintext ^ key
    # SubBytes
    rows, cols = to_coords(sbox_in)
    sbox_out = constants.SBOX_DEC[rows, cols]
    # HW Computation
    hw = [bin(val).replace('0b', '').count('1') for val in sbox_out]
    return hw

def hw_sbox_in(plaintext, key):
    # AddRoundKey
    sbox_in = plaintext ^ key
    # HW Computation
    hw = [bin(val).replace('0b', '').count('1') for val in sbox_in]
    return hw