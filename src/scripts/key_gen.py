# Random generation of the keys used in MDM

import sys
import random
import json


def main():

    """
    Generates a specified number of random keys for AES-128.
    Settings parameters (provided in order via command line):
        - n_keys: Number of keys to be generated
        
    The results are a command line print of the generated keys and a JSON file
    containing them.
    """
    
    _, n_keys = sys.argv
    
    n_keys = int(n_keys)

    random.seed(24)

    keys = {}
    for i in range(n_keys):
        key = hex(random.getrandbits(128)).replace('0x', '')
        if len(key) == 31:
            key = '0' + key

        keys[f'K{i}'] = key


    # Print the keys as byte-lists
    for k, el in keys.items():
        byte_list = [el[i:i+2] for i in range(0, 31, 2)]
        print(f"'{k}': {byte_list},")

    # Save the keys in a JSON file
    with open('/home/lcastellazzi/MDM32/keys.json', 'w') as jfile:
        json.dump(keys, jfile)


if __name__ == '__main__':
    main()
