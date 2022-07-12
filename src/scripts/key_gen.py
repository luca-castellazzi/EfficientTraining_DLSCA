# Random generation of the keys used in MDM

import sys
import random
import json


def main():
    n_keys = int(sys.argv[1])

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


    # Original keys
    #test_key = 'c4552e063bfbff50266d237d91654ef5' #K1
    #k2 = '5d634de10ee32ba7bba19f4e882f4965'
    #k3 = '6f5691ef569b829c11ad31fa8cfda643'  

    # New keys
    #random.seed(24)
    #keys = [test_key, k2, k3]
    #for i in range(n_keys):
        #k = []
        #for j in range(16):
            #byte = hex(random.randint(0, 255)).replace('0x', '')
            #if len(byte) == 1:
                #byte = f'0{byte}'
            #k.append(byte)
        #keys.append(''.join(k))


if __name__ == '__main__':
    main()
