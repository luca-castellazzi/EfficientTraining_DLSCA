# Script to test Tensorflow installation.

import tensorflow as tf
import tensorflow.python.platform.build_info as build


def main():
    
    print()
    print('#####################################################\n')
    
    print(f'Full info: {build.build_info}\n')
    print(f'CUDA version: {build.build_info["cuda_version"]}\n')
    print(f'CUDNN version: {build.build_info["cudnn_version"]}\n')
    print(f'GPU: {tf.config.list_physical_devices("GPU")}\n')
    
    print('#####################################################\n')
    
if __name__ == '__main__':
    main()