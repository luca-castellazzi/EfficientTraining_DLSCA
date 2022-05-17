def int_to_hex(int_values):

    """
    Converts int values into hex values, where the eventual 0 in front is explicit.
    
    Parameters:
        - int_values (int np.array or list): 
            int values to be converted to hex.
  
    Returns:
        - hex_values (str list): 
            element-to-element hex-conversion of the input.
            The eventual 0 in front is explicit.
    """

    hex_values = []
    for val in int_values:
        tmp = hex(val).replace('0x', '') # Get rid of '0x' that is generated by the cast to hex
        if len(tmp) == 1:
            tmp = f'0{tmp}' # Add 0 in front of the conversion if its len is 1
        hex_values.append(tmp)

    return hex_values


def hex_to_int(hex_str):
    
    """
    Converts the given hex number into an array of int, each one relative to
    a single byte of the input.

    Parameters:
        - hex_str (str):
            hex value to be converted.

    Returns:
        Conversion of the given hex value as an int list, where each value is
        relative to a single byte of the input.
    """

    split_hex_str = [hex_str[i:i+2] for i in range(0, len(hex_str), 2)]

    return [int(sb, 16) for sb in split_hex_str]