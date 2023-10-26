import numpy as np

def bytes_to_readable(size_in_bytes, digits=2):
    """
    Convert the number of bytes to a human-readable string format.

    Parameters:
    size_in_bytes (int): The size in bytes that you want to convert.
    digits (int, optional): The number of decimal places to format the output. Default is 2.

    Returns:
    str: A string representing the human-readable format of the size.

    Examples:
    >>> bytes_to_readable(123456789)
    '117.74 MB'

    >>> bytes_to_readable(9876543210)
    '9.20 GB'

    >>> bytes_to_readable(123456789, digits=4)
    '117.7383 MB'

    >>> bytes_to_readable(999, digits=1)
    '999.0 B'
    """
    for unit in ['B', 'kB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB']:
        if abs(size_in_bytes) < 1024.0:
            return f"{size_in_bytes:.{digits}f} {unit}"
        size_in_bytes /= 1024.0
    return f"{size_in_bytes:.{digits}f} YB"


def array_memory_usage(array, digits=2):
    """
    Return a string representation of the memory usage of a NumPy array.

    Parameters:
    array (np.ndarray): The NumPy array for which to calculate memory usage.
    digits (int, optional): The number of decimal places to format the output. Default is 2.

    Returns:
    str: A string representing the memory usage of the array.

    Examples:
    >>> arr = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    >>> array_memory_usage(arr)
    '20.00 B'

    >>> arr = np.random.rand(1000, 1000)
    >>> array_memory_usage(arr, digits=3)
    '7.630 MB'
    """
    size_in_bytes = array.nbytes
    return bytes_to_readable(size_in_bytes)

def print_memory_usage(data, indent=0, digits=2):
    """
    Print the memory usage of a nested dictionary of NumPy arrays.

    Parameters:
    data (dict): The nested dictionary of NumPy arrays or tuples of NumPy arrays.
    indent (int, optional): The indentation level for pretty printing. Default is 0.
    digits (int, optional): The number of decimal places to format the output. Default is 2.

    Returns:
    int: The total memory usage of the NumPy arrays in bytes.

    Examples:
    >>> data = {
    ...     'array1': np.array([1, 2, 3], dtype=np.float64),
    ...     'tuple_arrays': (np.array([1, 2]), np.array([[3, 4], [5, 6]])),
    ...     'nested_dict': {
    ...         'array2': np.array([4, 5, 6], dtype=np.int32),
    ...         'array3': np.array([[1, 2], [3, 4]], dtype=np.float32)
    ...     }
    ... }
    >>> print_memory_usage(data)
    array1: 24.00 B
    tuple_arrays:
        0: 16.00 B
        1: 64.00 B
    nested_dict:
        array2: 12.00 B
        array3: 32.00 B
    Total Memory Usage: 148.00 B
    """
    total_memory_usage = 0

    for key, value in data.items():
        if isinstance(value, np.ndarray):
            size_in_bytes = value.nbytes
            readable_size = bytes_to_readable(size_in_bytes, digits)
            print('    ' * indent + f"{key}: {readable_size}")
            total_memory_usage += size_in_bytes
        elif isinstance(value, tuple):
            print('    ' * indent + str(key) + ':')
            for i, array in enumerate(value):
                if isinstance(array, np.ndarray):
                    size_in_bytes = array.nbytes
                    readable_size = bytes_to_readable(size_in_bytes, digits)
                    print('    ' * (indent + 1) + f"{i}: {readable_size}")
                    total_memory_usage += size_in_bytes
                else:
                    print('    ' * (indent + 1) + f"{i}: Not a NumPy array")
        elif isinstance(value, dict):
            print('    ' * indent + str(key) + ':')
            total_memory_usage += print_memory_usage(value, indent + 1, digits)
        else:
            print('    ' * indent + f"{key}: Not a NumPy array, tuple, or nested dictionary")

    if indent == 0:
        print("Total Memory Usage:", bytes_to_readable(total_memory_usage, digits))
    return total_memory_usage