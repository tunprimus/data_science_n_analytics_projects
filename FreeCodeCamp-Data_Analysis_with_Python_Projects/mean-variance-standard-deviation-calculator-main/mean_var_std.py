"""
Create a function named calculate() in mean_var_std.py that uses Numpy to output the mean, variance, standard deviation, max, min, and sum of the rows, columns, and elements in a 3 x 3 matrix.

The input of the function should be a list containing 9 digits. The function should convert the list into a 3 x 3 Numpy array, and then return a dictionary containing the mean, variance, standard deviation, max, min, and sum along both axes and for the flattened matrix.

The returned dictionary should follow this format:

{
    'mean': [axis1, axis2, flattened],
    'variance': [axis1, axis2, flattened],
    'standard deviation': [axis1, axis2, flattened],
    'max': [axis1, axis2, flattened],
    'min': [axis1, axis2, flattened],
    'sum': [axis1, axis2, flattened]
}
If a list containing less than 9 elements is passed into the function, it should raise a ValueError exception with the message: "List must contain nine numbers." The values in the returned dictionary should be lists and not Numpy arrays.

For example, calculate([0,1,2,3,4,5,6,7,8]) should return:

{
    'mean': [[3.0, 4.0, 5.0], [1.0, 4.0, 7.0], 4.0],
    'variance': [[6.0, 6.0, 6.0], [0.6666666666666666, 0.6666666666666666, 0.6666666666666666], 6.666666666666667],
    'standard deviation': [[2.449489742783178, 2.449489742783178, 2.449489742783178], [0.816496580927726, 0.816496580927726, 0.816496580927726], 2.581988897471611],
    'max': [[6, 7, 8], [2, 5, 8], 8],
    'min': [[0, 1, 2], [0, 3, 6], 0],
    'sum': [[9, 12, 15], [3, 12, 21], 36]
}
"""

import numpy as np

def calculate(list):
    if len(list) != 9:
        raise ValueError("List must contain nine numbers.")

    # Convert list to numpy array
    arr = np.array(list).reshape((3,3))
    
    NUM_AXIS = arr.shape[0] - 1

    # Calculate statistics
    mean = [arr.mean(axis=i).tolist() for i in range(NUM_AXIS)]
    variance = [arr.var(axis=i).tolist() for i in range(NUM_AXIS)]
    std = [arr.std(axis=i).tolist() for i in range(NUM_AXIS)]
    max_val = [arr.max(axis=i).tolist() for i in range(NUM_AXIS)]
    min_val = [arr.min(axis=i).tolist() for i in range(NUM_AXIS)]
    sum_val = [arr.sum(axis=i).tolist() for i in range(NUM_AXIS)]

    calculations = {
        'mean': [mean[0], mean[1], arr.mean()],
        'variance': [variance[0], variance[1], arr.var()],
        'standard deviation': [std[0], std[1], arr.std()],
        'max': [max_val[0], max_val[1], arr.max()],
        'min': [min_val[0], min_val[1], arr.min()],
        'sum': [sum_val[0], sum_val[1], arr.sum()],
    }
    # Return dictionary
    return calculations