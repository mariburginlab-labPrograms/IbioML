import numpy as np
from itertools import groupby

def find_stationary_window(y, threshold):
    """
    Finds the stationary window in a position signal.

    Parameters:
    y (array): Position array to evaluate (should be 1D or flattened).
    threshold (float): Threshold to define how small the change in y must be to be considered stationary.

    Returns:
    tuple: Start and end indices of the longest stationary window, or (None, None) if no window is detected.
    """
    # Flatten y to ensure it's a 1D array
    y = y.flatten()

    # Calculate the derivative of the position signal
    y_diff = np.diff(y)

    # Identify indices where the derivative is less than the threshold
    stationary_indices = np.where(np.abs(y_diff) < threshold)[0]

    # Group consecutive indices to find continuous stationary windows
    groups = [list(group) for key, group in groupby(stationary_indices, key=lambda i, c=iter(range(len(stationary_indices))): next(c) - i)]

    # Find the longest stationary window
    longest_window = max(groups, key=len) if groups else []

    if longest_window:
        start_index = longest_window[0]
        end_index = longest_window[-1]
        return start_index, end_index
    else:
        return None, None


def map_to_discrete_hidden_size(value):
    """
    Maps a continuous value between 64 and 512 to the nearest value in [64, 128, 256, 512]
    """
    discrete_values = [64, 128, 256, 512]
    return min(discrete_values, key=lambda x: abs(x - value))