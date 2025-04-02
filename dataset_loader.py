import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm.auto import tqdm

def MinMaxScale(arr, min_val=-1, max_val=1):
    mm_scaler = None
    print(len(arr[0]))
    mm_scaler = MinMaxScaler(feature_range=(min_val, max_val))
    mm_scaler.fit(arr)
    return mm_scaler.transform(arr)

def sine_data_generation(no, seq_len, dim, seed, normalize):
    """Sine data generation.

    Args:
        - no: the number of samples
        - seq_len: sequence length of the time-series
        - dim: feature dimensions

    Returns:
        - data: generated data
    """
    # Store the state of the RNG to restore later.
    st0 = np.random.get_state()
    np.random.seed(seed)

    # Initialize the output
    data = list()
    # Generate sine data
    for i in tqdm(range(0, no), total=no, desc="Sampling sine-dataset"):
        # Initialize each time-series
        temp = list()
        # For each feature
        for k in range(dim):
            # Randomly drawn frequency and phase
            freq = np.random.uniform(0, 0.1)
            phase = np.random.uniform(0, 0.1)

            # Generate sine signal based on the drawn frequency and phase
            temp_data = [np.sin(freq * j + phase) for j in range(seq_len)]
            temp.append(temp_data)

        # Align row/column
        temp = np.transpose(np.asarray(temp))
        # Normalize to [0,1]
        if normalize:
            temp = (temp + 1) * 0.5
        # Stack the generated data
        data.append(temp)

    # Restore RNG.
    np.random.set_state(st0)
    data = np.array(data)

    return data

def create_sliding_windows(ori_data, seq_len, stp):
    temp_data = []  
    for i in range(0, len(ori_data) - seq_len, stp):
        _x = ori_data[i:i + seq_len]
        temp_data.append(_x)
    return temp_data

def get_sine_train_data(seq_len, no=10000, dim=5, seed=123, normalize=False):
    return sine_data_generation(no, seq_len, dim, seed, normalize)


def get_sine_test_data(seq_len, no=5000, dim=5, seed=420, normalize=False):
    return sine_data_generation(no, seq_len, dim, seed, normalize)

def load_dataset(data_name, sequence_length):
    stp=1
    if data_name =='ett':
        ori_data = np.loadtxt('datasets/ett/ETTm2.csv', delimiter = ",",skiprows = 1, usecols = (1,2,3,4,5,6,7))
        stp = int(0.04*sequence_length)
    elif data_name == 'ecg':
        ori_data = np.loadtxt('datasets/ecg/14157.csv', delimiter = ",",skiprows = 1, usecols = (0,1))
        ori_data = ori_data[:500000]
        stp = int(0.1*sequence_length)
    elif data_name =='em':
        ori_data = np.loadtxt('datasets/lenze/bhn/900rpm/H1.5.csv', delimiter = ",",skiprows = 1, usecols = (1,2,3,4,5,6,7,9))
        stp = int(0.1*sequence_length)
    elif data_name =='metropt3':
        ori_data = np.loadtxt('datasets/metropt3/MetroPT3.csv', delimiter = ",",skiprows = 1, usecols = (2,3,4,5,6,7,8))
        ori_data = ori_data[:500000]
        stp = int(0.1*sequence_length)
    elif data_name =='sine':
        ori_data = get_sine_train_data(sequence_length)

    if data_name != 'sine':
        ori_data = MinMaxScale(ori_data)
        ori_data = create_sliding_windows(ori_data, sequence_length, stp)

    return np.asarray(ori_data)

    
