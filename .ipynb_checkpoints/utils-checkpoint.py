import os
import numpy as np
from torch import nn, Tensor
from typing import Optional, Any, Union, Callable, Tuple
import torch
import pandas as pd
from pathlib import Path


def generate_square_subsequent_mask(dim1: int, dim2: int) -> Tensor:
    """
    Generates an upper-triangular matrix with -inf, and zeros on the diagonal.
    
    Modified from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    
    Args:
        dim1 (int): Target sequence length for both source (src) and target (tgt) masking.
        dim2 (int): For source (src) masking, this is the encoder sequence length (length of the input sequence to the model).
                    For target (tgt) masking, this is the target sequence length.
    
    Return:
        torch.Tensor: A matrix of shape [dim1, dim2].
    """
    return torch.triu(torch.ones(dim1, dim2) * float('-inf'), diagonal=1)


def get_indices_input_target(num_obs, input_len, step_size, forecast_horizon, target_len):

    """
    Produce start and end indices for all sub-sequences to split the data for training.
    
    Returns a tuple:
    1) Index of the first element in the input sequence.
    2) Index of the last element in the input sequence.
    3) Index of the first element in the target sequence.
    4) Index of the last element in the target sequence.
    
    Args:
        num_obs (int): Total number of observations in the dataset.
        input_len (int): Length of each input sub-sequence.
        step_size (int): Size of each step as the data sequence is traversed.
        forecast_horizon (int): Distance from the last input index to the first target index.
        target_len (int): Length of the target/output sequence.
    """

    input_len = round(input_len) # just a precaution
    start_position = 0
    stop_position = num_obs-1 # because of 0 indexing
    
    subseq_first_idx = start_position
    subseq_last_idx = start_position + input_len
    target_first_idx = subseq_last_idx + forecast_horizon
    target_last_idx = target_first_idx + target_len 
    print("target_last_idx is {}".format(target_last_idx))
    print("stop_position is {}".format(stop_position))
    indices = []
    while target_last_idx <= stop_position:
        indices.append((subseq_first_idx, subseq_last_idx, target_first_idx, target_last_idx))
        subseq_first_idx += step_size
        subseq_last_idx += step_size
        target_first_idx = subseq_last_idx + forecast_horizon
        target_last_idx = target_first_idx + target_len

    return indices

def get_indices_entire_sequence(data: pd.DataFrame, window_size: int, step_size: int) -> list:

    """
    Produce start and end indices for sub-sequences in a dataset.
    
    Returns a list of tuples, where each tuple is (start_idx, end_idx) for a sub-sequence.
    These tuples are used to slice the dataset into sub-sequences, which can then be further
    processed for input and target sequences.
    
    Args:
        num_obs (int): Total number of observations (time steps) in the dataset.
        window_size (int): Desired length of each sub-sequence (input_sequence_length + target_sequence_length).
                          For example, if considering the past 100 steps to predict the next 50, window_size = 150.
        step_size (int): Size of each step as the data sequence is traversed by the moving window.
                         If 1, the first sub-sequence will be [0:window_size], and the next will be [1:window_size].
    
    Return:
        indices (list): List of tuples representing start and end indices for sub-sequences.
    """

    stop_position = len(data)-1 # 1- because of 0 indexing
    
    # Start the first sub-sequence at index position 0
    subseq_first_idx = 0
    
    subseq_last_idx = window_size
    
    indices = []
    
    while subseq_last_idx <= stop_position:

        indices.append((subseq_first_idx, subseq_last_idx))
        
        subseq_first_idx += step_size
        
        subseq_last_idx += step_size

    return indices

def read_data(data_dir: Union[str, Path] = "data",  timestamp_col_name: str="timestamp") -> pd.DataFrame:
    """
    Read data from a CSV file and return a pd.DataFrame object.
    
    Args:
        data_dir (str or Path): Path to the directory containing the data.
        target_col_name (str): Name of the column containing the target variable.
        timestamp_col_name (str): Name of the column or named index containing timestamps.
    """

    # Ensure that `data_dir` is a Path object
    data_dir = Path(data_dir)

    # Read csv file
    csv_files = list(data_dir.glob("*.csv"))
    
    if len(csv_files) > 1:
        raise ValueError("data_dir contains more than 1 csv file. Must only contain 1")
    elif len(csv_files) == 0:
    	raise ValueError("data_dir must contain at least 1 csv file.")

    data_path = csv_files[0]

    print("Reading file in {}".format(data_path))

    data = pd.read_csv(
        data_path, 
        parse_dates=[timestamp_col_name], 
        index_col=[timestamp_col_name], 
        infer_datetime_format=True,
        low_memory=False
    )

    # Make sure all "n/e" values have been removed from df. 
    if is_ne_in_df(data):
        raise ValueError("data frame contains 'n/e' values. These must be handled")

    data = to_numeric_and_downcast_data(data)

    # Make sure data is in ascending order by timestamp
    data.sort_values(by=[timestamp_col_name], inplace=True)

    return data

def is_ne_in_df(df:pd.DataFrame):
    """
    Some raw data files contain cells with "n/e". This function checks whether
    any column in a df contains a cell with "n/e". Returns False if no columns
    contain "n/e", True otherwise
    """
    
    for col in df.columns:

        true_bool = (df[col] == "n/e")

        if any(true_bool):
            return True

    return False


def to_numeric_and_downcast_data(df: pd.DataFrame):
    """
    Downcast columns in df to smallest possible version of it's existing data
    type
    """
    fcols = df.select_dtypes('float').columns
    
    icols = df.select_dtypes('integer').columns

    df[fcols] = df[fcols].apply(pd.to_numeric, downcast='float')
    
    df[icols] = df[icols].apply(pd.to_numeric, downcast='integer')

    return df