import numpy as np
import h5py
import hdf5plugin
import torch
from torch.utils.data import TensorDataset, random_split


# ----------- Preprocessing Pipeline for the ATLAS Jet Tagging Dataset ---------------


def list_contents(group, prefix=''):
    """
    Recursively lists the contents of an HDF5 group.
    """
    for key in group.keys():
        item = group[key]
        path = f'{prefix}/{key}'
        print(path)

        # If the item is a Group, recurse into it
        if isinstance(item, h5py.Group):
            list_contents(item, path)
        else:  # Item is a Dataset
            print(f' - Dataset shape: {item.shape}, Dataset type: {item.dtype}')




def extract(filename, verbose=False):
    """
    Given the h5 file containing the ATLAS data, extract all the data.

    Params:
    ------------------------------
    filename: path (str) to h5 file
    verbose: print extra information? (bool, default False)

    Returns:
    ------------------------------
    dictionary mapping the key name to the data it contains
    """

    mapping = {}
    with h5py.File(filename, "r") as f:
        if verbose:
            list_contents(f)

        for key in f.keys():

            mapping[key] = f[key][:]

    return mapping


def get_data(filename, attribute):
    """
    Given the h5 file containing the ATLAS data, extract all the data corresponding
    to the appropriate attribute

    Allowed values for attribute are: 'jet', 'constituents', and 'high_level'

    Return the data, labels, weights, and feature names
    """

    jet_keys = ['fjet_pt', 'fjet_eta', 'fjet_phi', 'fjet_m']
    const_keys = ['fjet_clus_pt', 'fjet_clus_eta', 'fjet_clus_phi', 'fjet_clus_E']
    hl_keys = ['fjet_C2', 'fjet_D2', 
               'fjet_ECF1', 'fjet_ECF2', 
               'fjet_ECF3', 'fjet_L2', 
               'fjet_L3', 'fjet_Qw', 
               'fjet_Split12', 'fjet_Split23', 
               'fjet_Tau1_wta', 'fjet_Tau2_wta', 
               'fjet_Tau3_wta', 'fjet_Tau4_wta', 'fjet_ThrustMaj']
    
    use_keys = jet_keys if attribute == 'jet' else (const_keys if attribute == 'constituents' else (hl_keys if attribute == 'high_level' else None))
    
    mapping = extract(filename)

    data = []
    for k in use_keys:
        data.append(mapping[k])

    return np.asarray(data).T, np.asarray(mapping["labels"]), np.asarray(mapping["training_weights"]), np.asarray(use_keys)


def preprocess_data(data, labels, train_split, val_split, seed):
    """
    Given data of shape [INPUT_SIZE, NUM_FEATURES] and labels of shape [INPUT_SIZE]
    Apply basic preprocessing steps:
    - standardize
    - cast to PyTorch TensorDataset
    - split into training, validation, and testing

    returns: training_dataset, val_dataset, test_dataset: PyTorch TensorSubset objects
    """

    # Standardization (mean=0, std=1)
    mean = torch.mean(data, dim=0)
    std = torch.std(data, dim=0)
    data_standardized = (data - mean) / std

    # Cast to PyTorch tensors
    data = torch.tensor(data, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)

    # Wrap into a TensorDataset
    dataset = TensorDataset(data, labels)

    # Split into train, validation, and test sets
    total_length = len(dataset)
    train_length = int(train_split * total_length)
    val_length = int(val_split * total_length)
    test_length = total_length - train_length - val_length  # To handle any rounding issues

    training_dataset, val_dataset, test_dataset = random_split(dataset, [train_length, val_length, test_length], generator=torch.Generator().manual_seed(seed))

    return training_dataset, val_dataset, test_dataset