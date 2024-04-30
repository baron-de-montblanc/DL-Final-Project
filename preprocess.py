import numpy as np
import h5py
import hdf5plugin
import torch
from torch.utils.data import TensorDataset, random_split, Subset
from torch_geometric.nn import knn_graph
from torch_geometric.data import Data, DataLoader
from helper import features_by_attribute


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


def get_data(filename, attribute, verbose=False):
    """
    Given the h5 file containing the ATLAS data, extract all the data corresponding
    to the appropriate attribute

    Allowed values for attribute are: 'jet', 'constituents', and 'high_level'

    Return the data, labels, weights, and feature names
    """
    
    use_keys = features_by_attribute(attribute)
    
    data = []
    ordered_keys = []  # since the dataset's native ordering differs from mine
    with h5py.File(filename, "r") as f:
        if verbose:
            list_contents(f)

        for key in f.keys():

            if key in use_keys:
                data.append(f[key][:])
                ordered_keys.append(key)
                
        try:  # the column name here isn't always consistent
            weights = np.asarray(f["training_weights"][:])
        except:
            weights = np.asarray(f["weights"][:])
            
        labels = np.asarray(f["labels"][:])

    if attribute == "jet":
        data = np.asarray(data).T

    elif attribute == "constituents":
        data = np.asarray(data)
        # reshape to make sure the input_size is first
        data = np.reshape(data, (data.shape[1],data.shape[0],data.shape[2]))

    return data, labels, weights, np.asarray(ordered_keys)



# --------------------- FCNN-specific functions -------------------------------------------


def preprocess_split(data, labels, weights, train_split, val_split, seed=None):
    """
    Given data of shape [INPUT_SIZE, NUM_FEATURES], labels of shape [INPUT_SIZE], and weights of shape [INPUT_SIZE]
    Apply basic preprocessing steps:
    - standardize
    - cast to PyTorch TensorDataset
    - split into training, validation, and testing

    Returns:
    - training_dataset, val_dataset, test_dataset: PyTorch TensorSubset objects
    """
    # Cast to tensor
    data = torch.tensor(data, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)
    weights = torch.tensor(weights, dtype=torch.float32)  # Ensure weights are also a tensor

    # Standardization (mean=0, std=1)
    mean = torch.mean(data, dim=0)
    std = torch.std(data, dim=0)
    data_standardized = (data - mean) / std

    # Wrap into a TensorDataset with weights
    dataset = TensorDataset(data_standardized, labels, weights)

    # Split into train, validation, and test sets
    total_length = len(dataset)
    train_length = int(train_split * total_length)
    val_length = int(val_split * total_length)
    test_length = total_length - train_length - val_length  # To handle any rounding issues

    if seed:
        training_dataset, val_dataset, test_dataset = random_split(
            dataset, 
            [train_length, val_length, test_length], 
            generator=torch.Generator().manual_seed(seed)
        )
    else:
        training_dataset, val_dataset, test_dataset = random_split(
            dataset, 
            [train_length, val_length, test_length]
        )

    return training_dataset, val_dataset, test_dataset




# ------------------------ GNN-specific functions -------------------------------------


class WeightedData(Data):
    def __init__(self, weight, **kwargs):
        super().__init__(**kwargs)
        self.weight = weight


def prepare_graphs(data, labels, weights, k, device, batch_size=64):
    """
    Given our input data, labels, and training weights, construct graphs with
    the KNN algorithm
    """

    # Convert data to PyTorch tensors and move to the appropriate device
    data_tensor = torch.tensor(data, dtype=torch.float).to(device)
    labels_tensor = torch.tensor(labels, dtype=torch.long).to(device)
    weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

    # Standardize the data (mean=0, std=1)
    mean = data_tensor.mean(dim=0, keepdim=True)
    std = data_tensor.std(dim=0, keepdim=True)
    data_tensor = (data_tensor - mean) / std

    # Create a dataset and dataloader for batch processing
    dataset = TensorDataset(data_tensor, labels_tensor, weights_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    graphs = []
    for data, label, weight in loader:
        x = data.transpose(1, 2)  # Ensure data is in the correct shape (N, C, num_features)

        # Construct graphs for each sample in the batch
        for i in range(x.size(0)):
            edge_index = knn_graph(x[i], k=k, loop=False)
            graph = WeightedData(weight=weight[i], x=x[i], edge_index=edge_index, y=label[i].unsqueeze(0))
            graphs.append(graph)

    return graphs


def split_graphs(graph_list, training_split, val_split):
    """
    Given a GraphDataset object, split it into training, validation, and test sets
    """

    # Calculate sizes for each split
    total_count = len(graph_list)
    train_count = int(training_split * total_count)
    valid_count = int(val_split * total_count)
    test_count = total_count - train_count - valid_count

    # Perform the split using random_split
    train_dataset, valid_dataset, test_dataset = random_split(graph_list, [train_count, valid_count, test_count])

    return train_dataset, valid_dataset, test_dataset