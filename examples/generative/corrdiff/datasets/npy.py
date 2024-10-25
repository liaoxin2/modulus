import numpy as np
import torch
from torch.utils.data import Dataset

class NpyDataset(Dataset):
    def __init__(self, data_file, labels_file, transform=None, **kwargs):
        """
        Args:
            data_file (str): Path to the NumPy file containing the input data.
            labels_file (str): Path to the NumPy file containing the labels.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = np.load(data_file)
        self.labels = np.load(labels_file)
        self.transform = transform

    def __len__(self):
        # Return the size of the dataset
        return len(self.data)

    def __getitem__(self, idx):
        # Load the data and label at specified index
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample_data = self.data[idx]
        sample_label = self.labels[idx]

        # Apply any transformations if specified
        if self.transform:
            sample_data = self.transform(sample_data)
            sample_label = self.transform(sample_label)

        return (sample_label,sample_data,idx)

# # Example usage
# data_file = '/public/home/huanggang/data/lpy/GFSCOLO200266.npy'
# labels_file = '/public/home/huanggang/data/lpy/UVUS01.npy'
# dataset = CustomDataset(data_file, labels_file)

# # Access the first data and label pair
# data, label = dataset[0]
# print(dataset)
# print(data.shape)
# print(label.shape)