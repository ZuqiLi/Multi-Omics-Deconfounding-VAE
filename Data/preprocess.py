import torch.utils.data as data
import torch

class ConcatDataset(data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)
    
def scale(X):
    '''Min-max normalization to [0,1] along columns'''
    X_min, _ = torch.min(X, dim=0, keepdim=True)
    X_max, _ = torch.max(X, dim=0, keepdim=True)
    X = (X - X_min) / (X_max - X_min)
    return X