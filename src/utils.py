import torch
from torch.utils.data import Dataset, DataLoader

class FusionDataset(Dataset):
    def __init__(self, X_image, X_tabular, y):
        self.X_image = X_image
        self.X_tabular = X_tabular
        self.y = y

    def __len__(self):
        return len(self.X_image)

    def __getitem__(self, index):

        return self.X_image[index], self.X_tabular[index]