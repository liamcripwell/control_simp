import torch
from torch import Dataset


class LazyTensorDataset(Dataset):

    def __init__(self, df, x_col, y_col, features, transform):
        self.df = df
        self.x_col = x_col
        self.y_col = y_col
        self.features = features
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        seq = item[self.x_col]
        label = item[self.y_col]
        data = self.transform([seq], [label])
        
        return torch.Tensor([data[f] for f in self.features])