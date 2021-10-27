import torch
import numpy as np
from torch.utils.data import Dataset


class LazyTensorDataset(Dataset):
    """
    A DataSet that returns tensors after applying some transformation to input.
    Transformations will be lazily applied as items are accessed.
    """

    def __init__(self, df, x_col, y_col, features, transform, fixed_len=64):
        self.df = df
        self.x_col = x_col
        self.y_col = y_col
        self.features = features
        self.transform = transform
        self.fixed_len = fixed_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        seq = item[self.x_col]
        label = item[self.y_col]

        # NOTE: we don't use any lists or dicts in this object because it
        # can lead to memory consumption issues when using distributed training.
        # See https://github.com/pytorch/pytorch/issues/13246
        data = self.transform(np.array([seq]), np.array([label]))

        # adjust to fixed length tensors to avoid dim issues when batching
        seq_len = len(data["input_ids"][0])
        if seq_len < self.fixed_len:
            data["input_ids"] = torch.cat(
                (data["input_ids"][0], torch.ones(self.fixed_len - seq_len, dtype=int)))
            data["attention_mask"] = torch.cat(
                (data["attention_mask"][0], torch.zeros(self.fixed_len - seq_len, dtype=int)))
            data["labels"] = data["labels"][0]
        else:
            data = pd.Series({
                k: v[0][:self.fixed_len] if v[0].dim() > 0 else v[0] 
                for k, v in data.items()})

        return np.array([data[f] for f in self.features])