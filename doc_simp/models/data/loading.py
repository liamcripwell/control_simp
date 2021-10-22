import torch
from torch.utils.data import Dataset


class LazyTensorDataset(Dataset):

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

        # NOTE: we're assume this function expects a mini-batch so we wrap in a list
        # and later extract the 0th index item
        data = self.transform([seq], [label])

        # pad to a fixed length to avoid dim issues when batching
        seq_len = len(data["input_ids"][0])
        if seq_len < self.fixed_len:
            data["input_ids"] = torch.cat(
                (data["input_ids"][0], torch.ones(self.fixed_len - seq_len, dtype=int)))
            data["attention_mask"] = torch.cat(
                (data["attention_mask"][0], torch.zeros(self.fixed_len - seq_len, dtype=int)))
            data["labels"] = data["labels"][0]
        else:
            data = {
                k: v[0][:self.fixed_len] if v[0].dim() > 0 else v[0] 
                for k, v in data.items()}

        return tuple([data[f] for f in self.features])