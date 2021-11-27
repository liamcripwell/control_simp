import random

import torch
from torch.utils.data import Dataset


class LazyClassifierDataset(Dataset):
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

        # NOTE: we're assume this function expects a mini-batch so we wrap 
        # inputs in a list and later extract the 0th index item
        data = self.transform([seq], [label])

        # adjust to fixed length tensors to avoid dim issues when batching
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


class LazyPreproDataset(Dataset):

    def __init__(self, df, data_dir, label_col=None, ctrl_tok_ids=None, mtl_tok_ids=None):
        self.df = df
        self.data_dir = data_dir
        self.label_col = label_col
        self.ctrl_tok_ids = ctrl_tok_ids
        self.mtl_tok_ids = mtl_tok_ids

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        tensors = torch.load(f"{self.data_dir}/{idx}_x.pt")

        # insert control tokens to input if needed
        if self.mtl_tok_ids is not None:
            x_ = tensors[0]
            m_ = tensors[1]

            # load tokenized y sequence if generation task
            y_ = torch.load(f"{self.data_dir}/{idx}_y.pt")

            # construct contrived y sequence for classification task
            label = self.df.iloc[idx][self.label_col]
            ctrl_tok = self.ctrl_tok_ids[label]
            l_ = torch.tensor([0, ctrl_tok, 2]) # a single special-token sequence
                
            item = (x_, m_, y_, l_)
        else:
            item = tuple([t for t in tensors])
            # load tokenized y sequence if standard generation task
            item += (torch.load(f"{self.data_dir}/{idx}_y.pt"),)

        return item

    def insert_mtl_tok(self, x, label):
        # substitute existing operation control-token for mtl-token
        mtl_tok_id = self.mtl_tok_ids[label]
        x[1] = mtl_tok_id
        return x