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

    def __init__(self, df, data_dir, clf=False, y_col=None, label_tok_ids=None):
        self.df = df
        self.data_dir = data_dir
        self.clf = clf
        self.y_col = y_col
        self.label_tok_ids = label_tok_ids

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        tensors = torch.load(f"{self.data_dir}/{idx}_x.pt")

        # insert control tokens to input if needed
        if not self.clf and self.label_tok_ids is not None:
            label = self.df.iloc[idx][self.y_col]
            tensors[0] = self.insert_control_tok(tensors[0], label)
            tensors[1] = torch.cat([tensors[1], torch.tensor([1])])

        item = tuple([t for t in tensors])
        if not self.clf:
            # load tokenized y sequence if generation task
            item += (torch.load(f"{self.data_dir}/{idx}_y.pt"),)
        else:
            # load label from df if classification task
            item += (torch.tensor(self.df.iloc[idx][self.y_col]),)

        return item

    def insert_control_tok(self, x, label):
        # inserts a label-dependent control token into the input sequence
        control_tok_id = self.label_tok_ids[label]
        return torch.cat([torch.tensor([0, control_tok_id]), x[1:]])