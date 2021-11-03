import os

import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset

from control_simp.utils import TokenFilter
from control_simp.data.loading import LazyPreproDataset


def pretokenize(model, data, save_dir, x_col="complex", y_col="simple", max_samples=None, chunk_size=32):
    if max_samples is not None:
        data = data[:max_samples]

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    dm = BartDataModule(model.tokenizer, hparams=model.hparams)

    # number of chunks needed
    chunk_count = int(len(data)/chunk_size)+1

    for _, chunk in enumerate(np.array_split(data, chunk_count)):
        tokd = dm.preprocess(list(chunk[x_col]), list(chunk[y_col]), pad_to_max_length=False)
        i = 0
        for j, _ in chunk.iterrows():
            a = torch.tensor(tokd["input_ids"][i])
            b = torch.tensor(tokd["attention_mask"][i])
            c = torch.tensor(tokd["labels"][i])
            x = torch.stack([a, b, c])
            torch.save(x, f"{save_dir}/{j}.pt")
            i += 1


class BartDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, hparams):
        super().__init__()
        self.tokenizer = tokenizer

        # set hyperparams
        self.hparams = hparams
        self.x_col = self.hparams.x_col
        self.y_col = self.hparams.y_col
        self.batch_size = self.hparams.batch_size
        self.data_file = self.hparams.data_file
        self.max_samples = self.hparams.max_samples  # defaults to no restriction
        self.train_split = self.hparams.train_split  # default split will be 90/5/5
        self.val_split = min(self.hparams.val_split, 1-self.train_split)
        self.val_file = self.hparams.val_file
        self.max_source_length = self.hparams.max_source_length
        self.max_target_length = self.hparams.max_target_length
        self.train_data_dir = self.hparams.train_data_dir
        self.valid_data_dir = self.hparams.valid_data_dir

    def prepare_data(self):
        # NOTE: shouldn't assign state in here
        pass

    def setup(self, stage):
        # read and prepare input data
        self.data = pd.read_csv(self.data_file)
        self.data = self.data.sample(frac=1)[:min(self.max_samples, len(self.data))] # NOTE: this will actually exclude the last item
        if self.val_file is not None:
            print("Loading specific validation samples...")
            self.validate = pd.read_csv(self.val_file)
        print("All data loaded.")

        # train, validation, test split
        if self.val_file is None:
            train_span = int(self.train_split * len(self.data))
            val_span = int((self.train_split + self.val_split) * len(self.data))
            self.train, self.validate, self.test = np.split(
                self.data, [train_span, val_span])
        else:
            self.train = self.data
            self.test = self.data[:12] # arbitrarily have 12 test samples as precaution

        self.build_datasets()

    def build_datasets(self):
        if self.train_data_dir is None:
            # preprocess datasets
            self.train = self.build_tensor_dataset(self.preprocess(
                list(self.train[self.x_col]), list(self.train[self.y_col])))
            self.validate = self.build_tensor_dataset(self.preprocess(
                list(self.validate[self.x_col]), list(self.validate[self.y_col])))
            self.test = self.build_tensor_dataset(self.preprocess(
                list(self.test[self.x_col]), list(self.test[self.y_col])))
        else:
            self.train = LazyPreproDataset(self.train, self.train_data_dir)
            if self.val_file is not None:
                self.validate = LazyPreproDataset(self.validate, self.valid_data_dir)
            else:
                self.validate = LazyPreproDataset(self.validate, self.train_data_dir)
            self.test = LazyPreproDataset(self.test, self.train_data_dir)

    def build_tensor_dataset(self, data):
        return TensorDataset(
            data['input_ids'],
            data['attention_mask'],
            data['labels']
        )

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, 
                            num_workers=os.cpu_count(), pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.validate, batch_size=self.batch_size, num_workers=1, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=1, pin_memory=True)

    def preprocess(self, source_sequences, target_sequences, pad_to_max_length=True, return_tensors="pt"):
        """Transforms data into tokenized input/output sequences."""
        source_sequences = TokenFilter(max_len=self.max_source_length, blacklist=["<SEP>"])(source_sequences)
        target_sequences = TokenFilter(max_len=self.max_target_length, blacklist=["<SEP>"])(target_sequences)

        transformed_x = self.tokenizer(source_sequences, max_length=self.max_source_length,
                            padding=pad_to_max_length, truncation=True, 
                            return_tensors=return_tensors, add_prefix_space=True)
        transformed_y = self.tokenizer(target_sequences, max_length=self.max_target_length,
                            padding=pad_to_max_length, truncation=True, 
                            return_tensors=return_tensors, add_prefix_space=True)

        return {
            "input_ids": transformed_x['input_ids'],
            "attention_mask": transformed_x['attention_mask'],
            "labels": transformed_y['input_ids'],
        }