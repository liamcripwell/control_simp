import os

import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset

from control_simp.utils import TokenFilter
from control_simp.data.loading import LazyTensorDataset


def pretokenize(model, data, save_dir, x_col="complex", max_samples=None, chunk_size=32):
    if max_samples is not None:
        data = data[:max_samples]

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    dm = BertDataModule(model.tokenizer, hparams=model.hparams)

    # number of chunks needed
    chunk_count = int(len(data)/chunk_size)+1

    for _, chunk in enumerate(np.array_split(data, chunk_count)):
        tokd = dm.preprocess(list(chunk[x_col]), pad=False)
        i = 0
        for j, _ in chunk.iterrows():
            # currently only works for RoBERTa tokenizer
            a = torch.tensor(tokd["input_ids"][i])
            b = torch.tensor(tokd["attention_mask"][i])
            x = torch.stack([a, b])
            torch.save(x, f"{save_dir}/{j}.pt")
            i += 1


class BertDataModule(pl.LightningDataModule):

    MAX_LEN = 64

    def __init__(self, tokenizer, hparams=None):
        super().__init__()
        self.tokenizer = tokenizer

        if hparams != None:
            # set hyperparams
            self.hparams = hparams
            self.x_col = self.hparams.x_col
            self.y_col = self.hparams.y_col
            self.data_file = self.hparams.data_file
            self.batch_size = self.hparams.batch_size
            self.max_samples = self.hparams.max_samples  # defaults to no restriction
            self.train_split = self.hparams.train_split  # default split will be 90/5/5
            self.val_split = min(self.hparams.val_split, 1 - self.train_split)
            self.val_file = self.hparams.val_file
            # self.train_workers = self.hparams.train_workers

            self.model_type = self.hparams.model_type

    def prepare_data(self):
        # NOTE: shouldn't assign state in here
        pass

    def setup(self, stage):
        self.data = pd.read_csv(self.data_file)
        # self.data = self.data[:min(self.max_samples, len(self.data))]
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
        if self.hparams.lazy_loading:
            # create lazy dataset that tokenizes as samples are accessed
            self.train = LazyTensorDataset(
                self.train, self.x_col, self.y_col, ["input_ids", "attention_mask", "labels"], self.preprocess, self.MAX_LEN)
            self.validate = LazyTensorDataset(
                self.validate, self.x_col, self.y_col, ["input_ids", "attention_mask", "labels"], self.preprocess, self.MAX_LEN)
            self.test = LazyTensorDataset(
                self.validate, self.x_col, self.y_col, ["input_ids", "attention_mask", "labels"], self.preprocess, self.MAX_LEN)
        else:
            # tokenize all data
            self.train = self.build_tensor_dataset(self.preprocess(
                list(self.train[self.x_col]), list(self.train[self.y_col])))
            self.validate = self.build_tensor_dataset(self.preprocess(
                list(self.validate[self.x_col]), list(self.validate[self.y_col])))
            self.test = self.build_tensor_dataset(self.preprocess(
                list(self.test[self.x_col]), list(self.test[self.y_col])))
            
    def build_tensor_dataset(self, data):
        if self.model_type == "roberta":
            dataset = TensorDataset(
                data['input_ids'],
                data['attention_mask'],
                data['labels'])
        else:
            dataset = TensorDataset(
                data['input_ids'],
                data['attention_mask'],
                data['token_type_ids'],
                data['labels'])
        return dataset

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=self.train_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.validate, batch_size=self.batch_size, num_workers=1, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=1, pin_memory=True)

    def preprocess(self, seqs, labels=None, pad=True):
        seqs = TokenFilter(max_len=self.MAX_LEN, blacklist=["<SEP>"])(seqs)
        tokd_sequences = self.tokenizer(seqs, padding=pad, truncation=True)

        # if not padding wrap in a list to maintain differing dims
        if not pad:
            input_ids = list(tokd_sequences["input_ids"])
            attention_mask = list(tokd_sequences["attention_mask"])
        else:
            input_ids = torch.tensor(tokd_sequences["input_ids"])
            attention_mask = torch.tensor(tokd_sequences["attention_mask"])

        data = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        
        if self.model_type != "roberta":
            data["token_type_ids"] = torch.tensor(tokd_sequences["token_type_ids"])

        if labels is not None:
            data["labels"] = torch.tensor(labels)

        return data