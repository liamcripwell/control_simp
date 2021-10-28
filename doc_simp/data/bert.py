import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset

from doc_simp.utils import TokenFilter
from doc_simp.data.loading import LazyTensorDataset


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
            self.train_workers = self.hparams.train_workers

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

    def preprocess(self, seqs, labels=None):
        seqs = TokenFilter(max_len=self.MAX_LEN, blacklist=["<SEP>"])(seqs)
        padded_sequences = self.tokenizer(seqs, padding=True, truncation=True)

        data = {
            "input_ids": torch.tensor(padded_sequences["input_ids"]),
            "attention_mask": torch.tensor(padded_sequences["attention_mask"]),
        }
        
        if self.model_type != "roberta":
            data["token_type_ids"] = torch.tensor(padded_sequences["token_type_ids"])

        if labels is not None:
            data["labels"] = torch.tensor(labels)

        return data