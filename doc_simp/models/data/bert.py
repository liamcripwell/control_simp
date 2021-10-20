import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset

from doc_simp.utils import TokenFilter


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

        # tokenize datasets
        self.train = self.preprocess(
            list(
                self.train[self.x_col]), list(
                self.train[self.y_col]))
        self.validate = self.preprocess(
            list(
                self.validate[self.x_col]), list(
                self.validate[self.y_col]))
        self.test = self.preprocess(
            list(
                self.test[self.x_col]), list(
                self.test[self.y_col]))

    def train_dataloader(self):
        if self.model_type == "roberta":
            dataset = TensorDataset(
                self.train['input_ids'],
                self.train['attention_mask'],
                self.train['labels'])
        else:
            dataset = TensorDataset(
                self.train['input_ids'],
                self.train['attention_mask'],
                self.train['token_type_ids'],
                self.train['labels'])
        train_data = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        return train_data

    def val_dataloader(self):
        if self.model_type == "roberta":
            dataset = TensorDataset(
                self.validate['input_ids'],
                self.validate['attention_mask'],
                self.validate['labels'])
        else:
            dataset = TensorDataset(
                self.validate['input_ids'],
                self.validate['attention_mask'],
                self.validate['token_type_ids'],
                self.validate['labels'])
        val_data = DataLoader(dataset, batch_size=self.batch_size, num_workers=1, pin_memory=True)
        return val_data

    def test_dataloader(self):
        if self.model_type == "roberta":
            dataset = TensorDataset(
                self.test['input_ids'],
                self.test['attention_mask'],
                self.test['labels'])
        else:
            dataset = TensorDataset(
                self.test['input_ids'],
                self.test['attention_mask'],
                self.test['token_type_ids'],
                self.test['labels'])
        test_data = DataLoader(dataset, batch_size=self.batch_size, num_workers=1, pin_memory=True)
        return test_data

    def preprocess(self, seqs, labels=None):
        seqs = TokenFilter(max_len=self.MAX_LEN, blacklist=["<SEP>"])(seqs)

        padded_sequences = self.tokenizer(seqs, padding=True, truncation=True)
        input_ids = padded_sequences["input_ids"]
        attention_mask = padded_sequences["attention_mask"]

        data = {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
        }
        if self.model_type != "roberta":
            token_type_ids = padded_sequences["token_type_ids"]
            data["token_type_ids"] = torch.tensor(token_type_ids)

        if labels is not None:
            data["labels"] = torch.tensor(labels)

        return data