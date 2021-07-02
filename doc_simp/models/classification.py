import argparse

import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, AdamW, BertForSequenceClassification

from doc_simp.models.utils import flatten_list


class LightningBert(pl.LightningModule):

    loss_names = ["loss"]

    def __init__(self, hparams):
        super().__init__()

        self.model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

        # basic hyperparams
        self.hparams = hparams
        self.learning_rate = self.hparams.learning_rate

        self.train_losses = []

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)

    def training_step(self, batch, batch_idx):
        labels = batch["label"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]

        loss, _ = self.model(
                input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                labels=labels
                )

        output = {"train_loss": loss}

        # wandb log
        self.train_losses.append(loss)
        if batch_idx % int(self.hparams.train_check_interval * self.trainer.num_training_batches) == 0:
            avg_loss = torch.stack(self.train_losses).mean()
            self.logger.experiment.log({'train_loss': avg_loss})
            self.train_losses = []

        return output

    def validation_step(self, batch, batch_idx):
        labels = batch["label"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]

        loss, _ = self.model(
                input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                labels=labels
                )

        output = {"val_loss": loss}

        return output

    def validation_epoch_end(self, outputs, prefix="val"):
        losses = {k: torch.stack([x[k] for x in outputs]).mean()
                  for k in self.loss_names}
        loss = losses["loss"]

        preds = flatten_list([x["preds"] for x in outputs])

        # wandb log
        self.logger.experiment.log({
            f"{prefix}_loss": loss,
        })

        return {
            "preds": preds,
            f"{prefix}_loss": loss,
        }


    def configure_optimizers(self):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "gamma", "beta"]
        optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                    "weight_decay_rate": 0.01
                    },
                {
                    "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                    "weight_decay_rate": 0.0
                    },
                ]
        optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=self.learning_rate,
                )

        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)

        parser.add_argument("--name", type=str, default=None, required=False,)
        parser.add_argument("--save_dir", type=str, default=None, required=False,)
        parser.add_argument("--project", type=str, default=None, required=False,)
        parser.add_argument("--checkpoint", type=str, default=None, required=False,)
        parser.add_argument("--x_col", type=str, default="x", required=False,)
        parser.add_argument("--y_col", type=str, default="y", required=False,)
        parser.add_argument("--train_check_interval", type=float, default=0.20)
        parser.add_argument("--freeze_encoder", action="store_true")
        parser.add_argument("--freeze_embeds", action="store_true")
        parser.add_argument("--learning_rate", type=float, default=2e-5)
        parser.add_argument("--batch_size", type=int, default=16)
        parser.add_argument("--data_file", type=str, default=None, required=True)
        parser.add_argument("--data_file2", type=str, default=None, required=False)
        parser.add_argument("--max_samples", type=float)
        parser.add_argument("--train_split", type=float, default=0.9)

        return parser


class BertDataModule(pl.LightningDataModule):

    MAX_LEN = 128

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
            self.max_samples = int(self.hparams.max_samples)  # defaults to no restriction
            self.train_split = self.hparams.train_split  # default split will be 90/5/5
            self.val_split = (1 - self.train_split) / 2

    def prepare_data(self):
        # NOTE: shouldn't assign state in here
        pass

    def setup(self):
        self.data = pd.read_csv(self.data_file)
        self.data = self.data.sample(frac=1)[:self.max_samples]
        print("All data loaded.")

        # train, validation, test split
        train_size = int(self.train_split * len(self.data))
        val_size = int((self.train_split + self.val_split) * len(self.data))
        self.train, self.validate, self.test = np.split(
            self.data, [train_size, val_size])

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
        dataset = TensorDataset(
            self.train['input_ids'],
            self.train['attention_mask'],
            self.train['labels'])
        train_data = DataLoader(dataset, batch_size=self.batch_size)
        return train_data

    def val_dataloader(self):
        dataset = TensorDataset(
            self.validate['input_ids'],
            self.validate['attention_mask'],
            self.validate['labels'])
        val_data = DataLoader(dataset, batch_size=self.batch_size)
        return val_data

    def test_dataloader(self):
        dataset = TensorDataset(
            self.test['input_ids'],
            self.test['attention_mask'],
            self.test['labels'])
        test_data = DataLoader(dataset, batch_size=self.batch_size)
        return test_data

    def preprocess(self, seqs, labels=None):
        # input ids
        seqs = ["[CLS]" + str(seq) + " [SEP]" for seq in seqs]
        tokenized_seqs = [self.tokenizer.tokenize(seq) for seq in seqs]
        input_ids = [self.tokenizer.convert_tokens_to_ids(x) for x in tokenized_seqs]
        input_ids = pad_sequences(input_ids, maxlen=self.MAX_LEN, dtype="long", truncating="post", padding="post")

        # attention masks
        attention_masks = []
        for seq in input_ids:
            seq_mask = [1] * len(seq)
            attention_masks.append(seq_mask)

        inputs = torch.tensor(input_ids)
        masks = torch.tensor(attention_masks)
        data = {
            "input_ids": inputs,
            "attention_mask": masks
        }
        if labels is not None:
            data["labels"] = torch.tensor(labels)

        return data