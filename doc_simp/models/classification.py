import math
import argparse

import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, AdamW, BertForSequenceClassification, RobertaTokenizer, RobertaForSequenceClassification

INPUTS = {
    "roberta": ["input_ids", "attention_mask", "labels"],
    "bert": ["input_ids", "attention_mask", "token_type_ids", "labels"]
}


def run_classifier(model, test_set, input_col="complex", max_samples=None, device="cuda", batch_size=16):
    if max_samples is not None:
        test_set = test_set[:max_samples]

    with torch.no_grad():
        # preprocess data
        dm = BertDataModule(model.tokenizer, hparams=model.hparams)
        test = dm.preprocess(list(test_set[input_col]))

        # prepare data loader
        features = INPUTS[model.model_type][:-1] # ignore labels
        dataset = TensorDataset(*[test[f].to(device) for f in features])
        test_data = DataLoader(dataset, batch_size=batch_size)

        # run predictions for each batch
        preds = []
        for batch in test_data:
            _batch = {features[i]:batch[i] for i in range(len(features))}
            output = model.model(**_batch, return_dict=True)
            _, logits = extract_results(output)
            preds += logits
    
    return preds

def extract_results(output):
    if type(output) is tuple:
        loss = output[0]
        logits = output[1]
    else:
        loss = None
        if "loss" in output:
            loss = output["loss"]
        logits = output["logits"]

    return loss, logits


class LightningBert(pl.LightningModule):

    def __init__(self, hparams, model_type=None, pt_model=None, num_labels=4):
        super().__init__()

        # resolve model type and pretrained base
        if model_type is None:
            model_type = hparams.model_type if hparams.model_type is not None else "roberta"
        if pt_model is None:
            pt_model = "roberta-base" if model_type == "roberta" else "bert-base-uncased"
        self.model_type = model_type
        self.pt_model = pt_model

        # set up model
        if model_type == "bert":
            self.model = BertForSequenceClassification.from_pretrained(pt_model, num_labels=num_labels)
            self.tokenizer = BertTokenizer.from_pretrained(pt_model, do_lower_case=True)
        elif model_type == "roberta":
            self.model = RobertaForSequenceClassification.from_pretrained(pt_model, num_labels=num_labels)
            self.tokenizer = RobertaTokenizer.from_pretrained(pt_model)
        else:
            raise ValueError("Unknown model type specified. Please choose one of [`bert`, `roberta`].")

        # basic hyperparams
        self.hparams = hparams
        self.learning_rate = self.hparams.learning_rate
        self.use_lr_scheduler = self.hparams.lr_scheduler

        self.train_losses = []

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)

    def training_step(self, batch, batch_idx):
        _batch = {INPUTS[self.model_type][i] : batch[i] for i in range(len(batch))}
        output = self.model(**_batch, return_dict=True)

        loss, _ = extract_results(output)

        output = {"loss": loss}

        # wandb log
        self.train_losses.append(loss)
        if batch_idx % math.ceil(self.hparams.train_check_interval * self.trainer.num_training_batches) == 0:
            avg_loss = torch.stack(self.train_losses).mean()
            self.logger.experiment.log({'train_loss': avg_loss})
            self.train_losses = []

        return output

    def validation_step(self, batch, batch_idx):
        _batch = {INPUTS[self.model_type][i] : batch[i] for i in range(len(batch))}
        output = self.model(**_batch, return_dict=True)

        loss, logits = extract_results(output)

        output = {
            "loss": loss,
            "preds": logits,
        }

        return output

    def validation_epoch_end(self, outputs, prefix="val"):
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        result = {
            f"{prefix}_loss": loss,
        }

        # wandb log
        self.logger.experiment.log(result)

        return result


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
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate)
        
        # use a learning rate scheduler if specified
        if self.use_lr_scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                },
            }

        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)

        parser.add_argument("--name", type=str, default=None, required=False,)
        parser.add_argument("--project", type=str, default=None, required=False,)
        parser.add_argument("--save_dir", type=str, default=None, required=False,)
        parser.add_argument("--checkpoint", type=str, default=None, required=False,)
        parser.add_argument("--wandb_id", type=str, default=None, required=False,)
        parser.add_argument("--model_type", type=str, default="roberta", required=False,)
        parser.add_argument("--x_col", type=str, default="x", required=False,)
        parser.add_argument("--y_col", type=str, default="y", required=False,)
        parser.add_argument("--train_check_interval", type=float, default=0.20)
        parser.add_argument("--freeze_encoder", action="store_true")
        parser.add_argument("--freeze_embeds", action="store_true")
        parser.add_argument("--learning_rate", type=float, default=2e-5)
        parser.add_argument("--lr_scheduler", action="store_true")
        parser.add_argument("--batch_size", type=int, default=16)
        parser.add_argument("--data_file", type=str, default=None, required=True)
        parser.add_argument("--data_file2", type=str, default=None, required=False)
        parser.add_argument("--max_samples", type=int, default=-1)
        parser.add_argument("--train_split", type=float, default=0.9)
        parser.add_argument("--val_split", type=float, default=0.05)
        parser.add_argument("--val_file", type=str, default=None, required=False)

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
        seqs = ["[CLS]" + str(seq).replace("<SEP> ", "") + " [SEP]" for seq in seqs] # TODO: add custom token replacer
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
