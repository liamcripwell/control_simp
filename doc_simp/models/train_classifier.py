import argparse

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from doc_simp.models.wandb_util import existing_checkpoints
from doc_simp.models.classification import LightningBert, BertDataModule

if __name__ == '__main__':
    """
    Train a BERT simplification-operation classification model.
    """

    # prepare argument parser
    parser = argparse.ArgumentParser()
    parser = LightningBert.add_model_specific_args(parser)

    # add all the available trainer options to argparse
    # ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # prepare data module and trainer class
    if args.checkpoint is None:
        model = LightningBert(hparams=args, model_type=args.model_type)
    else:
        model = LightningBert.load_from_checkpoint(args.checkpoint, hparams=args, model_type=args.model_type)
    dm = BertDataModule(model.tokenizer, hparams=args)

    # construct default run name
    if args.name is None:
        args.name = f"{args.max_samples}_{args.batch_size}_{args.learning_rate}"

    # NOTE: use args.wandb_id to resume training on an existing wandb run.
    # However, existing checkpoint files must be removed from the project's run folder to avoid errors.
    if args.wandb_id is not None and existing_checkpoints(args):
        raise FileExistsError(
            "The specified wandb run already has local checkpoints. Please remove them before continuing.")

    wandb_logger = WandbLogger(
        name=args.name, project=args.project, save_dir=args.save_dir, id=args.wandb_id)

    trainer = pl.Trainer.from_argparse_args(
        args,
        val_check_interval=args.val_check_interval,
        logger=wandb_logger,
        accelerator="ddp",
        precision=16,)

    trainer.fit(model, dm)
