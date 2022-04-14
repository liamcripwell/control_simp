# Controllable Simplification via Operation Classification

This repo contains code used to run experiments presented in the NAACL 2022 paper, "Controllable Sentence Simplification via Operation Classification".

## IRSD data

The IRSD dataset is available in [data/](data/). The publicly available version does not include any examples from Newsela-auto. To obtain the Newsela-auto examples, please obtain permission from Newsela before contacting the authors directly.

## Training models

The following is an example of how to train a RoBERTa-based operation classifier:

```bash
doc_simp/models/train.py --data_file=<train_file.csv> --val_file=<val_file.csv> --learning_rate=3e-5 --x_col=complex --y_col=label --batch_size=32 --arch=classifier --model_type=roberta
```

Generative models can be trained as follows:

```bash
python control_simp/models/train.py --arch=bart --data_file=<train_file.csv> --val_file=<val_file.csv> --learning_rate=3e-5 --batch_size=16 --max_source_length=128 --max_target_length=128 --eval_beams=4 --x_col=complex --y_col=simple 
```

See the source code for additional arguments.

## Running models

Operation classifiers can be run from the terminal as follows (`test_file` should be a `.csv`):

```bash
python control_simp/models/eval.py clf <model_ckpt> <test_file> <out_file> --input_col=<sentence_col>
```

Generative models can be used for inference/evaluation in a similar fashion. For pipeline models, operation control tokens are dictated from a column in the input data specified via the `--ctrl_toks` flag. Disabling SAMSA with `--samsa=False` dramatically reduces runtime.

```bash
# End-to-End Model
python control_simp/models/eval.py bart <model_ckpt> <test_file> <output_dir> <run_name> --samsa=False

# CTRL Model
python control_simp/models/eval.py bart <model_ckpt> <test_file> <output_dir> <run_name> --ctrl_toks=<label_col> --samsa=False
```
