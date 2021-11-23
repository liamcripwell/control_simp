
import os
import time

import fire
import pandas as pd
from easse.sari import corpus_sari
from easse.bleu import sentence_bleu
from easse.samsa import get_samsa_sentence_scores
from easse.bertscore import get_bertscore_sentence_scores

from control_simp.models.recursive import RecursiveGenerator
from control_simp.models.end_to_end import run_generator, BartFinetuner
from control_simp.models.classification import run_classifier, LightningBert

FAST_METRICS = ["sari", "bleu", "bertscore"]


def calculate_bertscore(yy_, yy):
    """
    Compute BERTScore for given prediction/ground-truth pairs (assumes single references).
    """
    if not isinstance(yy[0], list): yy = [[y] for y in yy]
    p, r, f = get_bertscore_sentence_scores(yy_, yy)

    # return precision sub-metric
    return p.tolist()

def calculate_bleu(yy_, yy):
    """
    Compute BLEU score for given prediction/ground-truth pairs (assumes single references).
    """
    if not isinstance(yy[0], list): yy = [[y] for y in yy]
    bleus = [sentence_bleu(yy_[i], yy[i]) for i in range(len(yy_))] 
    return bleus

def calculate_sari(xx, yy_, yy):
    """
    Compute SARI score for a full set of predictions (assumes single references).
    """
    if not isinstance(yy[0], list): yy = [[y] for y in yy]
    saris = [corpus_sari([xx[i]], [yy_[i]], [yy[i]]) for i in range(len(xx))]
    return saris

def calculate_samsa(xx, yy_):
    """Compute SAMSA for given input/prediction pairs."""
    samsas = get_samsa_sentence_scores(xx, yy_)
    return samsas

def calculate_metrics(inputs, preds, refs, metrics=["blue", "sari"]):
    """Compute all evaluation metrics for provided data. SAMSA disabled by default."""
    results = {}
    if "bertscore" in metrics:
        print("Calculating BERTScores...")
        results["bertscore"] = calculate_bertscore(preds, refs)
    if "bleu" in metrics:
        print("Calculating BLEUs...")
        results["bleu"] = calculate_bleu(preds, refs)
    if "sari" in metrics:
        print("Calculating SARIs...")
        results["sari"] = calculate_sari(inputs, preds, refs)
    if "samsa" in metrics:
        print("Calculating SAMSAs...")
        results["samsa"] = calculate_samsa(inputs, preds)

    return results

def clean_refs(refs, tokenizer):
    """
    Apply tokenization and decoding to reference sequences to confirm same format as predictions.
    """
    clean = []
    for y in refs:
        y_ids = tokenizer(y)["input_ids"]
        y_ = tokenizer.decode(y_ids, skip_special_tokens=True)
        clean.append(y_)

    return clean

def run_evaluation(df, x_col="complex", y_col="simple", pred_col="pred", metrics=["bleu", "sari"], tokenizer=None):
    """
    Handles evaluation for `pandas.DataFrame` containing columns for inputs, references, and predictsions.
    """
    inputs = list(df[x_col])
    preds = list(df[pred_col])
    refs = list(df[y_col])
    if tokenizer is not None:
        refs = clean_refs(refs, tokenizer)

    return calculate_metrics(inputs, preds, refs, metrics=metrics)


class Launcher(object):

    def bart(self, model_loc, test_file, out_dir, name, ctrl_toks=None, max_samples=None, samsa=True, do_pred=True, device="cuda", ow=False, num_workers=8, mtl=False, beams=10):
        start = time.time()

        pred_file = f"{out_dir}/{name}_preds.csv"
        eval_file = f"{out_dir}/{name}_eval.csv"

        print("Loading data...")
        test_set = pd.read_csv(test_file)
        if max_samples is not None:
            test_set = test_set[:max_samples]

        print("Loading model...")
        model = BartFinetuner.load_from_checkpoint(model_loc, mtl=mtl, strict=False).to(device).eval()

        # run generation on test data
        if do_pred and (ow or not os.path.isfile(pred_file)):
            print("Generating predictions...")
            test_set["pred"] = run_generator(model, test_set, ctrl_toks=ctrl_toks, max_samples=max_samples, num_workers=num_workers, beams=beams)
            test_set.to_csv(pred_file, index=False)
            print(f"Predictions written to {pred_file}.")
        else:
            test_set = pd.read_csv(pred_file)

        print("Evaluating predictions...")
        metrics = FAST_METRICS
        if samsa:
            metrics.append("samsa")
        if not ow and os.path.isfile(eval_file):
            # don't re-compute existing metrics
            test_set = pd.read_csv(eval_file)
            metrics = [m for m in metrics if m not in test_set.columns]
            print(f"New evaluation metrics to be computed: {metrics}")

        # run evaluation process
        results = run_evaluation(test_set, metrics=metrics, tokenizer=model.tokenizer)
        for metric, vals in results.items():
            test_set[metric] = vals

        test_set.to_csv(eval_file, index=False)
        print(f"Scores written to {eval_file}.")

        end = time.time()
        elapsed = end - start
        print(f"Done! (Took {elapsed}s in total)")

    def clf(self, model_loc, test_file, out_file, input_col="complex", max_samples=None, device="cuda", num_workers=8):
        start = time.time()

        print("Loading data...")
        test_set = pd.read_csv(test_file)
        if max_samples is not None:
            test_set = test_set[:max_samples]

        print("Loading model...")
        model = LightningBert.load_from_checkpoint(model_loc, model_type="roberta").to(device).eval()

        print("Running predictions...")
        test_set["pred_l"] = run_classifier(model, test_set, input_col, max_samples=max_samples, device=device, num_workers=num_workers, return_logits=False)

        # check if predictions are correct
        correct = []
        for i, row in test_set[:max_samples].iterrows():
            correct.append(int(row.pred_l) == int(row.label))
        test_set["correct"] = correct
        print(f"Overall accuracy: {test_set['correct'].sum() / len(test_set)}")

        test_set.to_csv(out_file, index=False)
        print(f"Predictions written to {out_file}.")

        end = time.time()
        elapsed = end - start
        print(f"Done! (Took {elapsed}s in total)")

    def recursive(self, clf_loc, gen_loc, test_file, out_dir, name, x_col="complex", k=2, max_samples=None, samsa=False, device="cuda", ow=False, num_workers=8):
        start = time.time()

        pred_file = f"{out_dir}/{name}_rec_preds.csv"
        eval_file = f"{out_dir}/{name}_rec_eval.csv"

        print("Loading data...")
        test_set = pd.read_csv(test_file)
        if max_samples is not None:
            test_set = test_set[:max_samples]

        model = RecursiveGenerator(clf_loc, gen_loc, device=device, num_workers=num_workers)

        # run generation on test data
        if not ow and os.path.isfile(pred_file):
            print("Loading previous generated outputs...")
            test_set = pd.read_csv(pred_file)
        test_set = model.generate(test_set, x_col, k=k)
        test_set.to_csv(pred_file, index=False)
        print(f"Predictions written to {pred_file}.")

        # run evaluation
        if not ow and os.path.isfile(eval_file):
            test_set = pd.read_csv(eval_file)

        for i in range(1, k+1):
            metrics = FAST_METRICS
            if samsa:
                metrics.append("samsa")
            metrics = [m for m in metrics if f"{m}_{i}" not in test_set.columns]
            print(f"New evaluation metrics to be computed: {metrics}")

            # run evaluation process
            results = run_evaluation(test_set, pred_col=f"pred_{i}", metrics=metrics, tokenizer=model.tokenizer)
            for metric, vals in results.items():
                test_set[f"{metric}_{i}"] = vals

            test_set.to_csv(eval_file, index=False)
            print(f"Scores written to {eval_file}.")

        end = time.time()
        elapsed = end - start
        print(f"Done! (Took {elapsed}s in total)")
        

if __name__ == '__main__':
    fire.Fire(Launcher)