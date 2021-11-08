
import fire
import pandas as pd
from easse.sari import corpus_sari
from easse.bleu import sentence_bleu
from easse.samsa import get_samsa_sentence_scores

from control_simp.models.end_to_end import run_generator, BartFinetuner


def calculate_bertscore():
    pass

def calculate_bleu(yy_, yy):
    """
    Compute BLEU score for given prediction/ground-truth pairs (assumes single references).
    """
    bleus = [sentence_bleu(yy_[i], [yy[i]]) for i in range(len(yy_))] 
    return bleus

def calculate_sari(xx, yy_, yy):
    """
    Compute SARI score for a full set of predictions (assumes single references).
    """
    saris = [corpus_sari([xx[i]], [yy_[i]], [[yy[i]]]) for i in range(len(xx))]
    return saris

def calculate_samsa(yy_, yy):
    """Compute SAMSA for given prediction/ground-truth pairs."""
    samsas = get_samsa_sentence_scores(yy_, yy)
    return samsas

def calculate_metrics(inputs, preds, refs, samsa=False):
    """Compute all evaluation metrics for provided data. SAMSA disabled by default."""
    results = {}
    results["bleu"] = calculate_bleu(preds, refs)
    results["sari"] = calculate_sari(inputs, preds, refs)
    if samsa:
        results["samsa"] = calculate_samsa(preds, refs)

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

def run_evaluation(df, x_col="complex", y_col="simple", pred_col="pred", samsa=False, tokenizer=None):
    """
    Handles evaluation for `pandas.DataFrame` containing columns for inputs, references, and predictsions.
    """
    inputs = df[x_col]
    preds = df[pred_col]
    refs = df[y_col]
    if tokenizer is not None:
        refs = clean_refs(df[y_col], tokenizer)

    return calculate_metrics(inputs, preds, refs, samsa=samsa)


class Launcher(object):

    def bart(self, model_loc, test_file, out_file, ctrl_toks=None, max_samples=None, samsa=True, device="cuda"):
        test_set = pd.read_csv(test_file)
        if max_samples is not None:
            test_set = test_set[:max_samples]
            
        model = BartFinetuner.load_from_checkpoint(model_loc, strict=False).to(device).eval()

        test_set["pred"] = run_generator(model, test_set, ctrl_toks=ctrl_toks, max_samples=max_samples)

        results = run_evaluation(test_set, samsa=samsa, tokenizer=model.tokenizer)
        for metric, vals in results.items():
            test_set[metric] = vals

        test_set.to_csv(out_file, index=False)
        


if __name__ == '__main__':
    fire.Fire(Launcher)