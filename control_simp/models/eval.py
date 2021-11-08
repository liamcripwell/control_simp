
from easse.sari import corpus_sari
from easse.bleu import sentence_bleu
from easse.samsa import get_samsa_sentence_scores


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
    """Apply tokenization and decoding to reference sequences to confirm same format as predictions."""
    clean = []
    for y in refs:
        y_ids = tokenizer(y)["input_ids"]
        y_ = tokenizer.decode(y_ids, skip_special_tokens=True)
        clean.append(y_)

    return clean