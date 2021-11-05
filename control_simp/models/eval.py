
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
    sari = corpus_sari(xx, yy_, [[y] for y in yy])
    return sari

def calculate_samsa(yy_, yy):
    """Compute SAMSA for given prediction/ground-truth pairs."""
    samsas = get_samsa_sentence_scores(yy_, yy)
    return samsas