import numpy as np
from Levenshtein import ratio
from nltk import sent_tokenize

from disco_split.processing.utils import find_adverbial


def add_lev_ratio(df, xcol="complex", ycol="simple", reset=False):
    levs = []
    for i, row in df.iterrows():
        if "lev_ratio" in row and not reset:
            if not np.isnan(row.lev_ratio):
                lev = row.lev_ratio
        else:
            lev = ratio(row[xcol], row[ycol])
        levs.append(lev)

    df["lev_ratio"] = levs

def add_ntoks(df, xcol="complex", reset=False):
    ntoks = []
    for i, row in df.iterrows():
        if "ntoks" in row and not reset:
            if not np.isnan(row.lntoks):
                ntok = row.ntoks
        else:
            ntok = len(row[xcol].split())
        ntoks.append(ntok)
    
    df["ntoks"] = ntoks

def add_labels(df, x_col="complex", y_col="simple", sent_det=None):
    labels = []
    for i, row in df.iterrows():
        label = None

        # split simple into sents
        if sent_det is None:
            sents = sent_tokenize(row[y_col])
        else:
            sents = row[y_col].split(sent_det)
        
        # determine label
        if len(sents) == 1:
            if "lev_ratio" not in row:
                raise AttributeError("Data needs 'lev_ratio' column to assign labels. Use the `add_lev_ratio()` function to do so.")
            if row.lev_ratio == 1.0:
                label = 0
            else:
                label = 1
        else:
            if find_adverbial(sents[-1]) is None:
                label = 2
            else: 
                label = 3
        
        labels.append(label)
    
    df["label"] = labels

