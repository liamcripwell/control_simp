import numpy as np
from Levenshtein import ratio

def add_lev_ratio(df, xcol="complex", ycol="simple", reset=False):
    levs = []
    for i, row in df.iterrows():
        if "lev_ratio" in row and not reset:
            if not np.isnan(row.lev_ratio):
                lev = row.lev_ratio
        else:
            lev = ratio(row[xcol], row.[ycol])
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
    
    df["ntoks"] = ntoks
