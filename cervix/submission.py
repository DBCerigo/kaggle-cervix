from __future__ import print_function
import math
import pandas as pd

def _log_prob(p):
    p = max(min(p, 1 - 10**-15), 10**-15)
    return math.log(p)

def _get_log_loss(row):
    print(row)
    print(row['Type'])
    ctype = row['Type']
    return -_log_prob(row[ctype])

def compute_losses(sub_csv_path_or_sub_df):
    if isinstance(sub_csv_path_or_sub_df, str):
        df = pd.read_csv(sub_csv_path_or_sub_df)
    df['log_l'] = df.apply(get_log_loss, axis=1)
    N = len(df)
    score = df.log_l.sum()/N
    df = df.sort_values('log_l', ascending=False)
    return score, df
