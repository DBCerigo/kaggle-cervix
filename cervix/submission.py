from __future__ import print_function
import math
import pandas as pd
import csv

def _log_prob(p):
    p = max(min(p, 1 - 10**-15), 10**-15)
    return math.log(p)

def _get_log_loss(row):
    num_type = row['Type']
    return -_log_prob(row['Type_'+num_type])

def compute_losses(sub_csv_or_df):
    if isinstance(sub_csv_or_df, str):
        df = pd.read_csv(sub_csv_or_df)
    else:
        df = sub_csv_or_df
    df['log_l'] = df.apply(_get_log_loss, axis=1)
    N = len(df)
    score = df.log_l.sum()/N
    print('Use: `df.sort_values(\'log_l\', ascending=False)` to order by log_l')
    return score, df

def write_submission_file(path, df):
    """Write a submission file from a df with predictions appended"""
    df['image_name'] = df.path.map(lambda x: os.path.basename(x))
    df.to_csv(path, columns=['image_name','Type_1','Type_2','Type_3'],
                index=False)
