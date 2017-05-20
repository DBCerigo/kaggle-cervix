from __future__ import print_function
import math
import pandas as pd
import csv

def _log_prob(p):
    p = max(min(p, 1 - 10**-15), 10**-15)
    return math.log(p)

def _get_log_loss(row):
    ctype = row['Type']
    return -_log_prob(row[ctype])

def compute_losses(sub_csv_path_or_sub_df):
    if isinstance(sub_csv_path_or_sub_df, str):
        df = pd.read_csv(sub_csv_path_or_sub_df)
    df['log_l'] = df.apply(_get_log_loss, axis=1)
    N = len(df)
    score = df.log_l.sum()/N
    df = df.sort_values('log_l', ascending=False)
    return score, df

def write_submission_file(path, ids, preds):
    """Write a submission file using the predictions returned from random forest
    and ids returns from random_forest_transform"""
    with open(path, 'w') as csv_f:
        writer = csv.writer(csv_f)
        writer.writerow(['image_name','Type_1','Type_2','Type_3'])
        for i,row in enumerate(preds):
            path = [ids[i]+'.jpg']
            writer.writerow(path+list(row))
