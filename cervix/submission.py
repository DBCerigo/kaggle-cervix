from __future__ import print_function
import math
import pandas as pd
import csv
import os
import keras
from keras import backend as K

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

def keras_log_loss(y_true, y_pred):
    clipped_y_pred = K.clip(y_pred, 1e-15, 1-1e-15)
    return keras.losses.categorical_crossentropy(y_true, clipped_y_pred)
    #Leaving for now incase needed in future
    #weighted_logs = K.multipy(y_true, probability_log)
    #Need to check axis is right
    #sum_weighted_logs = K.sum(weighted_logs, axis=1)
    #negative_sum_weighted_logs = K.scalar_mul(-1, sum_weighted_logs)
    #return K.mean(negative_sum_weighted_logs, axis=-1)

def write_submission_file(path, df):
    """Write a submission file from a df with predictions appended"""
    df['image_name'] = df.path.map(lambda x: os.path.basename(x))
    df.to_csv(path, columns=['image_name','Type_1','Type_2','Type_3'],
                index=False)
