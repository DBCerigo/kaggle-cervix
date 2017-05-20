from __future__ import print_function
import cv2
import os
import pandas as pd
import numpy as np

def save_img(img, path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    cv2.imwrite(path,img)
    return path
    
def transform_save_imgs(df, transform_func):
    for index, row in df.iterrows():
        desc = transform_func.__name__
        df.loc[index,desc+'_path'] = _transform_save_img(row,transform_func)
    return df

def _transform_save_img(row, transform_func):
    path = row.path
    return transform_func(path)

def grayscale_resize(path):
    desc = 'grayscale_resize'

    filename = os.path.basename(path)
    directories = path.split('/')
    #subdir here is either 'train' or 'test'
    subdir = directories[2]

    gray_path = "../data/processed/"+desc+'/'+subdir+"/"+filename
    if os.path.exists(gray_path):
        return gray_path
    print(filename.split('.')[0], end='.')
    img = cv2.imread(path)
    rescaled = cv2.resize(img, (100, 100), cv2.INTER_LINEAR)
    gray = cv2.cvtColor(rescaled, cv2.COLOR_RGB2GRAY).astype('float')
    return save_img(gray, gray_path)

def resize_n(n):
    return lambda x: __resize_n(x, n)


def __resize_n(path, n):
    assert isinstance(n, (int)), 'n must be an int'
    desc = 'resize_'+str(n) 
    filename = os.path.basename(path)
    directories = path.split('/')
    #subdir here is either 'train' or 'test'
    subdir = directories[2]
    resize_path = "../data/processed/"+desc+'/'+subdir+"/"+filename
    if os.path.exists(resize_path):
        return resize_path
    print(filename.split('.')[0], end='.')
    img = cv2.imread(path)
    rescaled = cv2.resize(img, (n, n), cv2.INTER_LINEAR)
    return save_img(rescaled, resize_path)

def random_forest_transform(df, img_path_column, grayscale=None):
    assert isinstance(grayscale, (bool)), 'grayscale must be set to a bool'
    imread_opt = 0 if grayscale else 1 # 1 is 3chan rbg, 0 is grayscale
    path_to_vec = lambda x: process_image(cv2.imread(x,imread_opt))
    df['vec'] = df[img_path_column].map(path_to_vec)
    return df

def append_probabilities(orig_df, preds, type_order):
    columns = ['Type_'+t for t in type_order]
    index = orig_df.index.values
    probs_df = pd.DataFrame.from_records(preds, index=index, columns=columns)
    return orig_df.join(probs_df)

def process_image(img):
    """Normalize image and turn into array of one long column"""
    normalized = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX)
    vec = normalized.reshape(1, np.prod(normalized.shape))
    return vec / np.linalg.norm(vec)
