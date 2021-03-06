# -*- coding: utf-8 -*-
from __future__ import print_function
import cv2
import os
import pandas as pd
import numpy as np
import platform

__processed_path = None
if 'c001' in platform.node():
    __processed_path = '../cervix/data/processed/'
    __subdir_index = 3
else:
    __processed_path = '../data/processed/'
    __subdir_index = 2

def save_img(img, path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    print(path.split('/')[-1].split('.')[0], end=' ')
    cv2.imwrite(path,img)
    return path
    
def transform_save_imgs(df, transform_func, **args):
    df['processed_path'] = df.path.map(lambda x: transform_func(x, **args))
    return df


# =======================================================
# BEGIN PROCESSING FUNCTIONS

# helper
def __make_processed_path(path, desc):
    filename = os.path.basename(path)
    directories = path.split('/')
    #subdir here is either 'train' or 'test'
    subdir = directories[__subdir_index]
    return __processed_path+desc+'/'+subdir+'/'+filename

# should follow format as:
# EXAMPLE
def __processing_func(path):
    processed_unique_path =__make_processed_path(path, 'unique_desc')
        #this defines the path where the images are saved locally
    #loading and processing here
    img_objct = None
    return save_img(img_objct, processed_unique_path)


def grayscale_resize(path):
    gray_path = __make_processed_path(path, 'grayscale_resize')
    if os.path.exists(gray_path):
        return gray_path
    img = cv2.imread(path)
    rescaled = cv2.resize(img, (100, 100), cv2.INTER_LINEAR)
    gray = cv2.cvtColor(rescaled, cv2.COLOR_RGB2GRAY).astype('float')
    return save_img(gray, gray_path)

def resize_n(path, n=299):
    assert isinstance(n, (int)), 'n must be an int'
    resize_path = __make_processed_path(path, 'resize_'+str(n))
    if os.path.exists(resize_path):
        return resize_path
    img = cv2.imread(path)
    rescaled = cv2.resize(img, (n, n), cv2.INTER_LINEAR)
    return save_img(rescaled, resize_path)

# END processing functions
# ===========================================



def random_forest_transform(df, img_path_column, grayscale=None):
    assert isinstance(grayscale, (bool)), 'grayscale must be set to a bool'
    imread_opt = 0 if grayscale else 1 # 1 is 3chan rbg, 0 is grayscale
    path_to_vec = lambda x: process_image(cv2.imread(x,imread_opt))
    df['vec'] = df[img_path_column].map(path_to_vec)
    return df

def process_image(img):
    '''Normalize image and turn into array of one long column'''
    normalized = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX)
    vec = normalized.reshape(1, np.prod(normalized.shape))
    return vec / np.linalg.norm(vec)

def default_img_process(row, grayscale):
    assert isinstance(grayscale, (bool)), 'grayscale must be set to a bool'
    imread_opt = 0 if grayscale else 1 # 1 is 3chan rbg, 0 is grayscale

    return cv2.imread(row['processed_path'], imread_opt)

def df_to_training_tuples(df, input_shape, img_read=default_img_process, grayscale=None):
    features_shape = np.insert(input_shape, 0, len(df))
    features = np.zeros(features_shape)
    labels = np.zeros((len(df), 3))

    counter = 0
    for _,row in df.iterrows():
        onehot = np.zeros(3)
        if 'Type' in row:
            onehot[int(row['Type'])-1] = 1

        features[counter] = img_read(row, grayscale)
        labels[counter] = onehot
        counter += 1

    return features, labels

def df_to_keras_generator(df, batch_size, grayscale=None):
    assert isinstance(grayscale, (bool)), 'grayscale must be set to a bool'
    imread_opt = 0 if grayscale else 1 # 1 is 3chan rbg, 0 is grayscale

    batch_features = np.zeros((batch_size, 299, 299, 3))
    batch_labels = np.zeros((batch_size, 3))

    batch_counter = 0
    while 1:
        for index, row in df.iterrows():
            onehot = np.zeros(3)
            onehot[int(row['Type'])-1] = 1
            batch_features[batch_counter] = cv2.imread(row['processed_path'],imread_opt)
            batch_labels[batch_counter] = onehot
            batch_counter += 1
            if batch_counter == batch_size:
                batch_counter = 0
                yield batch_features, batch_labels

def append_probabilities(orig_df, preds, type_order):
    type_columns = ['Type_'+t for t in type_order]
    index = orig_df.index.values
    probs_df = pd.DataFrame.from_records(preds, index=index, columns=type_columns)
    orig_df[type_columns] = probs_df[type_columns]
    return orig_df
