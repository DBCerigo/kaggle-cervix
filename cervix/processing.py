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

def resize_100(path):
    desc = 'resize_100'

    filename = os.path.basename(path)
    directories = path.split('/')
    #subdir here is either 'train' or 'test'
    subdir = directories[2]

    resize_100_path = "../data/processed/"+desc+'/'+subdir+"/"+filename
    if os.path.exists(resize_100_path):
        return resize_100_path
    print(filename.split('.')[0], end='.')
    img = cv2.imread(path)
    rescaled = cv2.resize(img, (100, 100), cv2.INTER_LINEAR)
    return save_img(rescaled, resize_100_path)

def random_forest_transform(df, img_path_column, grayscale=True, test=False):
    vecs = []
    types = []
    ids = []
    imread_opt = 0 if grayscale else 1 # 1 is 3chan rbg, 0 is grayscale
    for _, row in df.iterrows():
        gray = cv2.imread(row.grayscale_resize_path, imread_opt)
        vec = process_image(gray)
        if not test:
            cervix_type = row.type
            types.append(cervix_type[-1])
        else:
            ids.append(row.num_id)
        vecs.append(vec)
    if not test:
        return np.squeeze(np.array(vecs)), np.array(types)
    else:
        return np.squeeze(np.array(vecs)), np.array(ids)
                              
def process_image(img):
    """Normalize image and turn into array of one long column"""
    normalized = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX)
    vec = normalized.reshape(1, np.prod(normalized.shape))
    return vec / np.linalg.norm(vec)
