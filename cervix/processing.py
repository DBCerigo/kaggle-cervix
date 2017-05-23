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
    
def transform_save_imgs(df, transform_func, **args):
    for index, row in df.iterrows():
        df.loc[index, 'processed_path'] = transform_func(row.path, **args)
    return df


# =======================================================
# BEGIN PROCESSING FUNCTIONS

# helper
def __make_processed_path(path, desc):
    filename = os.path.basename(path)
    directories = path.split('/')
    #subdir here is either 'train' or 'test'
    subdir = directories[2]
    return '../data/processed/'+desc+'/'+subdir+'/'+filename

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
    print(filename.split('.')[0], end='.')
    img = cv2.imread(path)
    rescaled = cv2.resize(img, (100, 100), cv2.INTER_LINEAR)
    gray = cv2.cvtColor(rescaled, cv2.COLOR_RGB2GRAY).astype('float')
    return save_img(gray, gray_path)

def resize(path, n=299):
    assert isinstance(n, (int)), 'n must be an int'
    resize_path = __make_processed_path(path, 'resize_'+str(n))
    if os.path.exists(resize_path):
        return resize_path
    print(filename.split('.')[0], end='.')
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

def df_to_keras_generator(df, grayscale=None):
    assert isinstance(grayscale, (bool)), 'grayscale must be set to a bool'
    imread_opt = 0 if grayscale else 1 # 1 is 3chan rbg, 0 is grayscale
    for _, row in df.iterrows():
        onehot = np.zeros(3)
        onehot[int(row['Type'])-1] = 1
        yield process_image(cv2.imread(row['processed_path'],imread_opt)), onehot

def append_probabilities(orig_df, preds, type_order):
    type_columns = ['Type_'+t for t in type_order]
    index = orig_df.index.values
    probs_df = pd.DataFrame.from_records(preds, index=index, columns=type_columns)
    orig_df[type_columns] = probs_df[type_columns]
    return orig_df
