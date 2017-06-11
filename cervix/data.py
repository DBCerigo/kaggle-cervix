from __future__ import print_function
from glob import glob
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import platform

__base_path = None
if 'c001' in platform.node():
    __base_path = '/data/kaggle/'
if '.local' in platform.node():
    __base_path = '../data/'


def make_base_df():
    base_path = __base_path +'train'
    image_paths = []
    for type_base_path in sorted(glob(base_path +'/*')):
        image_paths = image_paths + glob(type_base_path + '/*')
    df = pd.DataFrame({'path':image_paths})
    df['Type'] = df.path.map(lambda x: x.split('/')[-2][-1])
    df['filetype'] = df.path.map(lambda x: x.split('.')[-1])
    df['num_id'] = df.path.map(lambda x:x.split('/')[-1].split('.')[0])
    return df

def make_test_df():
    base_path = __base_path +'test'
    image_paths = glob(base_path+'/*')
    df = pd.DataFrame({'path':image_paths})
    df['num_id'] = df.path.map(lambda x:x.split('/')[-1].split('.')[0])
    return df

def get_img_paths_for(directory, num_id_series):
    base_path = '../data/'
    s = num_id_series.map(
            lambda id_x: base_path+directory+'/'+id_x+'.jpg')
    return s

def split_df(df):
    train, validate, test = np.split(df.sample(frac=1, random_state=12), 
                                     [int(.6*len(df)), int(.8*len(df))])
    return train, validate, test

def parse_json(fp):
    with open(fp, 'rb') as f:
        config = json.load(f)
    return config

def check_image(path):
    print(path)
    img = plt.imread(path)
    print(img.shape)
    plt.imshow(img)
