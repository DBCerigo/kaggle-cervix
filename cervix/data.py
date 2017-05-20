from __future__ import print_function
from glob import glob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def make_base_df():
    base_path = '../data/train'
    image_paths = []
    for type_base_path in sorted(glob(base_path +'/*')):
        image_paths = image_paths + glob(type_base_path + '/*')
    df = pd.DataFrame({'path':image_paths})
    df['type'] = df.path.map(lambda x: x.split('/')[-2])
    df['filetype'] = df.path.map(lambda x: x.split('.')[-1])
    df['num_id'] = df.path.map(lambda x:x.split('/')[-1].split('.')[0])
    return df

def make_test_df():
    base_path = '../data/test'
    image_paths = glob(base_path+'/*')
    df = pd.DataFrame({'path':image_paths})
    df['num_id'] = df.path.map(lambda x:x.split('/')[-1].split('.')[0])
    return df

def get_img_paths_for(directory, num_id_series):
    base_path = '../data/'
    s = num_id_series.map(
            lambda id_x: base_path+directory+'/'+id_x+'.jpg')
    return s


def check_image(path):
    print(path)
    img = plt.imread(path)
    print(img.shape)
    print(img)
    plt.imshow(img)
