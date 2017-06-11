from __future__ import print_function, division
import os
import sys
import cv2
import numpy as np
from keras.models import load_model
base_module_path = os.path.abspath(os.path.join('../'))
if base_module_path not in sys.path:
    sys.path.append(base_module_path)
import cervix as c

config = c.data.parse_json('config.json')

model_path = config['submission']['model_path']
model = load_model(model_path)
test = c.data.make_test_df()
test = c.processing.transform_save_imgs(test, c.processing.resize_n, n=299)

counter = 0
test_data = np.zeros((512,299,299,3))
for _, row in test.iterrows():
    test_data[counter] = cv2.imread(row['processed_path'],1)
    counter+=1

predictions = model.predict(test_data)
test = c.processing.append_probabilities(test, predictions, ['1','2','3'])

sub_path = config['submission']['submission_path']
c.submission.write_submission_file(sub_path, test)
