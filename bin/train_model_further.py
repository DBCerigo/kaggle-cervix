from __future__ import print_function, division
import os
import sys
from keras.models import load_model
base_module_path = os.path.abspath(os.path.join('../'))
if base_module_path not in sys.path:
    sys.path.append(base_module_path)
import cervix as c

config = c.data.parse_json('config.json')

df = c.data.make_base_df()
train  = c.processing.transform_save_imgs(df, c.processing.resize_n, n=299)

last_model_fp = config['further_training']['old_model_path']
next_model_fp = config['further_training']['new_model_path']

model = load_model(last_model_fp)

batch_size = 20
generator = c.processing.df_to_keras_generator(train,batch_size,grayscale=False)
history = model.fit_generator(generator,
                                    steps_per_epoch=len(train)//batch_size+1,
                                    epochs=20)

history_fp = config['further_training']['history_path']

if c.analysis.save_history(history.history, history_fp):
    print('Model history saved to '+history_fp+'.')
else:
    print('Model history save to '+history_fp+' failed.')
model.save(next_model_fp)
