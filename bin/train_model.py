from __future__ import print_function, division
import os
import sys
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
base_module_path = os.path.abspath(os.path.join('../'))
if base_module_path not in sys.path:
    sys.path.append(base_module_path)
import cervix as c

df = c.data.make_base_df()
df = c.processing.transform_save_imgs(df, c.processing.resize_n, n=299)
train, validate, test = c.data.split_df(df)

# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
    # we have three classes so usung 3 on dense predictions
predictions = Dense(3, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 172 layers and unfreeze the rest:
for layer in model.layers[:172]:
    layer.trainable = False
for layer in model.layers[172:]:
    layer.trainable = True

# compile the model (should be done *after* setting layers to non-trainable)
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), 
              loss='categorical_crossentropy')

# train the model on the new data for a few epochs
batch_size = 20
generator = c.processing.df_to_keras_generator(train,batch_size,grayscale=False)
history = model.fit_generator(generator,
                                    steps_per_epoch=len(train)//batch_size+1,
                                    epochs=20)

history_fp = '/home/u3760/model/history/v3_172_SGD_v3.pk'
model_fp = '/home/u3760/model/v3_172_SGD_v3.h5'
if c.analysis.save_history(history.history, history_fp):
    print('Model history saved to '+history_fp+'.')
else:
    print('Model history save to '+history_fp+' failed.')
model.save(model_fp)

