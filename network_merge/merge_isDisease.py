#coding: utf-8

import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape, Merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from Load_data import load_data

data, label=load_data()
label = np_utils.to_categorical(label, 2)
print ('finish loading data')

batch_size = 111
nb_classes = 2
nb_epoch = 200
data_augmentation = True

img_rows, img_cols = 64, 64
img_channels = 3


#init a model
model64 = Sequential()

#conv1
model64.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(img_channels, img_rows, img_cols)))
model64.add(Activation('relu'))
model64.add(Convolution2D(32, 3, 3)) 
model64.add(Activation('relu'))
model64.add(MaxPooling2D(pool_size=(2, 2)))
model64.add(Dropout(0.25))

#conv2
model64.add(Convolution2D(64, 3, 3, border_mode='same')) 
model64.add(Activation('relu'))
model64.add(Convolution2D(64, 3, 3)) 
model64.add(Activation('relu'))
model64.add(MaxPooling2D(pool_size=(2, 2)))
model64.add(Dropout(0.25))

model64.add(Flatten())
model64.add(Dense(512))
model64.add(Activation('relu'))
model64.add(Dropout(0.5))
model64.add(Dense(nb_classes))
model64.add(Activation('softmax'))

#init a model
model32 = Sequential()

#conv1
model32.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(img_channels, img_rows, img_cols)))
model32.add(Activation('relu'))
model32.add(AveragePooling2D(pool_size=(2,2)))
model32.add(Convolution2D(32, 3, 3)) 
model32.add(Activation('relu'))
model32.add(MaxPooling2D(pool_size=(2, 2)))
model32.add(Dropout(0.25))

#conv2
model32.add(Convolution2D(64, 3, 3)) 
model32.add(Activation('relu'))
model32.add(MaxPooling2D(pool_size=(2, 2)))
model32.add(Dropout(0.5))

model32.add(Flatten())
model32.add(Dense(512))
model32.add(Activation('relu'))
model32.add(Dropout(0.5))
model32.add(Dense(nb_classes))
model32.add(Activation('softmax'))

model = Sequential()
model.add(Merge([model64, model32], mode='concat', concat_axis=-1))
model.add(Activation('softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

#nb_label = np.append(label, label)
result = model.fit([data, data], label, batch_size=batch_size,nb_epoch=nb_epoch,shuffle=True,verbose=1,show_accuracy=True,validation_split=0.3)

print model.evaluate([data,data], label, batch_size=batch_size)
