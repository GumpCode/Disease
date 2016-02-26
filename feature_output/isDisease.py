#coding: utf-8

import theano
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from Load_data import load_data

data, label=load_data()
label = np_utils.to_categorical(label, 2)
print ('finish loading data')

batch_size = 111
nb_classes = 2
nb_epoch = 2
data_augmentation = True

img_rows, img_cols = 64, 64
img_channels = 3


#init a model
model = Sequential()

#conv1
model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(img_channels, img_rows, img_cols)))
model.add(Activation('relu'))
model.compile(loss='categorical_crossentropy', optimizer='adamax')
a1 = model.predict(data, batch_size=batch_size)
for i in range(793):
    for j in range(32):
        cv2.imwrite('/home/ganlinhao/image/' + str(i) + '_' + str(j) + '.jpg', a1[i][j])

#conv2
model.add(Convolution2D(32, 3, 3)) 
model.add(Activation('relu'))
#pool1
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
#conv3
model.add(Convolution2D(64, 3, 3, border_mode='same')) 
model.add(Activation('relu'))
#conv4
model.add(Convolution2D(64, 3, 3)) 
model.add(Activation('relu'))
#pool2
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
#connect1
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
#softmax
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(loss='categorical_crossentropy', optimizer=sgd)
model.compile(loss='categorical_crossentropy', optimizer='adamax')

result = model.fit(data, label, batch_size=batch_size,nb_epoch=nb_epoch,shuffle=True,verbose=1,show_accuracy=True,validation_split=0.3)
