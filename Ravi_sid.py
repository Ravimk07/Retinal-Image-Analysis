# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 14:52:45 2018

@author: admin
"""

from keras.layers import Flatten, Dense, Input
from keras.layers import Conv2D, MaxPooling2D, concatenate
from keras.models import Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.layers import Activation, Dropout, GlobalAveragePooling2D, Concatenate, Dense, Input, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import TensorBoard

weight_decay=1e-4
input_shape = (32, 32, 3)
#def get_model(input_shape, weight_decay):
img_input = Input(shape=input_shape)
x = Conv2D(32, (3,3), padding='same',strides=(1,1),use_bias=False, kernel_regularizer=l2(weight_decay))(img_input)
tower_1 = Conv2D(32, (1,1), padding='same', activation='relu')(x)
tower_1 = Conv2D(32, (3,3), padding='same', activation='relu')(tower_1)

tower_2 = Conv2D(32, (1,1), padding='same', activation='relu')(x)
tower_2 = Conv2D(32, (5,5), padding='same', activation='relu')(tower_2)

tower_3 = MaxPooling2D((3,3), strides=(1,1), padding='same')(x)
tower_3 = Conv2D(32, (1,1), padding='same', activation='relu')(tower_3)

tower_4 = Conv2D(32, (1,1), padding='same', activation='relu')(x)

output = concatenate([tower_1, tower_2, tower_3, tower_4], axis = 3)

tower_1 = Conv2D(32, (1,1), padding='same', activation='relu')(output)
tower_1 = Conv2D(32, (3,3), padding='same', activation='relu')(tower_1)

tower_2 = Conv2D(32, (1,1), padding='same', activation='relu')(output)
tower_2 = Conv2D(32, (5,5), padding='same', activation='relu')(tower_2)

tower_3 = MaxPooling2D((3,3), strides=(1,1), padding='same')(output)
tower_3 = Conv2D(32, (1,1), padding='same', activation='relu')(tower_3)

tower_4 = Conv2D(32, (1,1), padding='same', activation='relu')(output)

output_2 = concatenate([output, tower_1, tower_2, tower_3, tower_4], axis = 3)

tower_1 = Conv2D(32, (1,1), padding='same', activation='relu')(output_2)
tower_1 = Conv2D(32, (3,3), padding='same', activation='relu')(tower_1)

tower_2 = Conv2D(32, (1,1), padding='same', activation='relu')(output_2)
tower_2 = Conv2D(32, (5,5), padding='same', activation='relu')(tower_2)

tower_3 = MaxPooling2D((3,3), strides=(1,1), padding='same')(output_2)
tower_3 = Conv2D(32, (1,1), padding='same', activation='relu')(tower_3)

tower_4 = Conv2D(32, (1,1), padding='same', activation='relu')(output_2)

output_3 = concatenate([output, output_2, tower_1, tower_2, tower_3, tower_4], axis = 3)

y = Conv2D(256, (1,1), padding='same',strides=(1,1),use_bias=False, kernel_regularizer=l2(weight_decay))(output_3)
y = AveragePooling2D((2, 2), strides=(2, 2))(y)

tower_1 = Conv2D(64, (1,1), padding='same', activation='relu')(y)
tower_1 = Conv2D(64, (3,3), padding='same', activation='relu')(tower_1)

tower_2 = Conv2D(64, (1,1), padding='same', activation='relu')(y)
tower_2 = Conv2D(64, (5,5), padding='same', activation='relu')(tower_2)

tower_3 = MaxPooling2D((3,3), strides=(1,1), padding='same')(y)
tower_3 = Conv2D(64, (1,1), padding='same', activation='relu')(tower_3)

tower_4 = Conv2D(64, (1,1), padding='same', activation='relu')(y)
output = concatenate([tower_1, tower_2, tower_3, tower_4], axis = 3)

tower_1 = Conv2D(64, (1,1), padding='same', activation='relu')(output)
tower_1 = Conv2D(64, (3,3), padding='same', activation='relu')(tower_1)

tower_2 = Conv2D(64, (1,1), padding='same', activation='relu')(output)
tower_2 = Conv2D(64, (5,5), padding='same', activation='relu')(tower_2)

tower_3 = MaxPooling2D((3,3), strides=(1,1), padding='same')(output)
tower_3 = Conv2D(64, (1,1), padding='same', activation='relu')(tower_3)

tower_4 = Conv2D(64, (1,1), padding='same', activation='relu')(output)
output_2 = concatenate([output,tower_1, tower_2, tower_3, tower_4], axis = 3)

tower_1 = Conv2D(64, (1,1), padding='same', activation='relu')(output_2)
tower_1 = Conv2D(64, (3,3), padding='same', activation='relu')(tower_1)

tower_2 = Conv2D(64, (1,1), padding='same', activation='relu')(output_2)
tower_2 = Conv2D(64, (5,5), padding='same', activation='relu')(tower_2)

tower_3 = MaxPooling2D((3,3), strides=(1,1), padding='same')(output_2)
tower_3 = Conv2D(64, (1,1), padding='same', activation='relu')(tower_3)

tower_4 = Conv2D(64, (1,1), padding='same', activation='relu')(output_2)
output_3 = concatenate([output, output_2, tower_1, tower_2, tower_3, tower_4], axis = 3)

z = Conv2D(512, (1,1), padding='same',strides=(1,1),use_bias=False, kernel_regularizer=l2(weight_decay))(output_3)
z = AveragePooling2D((2, 2), strides=(2, 2))(z)
tower_1 = Conv2D(96, (1,1), padding='same', activation='relu')(z)
tower_1 = Conv2D(96, (3,3), padding='same', activation='relu')(tower_1)

tower_2 = Conv2D(96, (1,1), padding='same', activation='relu')(z)
tower_2 = Conv2D(96, (5,5), padding='same', activation='relu')(tower_2)

tower_3 = MaxPooling2D((3,3), strides=(1,1), padding='same')(z)
tower_3 = Conv2D(96, (1,1), padding='same', activation='relu')(tower_3)

tower_4 = Conv2D(96, (1,1), padding='same', activation='relu')(z)
output = concatenate([tower_1, tower_2, tower_3, tower_4], axis = 3)

tower_1 = Conv2D(96, (1,1), padding='same', activation='relu')(output)
tower_1 = Conv2D(96, (3,3), padding='same', activation='relu')(tower_1)

tower_2 = Conv2D(96, (1,1), padding='same', activation='relu')(output)
tower_2 = Conv2D(96, (5,5), padding='same', activation='relu')(tower_2)

tower_3 = MaxPooling2D((3,3), strides=(1,1), padding='same')(output)
tower_3 = Conv2D(96, (1,1), padding='same', activation='relu')(tower_3)

tower_4 = Conv2D(96, (1,1), padding='same', activation='relu')(output)
output_2 = concatenate([output,tower_1, tower_2, tower_3, tower_4], axis = 3)

tower_1 = Conv2D(96, (1,1), padding='same', activation='relu')(output_2)
tower_1 = Conv2D(96, (3,3), padding='same', activation='relu')(tower_1)

tower_2 = Conv2D(96, (1,1), padding='same', activation='relu')(output_2)
tower_2 = Conv2D(96, (5,5), padding='same', activation='relu')(tower_2)

tower_3 = MaxPooling2D((3,3), strides=(1,1), padding='same')(output_2)
tower_3 = Conv2D(96, (1,1), padding='same', activation='relu')(tower_3)

tower_4 = Conv2D(96, (1,1), padding='same', activation='relu')(output_2)
output_3 = concatenate([output, output_2, tower_1, tower_2, tower_3, tower_4], axis = 3)
output_3 = GlobalAveragePooling2D()(output_3)
output_3 = Dense(10, activation = 'softmax')(output_3)
#    return output_3

model = Model(input=img_input, output = output_3)
model.compile(optimizer = 'adam', loss  = 'categorical_crossentropy', metrics = ['accuracy'])
model.summary()

batch_size = 128
num_classes = 10
epochs = 12

import keras
from keras.datasets import cifar10
# input image dimensions
img_rows, img_cols = 32, 32

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
    input_shape = (3, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 3)

print(input_shape)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_train = y_train.reshape(50000, 10)
y_test = keras.utils.to_categorical(y_test, num_classes)
y_test = y_test.reshape(10000, 10)

newpath = './hello'
tensorboard = TensorBoard(log_dir = newpath)

cnn = model.fit(x_train, y_train, batch_size = 32, epochs = 100, validation_data = (x_test, y_test), shuffle = True, callbacks = [tensorboard])
