import keras
import numpy as np
import pandas as pd
import os
import cv2
from keras.applications.imagenet_utils import _obtain_input_shape
import keras.backend as K
import glob
import time 
import warnings
#from warnings import RuntimeWarning
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
import pickle
import h5py
import matplotlib.pyplot as plt
from keras.models import model_from_json
import datetime
from keras.utils import np_utils
from shutil import copy2
from numpy.random import permutation
from keras.models import Sequential, Model
from keras.regularizers import l2, l1
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import  MaxPooling2D
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import SGD
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, TensorBoard
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
from keras.layers import Flatten, Dense, Input
from keras.layers import Conv2D, MaxPooling2D, concatenate
from keras.layers import Activation, Dropout, GlobalAveragePooling2D, Concatenate, Dense, Input, AveragePooling2D
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, TensorBoard

from keras.applications import densenet
def get_im_resizer(path):
    img = cv2.imread(path)
    resized_img = cv2.resize(img, (300, 300), cv2.INTER_LINEAR)
    return resized_img

def load_data():
    X_train = []
    y_train = []
    start_time = time.time()
    print('Reading Images')
    for j in range(4):
        print('Loading Folder %s' % str(j))
        path = os.path.join(r'D:\Ravi\Database\messidor data\DR\More_DR_patches_with_grading', 'Grade ' + str(j), '*.tif')
        files = glob.glob(path)
        for fl in files:
            img = get_im_resizer(fl)
            X_train.append(img)
            y_train.append(j)
    end_time = time.time()
    print('Time taken for loading:', end_time - start_time)
    return X_train, y_train

def store_cache_data(data, path):
    if os.path.isdir(os.path.dirname(path)):
        file = open(path, 'wb')
        pickle.dump(data, file)
        file.close()
        print('I just dumped the data bro!')
    else:
        print('Directory does exist bro!')
        
def reload_data(path):
    print('Reloading from Cache data.')
    data = dict()
    if os.path.isfile(path):
        file = open(path, 'rb')
        data = pickle.load(file)
    return data

def save_model(model):
    json_string = model.to_json()
    if not os.path.isdir('cache'):
        os.mkdir('cache')
    open(os.path.join('cache', 'architecure_network.json'), 'w').write(json_string)
    model.save_weights(os.path.join('cache', 'model_weights.h5'), overwrite = True)

def read_model():
    model = model_from_json(open(os.path.join('cache', 'architecture_network.json')).read())
    model.load_weights(os.path.join('cache', 'model_weights.h5'))
    return model

def preprocess_data():
    cache_path = os.path.join('cache', 'data_r_' + str(224) + '_c_' + str(224) + '_t_' + str(3) + '.dat')
    if not os.path.isfile(cache_path):
        data, target = load_data()
        store_cache_data((data, target), cache_path)
    else:
        print('Restoring train from cache bro!')
        data, target = reload_data(cache_path)
    print('Convert this to numpy coz cant wont with list bruv!')
    data = np.array(data, dtype = np.uint8)
    target = np.array(target, dtype = np.uint8)
    print('Reshaping the train data bro! Naah! Just kidding, We are TensorFlow!')
    data = np.reshape(data,(3054, 300, 300, 3))
    print('The new shape of this data is:', data.shape)
    print('But we will convert it to float!')
    data = np.array(data, dtype = np.float16)
    target = np_utils.to_categorical(target, 4)
    #Shuffle experiment START!
    perm = permutation(len(target))
    data = data[perm]
    target = target[perm]
    #EVeryday I am shuffling!

    print('Number of training samples: ', data.shape[0])
    return data, target

def base_model():
    
    weight_decay=1e-4
    input_shape = (300, 300, 3)
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
    output_3 = Dense(1536, activation = 'relu')(output_3)
    output_3 = Dropout(0.25)(output_3)
    output_3 = Dense(4, activation = 'softmax')(output_3)
    #    return output_3
    
    model = Model(input=img_input, output = output_3)
    model.compile(optimizer = 'adam', loss  = 'categorical_crossentropy', metrics = ['accuracy'])
    model.summary()
    
    return model

def __conv_block(ip, nb_filter, bottleneck=False, dropout_rate=None, weight_decay=1e-4):
    ''' Apply BatchNorm, Relu, 3x3 Conv2D, optional bottleneck block and dropout
    Args:
        ip: Input keras tensor
        nb_filter: number of filters
        bottleneck: add bottleneck block
        dropout_rate: dropout rate
        weight_decay: weight decay factor
    Returns: keras tensor with batch_norm, relu and convolution2d added (optional bottleneck)
    '''
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(ip)
    x = Activation('relu')(x)

    if bottleneck:
        inter_channel = nb_filter * 4  # Obtained from https://github.com/liuzhuang13/DenseNet/blob/master/densenet.lua

        x = Conv2D(inter_channel, (1, 1), kernel_initializer='he_normal', padding='same', use_bias=False,
                   kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)
        x = Activation('relu')(x)

    x = Conv2D(nb_filter, (3, 3), kernel_initializer='he_normal', padding='same', use_bias=False)(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x

def DenseNet(input_shape=None, depth=40, nb_dense_block=3, growth_rate=12, nb_filter=-1, nb_layers_per_block=-1,
             bottleneck=False, reduction=0.0, dropout_rate=0.0, weight_decay=1e-4, subsample_initial_block=False,
             include_top=True, weights=None, input_tensor=None,
             classes=10, activation='softmax'):
    '''Instantiate the DenseNet architecture,
        optionally loading weights pre-trained
        on CIFAR-10. Note that when using TensorFlow,
        for best performance you should set
        `image_data_format='channels_last'` in your Keras config
        at ~/.keras/keras.json.
        The model and the weights are compatible with both
        TensorFlow and Theano. The dimension ordering
        convention used by the model is the one
        specified in your Keras config file.
        # Arguments
            input_shape: optional shape tuple, only to be specified
                if `include_top` is False (otherwise the input shape
                has to be `(32, 32, 3)` (with `channels_last` dim ordering)
                or `(3, 32, 32)` (with `channels_first` dim ordering).
                It should have exactly 3 inputs channels,
                and width and height should be no smaller than 8.
                E.g. `(200, 200, 3)` would be one valid value.
            depth: number or layers in the DenseNet
            nb_dense_block: number of dense blocks to add to end (generally = 3)
            growth_rate: number of filters to add per dense block
            nb_filter: initial number of filters. -1 indicates initial
                number of filters is 2 * growth_rate
            nb_layers_per_block: number of layers in each dense block.
                Can be a -1, positive integer or a list.
                If -1, calculates nb_layer_per_block from the network depth.
                If positive integer, a set number of layers per dense block.
                If list, nb_layer is used as provided. Note that list size must
                be (nb_dense_block + 1)
            bottleneck: flag to add bottleneck blocks in between dense blocks
            reduction: reduction factor of transition blocks.
                Note : reduction value is inverted to compute compression.
            dropout_rate: dropout rate
            weight_decay: weight decay rate
            subsample_initial_block: Set to True to subsample the initial convolution and
                add a MaxPool2D before the dense blocks are added.
            include_top: whether to include the fully-connected
                layer at the top of the network.
            weights: one of `None` (random initialization) or
                'imagenet' (pre-training on ImageNet)..
            input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
                to use as image input for the model.
            classes: optional number of classes to classify images
                into, only to be specified if `include_top` is True, and
                if no `weights` argument is specified.
            activation: Type of activation at the top layer. Can be one of 'softmax' or 'sigmoid'.
                Note that if sigmoid is used, classes must be 1.
        # Returns
            A Keras model instance.
        '''

    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `cifar10` '
                         '(pre-training on CIFAR-10).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as ImageNet with `include_top`'
                         ' as true, `classes` should be 1000')

    if activation not in ['softmax', 'sigmoid']:
        raise ValueError('activation must be one of "softmax" or "sigmoid"')

    if activation == 'sigmoid' and classes != 1:
        raise ValueError('sigmoid activation can only be used when classes = 1')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=32,
                                      min_size=8,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = __create_dense_net(classes, img_input, include_top, depth, nb_dense_block,
                           growth_rate, nb_filter, nb_layers_per_block, bottleneck, reduction,
                           dropout_rate, weight_decay, subsample_initial_block, activation)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    from keras.engine.topology import get_source_inputs
    from keras.utils.data_utils import get_file

    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='densenet')

    # load weights
    if weights == 'imagenet':
        weights_loaded = False

        if (depth == 121) and (nb_dense_block == 4) and (growth_rate == 32) and (nb_filter == 64) and \
                (bottleneck is True) and (reduction == 0.5) and (dropout_rate == 0.0) and (subsample_initial_block):
            if include_top:
                weights_path = get_file('DenseNet-BC-121-32.h5',
                                        DENSENET_121_WEIGHTS_PATH,
                                        cache_subdir='models',
                                        md5_hash='a439dd41aa672aef6daba4ee1fd54abd')
            else:
                weights_path = get_file('DenseNet-BC-121-32-no-top.h5',
                                        DENSENET_121_WEIGHTS_PATH_NO_TOP,
                                        cache_subdir='models',
                                        md5_hash='55e62a6358af8a0af0eedf399b5aea99')
            model.load_weights(weights_path)
            weights_loaded = True

        if (depth == 161) and (nb_dense_block == 4) and (growth_rate == 48) and (nb_filter == 96) and \
                (bottleneck is True) and (reduction == 0.5) and (dropout_rate == 0.0) and (subsample_initial_block):
            if include_top:
                weights_path = get_file('DenseNet-BC-161-48.h5',
                                        DENSENET_161_WEIGHTS_PATH,
                                        cache_subdir='models',
                                        md5_hash='6c326cf4fbdb57d31eff04333a23fcca')
            else:
                weights_path = get_file('DenseNet-BC-161-48-no-top.h5',
                                        DENSENET_161_WEIGHTS_PATH_NO_TOP,
                                        cache_subdir='models',
                                        md5_hash='1a9476b79f6b7673acaa2769e6427b92')
            model.load_weights(weights_path)
            weights_loaded = True

        if (depth == 169) and (nb_dense_block == 4) and (growth_rate == 32) and (nb_filter == 64) and \
                (bottleneck is True) and (reduction == 0.5) and (dropout_rate == 0.0) and (subsample_initial_block):
            if include_top:
                weights_path = get_file('DenseNet-BC-169-32.h5',
                                        DENSENET_169_WEIGHTS_PATH,
                                        cache_subdir='models',
                                        md5_hash='914869c361303d2e39dec640b4e606a6')
            else:
                weights_path = get_file('DenseNet-BC-169-32-no-top.h5',
                                        DENSENET_169_WEIGHTS_PATH_NO_TOP,
                                        cache_subdir='models',
                                        md5_hash='89c19e8276cfd10585d5fadc1df6859e')
            model.load_weights(weights_path)
            weights_loaded = True

        if weights_loaded:
            if K.backend() == 'theano':
                convert_all_kernels_in_model(model)

            if K.image_data_format() == 'channels_first' and K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image data format convention '
                              '(`image_data_format="channels_first"`). '
                              'For best performance, set '
                              '`image_data_format="channels_last"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')

            print("Weights for the model were loaded successfully")

    return model

def __transition_block(ip, nb_filter, compression=1.0, weight_decay=1e-4):
    ''' Apply BatchNorm, Relu 1x1, Conv2D, optional compression, dropout and Maxpooling2D
    Args:
        ip: keras tensor
        nb_filter: number of filters
        compression: calculated as 1 - reduction. Reduces the number of feature maps
                    in the transition block.
        dropout_rate: dropout rate
        weight_decay: weight decay factor
    Returns: keras tensor, after applying batch_norm, relu-conv, dropout, maxpool
    '''
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(ip)
    x = Activation('relu')(x)
    x = Conv2D(int(nb_filter * compression), (1, 1), kernel_initializer='he_normal', padding='same', use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    x = AveragePooling2D((2, 2), strides=(2, 2))(x)

    return x
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D

def __transition_up_block(ip, nb_filters, type='deconv', weight_decay=1E-4):
    ''' SubpixelConvolutional Upscaling (factor = 2)
    Args:
        ip: keras tensor
        nb_filters: number of layers
        type: can be 'upsampling', 'subpixel', 'deconv'. Determines type of upsampling performed
        weight_decay: weight decay factor
    Returns: keras tensor, after applying upsampling operation.
    '''

    if type == 'upsampling':
        x = UpSampling2D()(ip)

    else:
        x = Conv2DTranspose(nb_filters, (3, 3), activation='relu', padding='same', strides=(2, 2),
                            kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(ip)

    return x


def __create_dense_net(nb_classes, img_input, include_top, depth=40, nb_dense_block=3, growth_rate=12, nb_filter=-1,
                       nb_layers_per_block=-1, bottleneck=False, reduction=0.0, dropout_rate=None, weight_decay=1e-4,
                       subsample_initial_block=False, activation='softmax'):
    ''' Build the DenseNet model
    Args:
        nb_classes: number of classes
        img_input: tuple of shape (channels, rows, columns) or (rows, columns, channels)
        include_top: flag to include the final Dense layer
        depth: number or layers
        nb_dense_block: number of dense blocks to add to end (generally = 3)
        growth_rate: number of filters to add per dense block
        nb_filter: initial number of filters. Default -1 indicates initial number of filters is 2 * growth_rate
        nb_layers_per_block: number of layers in each dense block.
                Can be a -1, positive integer or a list.
                If -1, calculates nb_layer_per_block from the depth of the network.
                If positive integer, a set number of layers per dense block.
                If list, nb_layer is used as provided. Note that list size must
                be (nb_dense_block + 1)
        bottleneck: add bottleneck blocks
        reduction: reduction factor of transition blocks. Note : reduction value is inverted to compute compression
        dropout_rate: dropout rate
        weight_decay: weight decay rate
        subsample_initial_block: Set to True to subsample the initial convolution and
                add a MaxPool2D before the dense blocks are added.
        subsample_initial:
        activation: Type of activation at the top layer. Can be one of 'softmax' or 'sigmoid'.
                Note that if sigmoid is used, classes must be 1.
    Returns: keras tensor with nb_layers of conv_block appended
    '''

    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    if reduction != 0.0:
        assert reduction <= 1.0 and reduction > 0.0, 'reduction value must lie between 0.0 and 1.0'

    # layers in each dense block
    if type(nb_layers_per_block) is list or type(nb_layers_per_block) is tuple:
        nb_layers = list(nb_layers_per_block)  # Convert tuple to list

        assert len(nb_layers) == (nb_dense_block), 'If list, nb_layer is used as provided. ' \
                                                   'Note that list size must be (nb_dense_block)'
        final_nb_layer = nb_layers[-1]
        nb_layers = nb_layers[:-1]
    else:
        if nb_layers_per_block == -1:
            assert (depth - 4) % 3 == 0, 'Depth must be 3 N + 4 if nb_layers_per_block == -1'
            count = int((depth - 4) / 3)

            if bottleneck:
                count = count // 2

            nb_layers = [count for _ in range(nb_dense_block)]
            final_nb_layer = count
        else:
            final_nb_layer = nb_layers_per_block
            nb_layers = [nb_layers_per_block] * nb_dense_block

    # compute initial nb_filter if -1, else accept users initial nb_filter
    if nb_filter <= 0:
        nb_filter = 2 * growth_rate

    # compute compression factor
    compression = 1.0 - reduction

    # Initial convolution
    if subsample_initial_block:
        initial_kernel = (7, 7)
        initial_strides = (2, 2)
    else:
        initial_kernel = (3, 3)
        initial_strides = (1, 1)

    x = Conv2D(nb_filter, initial_kernel, kernel_initializer='he_normal', padding='same',
               strides=initial_strides, use_bias=False, kernel_regularizer=l2(weight_decay))(img_input)

    if subsample_initial_block:
        x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        x, nb_filter = __dense_block(x, nb_layers[block_idx], nb_filter, growth_rate, bottleneck=bottleneck,
                                     dropout_rate=dropout_rate, weight_decay=weight_decay)
        # add transition_block
        x = __transition_block(x, nb_filter, compression=compression, weight_decay=weight_decay)
        nb_filter = int(nb_filter * compression)

    # The last dense_block does not have a transition_block
    x, nb_filter = __dense_block(x, final_nb_layer, nb_filter, growth_rate, bottleneck=bottleneck,
                                 dropout_rate=dropout_rate, weight_decay=weight_decay)

    x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)

    if include_top:
        x = Dense(nb_classes, activation=activation)(x)

    return x
def __dense_block(x, nb_layers, nb_filter, growth_rate, bottleneck=False, dropout_rate=None, weight_decay=1e-4,
                  grow_nb_filters=True, return_concat_list=False):
    ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
    Args:
        x: keras tensor
        nb_layers: the number of layers of conv_block to append to the model.
        nb_filter: number of filters
        growth_rate: growth rate
        bottleneck: bottleneck block
        dropout_rate: dropout rate
        weight_decay: weight decay factor
        grow_nb_filters: flag to decide to allow number of filters to grow
        return_concat_list: return the list of feature maps along with the actual output
    Returns: keras tensor with nb_layers of conv_block appended
    '''
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x_list = [x]

    for i in range(nb_layers):
        cb = __conv_block(x, growth_rate, bottleneck, dropout_rate, weight_decay)
        x_list.append(cb)

        x = concatenate([x, cb], axis=concat_axis)

        if grow_nb_filters:
            nb_filter += growth_rate

    if return_concat_list:
        return x, nb_filter, x_list
    else:
        return x, nb_filter

def __create_fcn_dense_net(nb_classes, img_input, include_top, nb_dense_block=5, growth_rate=12,
                           reduction=0.0, dropout_rate=None, weight_decay=1e-4,
                           nb_layers_per_block=4, nb_upsampling_conv=128, upsampling_type='upsampling',
                           init_conv_filters=48, input_shape=None, activation='deconv'):
    ''' Build the DenseNet model
    Args:
        nb_classes: number of classes
        img_input: tuple of shape (channels, rows, columns) or (rows, columns, channels)
        include_top: flag to include the final Dense layer
        nb_dense_block: number of dense blocks to add to end (generally = 3)
        growth_rate: number of filters to add per dense block
        reduction: reduction factor of transition blocks. Note : reduction value is inverted to compute compression
        dropout_rate: dropout rate
        weight_decay: weight decay
        nb_layers_per_block: number of layers in each dense block.
            Can be a positive integer or a list.
            If positive integer, a set number of layers per dense block.
            If list, nb_layer is used as provided. Note that list size must
            be (nb_dense_block + 1)
        nb_upsampling_conv: number of convolutional layers in upsampling via subpixel convolution
        upsampling_type: Can be one of 'upsampling', 'deconv' and 'subpixel'. Defines
            type of upsampling algorithm used.
        input_shape: Only used for shape inference in fully convolutional networks.
        activation: Type of activation at the top layer. Can be one of 'softmax' or 'sigmoid'.
                    Note that if sigmoid is used, classes must be 1.
    Returns: keras tensor with nb_layers of conv_block appended
    '''

    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    if concat_axis == 1:  # channels_first dim ordering
        _, rows, cols = input_shape
    else:
        rows, cols, _ = input_shape

    if reduction != 0.0:
        assert reduction <= 1.0 and reduction > 0.0, 'reduction value must lie between 0.0 and 1.0'

    # check if upsampling_conv has minimum number of filters
    # minimum is set to 12, as at least 3 color channels are needed for correct upsampling
    assert nb_upsampling_conv > 12 and nb_upsampling_conv % 4 == 0, 'Parameter `upsampling_conv` number of channels must ' \
                                                                    'be a positive number divisible by 4 and greater ' \
                                                                    'than 12'

    # layers in each dense block
    if type(nb_layers_per_block) is list or type(nb_layers_per_block) is tuple:
        nb_layers = list(nb_layers_per_block)  # Convert tuple to list

        assert len(nb_layers) == (nb_dense_block + 1), 'If list, nb_layer is used as provided. ' \
                                                       'Note that list size must be (nb_dense_block + 1)'

        bottleneck_nb_layers = nb_layers[-1]
        rev_layers = nb_layers[::-1]
        nb_layers.extend(rev_layers[1:])
    else:
        bottleneck_nb_layers = nb_layers_per_block
        nb_layers = [nb_layers_per_block] * (2 * nb_dense_block + 1)

    # compute compression factor
    compression = 1.0 - reduction

    # Initial convolution
    x = Conv2D(init_conv_filters, (7, 7), kernel_initializer='he_normal', padding='same', name='initial_conv2D',
               use_bias=False, kernel_regularizer=l2(weight_decay))(img_input)
    x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)
    x = Activation('relu')(x)

    nb_filter = init_conv_filters

    skip_list = []

    # Add dense blocks and transition down block
    for block_idx in range(nb_dense_block):
        x, nb_filter = __dense_block(x, nb_layers[block_idx], nb_filter, growth_rate, dropout_rate=dropout_rate,
                                     weight_decay=weight_decay)

        # Skip connection
        skip_list.append(x)

        # add transition_block
        x = __transition_block(x, nb_filter, compression=compression, weight_decay=weight_decay)

        nb_filter = int(nb_filter * compression)  # this is calculated inside transition_down_block

    # The last dense_block does not have a transition_down_block
    # return the concatenated feature maps without the concatenation of the input
    _, nb_filter, concat_list = __dense_block(x, bottleneck_nb_layers, nb_filter, growth_rate,
                                              dropout_rate=dropout_rate, weight_decay=weight_decay,
                                              return_concat_list=True)

    skip_list = skip_list[::-1]  # reverse the skip list

    # Add dense blocks and transition up block
    for block_idx in range(nb_dense_block):
        n_filters_keep = growth_rate * nb_layers[nb_dense_block + block_idx]

        # upsampling block must upsample only the feature maps (concat_list[1:]),
        # not the concatenation of the input with the feature maps (concat_list[0].
        l = concatenate(concat_list[1:], axis=concat_axis)

        t = __transition_up_block(l, nb_filters=n_filters_keep, type=upsampling_type, weight_decay=weight_decay)

        # concatenate the skip connection with the transition block
        x = concatenate([t, skip_list[block_idx]], axis=concat_axis)

        # Dont allow the feature map size to grow in upsampling dense blocks
        x_up, nb_filter, concat_list = __dense_block(x, nb_layers[nb_dense_block + block_idx + 1], nb_filter=growth_rate,
                                                     growth_rate=growth_rate, dropout_rate=dropout_rate,
                                                     weight_decay=weight_decay, return_concat_list=True,
                                                     grow_nb_filters=False)

    if include_top:
        x = Conv2D(nb_classes, (1, 1), activation='linear', padding='same', use_bias=False)(x_up)

        if K.image_data_format() == 'channels_first':
            channel, row, col = input_shape
        else:
            row, col, channel = input_shape

        x = Reshape((row * col, nb_classes))(x)
        x = Activation(activation)(x)
        x = Reshape((row, col, nb_classes))(x)
    else:
        x = x_up

    return x


def run_cross_validation_create_models(num_fold = 5):
    #Input image dimensions
    batch_size = 4
    nb_epoch = 50
    
    restore_from_last_checkpoint = 1
    
    data, target = preprocess_data() 
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=42)
    t1 = DenseNet(classes=4, input_shape=(300, 300, 3), depth=40, growth_rate=12, bottleneck=True, reduction=0.5)
#    model = Model(input=img_input, output = output_3)
#    model.compile(optimizer = 'adam', loss  = 'categorical_crossentropy', metrics = ['accuracy'])
#    model.summary()
#    model = base_model()
    top_model = Sequential()
    top_model.add(t1)
    top_model.add(Flatten(input_shape=t1.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(4, activation='softmax'))
    
    # note that it is necessary to start with a fully-trained
    # classifier, including the top classifier,
    # in order to successfully do fine-tuning
    
    # add the model on top of the convolutional base
    model = Sequential()
    model.add(top_model)
    model.summary()
    
    model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=1e-2, momentum=0.9),
              metrics=['accuracy'])

    num_fold += 1
    print('Start KFold number {} from {}'.format(num_fold, num_fold))
    print('Split train:', len(X_train), len(y_train))
    print('Split test:', len(X_test), len(y_test))
    kfold_weights_path = os.path.join('cache', 'weights_kfold_vgg16_' + str(num_fold) + '.h5')
#    if not os.path.isfile(kfold_weights_path) or restore_from_last_checkpoint == 0:
    callbacks = [
#                EarlyStoppingbyLossVal(monitor = 'val_loss', value = 0.00001, verbose = 1), 
#                EarlyStopping(monitor = 'val_loss', patience = 5, verbose = 1), 
            ModelCheckpoint(kfold_weights_path, monitor = 'val_loss', save_best_only = True, verbose = 0),
            TensorBoard(log_dir='./LogsForAUC', write_images = True)
    ]
    cnn = model.fit(X_train, y_train, batch_size = batch_size, epochs = nb_epoch, shuffle = True, verbose = 1, validation_data = (X_test, y_test), callbacks = callbacks)
    
    if os.path.isfile(kfold_weights_path):
        model.load_weights(kfold_weights_path)
    score1 = model.evaluate(X_test, y_test, show_accuracy = True, verbose = 0)
    print('Score on test was : ', score1)
    predictions = model.predict(X_train.astype('float32'), batch_size = batch_size, verbose = 1)
    score = log_loss(y_test, predictions)
    print('Score log_loss on test is', score)
    
    plt.plot(cnn.history['acc'])
    plt.plot(cnn.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(cnn.history['loss'])
    plt.plot(cnn.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    pd.DataFrame(cnn.history).to_csv("/historyAUC.csv")

def append_chunk(main, part):
    for p in part:
        main.append(p)
    return main

if __name__ == '__main__':
    num_fold = 5
    if not os.path.isdir('subm'):
        os.mkdir('subm')
    if not os.path.isdir('cache'):
        os.mkdir('cache')
    if not os.path.isfile(r"weights\vgg19_weights.h5"):
        print("Please put the pretained vgg19 weights in weights\\vgg19_weights.h5")
    run_cross_validation_create_models(num_fold = 5)