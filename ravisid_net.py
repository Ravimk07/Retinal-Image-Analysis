from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import AveragePooling2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Input
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import keras.backend as K
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import TensorBoard


def DenseNet(input_shape = None, nb_dense_block = 4, growth_rate = 12, nb_filter = -1,
             nb_layers_per_block = 4, reduction = 0.5, dropout_rate = 0.0, weight_decay = 1e-4,
             classes = 10):
    
    #setting up the input
    img_input = Input(shape = input_shape)
    
    #start creating densenet
    x = __create_dense_net(classes, img_input, nb_dense_block, growth_rate, nb_filter,
                           nb_layers_per_block, reduction, dropout_rate, weight_decay)
    
    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.

    inputs = img_input
    
    # Create model.
    model = Model(inputs, x, name='densenet')

    return model


def __conv_block(ip, nb_filter, dropout_rate = 0.1, weight_decay = 1e-4):
#    print("In conv")
    
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1
    
    x = BatchNormalization(axis = concat_axis, epsilon = 1.1e-5)(ip)
    x = LeakyReLU(0.3)(x)
    
    t1 = Conv2D(nb_filter, (1, 1), kernel_initializer = 'orthogonal', activation = 'relu', padding = 'same',
                kernel_regularizer = l2(weight_decay))(x)
    
    t2 = Conv2D(nb_filter, (1, 1), kernel_initializer = 'orthogonal', activation = 'relu', padding = 'same',
                kernel_regularizer = l2(weight_decay))(x)
    t2 = Conv2D(nb_filter, (3, 3), kernel_initializer = 'orthogonal', activation = 'relu', padding = 'same',
                kernel_regularizer = l2(weight_decay))(t2)
    
    t3 = Conv2D(nb_filter, (1, 1), kernel_initializer = 'orthogonal', activation = 'relu', padding = 'same',
                kernel_regularizer = l2(weight_decay))(x)
    t3 = Conv2D(nb_filter, (5, 5), kernel_initializer = 'orthogonal', activation = 'relu', padding = 'same',
                kernel_regularizer = l2(weight_decay))(t3)
    
    t4 = MaxPooling2D((3, 3), strides = (1, 1), padding = 'same')(x)
    t4 = Conv2D(nb_filter, (1, 1), kernel_initializer = 'orthogonal', activation = 'relu', padding = 'same',
                kernel_regularizer = l2(weight_decay))(t4)   
    
    x = concatenate([t1, t2, t3, t4])
    
    return x

def __dense_block(x, nb_filter, nb_layers_per_block, dropout_rate, weight_decay, growth_rate):
    
    print("In Dense")

    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    for i in range(nb_layers_per_block):
        cb = __conv_block(x, nb_filter, dropout_rate, weight_decay)
        
        x = concatenate([x, cb])
        
#        nb_filter += growth_rate
    
    return x, nb_filter

def __transition_block(ip, nb_filter, compression = 0.5, weight_decay = 1e-4):
    
    print("In transition")
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = BatchNormalization(axis = concat_axis, epsilon=1.1e-5)(ip)
    x = LeakyReLU(0.3)(x)
    x = Conv2D(int(nb_filter * compression), (1, 1), kernel_initializer = 'he_normal', padding = 'same',
               kernel_regularizer = l2(weight_decay))(x)
    x = AveragePooling2D((2, 2), strides = (2, 2))(x)
    
    return x


def __create_dense_net(classes, img_input, nb_dense_block=3, growth_rate=12, nb_filter=-1,
                       nb_layers_per_block = 4, reduction=0.5, dropout_rate = 0.0,
                       weight_decay = 1e-3, subsample_initial_block = False):
    
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1
    
    if nb_filter <= 0:
        nb_filter = 2 * growth_rate
        
    compression = 1 - reduction
    
    if subsample_initial_block:
        initial_kernel = (7, 7)
        initial_strides = (3, 3)
    else:
        initial_kernel = (3, 3)
        initial_strides = (1, 1)

    x = Conv2D(nb_filter, initial_kernel, kernel_initializer = 'he_normal', padding = 'same',
               strides = initial_strides, kernel_regularizer = l2(weight_decay))(img_input)

    if subsample_initial_block:
        x = BatchNormalization(axis = concat_axis, epsilon=1e-5)(x)
        x = LeakyReLU(0.3)(x)
        x = MaxPooling2D((3, 3), strides = (2, 2), padding = 'same')(x)
        
    for block_idx in range(nb_dense_block - 1):
        x , nb_filter = __dense_block(x, nb_filter, nb_layers_per_block, dropout_rate = dropout_rate,
                                      weight_decay = weight_decay, growth_rate = growth_rate)
        
        #add transition block
        x = __transition_block(x, nb_filter, compression = compression, weight_decay = weight_decay)
        nb_filter += int(nb_filter * compression)
        
    #
    x, nb_filter = __dense_block(x, nb_filter, nb_layers_per_block, dropout_rate = dropout_rate,
                                 weight_decay = weight_decay,  growth_rate = growth_rate)
    x = BatchNormalization(axis = concat_axis, epsilon = 1.1e-5)(x)
    x = LeakyReLU(0.3)(x)
    x = GlobalAveragePooling2D()(x)
    
    x = Dense(classes, activation = 'softmax')(x)

    return x



model = DenseNet(input_shape = (32, 32, 3), nb_dense_block = 4, growth_rate = 12, nb_filter = -1,
                 nb_layers_per_block = 4, reduction = 0.5, dropout_rate = 0.1, weight_decay = 1e-4,
                 classes = 10)

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

tensorboard = TensorBoard('/home/ujjwal/my_logs')
cnn = model.fit(x_train, y_train, batch_size = 32, epochs = 100, validation_data = (x_test, y_test), shuffle = True, callbacks = [tensorboard])