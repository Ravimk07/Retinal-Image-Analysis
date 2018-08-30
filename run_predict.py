from random import shuffle
import glob
import cv2
import numpy as np
import h5py
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import SGD, Adam, RMSprop
import scipy.io as sci

path = 'C:\\Users\\admin\PycharmProjects\OCT_update\Experiment_InceptionV3\checkpoint\InceptionV3_vol_30.hdf5'
dme_path = 'C:\\Users\\admin\PycharmProjects\OCT_update\HK_Cropped_BM3D\dme\*.png'
normal_path = 'C:\\Users\\admin\PycharmProjects\OCT_update\HK_Cropped_BM3D\\normal\*.png'
nb_classes = 2
img_width, img_height = 224, 224
data_order = 'tf'  # 'th' for Theano, 'tf' for Tensorflow
one_hot_encoding = True

base_model = InceptionV3(include_top=False, weights=None)
model_name='InceptionV3'

# read addresses and labels from the folder
addrs_dme = glob.glob(dme_path)
labels_dme = [1 for addr in addrs_dme]  # 0 = Normal, 1 = DME
addrs_normal = glob.glob(normal_path)
labels_normal = [0 for addr in addrs_normal]  # 0 = Normal, 1 = DME
addrs = addrs_dme+addrs_normal
labels = labels_dme+labels_normal

# one hot encoding
if one_hot_encoding:
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(labels)

    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)


# check the order of data and chose proper data shape to save images
if data_order == 'th':
    shape = (len(addrs), 3, 224, 224)
elif data_order == 'tf':
    shape = (len(addrs), 224, 224, 3)

print('shape', shape)

# loop over addresses
X_test = np.empty(shape=(len(addrs), img_height, img_width, 3))
for i in range(len(addrs)):
    # print how many images are saved every 1000 images

    if i % 1000 == 0 and i > 1:
        print('Data: {}/{}'.format(i, len(addrs)))

    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    addr = addrs[i]
    img = cv2.imread(addr)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    X_test[i] = img

print('X_test.shape: ', X_test.shape)

def add_new_last_layer(base_model, nb_classes):
    print('Add last layer to the convnet..')
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(nb_classes, activation='relu')(x)  # new FC layer, random init
    predictions = Dense(nb_classes, activation='softmax')(x)  # new softmax layer
    model = Model(inputs=base_model.input, outputs=predictions)  # Combine the network

    return model


# Add the new last layer to the model
model = add_new_last_layer(base_model, nb_classes)

learning_rate = 0.0001
decay_rate = learning_rate / 100
momentum = 0.8
# SGD = optimizers.SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
SGD = SGD(lr=learning_rate)

""""""
model.compile(optimizer=RMSprop(lr=learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.load_weights(path)

predict = model.predict(X_test)

dd=np.argmax(predict,axis=1)

print('y_true:', dd)

sci.savemat('C:\\Users\\admin\PycharmProjects\OCT_update\data_'+ model_name +'.mat', mdict={'y_pred':dd})
