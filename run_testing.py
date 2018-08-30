from keras.deeplearningmodels.inception_v3 import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import SGD, Adam, RMSprop
from keras.preprocessing.image import ImageDataGenerator

path = 'C:\\Users\\admin\PycharmProjects\OCT_update\Experiment_InceptionV3\checkpoint\InceptionV3_vol_13.hdf5'
validation_data_dir = 'C:\\Users\\admin\PycharmProjects\OCT_update\HK_Cropped_BM3D'
img_width, img_height = 224, 224

base_model = InceptionV3(include_top=False, weights=None)

def add_new_last_layer(base_model, nb_classes):
    print('Add last layer to the convnet..')
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(nb_classes, activation='relu')(x)  # new FC layer, random init
    predictions = Dense(nb_classes, activation='softmax')(x)  # new softmax layer
    model = Model(inputs=base_model.input, outputs=predictions)  # Combine the network

    return model


# Add the new last layer to the model
nb_classes = 2

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


test_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        color_mode= "rgb",
        target_size=(img_width, img_height),
        batch_size=128,
        class_mode='categorical')

scoreSeg = model.evaluate_generator(validation_generator, 400)
#predict = model.predict_generator(validation_generator, 400)
print("Accuracy = ", scoreSeg[1])
