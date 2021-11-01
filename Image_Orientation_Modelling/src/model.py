import os
from keras import layers
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, GlobalMaxPooling2D
from keras import applications
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

def build_model_vgg16(image_size, input_shape, epochs, log_dir, model_path):

    model_name = "VGG16"

    pre_trained_model = VGG16(input_shape=input_shape, include_top=False, weights="imagenet")

    for layer in pre_trained_model.layers[:15]:
        layer.trainable = False

    for layer in pre_trained_model.layers[15:]:
        layer.trainable = True

    last_layer = pre_trained_model.get_layer('block5_pool')
    last_output = last_layer.output

    # Flatten the output layer to 1 dimension
    x = GlobalMaxPooling2D()(last_output)
    # Add a fully connected layer with 512 hidden units and ReLU activation
    x = Dense(512, activation='relu')(x)
    # Add a dropout rate of 0.5
    x = Dropout(0.5)(x)
    # Add a final sigmoid layer for classification
    x = layers.Dense(4, activation='softmax')(x)

    model = Model(pre_trained_model.input, x)

    # Stochastic gradient decent optimizer
    optimizer = 'sgd'

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    # Tensorboard callback
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True)

    # simple early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)
    mc = ModelCheckpoint(os.path.join(model_path, '{}_best_model.h5'.format(model_name)),
                         monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

    model.summary()

    return model