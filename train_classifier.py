from random import seed, random
from shutil import copyfile
import os
import sys

# from keras.layers import BatchNormalization, Dropout
from matplotlib import pyplot
from matplotlib.image import imread
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import layers
from datetime import datetime
import numpy as np
import cv2
import tensorflow as tf

# import tuning as tuning

dataset_home = 'db/'

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

HAND_GESTURES = ['right', 'play', 'left', 'close', 'pointer', 'open', 'volume_up', 'volume_down']
HAND_GESTURES = sorted(HAND_GESTURES)

INPUT_SHAPE = (200, 200, 3)
EPOCHS = 10
NUM_OF_CLASSES = len(HAND_GESTURES)


def prepare_dataset():
    seed(1)
    # define ratio of pictures to use for validation
    val_ratio = 0.25
    # copy training dataset images into subdirectories

    for hand_gesture in HAND_GESTURES:
        os.makedirs(dataset_home + 'train/' + hand_gesture)
        os.makedirs(dataset_home + 'test/' + hand_gesture)
        for file in os.listdir(dataset_home + hand_gesture):
            dst_dir = 'train/'
            if random() < val_ratio:
                dst_dir = 'test/'
            src = dataset_home + hand_gesture + '/' + file
            dst = dataset_home + dst_dir + hand_gesture + '/' + file
            copyfile(src, dst)


def define_one_block_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                     input_shape=INPUT_SHAPE))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model


def define_three_block_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                     input_shape=INPUT_SHAPE))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    # model.add(Dense(1, activation='sigmoid'))
    model.add(Dense(NUM_OF_CLASSES, activation='softmax'))
    # compile model
    opt = SGD(lr=0.001, momentum=0.9)
    # model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
    return model


def define_conv_4_layer_model():
    cnn4 = Sequential()
    cnn4.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=INPUT_SHAPE))
    cnn4.add(BatchNormalization())

    cnn4.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    cnn4.add(BatchNormalization())
    cnn4.add(MaxPooling2D(pool_size=(2, 2)))
    cnn4.add(Dropout(0.25))

    cnn4.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    cnn4.add(BatchNormalization())
    cnn4.add(Dropout(0.25))

    cnn4.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    cnn4.add(BatchNormalization())
    cnn4.add(MaxPooling2D(pool_size=(2, 2)))
    cnn4.add(Dropout(0.25))

    cnn4.add(Flatten())

    cnn4.add(Dense(512, activation='relu'))
    cnn4.add(BatchNormalization())
    cnn4.add(Dropout(0.5))

    cnn4.add(Dense(128, activation='relu'))
    cnn4.add(BatchNormalization())
    cnn4.add(Dropout(0.5))

    cnn4.add(Dense(NUM_OF_CLASSES, activation='softmax'))

    cnn4.compile(loss=keras.losses.categorical_crossentropy,
                 optimizer=keras.optimizers.Adam(),
                 metrics=['accuracy'])
    return cnn4


# plot diagnostic learning curves
def summarize_diagnostics(history):
    # plot loss
    pyplot.subplot(211)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.plot(history.history['val_loss'], color='orange', label='test')
    # plot accuracy
    pyplot.subplot(212)
    pyplot.title('Classification Accuracy')
    pyplot.plot(history.history['accuracy'], color='blue', label='train')
    pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
    # save plot to file
    filename = sys.argv[0].split('/')[-1]
    pyplot.savefig(filename + '_plot.png')
    pyplot.close()


# run the test harness for evaluating a model
def caclulate_model():
    # define model
    # model = define_one_block_model()
    model = define_three_block_model()
    # model = define_conv_4_layer_model()

    checkpoint_path = 'checkpoints/checkpoin-{epoch:02d}-{val_accuracy:.2f}.hdf5'
    keras_callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, mode='min', min_delta=0.0001),
        # ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=False, mode='min')
    ]

    # create data generator
    train_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    # width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    # prepare iterators
    train_it = train_datagen.flow_from_directory(dataset_home + 'train/',
                                                 class_mode='categorical', batch_size=16,  # categorical / binary
                                                 target_size=(INPUT_SHAPE[0], INPUT_SHAPE[1]))
    test_it = test_datagen.flow_from_directory(dataset_home + 'test/',
                                               class_mode='categorical', batch_size=16,  # categorical / binary
                                               target_size=(INPUT_SHAPE[0], INPUT_SHAPE[1]))
    # fit model
    history = model.fit_generator(train_it, steps_per_epoch=len(train_it),
                                  validation_data=test_it, validation_steps=len(test_it), epochs=EPOCHS,
                                  use_multiprocessing=True, callbacks=keras_callbacks)
    model.summary()
    model.save("latest_model.h5")
    model.save("trained_models/model " + str(datetime.now()) + ".h5")

    # evaluate model
    _, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=0)
    print('> %.3f' % (acc * 100.0))
    # learning curves
    summarize_diagnostics(history)

    return model, history


def test_one_example(model: tf.keras.Model, img_path: str):
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (INPUT_SHAPE[0], INPUT_SHAPE[1]))
    img_expanded = np.expand_dims(img_resized, axis=0)
    print(model.predict(img_expanded))


def run_tests(model: tf.keras.Model):
    print('Train data tests:')
    print('Right: ', end='')
    test_one_example(model, 'db/test/right/2.jpg')
    test_one_example(model, 'db/test/right/8.jpg')
    print('Left: ', end='')
    test_one_example(model, 'db/test/left/3.jpg')
    test_one_example(model, 'db/test/left/11.jpg')
    print('Play: ', end='')
    test_one_example(model, 'db/test/play/7.jpg')
    test_one_example(model, 'db/test/play/10.jpg')
    print('Close: ', end='')
    test_one_example(model, 'db/test/close/8.jpg')
    test_one_example(model, 'db/test/close/23.jpg')
    print('Pointer: ', end='')
    test_one_example(model, 'db/test/pointer/3.jpg')
    test_one_example(model, 'db/test/pointer/7.jpg')
    print('Open: ', end='')
    test_one_example(model, 'db/test/open/1.jpg')
    test_one_example(model, 'db/test/open/3.jpg')
    print('Volume up: ', end='')
    test_one_example(model, 'db/test/volume_up/4.jpg')
    test_one_example(model, 'db/test/volume_up/5.jpg')
    print('Volume down: ', end='')
    test_one_example(model, 'db/test/volume_down/2.jpg')
    test_one_example(model, 'db/test/volume_down/9.jpg')


def convert_prediction_array_to_gesture_name(prediction: np.ndarray):
    return HAND_GESTURES[np.where(prediction[0] == 1)[0][0]]


def predict_hand_capture_gesture(model: tf.keras.Model, image: np.ndarray):
    img_resized = cv2.resize(image, (INPUT_SHAPE[0], INPUT_SHAPE[1]))
    img_expanded = np.expand_dims(img_resized, axis=0)
    return convert_prediction_array_to_gesture_name(model.predict(img_expanded))


# prepare_dataset()
# model, history = caclulate_model()
# run_tests(model)
# model.save_weights('trained_classifier_0')

model = define_three_block_model()
model.load_weights('trained_classifier_0/trained_classifier_0')
