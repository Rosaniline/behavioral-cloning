import os

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
from keras.layers import GlobalAveragePooling2D
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import Sequence
from scipy import misc
from tensorflow.contrib.learn.python.learn.estimators._sklearn import train_test_split

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_integer('features_epochs', 1, 'The number of epochs when training features.')
flags.DEFINE_integer('full_epochs', 100, 'The number of epochs when end-to-end training.')
flags.DEFINE_integer('batch_size', 128, 'The batch size.')
flags.DEFINE_integer('samples_per_epoch', 12800, 'The number of samples per epoch.')
flags.DEFINE_integer('img_h', 60, 'The image height.')
flags.DEFINE_integer('img_w', 200, 'The image width.')
flags.DEFINE_integer('img_c', 3, 'The number of channels.')

flags.DEFINE_string('img_dir', 'log/IMG/', 'image directory')
flags.DEFINE_string('driving_log', 'log/driving_log.csv', 'driving logs')
flags.DEFINE_boolean('preload', False, 'preload training data')


def img_pre_processing(img, old=False):
    img = cv2.resize(img, (200, 100)).astype('float')
    img = img[40:]

    # normalize
    img /= 255.
    img -= 0.5
    img *= 2.
    return img


def read_img(path):
    return cv2.imread(os.path.join(FLAGS.img_dir, path))


def load_img(img_list):
    imgs = np.asarray(list(map(img_pre_processing, map(read_img, img_list))))

    return imgs


def save_model(model):
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights('model.h5')


class DataGenerator(Sequence):
    def __init__(self, x_set, y_set, shuffle=True):
        self.x = x_set
        self.y = y_set

        self.indexes = np.arange(len(self.x))
        self.batch_size = FLAGS.batch_size
        self.shuffle = shuffle

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_index = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_x = [self.x[:, 0][k] for k in batch_index]
        batch_y = [self.y[:, 0][k] for k in batch_index]

        if FLAGS.preload:
            return np.array(batch_x), np.array(batch_y)
        else:
            return load_img(batch_x), np.array(batch_y)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)


def read_training_data():
    with open(FLAGS.driving_log, 'rb') as f:
        log_data = pd.read_csv(f, header=None, names=['img', 'angle', 'throttle', 'break', 'speed', 'time', 'lap', 'sign'])

    print("Got", len(log_data), "samples for training")

    if FLAGS.preload:
        print("preloading image files...")
        log_data['img'] = list(map(img_pre_processing, map(read_img, log_data['img'])))

    X = log_data[['img', 'speed']].values
    y = log_data[['angle', 'throttle']].values

    return X, y


def declare_model():
    # create and train the model
    input_shape = (FLAGS.img_h, FLAGS.img_w, FLAGS.img_c)
    input_tensor = Input(shape=input_shape)

    # get the VGG16 network
    base_model = VGG16(input_tensor=input_tensor,
                       weights='imagenet',
                       include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # add the fully-connected
    # layer similar to the NVIDIA paper
    x = Dense(512, activation='elu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='elu')(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='elu')(x)
    x = Dropout(0.1)(x)
    predictions = Dense(1, init='zero')(x)

    # create the full model
    model = Model(input=base_model.input, output=predictions)

    # freeze all convolutional layers to initialize the top layers
    for layer in base_model.layers:
        layer.trainable = False

    return model


def main(_):
    # fix random seed for reproducibility
    np.random.seed(123)

    X, y = read_training_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    if FLAGS.preload:
        # preload X_test images
        X_test = np.stack(X_test[:, 0], axis=0)

    else:
        X_test = load_img(X_test[:, 0])

    y_test = y_test[:, 0]

    model = declare_model()

    model.compile(optimizer='adam', loss='mse')

    training_generator = DataGenerator(X_train, y_train)

    model.fit_generator(
        training_generator,
        nb_epoch=FLAGS.features_epochs,
        verbose=1
    )

    # for VGG we choose to include the
    # top 2 blocks in training
    for layer in model.layers[:11]:
        layer.trainable = False
    for layer in model.layers[11:]:
        layer.trainable = True

    # recompile and train with a finer learning rate
    opt = Adam(lr=1e-03, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.7)
    model.compile(optimizer=opt, loss='mse')
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=1,
                                   min_delta=0.00009)

    training_generator_2 = DataGenerator(X_train, y_train)

    print('Train top 2 conv blocks and fully-connected layers:')
    model.fit_generator(
        training_generator_2,
        validation_data=(X_test, y_test),
        nb_epoch=FLAGS.full_epochs,
        callbacks=[early_stopping],
        verbose=1
    )

    # save model to disk
    save_model(model)
    print('model saved')


# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
