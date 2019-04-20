import logging

from keras import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

import numpy as np

from utils import PATCH_SIZE


class Model:
    def __init__(self):
        self.model = self._build_model()

    def load_weights(self, filename):
        logging.info('Load weights: {}'.format(filename))
        self.model.load_weights(filename)

    def fit(self, X, y):
        filepath = "weights/weights-{epoch:02d}-{val_acc:.4f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        self.model.fit(X, y, validation_split=0.1, epochs=500, batch_size=10, callbacks=callbacks_list, verbose=1)

    def compile(self):
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        logging.info('Compiled keras model')

    def evaluate(self, X, y):
        result = self.model.evaluate(X, y)
        logging.info('Model evaluated, loss: {} acc: {}'.format(result[0], result[1]))
        return result

    def predict(self, X):
        input = X
        if len(input.shape) == 2:
            input = input.reshape(1, PATCH_SIZE, PATCH_SIZE, 1)
            logging.debug('Reshaped before prediction')
        result = self.model.predict(input)
        logging.info('Predicted: {}, return {}'.format(result, np.argmax(result)))
        logging.debug('For: {}'.format(input))
        return result

    @staticmethod
    def _build_model():
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=(PATCH_SIZE, PATCH_SIZE, 1)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='softmax'))

        logging.info('Created keras model')
        return model