import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, initializers, Flatten, Dropout, Reshape
from keras.layers import Conv2D, InputLayer
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import SGD, Adagrad, Adam
from keras import regularizers


def LeNet():
    model = Sequential()
    model.add(Reshape(input_shape=[784],
                      target_shape=[28, 28, 1],
                      name='Reshape_2d'))
    model.add(Conv2D(filters=20,
                     kernel_size=[5, 5],
                     padding='same',
                     activation='relu',
                     data_format='channels_last'))
    model.add(MaxPooling2D(pool_size=[2, 2],
                           strides=2,
                           padding='valid'))
    model.add(Conv2D(filters=50,
                     kernel_size=[5, 5],
                     padding='same',
                     activation='relu',
                     data_format='channels_last'))
    model.add(MaxPooling2D(pool_size=[2, 2],
                           strides=2,
                           padding='valid'))
    model.add(Flatten(name='Flatten'))
    model.add(Dense(units=500,
                    activation='relu'))
    model.add(Dense(units=10,
                    activation='softmax'))
    optimizer = Adagrad(lr=0.01)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model
