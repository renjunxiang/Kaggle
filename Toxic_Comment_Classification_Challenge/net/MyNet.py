from keras.models import Sequential
from keras.layers.core import Dense, initializers, Flatten, Dropout, Masking
from keras.layers import Conv1D, InputLayer
from keras.layers.recurrent import LSTM
from keras.layers.pooling import MaxPooling1D

def MyNet(input_shape):

    model = Sequential()
    model.add(InputLayer())
    model.add(Reshape(input_shape=[784],
                      target_shape=[28, 28, 1],
                      name='Reshape_2d'))
    model.add(Conv2D(filters=16,
                     kernel_size=[3, 3],
                     padding='same',
                     activation='relu',
                     data_format='channels_last'))
    model.add(MaxPooling2D(pool_size=[2, 2],
                           strides=2,
                           padding='valid'))
    model.add(Conv2D(filters=64,
                     kernel_size=[3, 3],
                     padding='same',
                     activation='relu',
                     data_format='channels_last'))
    model.add(MaxPooling2D(pool_size=[2, 2],
                           strides=2,
                           padding='valid'))
    model.add(Flatten(name='Flatten'))
    model.add(Dense(units=64,
                    activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(units=10,
                    activation='softmax'))
    optimizer = Adagrad(lr=0.01)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model



