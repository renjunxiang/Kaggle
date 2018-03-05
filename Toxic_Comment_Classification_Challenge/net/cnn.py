from keras.models import Sequential
from keras.layers.core import Dense, initializers, Flatten, Dropout, Masking
from keras.layers import Conv1D, InputLayer
from keras.layers.recurrent import LSTM
from keras.layers.pooling import MaxPooling1D
from keras.optimizers import SGD, Adagrad, Adam

def cnn1d(input_shape):
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))
    model.add(Masking(mask_value=0))
    model.add(Conv1D(filters=16,  # 卷积核数量
                     kernel_size=5,  # 卷积核尺寸，或者[3]
                     strides=1,
                     padding='same',
                     kernel_initializer=initializers.normal(stddev=0.1),
                     bias_initializer=initializers.normal(stddev=0.1),
                     activation='relu'))
    model.add(MaxPooling1D(pool_size=2,
                           strides=2,
                           padding='valid'))
    model.add(Conv1D(filters=32,  # 卷积核数量
                     kernel_size=5,  # 卷积核尺寸，或者[3]
                     strides=1,
                     padding='same',
                     activation='relu',
                     kernel_initializer=initializers.normal(stddev=0.1),
                     bias_initializer=initializers.normal(stddev=0.1)))
    model.add(MaxPooling1D(pool_size=2,
                           strides=2,
                           padding='valid'))
    model.add(Flatten(name='Flatten'))
    model.add(Dense(units=64,
                    activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(units=2,
                    activation='softmax'))
    optimizer = Adagrad(lr=0.01)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
