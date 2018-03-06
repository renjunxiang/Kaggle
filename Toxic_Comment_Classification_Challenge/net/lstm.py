from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, initializers, Dropout, Masking
from keras.layers import Conv1D, InputLayer
from keras.layers.recurrent import LSTM
from keras.optimizers import SGD, Adagrad, Adam


def lstm(input_shape):
    model = Sequential()
    model.add(InputLayer(input_shape=[20]))
    model.add(Embedding(output_dim=[20,50],mask_zero=0))
    model.add(Masking(mask_value=0))
    model.add(LSTM(units=16,
                   activation='tanh',
                   recurrent_activation='hard_sigmoid',
                   implementation=1,
                   dropout=0,
                   recurrent_dropout=0))
    model.add(Dense(units=64,
                    activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(units=2,
                    activation='softmax'))
    optimizer = Adagrad(lr=0.01)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
