from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, initializers, Flatten, Dropout, Masking
from keras.layers import Conv1D, InputLayer
from keras.layers.recurrent import LSTM
from keras.layers.pooling import MaxPooling1D
from keras.optimizers import SGD, Adagrad, Adam


def cnn1d(input_dim, input_length=100, output_dim=50, label_n=6,
          loss='categorical_crossentropy'):
    '''

    :param input_dim: 字典长度，即onehot的长度
    :param input_length: 文本长度
    :param output_dim: 词向量长度
    :return: 
    '''
    model = Sequential()
    model.add(Embedding(input_dim=input_dim + 1,
                        input_length=input_length,
                        output_dim=output_dim,
                        mask_zero=0))
    model.add(Conv1D(filters=64,  # 卷积核数量
                     kernel_size=5,  # 卷积核尺寸，或者[3]
                     strides=1,
                     padding='same',
                     kernel_initializer=initializers.normal(stddev=0.1),
                     bias_initializer=initializers.normal(stddev=0.1),
                     activation='relu'))
    model.add(MaxPooling1D(pool_size=2,
                           strides=2,
                           padding='valid'))
    model.add(Conv1D(filters=64,  # 卷积核数量
                     kernel_size=5,  # 卷积核尺寸，或者[3]
                     strides=1,
                     padding='same',
                     kernel_initializer=initializers.normal(stddev=0.1),
                     bias_initializer=initializers.normal(stddev=0.1),
                     activation='relu'))
    model.add(MaxPooling1D(pool_size=2,
                           strides=2,
                           padding='valid'))
    model.add(Flatten(name='Flatten'))
    model.add(Dense(units=64,
                    activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(units=label_n,
                    activation='relu'))
    optimizer = Adagrad(lr=0.01)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model
