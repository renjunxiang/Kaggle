import pandas as pd
import numpy as np
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, initializers, Dropout, Masking,Flatten
from keras.optimizers import SGD, Adagrad, Adam
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.layers import CuDNNGRU
from keras.layers import Conv1D
from keras.layers.pooling import MaxPooling1D

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

K.clear_session()

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
set_session(tf.Session(config=config))


# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""


def multiply_loss(y_true, y_pred):
    return -K.mean(y_true * K.log(y_pred) + (1 - y_true) * K.log(1 - y_pred), axis=-1)


DIR = 'D:/github/Kaggle/Toxic_Comment_Classification_Challenge'


def CuDNNGRU_drop(input_dim=100, input_length=100, output_dim=200, label_n=6):
    '''

    :param input_dim: 字典长度，即onehot的长度
    :param input_length: 文本长度
    :param output_dim: 词向量长度
    :return: 
    '''
    # input_dim = 100
    # input_length = 100
    # output_dim = 200
    # label_n = 6
    model = Sequential()
    model.add(Embedding(input_dim=input_dim + 1,
                        input_length=input_length,
                        output_dim=output_dim,
                        mask_zero=0))
    model.add(CuDNNGRU(units=32,
                       return_sequences=True))
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
    model.add(Flatten())
    model.add(Dense(units=128,
                    activation='tanh'))
    model.add(Dropout(0.20))
    model.add(Dense(units=label_n,
                    activation='sigmoid'))
    optimizer = Adagrad(lr=0.001)
    model.compile(optimizer=optimizer, loss=multiply_loss, metrics=['accuracy'])
    return model


train_data = pd.read_csv('./data/train.csv')
train_label = np.array(train_data.iloc[:, 2:])
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts=train_data.iloc[:, 1])
train_data_seq = tokenizer.texts_to_sequences(texts=train_data.iloc[:, 1])
train_data_new = pad_sequences(train_data_seq, maxlen=200, padding='post', value=0, dtype='float32')
# np.save('./data_seq/train_data_new.npy',train_data_seq)

test_data = pd.read_csv('./data/test.csv')
test_data_seq = tokenizer.texts_to_sequences(texts=test_data.iloc[:, 1])
test_data_new = pad_sequences(test_data_seq, maxlen=200, padding='post', value=0, dtype='float32')
# np.save('./data_seq/test_data_new.npy',train_data_seq)

input_dim = len(tokenizer.word_index)
model_CuDNNGRU_drop = CuDNNGRU_drop(input_dim=input_dim, input_length=200, output_dim=200, label_n=6)

samples = 159571
batch_size = 200
epochs = 1

train_train_x, train_test_x, train_train_y, train_test_y = train_test_split(train_data_new[0:samples],
                                                                            train_label[0:samples],
                                                                            test_size=0.2)
np.save('./data_seq/train_train_x.npy', train_train_x)
np.save('./data_seq/train_test_x.npy', train_test_x)
np.save('./data_seq/train_train_y.npy', train_train_y)
np.save('./data_seq/train_test_y.npy', train_test_y)

optimizer = Adagrad(lr=0.001)
model_CuDNNGRU_drop.compile(optimizer=optimizer, loss=multiply_loss, metrics=['accuracy'])

model_CuDNNGRU_drop.fit(x=train_train_x, y=train_train_y,
                        validation_data=[train_test_x, train_test_y],
                        batch_size=batch_size, epochs=epochs, verbose=1)

model_CuDNNGRU_drop.save('./models/CuDNNGRU_drop_200_200_%d_%d_%d.h5' % (samples, batch_size, 3))

test_label = model_CuDNNGRU_drop.predict(test_data_new)
test_predict = pd.DataFrame(y_test, columns=train_data.columns[2:])

test_result = pd.concat([test_data.iloc[:, 0:1], test_predict], axis=1)
test_result.to_csv('./result/CuDNNGRU_drop_100_128_%d_%d_%d.csv' % (samples, batch_size, 3), index=False)
