import pandas as pd
import numpy as np
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, initializers, Dropout, Masking
from keras.layers import Conv1D, InputLayer
from keras.layers.recurrent import LSTM,GRU
from keras.optimizers import SGD, Adagrad, Adam
# from net.lstm import lstm
# from net.cnn1d import cnn1d
from sklearn.model_selection import train_test_split
from keras import backend as K


def multiply_loss(y_true, y_pred):
    return -K.mean(y_true * K.log(y_pred) + (1 - y_true) * K.log(1 - y_pred), axis=-1)

DIR = 'D:/github/Kaggle/Toxic_Comment_Classification_Challenge'


def gru(input_dim, input_length=100, output_dim=50, label_n=6):
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
    model.add(Masking(mask_value=0))
    model.add(GRU(units=32,
                   activation='tanh',
                   recurrent_activation='hard_sigmoid',
                   return_sequences=False))
    model.add(Dense(units=64,
                    activation='relu'))
    # model.add(Dropout(0.25))
    model.add(Dense(units=label_n,
                    activation='sigmoid'))
    optimizer = Adagrad(lr=0.01)
    model.compile(optimizer=optimizer, loss=multiply_loss, metrics=['accuracy'])
    return model

train_data = pd.read_csv('./data/train.csv')
train_label = np.array(train_data.iloc[:, 2:])
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts=train_data.iloc[:, 1])
train_data_seq = tokenizer.texts_to_sequences(texts=train_data.iloc[:, 1])
train_data_new = pad_sequences(train_data_seq, maxlen=100, padding='post', value=0, dtype='float32')
np.save('./data_seq/train_data_new.npy',train_data_seq)

test_data = pd.read_csv('./data/test.csv')
test_data_seq = tokenizer.texts_to_sequences(texts=test_data.iloc[:, 1])
test_data_new = pad_sequences(test_data_seq, maxlen=100, padding='post', value=0, dtype='float32')
np.save('./data_seq/test_data_new.npy',train_data_seq)

input_dim = len(tokenizer.word_index)
model_lstm=gru(input_dim=input_dim, input_length=100, output_dim=100, label_n=6)

samples=159571
batch_size=1000
epochs=4

train_train_x, train_test_x, train_train_y, train_test_y = train_test_split(train_data_new[0:samples],
                                                                            train_label[0:samples],
                                                                            test_size=0.3)
np.save('./data_seq/train_train_x.npy',train_train_x)
np.save('./data_seq/train_test_x.npy',train_test_x)
np.save('./data_seq/train_train_y.npy',train_train_y)
np.save('./data_seq/train_test_y.npy',train_test_y)

model_lstm.fit(x=train_train_x, y=train_train_y,
               validation_data=[train_test_x,train_test_y],
               batch_size=batch_size, epochs=epochs,verbose=1)
model_lstm.save('./models/gru_%d_%d_%d.h5' % (samples, batch_size, epochs))

test_label=model_lstm.predict(test_data_new)
test_predict=pd.DataFrame(test_label,columns=train_data.columns[2:])

test_result=pd.concat([test_data.iloc[:,0:1],test_predict],axis=1)
test_result.to_csv('./result/gru_%d_%d_%d.csv'%(samples,batch_size,epochs),index=False)
