from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, initializers, Dropout, Masking
from keras.layers import Conv1D, InputLayer
from keras.layers.recurrent import LSTM
from keras.optimizers import SGD, Adagrad, Adam


def lstm(input_dim, input_length=100, output_dim=50, label_n=6):
    '''
    
    :param input_dim: 字典长度，即onehot的长度
    :param input_length: 文本长度
    :param output_dim: 词向量长度
    :return: 
    '''
    model = Sequential()
    model.add(Embedding(input_dim=input_dim+1,
                        input_length=input_length,
                        output_dim=output_dim,
                        mask_zero=0))
    model.add(Masking(mask_value=0))
    model.add(LSTM(units=16,
                   activation='tanh',
                   recurrent_activation='hard_sigmoid',
                   return_sequences=False))
    model.add(Dense(units=64,
                    activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(units=label_n,
                    activation='relu'))
    optimizer = Adagrad(lr=0.01)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model
