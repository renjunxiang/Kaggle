import pandas as pd
import numpy as np
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from net.lstm import lstm
from net.cnn1d import cnn1d

DIR = 'D:/github/Kaggle/Toxic_Comment_Classification_Challenge'

train_data = pd.read_csv('./data/train.csv')
train_label = np.array(train_data.iloc[:, 2:])
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts=train_data.iloc[:, 1])
train_data_seq = tokenizer.texts_to_sequences(texts=train_data.iloc[:, 1])
np.save('./data_seq/train_data_seq.npy',train_data_seq)
train_data_new = pad_sequences(train_data_seq, maxlen=100, padding='post', value=0, dtype='float32')

test_data = pd.read_csv('./data/test.csv')
test_data_seq = tokenizer.texts_to_sequences(texts=test_data.iloc[:, 1])
np.save('./data_seq/test_data_seq.npy',train_data_seq)
test_data_new = pad_sequences(test_data_seq, maxlen=100, padding='post', value=0, dtype='float32')

input_dim = len(tokenizer.word_index)
model_lstm=lstm(input_dim=input_dim, input_length=100, output_dim=50, label_n=6)
model_cnn1d=cnn1d(input_dim=input_dim, input_length=100, output_dim=50,label_n=6)

samples=159571
batch_size=1000
epochs=15

model_lstm.fit(x=train_data_new[0:samples], y=train_label[0:samples], validation_split=0.3,
               batch_size=batch_size, epochs=epochs)
model_lstm.save('./models/lstm_%d_%d_%d.h5' % (samples, batch_size, epochs))

test_label=model_lstm.predict(test_data_new)
test_predict=pd.DataFrame(test_label,columns=train_data.columns[2:])

test_result=pd.concat([test_data.iloc[:,0:1],test_predict],axis=1)
test_result.to_csv('./result/lstm_%d_%d_%d.csv'%(samples,batch_size,epochs),index=False)