import pandas as pd
import numpy as np
import os
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from gensim.models.word2vec import Word2Vec


train_data = pd.read_csv('./data/train.csv')
train_data_seq=[text_to_word_sequence(i) for i in train_data]
model=Word2Vec(train_data_seq,sg=1, size=100, window=5, min_count=1)
model.save('./models/Word2Vec.model')


test_data = pd.read_csv('./data/test.csv')
test_data_seq=[text_to_word_sequence(i) for i in test_data]

data = pad_sequences(data, maxlen=maxlen, padding='post', value=0, dtype='float32')



jieba.lcut(train_data.iloc[0,1])

train_data.iloc[0,1].split(' ')


text_to_word_sequence(train_data.iloc[0,1])