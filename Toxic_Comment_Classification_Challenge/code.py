import pandas as pd
import numpy as np
import os
from keras.preprocessing.sequence import pad_sequences
from function.text2vec import Text_transform

DIR='D:/github/Kaggle/Toxic_Comment_Classification_Challenge'

train_data = pd.read_csv('./data/train.csv')

text_transform = Text_transform(texts=list(train_data.iloc[:, 1]))
text_transform.creat_vocab_word2vec(sg=0,
                                    size=5,
                                    window=5,
                                    min_count=1,
                                    vocab_savepath=DIR + '/models/vocab_word2vec.model')
text_transform.text2vec()
train_data_seq = text_transform.text_vec
train_data_new = pad_sequences(train_data_seq, maxlen=100, padding='post', value=0, dtype='float32')
####################################################################################################################3
test_data = pd.read_csv('./data/test.csv')
text_transform = Text_transform(texts=list(test_data.iloc[:, 1]))
text_transform.load_vocab_word2vec(vocab_loadpath=DIR + '/models/vocab_word2vec.model')
text_transform.text2vec()
test_data_seq = text_transform.text_vec
test_data_new = pad_sequences(test_data_seq, maxlen=100, padding='post', value=0, dtype='float32')