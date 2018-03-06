import pandas as pd
import numpy as np
import os
from keras.preprocessing.text import text_to_word_sequence,one_hot
from keras.preprocessing.sequence import pad_sequences


def text2seq(texts):
    text_seq = [text_to_word_sequence(i) for i in texts]
    return text_seq

DIR = 'D:/github/Kaggle/Toxic_Comment_Classification_Challenge'

train_data = pd.read_csv('./data/train.csv')
train_data_seq = text2seq.text_vec

####################################################################################################################3
test_data = pd.read_csv('./data/test.csv')
text_transform = Text_transform(texts=list(test_data.iloc[:, 1]))
text_transform.load_vocab_word2vec(vocab_loadpath=DIR + '/models/vocab_word2vec.model')
text_transform.text2vec()
test_data_seq = text_transform.text_vec
test_data_new = pad_sequences(test_data_seq, maxlen=100, padding='post', value=0, dtype='float32')
