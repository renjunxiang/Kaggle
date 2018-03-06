from keras.preprocessing.text import text_to_word_sequence
from gensim.models.word2vec import Word2Vec

DIR = 'D:/github/Kaggle/Toxic_Comment_Classification_Challenge'

class Text_transform():
    def __init__(self, texts):
        self.texts = texts
        text_seq = [text_to_word_sequence(i) for i in texts]
        self.text_seq = text_seq

    def text_one_hot(self):
        text_seq=self.text_seq

