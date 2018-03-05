from keras.preprocessing.text import text_to_word_sequence
from gensim.models.word2vec import Word2Vec

DIR = 'D:/github/Kaggle/Toxic_Comment_Classification_Challenge'

class Text_transform():
    def __init__(self, texts):
        self.texts = texts
        text_seq = [text_to_word_sequence(i) for i in texts]
        self.text_seq = text_seq

    def creat_vocab_word2vec(self,
                             sg=0,
                             size=5,
                             window=5,
                             min_count=1,
                             vocab_savepath=None):
        text_seq = self.text_seq
        vocab_word2vec = Word2Vec(text_seq, sg=sg, size=size, window=window, min_count=min_count)
        self.vocab_word2vec = vocab_word2vec
        if vocab_savepath is not None:
            vocab_word2vec.save(vocab_savepath)

    def load_vocab_word2vec(self,
                            vocab_loadpath=DIR + '/models/vocab_word2vec.model'):
        '''
        load dictionary
        :param vocab_loadpath: path to load word2vec dictionary
        :return: 
        '''
        self.vocab_word2vec = Word2Vec.load(vocab_loadpath)

    def text2vec(self):
        text_seq = self.text_seq
        vocab_word2vec = self.vocab_word2vec
        text_vec = [[vocab_word2vec[word] for word in one_text if word in vocab_word2vec] for one_text in text_seq]
        self.text_vec = text_vec
