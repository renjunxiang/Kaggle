from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from gensim.models.word2vec import Word2Vec


def text2vec(text_seq=None,
             sg=0,
             vocab_savepath='./models/Word2Vec.model',
             size=5,
             window=5,
             min_count=1):
    model = Word2Vec(text_seq, sg=sg, size=size, window=window, min_count=min_count)
    model.save(vocab_savepath)

