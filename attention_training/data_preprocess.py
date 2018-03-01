from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import text_to_word_sequence
from keras.models import Sequential
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, recurrent, Embedding
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam, RMSprop
from nltk import FreqDist
import numpy as np
import os
import datetime

def load_data(source, dst, max_len, vocab_size, classes):
    # Reading raw text from source and destination files
    f = open(source, 'r')
    X_data = f.read()
    f.close()
    f = open(dst, 'r')
    y_data = f.read()
    f.close()

    # Splitting raw text into array of sequences
    # X = [text_to_word_sequence(x)[::-1] for x, y in zip(X_data.split('\n'), y_data.split('\n')) if len(x) > 0 and len(y) > 0 and len(x) <= max_len and len(y) <= max_len]
    # y = [text_to_word_sequence(y) for x, y in zip(X_data.split('\n'), y_data.split('\n')) if len(x) > 0 and len(y) > 0 and len(x) <= max_len and len(y) <= max_len]

    X = [text_to_word_sequence(x) for x in X_data.split('\n')]
    y = [text_to_word_sequence(y) for y in y_data.split('\n')]

    # Creating the vocabulary set with the most common words
    dist = FreqDist(np.hstack(X))
    X_vocab = dist.most_common(vocab_size)
    dist = FreqDist(np.hstack(y))
    y_vocab = dist.most_common(classes)

    # Creating an array of words from the vocabulary set, we will use this array as index-to-word dictionary
    X_ix_to_word = [word[0] for word in X_vocab]    # (lc)
    # Adding the word "ZERO" to the beginning of the array
    X_ix_to_word.insert(0, 'ZERO')
    # Adding the word 'UNK' to the end of the array (stands for UNKNOWN words)
    X_ix_to_word.append('UNK')

    # Creating the word-to-index dictionary from the array created above
    X_word_to_ix = {word:ix for ix, word in enumerate(X_ix_to_word)}

    # Converting each word to its index value
    for i, sentence in enumerate(X):
        for j, word in enumerate(sentence):
            if word in X_word_to_ix:
                X[i][j] = X_word_to_ix[word]
            else:
                X[i][j] = X_word_to_ix['UNK']

    y_ix_to_word = [word[0] for word in y_vocab]    # (lc)
    y_ix_to_word.insert(0, 'ZERO')
    y_ix_to_word.append('UNK')

    y_word_to_ix = {word:ix for ix, word in enumerate(y_ix_to_word)}

    for i, sentence in enumerate(y):
        for j, word in enumerate(sentence):
            if word in y_word_to_ix:
                y[i][j] = y_word_to_ix[word]
            else:
                y[i][j] = y_word_to_ix['UNK']
    return (X, len(X_vocab)+2, X_word_to_ix, X_ix_to_word, y, len(y_vocab)+2, y_word_to_ix, y_ix_to_word)

# src = '/home/shenzhen/PycharmProjects/Attention_RNN_Feb6/music_data/train_words'
# dst = '/home/shenzhen/PycharmProjects/Attention_RNN_Feb6/music_data/train_label'
# max_len= 25
# vocab_size = 5000
# X, X_vocab_len, X_word_to_ix, X_ix_to_word, y, y_vocab_len, y_word_to_ix, y_ix_to_word = load_data(src, dst, max_len, vocab_size)
# print("Processed Data")
#
# X_max_len = max([len(sentence) for sentence in X])
# y_max_len = max([len(sentence) for sentence in y])
#
# # Padding zeros to make all sequences have a same length with the longest one
# print('[INFO] Zero padding...')
# X = pad_sequences(X, maxlen=X_max_len, dtype='int32')
# y = pad_sequences(y, maxlen=y_max_len, dtype='int32')
#
# print("Input with zero padding")