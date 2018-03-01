from random import randint
from numpy import array
from numpy import argmax
from numpy import array_equal
from keras.models import Sequential
from keras.layers import LSTM
from attention_training.attention_decoder import AttentionDecoder
from attention_training.data_preprocess import load_data

# generate a sequence of random integers
def generate_sequence(length, n_unique):
    return [randint(0, n_unique - 1) for _ in range(length)]

def all_data():
    sentence_words = '/home/jugs/PycharmProjects/FinalAttentionMusicDataSlot/Database/whilelist_X'
    sentence_labels = '/home/jugs/PycharmProjects/FinalAttentionMusicDataSlot/Database/whitelist_y'
    data = load_data(sentence_words, sentence_labels, max_len=25, vocab_size=500, classes=7)
    return data

def validation_data():
    sentence_words = '/home/jugs/PycharmProjects/FinalAttentionMusicDataSlot/Database/validation_X'
    sentence_labels = '/home/jugs/PycharmProjects/FinalAttentionMusicDataSlot/Database/validation_y'
    data = load_data(sentence_words, sentence_labels, max_len=25, vocab_size=500, classes=7)
    return data

def get_data(X, y, n_in, n_out, features):
    while len(X) < n_in:
        X.append(0)
    while len(y) < n_in:
        y.append(0)

    if len(y)==n_in & len(X)==n_in:
        X = one_hot_encode(X, features)
        y= one_hot_encode(y, n_out)

        X = X.reshape((1, X.shape[0], X.shape[1]))
        y = y.reshape((1, y.shape[0], y.shape[1]))

        return X, y


def one_hot_encode(sequence, n_unique):
    encoding = list()
    for value in sequence:
        vector = [0 for _ in range(n_unique)]
        vector[value] = 1
        encoding.append(vector)
    return array(encoding)
# one hot encode sequence

# decode a one hot encoded string
def one_hot_decode(encoded_seq):
    return [argmax(vector) for vector in encoded_seq]


# prepare data for the LSTM
def get_pair(n_in, n_out, cardinality):
    # generate random sequence
    sequence_in = generate_sequence(n_in, cardinality)
    sequence_out = sequence_in[:n_out] + [0 for _ in range(n_in - n_out)]
    # one hot encode
    X = one_hot_encode(sequence_in, cardinality)
    y = one_hot_encode(sequence_out, cardinality)
    # reshape as 3D
    X = X.reshape((1, X.shape[0], X.shape[1]))
    y = y.reshape((1, y.shape[0], y.shape[1]))
    return X, y


# configure problem
n_features = 500
n_timesteps_in = 25
n_timesteps_out = 8

# define model
model = Sequential()
model.add(LSTM(150, input_shape=(n_timesteps_in, n_features), return_sequences=True))
model.add(AttentionDecoder(150, n_timesteps_out))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
# train LSTM
X, X_vocab_len, X_word_to_ix, X_ix_to_word, y, y_vocab_len, y_word_to_ix, y_ix_to_word  = all_data()        # Data
for epoch in range(len(X)):
    # generate new random sequence
    if (X[epoch] is not None) & (y[epoch] is not None):
        Xi, yi = get_data(X[epoch], y[epoch], n_timesteps_in, n_timesteps_out, n_features)
        # X, y = get_pair(n_timesteps_in, n_timesteps_out, n_features)
        # fit model for one epoch on this sequence
        model.fit(Xi, yi, epochs=1, verbose=2)
# evaluate LSTM
Xv, Xv_vocab_len, Xv_word_to_ix, Xv_ix_to_word, yv, yv_vocab_len, yv_word_to_ix, yv_ix_to_word  = validation_data()     # Data
total, correct = 100, 0
for i in range(len(Xv)):
    if (Xv[i] is not None) & (yv[i] is not None):
        Xiv ,yiv = get_data(Xv[i], yv[i], n_timesteps_in, n_timesteps_out, n_features)
        # X, y = get_pair(n_timesteps_in, n_timesteps_out, n_features)
        yhat = model.predict(Xiv, verbose=0)
        if array_equal(one_hot_decode(y[0]), one_hot_decode(yhat[0])):
            correct += 1
# print('Accuracy: %.2f%%' % (float(correct) / float(total) * 100.0))
# # spot check some examples
# for i in range(10):
#     X, y = get_pair(n_timesteps_in, n_timesteps_out, n_features)
#     yhat = model.predict(X, verbose=0)
#     print('Expected:', one_hot_decode(y[0]), 'Predicted', one_hot_decode(yhat[0]))
