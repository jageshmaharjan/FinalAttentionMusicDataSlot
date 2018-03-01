import argparse

from keras.layers import LSTM, TimeDistributed, Dense
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import text_to_word_sequence
import numpy as np

from attention_training.attention_decoder import AttentionDecoder
# from brownlee_attention import one_hot_encode
from attention_training.data_preprocess import load_data

# one hot encode sequence
def one_hot_encode(sequence, n_unique):
	encoding = list()
	for value in sequence:
		vector = [0 for _ in range(n_unique)]
		vector[value] = 1
		encoding.append(vector)
	return np.array(encoding)

# decode a one hot encoded string
def one_hot_decode(encoded_seq):
	return [(vector) for vector in encoded_seq]

def attention_model(seq_len, vocab_size):
	model = Sequential()
	# model.add(Embedding(vocab_size, n_features, trainable=True))
	model.add(LSTM(150, input_shape=(seq_len, vocab_size), return_sequences=True))
	model.add(AttentionDecoder(150, vocab_size))
	# model.add(TimeDistributed(Dense(n_classes, activation='softmax')))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
	return model

def get_attention_label(query):
    n_features = 50
    n_timesteps_in = 25  # length of the sentence
    n_timesteps_out = 2
    vocab_size = 714
    n_classes = 5
    src = '/home/jugs/PycharmProjects/FinalAttentionMusicDataSlot/Database/whilelist_X'
    dst = '/home/jugs/PycharmProjects/FinalAttentionMusicDataSlot/Database/whilelist_y'
    X, X_vocab_len, X_word_to_ix, X_ix_to_word, y, y_vocab_len, y_word_to_ix, y_ix_to_word = \
        load_data(src, dst, max_len=30, vocab_size=vocab_size, classes=n_classes)

    weight = '/home/shenzhen/PycharmProjects/Attention_RNN_Feb6/JugsAttention/simpleData/models/weight_model.83-0.09.hdf5'
    model = attention_model(n_timesteps_in, vocab_size)
    model.summary()
    model.load_weights(weight)

    x_input = list()
    sample_tokens = text_to_word_sequence(query)  # [text_to_word_sequence(x)[::-1] for x in sample.split(" ")]
    for j, word in enumerate(sample_tokens):
        if word in X_word_to_ix:
            x_input.append(X_word_to_ix[word])
        else:
            x_input.append(X_word_to_ix['UNK'])

    while len(x_input) < 25:
        x_input.append(0)

    x_input = one_hot_encode(x_input, vocab_size)
    x_input = x_input.reshape((1, x_input.shape[0], x_input.shape[1]))

    predict = model.predict(x_input)
    predict = np.argmax(one_hot_decode(predict[0]), axis=1)
    predict = predict
    result = [y_ix_to_word[int(predict[i])] for i in range(n_timesteps_in)]

    return result