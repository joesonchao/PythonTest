import numpy as np
from keras import layers, models
from keras.datasets import imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
print(train_data[0])
print(max([max(sequence) for sequence in train_data]))
word_index = imdb.get_word_index()
print(type(word_index))
# print(word_index)
reverse_word_index = dict([(v, k) for k, v in word_index.items()])
decoded_1_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
print(train_data[0])
print(decoded_1_review)
decoded_2_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[1]])
print(decoded_2_review)
