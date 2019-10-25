import tarfile
import sys
import os
import tensorflow as tf
from tensorflow.keras import layers
import shutil
import glob
import numpy as np
import zipfile as zp


# Task 1: Load the data
# For this task you will load the data, create a vocabulary and encode the reviews with integers

def read_file(path_to_dataset):
    """
    :param path_to_dataset: a path to the tar file (dataset)
    :return: two lists, one containing the movie reviews and another containing the corresponding labels
    """

    if not tarfile.is_tarfile(path_to_dataset):
        sys.exit("Input path is not a tar file")
    dirent = tarfile.open(path_to_dataset)
    
    dirent.extractall()

    pos_train_data = []
    for data in glob.glob("movie_reviews/train/pos/cv*_*.txt"):
        with open(data, 'r') as d:
            contents = d.read().splitlines()
            pos_train_data.extend(contents)
    pos_train_labels = ["POSITIVE" for k in pos_train_data]

    neg_train_data = []
    for data in glob.glob("movie_reviews/train/neg/cv*_*.txt"):
        with open(data, 'r') as d:
            contents = d.read().splitlines()
            neg_train_data.extend(contents)
    neg_train_labels = ["NEGATIVE" for k in neg_train_data]

    train_data = pos_train_data + neg_train_data
    train_labels = pos_train_labels + neg_train_labels


    pos_test_data = []
    for data in glob.glob("movie_reviews/test/pos/cv*_*.txt"):
        with open(data, 'r') as d:
            contents = d.read().splitlines()
            pos_test_data.extend(contents)
    pos_test_labels = ["POSITIVE" for k in pos_test_data]

    neg_test_data = []
    for data in glob.glob("movie_reviews/test/neg/cv*_*.txt"):
        with open(data, 'r') as d:
            contents = d.read().splitlines()
            neg_test_data.extend(contents)
    neg_test_labels = ["NEGATIVE" for k in neg_test_data]

    test_data = pos_test_data + neg_test_data
    test_labels = pos_test_labels + neg_test_labels
    
    dirent.close()
    return (train_data, train_labels, test_data, test_labels)


def preprocess(text):
    """
    :param text: list of sentences or movie reviews
    :return: a dict of all tokens you encounter in the dataset. i.e. the vocabulary of the dataset
    Associate each token with a unique integer
    """

    if type(text) is not list:
        sys.exit("Please provide a list to the method")

    all_words = ''.join(text).split(' ')
    return {k:i for i,k in enumerate(list(dict.fromkeys(all_words)) + [None])}



def encode_review(vocab, text):
    """
    :param vocab: the vocabulary dictionary you obtained from the previous method
    :param text: list of movie reviews obtained from the previous method
    :return: encoded reviews
    """

    if type(vocab) is not dict or type(text) is not list:
        sys.exit("Please provide a list to the method")
    return [[vocab[k] for k in sentence.split(' ')] for sentence in text]


def encode_labels(labels): # Note this method is optional (if you have not integer-encoded the labels)
    """
    :param labels: list of labels associated with the reviews
    :return: encoded labels
    """

    if type(labels) is not list:
        sys.exit("Please provide a list to the method")
    return [0 if k == "NEGATIVE" else 1 for k in labels]


def pad_zeros(encoded_reviews, seq_length = 200):
    """
    :param encoded_reviews: integer-encoded reviews obtained from the previous method
    :param seq_length: maximum allowed sequence length for the review
    :return: encoded reviews after padding zeros
    """

    if type(encoded_reviews) is not list:
        sys.exit("Please provide a list to the method")
    """
    COMPLETE THE REST OF THE METHOD
    """
    return [(k+[0]*seq_length)[0:seq_length] for k in encoded_reviews]

# Task 2: Load the pre-trained embedding vectors
# For this task you will load the pre-trained embedding vectors from Word2Vec

def load_embedding_file(embedding_file, token_dict):
    """
    :param embedding_file: path to the embedding file
    :param token_dict: token-integer mapping dict obtained from previous step
    :return: embedding dict: embedding vector-integer mapping
    """

    if not os.path.isfile(embedding_file):
        sys.exit("Input embedding path is not a file")
    if type(token_dict) is not dict:
        sys.exit("Input a dictionary!")
    
    embedding_mapping = np.zeros([len(token_dict), 300])
    i = 0
    with zp.ZipFile(embedding_file) as myzip:
        myzip.extractall()
    with open('.'.join(embedding_file.split('.')[:-1]), 'r') as f:
        next(f)
        for line in f:
            row = line.split(' ')
            try:
                index = token_dict[row[0].lower()]
                embedding_mapping[index] = row[1:]
            except KeyError as k:
                # print("Could not find", row[0].lower())
                pass
            i += 1
            if i % 50000 == 0:
                print("Processed", i, "lines from vec file...")
    os.remove('.'.join(embedding_file.split('.')[:-1]))
    return embedding_mapping
    
# Task 3: Create a TensorDataset and DataLoader

class SentimentDataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, encoded_reviews, labels, batch_size = 32):
        'Initialization'
        self.encoded_reviews, self.labels = encoded_reviews, labels
        self.batch_size = batch_size

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.encoded_reviews)//self.batch_size

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch

        x = self.encoded_reviews[index*self.batch_size:(index+1)*self.batch_size]
        y = self.labels[index*self.batch_size:(index+1)*self.batch_size]
        return x, y

def create_data_loader(encoded_reviews, labels, batch_size = 32):
    """
    :param encoded_reviews: zero-paddded integer-encoded reviews
    :param labels: integer-encoded labels
    :param batch_size: batch size for training
    :return: DataLoader object
    """
    def unison_shuffled_copies(a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]

    if type(encoded_reviews) is not list or type(labels) is not list:
        sys.exit("Please provide a list to the method")
    
    split = int(0.8*len(encoded_reviews))
    encoded_reviews, labels = unison_shuffled_copies(np.asfarray(encoded_reviews), np.asfarray(labels))
    train_gen = SentimentDataGenerator(encoded_reviews[0:split], labels[0:split], batch_size=batch_size)
    val_gen = SentimentDataGenerator(encoded_reviews[split:], labels[split:], batch_size=batch_size)
    return train_gen, val_gen 
    


seq_length = 200
batch_size = 64

train_data, train_labels, test_data, test_labels = read_file("movie_reviews.tar.gz")
vocab = preprocess(train_data + test_data)

train_data = encode_review(vocab, train_data)
train_data = pad_zeros(train_data, seq_length=seq_length)
train_labels = encode_labels(train_labels)

test_data = encode_review(vocab, test_data)
test_data = pad_zeros(test_data, seq_length=seq_length)
test_labels = encode_labels(test_labels)

embedding_mapping = load_embedding_file("wiki-news-300d-1M.vec.zip", vocab)

gen, vgen = create_data_loader(train_data, train_labels, batch_size=batch_size)



# # Task 4: Define the Baseline model here

# # This is the baseline model that contains an embedding layer and an fcn for classification

inputs = layers.Input(shape=(seq_length))
embed = layers.Embedding(len(vocab), 300, input_length=seq_length, trainable=False, weights=[embedding_mapping])(inputs)
x = layers.Flatten()(embed)
output = layers.Dense(1, activation="sigmoid")(x)

baseline_model = tf.keras.Model(inputs=inputs, outputs=output)

# class BaseSentiment(nn.Module):
#     def __init__(self):
#         super(BaseSentiment).__init__()


#     def forward (self, input_words):
#         pass



# # Task 5: Define the RNN model here

# # This model contains an embedding layer, an rnn and an fcn for classification

inputs = layers.Input(shape=(seq_length))
embed = layers.Embedding(len(vocab), 300, input_length=seq_length, trainable=False, weights=[embedding_mapping])(inputs)
x = layers.SimpleRNN(64, unroll=True)(embed)
output = layers.Dense(1, activation="sigmoid")(x)

rnn_model = tf.keras.Model(inputs=inputs, outputs=output)
print(rnn_model.summary())

sgd = tf.keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=1.)
rnn_model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(lr=0.001),
            #   optimizer=sgd,
              metrics=['binary_accuracy'])

history = rnn_model.fit_generator(
    gen, epochs=500, validation_data=vgen, shuffle=False,
    # workers=8, use_multiprocessing=True
)


# class RNNSentiment(nn.Module):
#     def __init__(self):
#         super(RNNSentiment).__init__()

#     def forward(self, input_words):
#         pass

# # Task 6: Define the Attention model here

# # This model contains an embedding layer, self-attention and an fcn for classification
# class AttentionSentiment(nn.Module):
#     def __init__(self):
#         super(RNNSentiment).__init__()

#     def forward(self, input_words):
#         pass

"""
ALL METHODS AND CLASSES HAVE BEEN DEFINED! TIME TO START EXECUTION!!
"""

# Task 7: Start model training and testing

# Instantiate all hyper-parameters and objects here

# Define loss and optimizer here

# Training starts!!

# Testing starts!!


# print(embedding_mapping[:10])
# print(train_data[:10])
# print(train_labels[:10])
# print(test_data[:10])
# print(test_labels[:10])

# shutil.rmtree("movie_reviews")
