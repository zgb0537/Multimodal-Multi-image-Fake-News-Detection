# -*- coding: utf-8 -*-
import numpy as np
np.random.seed(100)
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import sys
from sklearn import preprocessing
import re
import aidrtokenize as aidrtokenize
from collections import Counter
import random

def file_exist(file_name):
    if os.path.exists(file_name):
        return True
    else:
        return False

def read_stop_words(file_name):
    if (not file_exist(file_name)):
        print("Please check the file for stop words, it is not in provided location " + file_name)
        sys.exit(0)
    stop_words = []
    with open(file_name, 'rU') as f:
        for line in f:
            line = line.strip()
            if (line == ""):
                continue
            stop_words.append(line)
    return stop_words

stop_words_file = "data/stop_words_english.txt"
stop_words = read_stop_words(stop_words_file)


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def get_tokenizer(data, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH):
    """
    Prepare the data
    """
    # Finally, vectorize the text samples into a 2D integer tensor
    tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(data)
    sequences = tokenizer.texts_to_sequences(data)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    print('Shape of data tensor:', data.shape)
    return word_index, tokenizer


def read_train_data_multimodal(data_file, text_file_path, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH):
    """
    Prepare the data
    """
    data = []
    image_list = []
    lab = []
    text_file = open(text_file_path,'r',encoding='utf8')
    for line in text_file.readlines():
        news_id = line.split('\t')[0]
        if news_id in data_file:
            text = line.split('\t')[1]
            im_list = line.split('\t')[2].split(' ')
            text = aidrtokenize.tokenize(text)
            label = news_id.split('_')[0]
            image_list.append(im_list)
            data.append(text)
            lab.append(label)
    print('training data length',len(data))
    # print(data[0])
    le = preprocessing.LabelEncoder()
    yL = le.fit_transform(lab)
    labels = list(le.classes_)
    print("training classes: " + " ".join(labels))
    label = yL.tolist()
    yC = len(set(label))
    yR = len(label)
    y = np.zeros((yR, yC))
    y[np.arange(yR), yL] = 1
    y = np.array(y, dtype=np.int32)

    # finally, vectorize the text samples into a 2D integer tensor
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, oov_token="OOV_TOK")
    tokenizer.fit_on_texts(data)
    sequences = tokenizer.texts_to_sequences(data)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    # labels = to_categorical(np.asarray(labels))
    print('Shape of data tensor:', data.shape)
    # print('Shape of label tensor:', labels.shape)
    # return data,labels,word_index,dim;
    return data, image_list, y, le, labels, word_index, tokenizer

def read_dev_data_multimodal(data_file, text_file_path, tokenizer, MAX_SEQUENCE_LENGTH):
    """
    Prepare the data
    """
    ids = []
    data = []
    image_list = []
    lab = []
    text_file = open(text_file_path, 'r', encoding='utf8')
    for line in text_file.readlines():
        news_id = line.split('\t')[0]
        if news_id in data_file:
            ids.append(news_id)
            text = line.split('\t')[1]
            im_list = line.split('\t')[2].split(' ')
            text = aidrtokenize.tokenize(text)
            label = news_id.split('_')[0]
            image_list.append(im_list)
            data.append(text)
            lab.append(label)
    counts = Counter(lab)
    print(counts)
    print(len(data))

    le = preprocessing.LabelEncoder()
    yL = le.fit_transform(lab)
    labels = list(le.classes_)
    print("training classes: " + " ".join(labels))

    label = yL.tolist()
    yC = len(set(label))
    yR = len(label)
    y = np.zeros((yR, yC))
    y[np.arange(yR), yL] = 1
    y = np.array(y, dtype=np.int32)

    sequences = tokenizer.texts_to_sequences(data)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    print('Shape of data tensor:', data.shape)
    return data, image_list, y, le, labels, ids

def convert_to_vector(inst_list, tokenizer, MAX_SEQUENCE_LENGTH, delim):
    """
    Prepare the data
    """
    # ids = []
    data = []
    lab = []
    for inst in inst_list:
        txt = aidrtokenize.tokenize(inst.text)
        text = " ".join(txt)
        if (len(txt) < 1):
            print("TEXT SIZE:" + txt)
            continue
        data.append(text)
        lab.append(inst.label)

    le = preprocessing.LabelEncoder()
    yL = le.fit_transform(lab)
    labels = list(le.classes_)

    label = yL.tolist()
    yC = len(set(label))
    yR = len(label)
    y = np.zeros((yR, yC))
    y[np.arange(yR), yL] = 1
    y = np.array(y, dtype=np.int32)

    sequences = tokenizer.texts_to_sequences(data)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    print('Shape of data tensor:', data.shape)
    return data, y, le, labels

def load_embedding(fileName):
    print('Indexing word vectors.')
    embeddings_index = {}
    f = open(fileName)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index

def prepare_embedding(word_index, model, MAX_NB_WORDS, EMBEDDING_DIM):
    # prepare embedding matrix
    nb_words = min(MAX_NB_WORDS, len(word_index) + 1)
    embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM), dtype=np.float32)
    print(len(embedding_matrix))
    for word, i in word_index.items():
        if i >= nb_words:
            continue
        try:
            embedding_vector = model[word][0:EMBEDDING_DIM]  # embeddings_index.get(word)
            embedding_matrix[i] = np.asarray(embedding_vector, dtype=np.float32)
        except KeyError:
            try:
                rng = np.random.RandomState()
                embedding_vector = rng.randn(EMBEDDING_DIM)  # np.random.random(num_features)
                embedding_matrix[i] = np.asarray(embedding_vector, dtype=np.float32)
            except KeyError:
                continue
    return embedding_matrix
