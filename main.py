#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#
from numpy.random import seed
seed(200)
from keras.models import Model
import warnings
import datetime
import optparse
import os, errno
import performance as performance
import keras.callbacks as callbacks
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from gensim.models import KeyedVectors
from keras.layers import Input, Dense, concatenate, Dropout, Flatten, LSTM, Bidirectional, GlobalAveragePooling1D, Multiply
import data_process_multimodal_pair as data_process
from time import time
import pickle
from keras.layers.normalization import BatchNormalization
from data_generator_image_optimized import DataGenerator
import keras
from keras.regularizers import l2
import sklearn.model_selection
import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K

def check_dir(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def file_exist(w2v_checkpoint):
    if os.path.exists(w2v_checkpoint):
        return True
    else:
        return False


def save_model(model, model_dir, model_file_name, tokenizer, label_encoder):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    base_name = os.path.basename(model_file_name)
    base_name = os.path.splitext(base_name)[0]
    model_file = model_dir + "/" + base_name + ".hdf5"
    tokenizer_file = model_dir + "/" + base_name + ".tokenizer"
    label_encoder_file = model_dir + "/" + base_name + ".label_encoder"

    configfile = model_dir + "/" + base_name + "_v2.config"
    configFile = open(configfile, "w")
    configFile.write("model_file=" + model_file + "\n")
    configFile.write("tokenizer_file=" + tokenizer_file + "\n")
    configFile.write("label_encoder_file=" + label_encoder_file + "\n")
    configFile.close()

    files = []
    files.append(configfile)

    # serialize weights to HDF5
    model.save(model_file)
    files.append(model_file)

    # saving tokenizer
    with open(tokenizer_file, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    files.append(tokenizer_file)

    # saving label_encoder
    with open(label_encoder_file, 'wb') as handle:
        pickle.dump(label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)
    files.append(label_encoder_file)

def single_test_data_generation(images, im_num, im_dict, sim_dict, bert_dict):
    images1_batch = np.empty([1, 1, 1000])
    images2_batch = np.empty([1, 1, 1000])
    images3_batch = np.empty([1, 1, 1000])
    images4_batch = np.empty([1, 1, 1000])
    images5_batch = np.empty([1, 1, 1000])

    sim_batch = np.empty([1, 1, 3])
    bert_batch = np.empty([1, 1, 768])

    image_file_name = str(images[0])
    news_ids = image_file_name.split('_')[0] + '_' + image_file_name.split('_')[1]
    sim = sim_dict[news_ids][:im_num]
    bert = bert_dict[news_ids]
    sim_batch[0, :, :] = np.array(sim)
    bert_batch[0, :, :] = np.array(bert)
    if len(images) == 5:
        image_file_name2 = str(images[1])
        image_file_name3 = str(images[2])
        image_file_name4 = str(images[3])
        image_file_name5 = str(images[4])

    elif len(images) == 4:
        image_file_name2 = str(images[1])
        image_file_name3 = str(images[2])
        image_file_name4 = str(images[3])
        image_file_name5 = ' '
    elif len(images) == 3:
        image_file_name2 = str(images[1])
        image_file_name3 = str(images[2])
        image_file_name4 = ' '
        image_file_name5 = ' '

    elif len(images) == 2:
        image_file_name2 = str(images[1])
        image_file_name3 = ' '
        image_file_name4 = ' '
        image_file_name5 = ' '

    else:
        image_file_name2 = ' '
        image_file_name3 = ' '
        image_file_name4 = ' '
        image_file_name5 = ' '

    if (image_file_name1 in im_dict):
        img1 = im_dict[image_file_name1]

        images1_batch[0, :, :] = img1
        text_batch[0] = text_batch
    else:
        img1 = np.zeros([1, 1000])
        images1_batch[0, :, :] = img1
        text_batch[0] = text_batch
    if (image_file_name2 in im_dict):
        img2 = im_dict[image_file_name2]
        images2_batch[0, :, :] = img2
    else:
        img2 = np.zeros([1, 1000])
        images2_batch[0, :, :] = img2
    if (image_file_name3 in im_dict):
        img3 = im_dict[image_file_name3]
        images3_batch[0, :, :] = img3
    else:
        img3 = np.zeros([1, 1000])
        images3_batch[0, :, :] = img3

    if (image_file_name4 in im_dict):
        img4 = im_dict[image_file_name4]
        images4_batch[0, :, :] = img4
    else:
        img4 = np.zeros([1, 1000])
        images4_batch[0, :, :] = img4
    if (image_file_name5 in im_dict):
        img5 = im_dict[image_file_name5]
        images5_batch[0, :, :] = img5
    else:
        img5 = np.zeros([1, 1000])
        images5_batch[0, :, :] = img5
        
    if im_num == 1:
        return [images1_batch, sim_batch, bert_batch]
    elif im_num == 2:
        return [images1_batch, images2_batch, sim_batch, bert_batch]
    elif im_num == 3:
        return [images1_batch, images2_batch, images3_batch, sim_batch, bert_batch]
    elif im_num == 4:
        return [images1_batch, images2_batch, images3_batch, images4_batch, sim_batch, bert_batch]
    elif im_num == 5:
        return [images1_batch, images2_batch, images3_batch, images4_batch, images5_batch, sim_batch, bert_batch]

def write_results(out_file, file_name, accu, P, R, F1, wAUC, AUC, report, conf_mat):
    accu = accu * 100
    wauc = wAUC * 100
    auc = AUC * 100
    precision = P * 100
    recall = R * 100
    f1_score = F1 * 100
    result = str("{0:.2f}".format(auc)) + "\t" + str("{0:.2f}".format(accu)) + "\t" + str(
        "{0:.2f}".format(precision)) + "\t" + str("{0:.2f}".format(recall)) + "\t" + str(
        "{0:.2f}".format(f1_score)) + "\n"
    print(result)
    print(report)
    out_file.write(file_name + "\n")
    out_file.write(result)
    out_file.write(report)
    out_file.write(conf_mat)

def dir_exist(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def get_RNN(unit=LSTM, cells=64, bi=False, return_sequences=True, return_state=False, dropout_U=0., consume_less='cpu', l2_reg=0):
    rnn = unit(cells, return_sequences=return_sequences, return_state=return_state,
               consume_less=consume_less, dropout_U=dropout_U,
               W_regularizer=l2(l2_reg))
    if bi:
        return Bidirectional(rnn)
    else:
        return rnn

def multi_image_fake_news_detection_model(image_num):
    input_sim = Input(shape=(1, image_num))
    sim_feature = Flatten()(input_sim)

    input_bert = Input(shape=(1, 768))
    bert_feature = Flatten()(input_bert)

    ######## Image text_network ########
    last_layer_output1 = Input(shape=(1, 1000))
    last_layer_output2 = Input(shape=(1, 1000))
    last_layer_output3 = Input(shape=(1, 1000))
    last_layer_output4 = Input(shape=(1, 1000))
    last_layer_output5 = Input(shape=(1, 1000))

    last_layer_output1 = Flatten()(last_layer_output1) 
    last_layer_output2 = Flatten()(last_layer_output2)
    last_layer_output3 = Flatten()(last_layer_output3)
    last_layer_output4 = Flatten()(last_layer_output4)
    last_layer_output5 = Flatten()(last_layer_output5)
    if image_num > 1:
        if image_num == 2:
            merged_images = concatenate([last_layer_output1, last_layer_output2], axis=-1)
        elif image_num == 3:
            merged_images = concatenate([last_layer_output1, last_layer_output2, last_layer_output3], axis=-1) 
        elif image_num == 4:
            merged_images = concatenate([last_layer_output1, last_layer_output2, last_layer_output3, last_layer_output4], axis=-1)  
        elif image_num == 5:
            merged_images = concatenate([last_layer_output1, last_layer_output2, last_layer_output3, last_layer_output4, last_layer_output5], axis=-1)
    
        im_lstm_input = keras.layers.core.Reshape((image_num, 1000))(merged_images)
        print('h_title', K.int_shape(im_lstm_input))
        rnn_out, h1, c1 = get_RNN(LSTM, 200, bi=False, return_sequences=True, return_state=True, dropout_U=0.2)(
            im_lstm_input)
        image_feature = GlobalAveragePooling1D()(rnn_out)
    else:
        image_feature = last_layer_output1

    ######## Merge image and text networks ########
    merged_network = concatenate([bert_feature, sim_feature, image_feature], axis=-1)
    attention_probs = Dense(971, activation='softmax', name='attention_vec')(merged_network)
    # multiply
    attention_mul = Multiply(name='attention_mul')([merged_network, attention_probs])
    merged_network = BatchNormalization()(attention_mul)
    merged_network = Dropout(0.2)(merged_network)
    merged_network = Dense(400, activation='relu')(merged_network)
    merged_network = Dropout(0.2)(merged_network)
    merged_network = Dense(100, activation='relu')(merged_network)
    merged_network = Dropout(0.2)(merged_network)
    out = Dense(nb_classes, activation='softmax')(merged_network)
    model = Model(
        inputs=[last_layer_output1, last_layer_output2, last_layer_output3, last_layer_output4, last_layer_output5, input_sim, input_bert],
        outputs=out)  #

    lr = 0.00005
    print("lr=" + str(lr) + ", beta_1=0.9, beta_2=0.999, amsgrad=False")
    adam = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    print(model.summary())

    callback = callbacks.EarlyStopping(monitor='val_accuracy', patience=patience_early_stop, verbose=1, mode='max')
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=patience_learning_rate, verbose=1,
                                                factor=0.1, min_lr=0.00001, mode='max')
    csv_logger = CSVLogger(log_file, append=False, separator='\t')
    checkpoint = ModelCheckpoint(best_model_path, monitor='val_accuracy', verbose=1,
                                 save_best_only=True, save_weights_only=True, mode='max')
    callbacks_list = [callback, learning_rate_reduction, csv_logger, checkpoint]

    history = model.fit_generator(generator=train_data_generator, epochs=nb_epoch, validation_data=val_data_generator,
                                  use_multiprocessing=True, workers=4, verbose=1, callbacks=callbacks_list)
    return model

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = optparse.OptionParser()
    parser.add_option('-m', action="store", dest="model_file", default="best_model.hdf5", type="string")
    parser.add_option('-o', action="store", dest="outputfile", default="results.tsv", type="string")
    parser.add_option("-w", "--w2v_checkpoint", action="store", dest="w2v_checkpoint",
                      default="data/GoogleNews_vectors_negative_300d.bin", type="string")
    parser.add_option("-d", "--log_dir", action="store", dest="log_dir", default="model_log/", type="string")
    # parser.add_option("-l","--log_file", action="store", dest="log_file", default="./log", type="string")
    parser.add_option("-c", "--checkpoint_log", action="store", dest="checkpoint_log", default="./checkpoint_log/",
                      type="string")
    parser.add_option("-x", "--vocab_size", action="store", dest="vocab_size", default=20000, type="int")
    parser.add_option("--embedding_dim", action="store", dest="embedding_dim", default=300, type="int")
    parser.add_option("--batch_size", action="store", dest="batch_size", default=32, type="int")
    parser.add_option("--nb_epoch", action="store", dest="nb_epoch", default=60, type="int")
    parser.add_option("--max_seq_length", action="store", dest="max_seq_length", default=25, type="int")
    parser.add_option("--patience", action="store", dest="patience", default=60, type="int")
    parser.add_option("--patience-lr", action="store", dest="patience_lr", default=10, type="int")
    parser.add_option("--im_num", action="store", dest="im_num", default=3, type="int")

    options, args = parser.parse_args()
    a = datetime.datetime.now().replace(microsecond=0)

    out_file = options.outputfile
    best_model_path = options.model_file
    log_path = options.checkpoint_log
    log_dir = os.path.abspath(os.path.dirname(log_path))
    dir_exist(log_dir)

    im_dict = np.load('feature_extraction/article_image_vgg16_1000_feature.npy').item() 
    sim_dict = np.load('feature_extraction/gossipcop_image_text_similarity_feature.npy').item()
    bert_dict = np.load('feature_extraction/bert_feature.npy').item()

    ######## Parameters ########
    MAX_SEQUENCE_LENGTH = options.max_seq_length
    MAX_NB_WORDS = options.vocab_size
    EMBEDDING_DIM = options.embedding_dim
    batch_size = options.batch_size
    nb_epoch = options.nb_epoch
    patience_early_stop = options.patience
    patience_learning_rate = options.patience
    im_num = options.im_num
    dir_exist(options.checkpoint_log)
    delim = "\t"

    file_name_list = []
    text_file_path = 'data/gossipcop_news_data.txt'
    text_file = open(text_file_path, 'r', encoding='utf8')
    news_ids = []
    labels = []
    for line in text_file.readlines():
        news_id = line.split('\t')[0]
        news_ids.append(news_id)
        label = news_id.split('_')[0]
        labels.append(label)
    print('all data number',len(news_ids))
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(news_ids, labels, test_size=0.2,
                                                                                random_state=10, stratify=labels)  #10
    X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X_train, y_train, test_size=0.1,
                                                                                random_state=10, stratify=y_train)
    #### training dataset
    train_x, train_image_list, train_y, train_le, train_labels, word_index, tokenizer = data_process.read_train_data_multimodal(X_train, text_file_path, MAX_NB_WORDS,MAX_SEQUENCE_LENGTH)

    #### development dataset
    dev_x, dev_image_list, dev_y, dev_le, dev_labels, _ = data_process.read_dev_data_multimodal(X_val, text_file_path,
                                                                                                tokenizer,
                                                                                                MAX_SEQUENCE_LENGTH)

    nb_classes = len(set(train_labels))
    print("Number of classes: " + str(nb_classes))
    params = {"max_seq_length": MAX_SEQUENCE_LENGTH, "batch_size": batch_size,
              "n_classes": nb_classes, "shuffle": True}
    train_data_generator = DataGenerator(train_image_list, im_num, train_x, im_dict, sim_dict, albert_dict, train_y, **params)

    params = {"max_seq_length": MAX_SEQUENCE_LENGTH, "batch_size": batch_size,
              "n_classes": nb_classes, "shuffle": False}
    val_data_generator = DataGenerator(dev_image_list,im_num, dev_x, im_dict, sim_dict, albert_dict, dev_y, **params)

    MAX_SEQUENCE_LENGTH = options.max_seq_length
    MAX_NB_WORDS = options.vocab_size

    ######## Word-Embedding ########
    if (options.w2v_checkpoint and file_exist(options.w2v_checkpoint)):
        options.emb_matrix = pickle.load(open(options.w2v_checkpoint, "rb"))
    else:
        word_vec_model_file = "data/GoogleNews_vectors_negative_300d.bin"
        emb_model = KeyedVectors.load_word2vec_format(word_vec_model_file, binary=True)
        embedding_matrix = data_process.prepare_embedding(word_index, emb_model, options.vocab_size,
                                                          options.embedding_dim)
        options.emb_matrix = embedding_matrix
        options.vocab_size, options.embedding_dim = embedding_matrix.shape
        pickle.dump(options.emb_matrix, open(options.w2v_checkpoint, "wb"))

    ######## Text text_network ########
    print("Embedding size: " + str(options.emb_matrix.shape))

    model = multi_image_fake_news_detection_model(im_num)
    ######## Save the model ########
    print("saving model...")
    model.load_weights(best_model_path)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print("Best saved model loaded...")

    dir_name = os.path.dirname(best_model_path)

    model_dir = dir_name + "/" + base_name
    save_model(model, model_dir, best_model_path, tokenizer, train_le)

    ############ Test data  ########
    dev_prob = model.predict_generator(val_data_generator, verbose=1)
    print("dev true len: " + str(len(dev_y)))
    print("dev pred len: " + str(len(dev_prob)))
    AUC, accu, P, R, F1, report = performance.performance_measure_cnn(dev_y, dev_prob, train_le)

    result = str("{0:.4f}".format(accu)) + "\t" + str("{0:.4f}".format(P)) + "\t" + str(
        "{0:.4f}".format(R)) + "\t" + str("{0:.4f}".format(F1)) + "\t" + str("{0:.4f}".format(AUC)) + "\n"
    print(result)
    print(report)

    test_x, test_image_list, test_y, test_le, test_labels, ids = data_process.read_dev_data_multimodal(X_test,
                                                                                                       text_file_path,
                                                                                                       tokenizer,
                                                                                                       MAX_SEQUENCE_LENGTH)
    print("Number of classes: " + str(nb_classes))
    params = {"max_seq_length": MAX_SEQUENCE_LENGTH, "batch_size": batch_size,
              "n_classes": nb_classes, "shuffle": False}
    print("image size: " + str(len(test_image_list)))
    print("test x: " + str(len(test_x)))
    print("test y: " + str(len(test_y)))

    test_result_file = open('all_input_result.txt','w',encoding='utf8')
    test_result_file.write('news_id' + '\t' + 'predict_label'+'\t'+ 'true_label'+ '\n')
    test_results = []
    
    #############single test data evaluation#################
    for id, test_data_image, real_label in zip(ids, test_image_list, test_y):
        single_test_data = single_test_data_generation(test_data_image, im_num, im_dict, sim_dict, bert_dict)
        single_result = model.predict(single_test_data,batch_size=1)
        test_results.append(single_result.tolist()[0])
        y_pred = np.argmax(single_result, axis=1)

        y_true = real_label.tolist().index(1)
        single_result = single_result.tolist()[0]

        id = id.split('_')[1]
        test_result_file.write(id+'\t'+str(y_pred[0])+'\t'+str(y_true)+'\n')
    test_results = np.array(test_results)
    AUC, accu, P, R, F1, report = performance.performance_measure_cnn(test_y, test_results, test_le)
    result = str("{0:.4f}".format(accu)) + "\t" + str("{0:.4f}".format(P)) + "\t" + str(
        "{0:.2f}".format(R)) + "\t" + str("{0:.4f}".format(F1)) + "\t" + str("{0:.4f}".format(AUC)) + "\n"
    print("results-cnn:\t" + base_name + "\t" + result)
    print(report)