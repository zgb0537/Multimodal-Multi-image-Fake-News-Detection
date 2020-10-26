import numpy as np
import keras
from keras.preprocessing.image import array_to_img
import warnings
import datetime
import optparse
import os, errno
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, image_file_list, im_num, text_vec, image_vec_dict, sim_dict, bert_dict,labels, max_seq_length=20, batch_size=32,
                 n_classes=2, shuffle=False):
        'Initialization'
        # self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.image_file_list = image_file_list
        self.text_vec = text_vec
        self.image_vec_dict = image_vec_dict
        self.im_num = im_num
        self.sim_dict = sim_dict
        self.bert_dict = bert_dict
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.max_seq_length = max_seq_length
        self.on_epoch_end()
        

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.image_file_list) / float(self.batch_size)))


    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        #print(" index starts at: "+str(index * self.batch_size) +" ends at: "+str((index + 1) * self.batch_size))
        start = index * self.batch_size
        end = (index + 1) * self.batch_size
        if(end > len(self.image_file_list)):
            #print(" index starts at: " + str(start) + " ends at: " + str(end))
            end = len(self.image_file_list)

        temp_indexes = self.indexes[start:end]
        #print(temp_indexes)
        # Generate data
        images1_batch, images2_batch, images3_batch, images4_batch,images5_batch, text_batch, sim_batch, bert_batch, y = self.__data_generation(temp_indexes)

        return [images1_batch, images2_batch, images3_batch,images4_batch, images5_batch, text_batch, sim_batch, bert_batch], y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.image_file_list))
        #print(" indexes len: "+str(len(self.indexes)))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples'
        # Initialization
        y = np.empty((len(indexes),self.n_classes), dtype=int)
        text_batch = np.empty((len(indexes), self.max_seq_length), dtype=int)
        images1_batch = np.empty([len(indexes), 1, 1000])
        images2_batch = np.empty([len(indexes), 1, 1000])
        images3_batch = np.empty([len(indexes), 1, 1000])
        images4_batch = np.empty([len(indexes), 1, 1000])
        images5_batch = np.empty([len(indexes), 1, 1000])
        sim_batch = np.empty([len(indexes), 1, self.im_num])
        bert_batch = np.empty([len(indexes), 1, 768])

        # Generate data
        for i, index in enumerate(indexes):
            try:
                if(index <= len(self.image_file_list)):
                    image_file_name1 = str(self.image_file_list[index][0])
                    news_ids = image_file_name1.split('_')[0]+'_'+ image_file_name1.split('_')[1]
                    sim = self.sim_dict[news_ids][:self.im_num]
                    bert = self.bert_dict[news_ids]
                    sim_batch[i, :, :]=np.array(sim)
                    bert_batch[i, :, :]=np.array(bert)  #albert.tolist()[0]

                    if len(self.image_file_list[index]) == 5:
                        image_file_name2 = str(self.image_file_list[index][1])
                        image_file_name3 = str(self.image_file_list[index][2])
                        image_file_name4 = str(self.image_file_list[index][3])
                        image_file_name5 = str(self.image_file_list[index][4])
                    elif len(self.image_file_list[index])==4:
                        image_file_name2 = str(self.image_file_list[index][1])
                        image_file_name3 = str(self.image_file_list[index][2])
                        image_file_name4 = str(self.image_file_list[index][3])
                        image_file_name5 = ' '

                    elif len(self.image_file_list[index])==3:
                        image_file_name2 = str(self.image_file_list[index][1])
                        image_file_name3 = str(self.image_file_list[index][2])
                        image_file_name4 = ' '
                        image_file_name5 = ' '

                    elif len(self.image_file_list[index])==2:
                        image_file_name2 = str(self.image_file_list[index][1])
                        image_file_name3 = ' '
                        image_file_name4 = ' '
                        image_file_name5 = ' '

                    else:
                        image_file_name2 = ' '
                        image_file_name3 = ' '
                        image_file_name4 = ' '
                        image_file_name5 = ' '

                    if(image_file_name1 in self.image_vec_dict):
                        # if(image_file_name=="image_null"):
                        #     img = np.zeros([1, 224, 224, 3])
                        # else:
                        img = self.image_vec_dict[image_file_name1]
                        images1_batch[i, :, :] = img
                        #images1_batch[i, :, :, :] = img
                        # Store class
                        #print(img)
                        y[i] = self.labels[index]
                        text_batch[i] = self.text_vec[index]

                        print('img:::::::::::::::::::', img)
                        print('sim:::::::::::::::::::', np.array(sim))
                        print('text:::::::::::::::::::', np.array(albert))
                        print('label::::::::::::::::::', self.labels[index])

                    else:
                        y[i] = self.labels[index]
                        text_batch[i] = self.text_vec[index]
                        img = np.zeros([1, 1000])
                        images1_batch[i, :, :] = img

                    if(image_file_name2 in self.image_vec_dict):
                        img = self.image_vec_dict[image_file_name2]
                        images2_batch[i, :, :] = img
                        # Store class
                    else:
                        img = np.zeros([1, 1000])
                        images2_batch[i, :, :] = img
                    if(image_file_name3 in self.image_vec_dict):
                        img = self.image_vec_dict[image_file_name3]
                        images3_batch[i, :, :] = img
                        # Store class
                    else:
                        img = np.zeros([1, 1000])
                        images3_batch[i, :, :] = img

                    if (image_file_name4 in self.image_vec_dict):

                        img = self.image_vec_dict[image_file_name3]
                        images4_batch[i, :, :] = img
                        # Store class
                    else:
                        img = np.zeros([1, 1000])
                        images4_batch[i, :, :] = img

                    if (image_file_name5 in self.image_vec_dict):
                        img = self.image_vec_dict[image_file_name3]
                        images5_batch[i, :, :] = img
                    else:
                        img = np.zeros([1, 1000])
                        images5_batch[i, :, :] = img
                else:
                    print("Exception in indexing in image list less "+str(index)+" " +str(self.image_file_list))
            except Exception as e:
                print("Exception in data generation.")
                print(e)
        if self.im_num == 1:
            return images1_batch, text_batch, sim_batch, bert_batch, y
        elif self.im_num == 2:
            return images1_batch, images2_batch, text_batch, sim_batch, bert_batch, y
        elif self.im_num == 3:
            return images1_batch, images2_batch, images3_batch, text_batch, sim_batch, bert_batch, y
        elif self.im_num == 4:
            return images1_batch, images2_batch, images3_batch, images4_batch, text_batch, sim_batch, bert_batch, y
        elif self.im_num == 5:
            return images1_batch, images2_batch, images3_batch, images4_batch, images5_batch, text_batch, sim_batch, bert_batch, y