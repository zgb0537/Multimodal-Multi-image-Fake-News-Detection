from keras.applications import VGG16, DenseNet201, Xception,VGG19
import os
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras.models import Model
import pickle, aidrtokenize
from keras.applications.densenet import decode_predictions
from tqdm import tqdm
from scipy.spatial.distance import cosine

def image_feature_extraction(model,im_path):
    train_data = []
    label_data = []
    im_dict = {}
    for im in im_list:
        print(im)
        try:
            img = image.load_img(im_path+im, target_size=(224, 224))
        except:
            continue
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        layer_model = Model(inputs=model.input, outputs=model.layers[-1].output)
        features = layer_model.predict(x)
        train_data.append(features)
        label_data.append(i.split('_')[0])
        im_dict[im] = features
    np.save('gossipcop_image_vgg16_1000_feature.npy',im_dict)

def embedding_load(embedding_path):
    embedding_vector = {}
    f = open(embedding_path,'r',encoding='utf8')
    for line in tqdm(f):
        value = line.split(' ')
        word = value[0]
        coef = np.array(value[1:], dtype='float32')
        embedding_vector[word] = coef
    f.close()
    return embedding_vector

def text_embedding(text,embedding_vector):
    token = Tokenizer()
    token.fit_on_texts(text)
    tags_matrix = []
    zero_array = np.zeros(300)
    for word in text.strip().split(' '):
        if word in embedding_vector.keys():
            tag_embedding = embedding_vector[word]
            tags_matrix.append(np.array(tag_embedding))
            zero_array = zero_array + np.array(tag_embedding)
        else:
            continue
    tag_feature = zero_array / len(tags_matrix)
    return list(tag_feature)

def image_text_similarity_extraction(model,im_path):
    similarity_feature_dict = {}
    embedding_vector = embedding_load('../data/GoogleNews_vectors_negative_300d.txt')
    data_file = open('../data/gossipcop_news_data.txt', 'r', encoding='utf8')
    not_read_images = []
    for news in data_file.readlines():
        similarity_values = []
        print(news)
        news_id = news.split('\t')[0]
        title = news.split('\t')[1].replace('\n', '').strip()
        title_embedding = text_embedding(title, embedding_vector)
        print("title", title)
        im_list = news.split('\t')[-1].replace('\n', '').strip().split(' ') 
        for im in im_list:
            try:
                img = image.load_img(im_path+im, target_size=(224, 224))
            except:
                not_read_images.append(im)
                continue
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            yhat = model.predict(x)
            print(im)
            labels = decode_predictions(yhat, top=10)
            print(labels)
            words = ''
            for label in labels[0]:
                word = label[1]
                #print(word)
                words = words + word.replace('_',' ') + ' '
            tags_embedding = text_embedding(words, embedding_vector)
            text_im_similarity = cosine(title_embedding, tags_embedding)
            similarity_values.append(text_im_similarity)
        if len(similarity_values)==1:
            similarity_values.extend([0,0,0,0])
        elif len(similarity_values)==2:
            similarity_values.extend([0,0,0])
        elif len(similarity_values)==3:
            similarity_values.extend([0,0])
        elif len(similarity_values)==4:
            similarity_values.extend([0])
        elif len(similarity_values)==5:
            print(similarity_values)
        elif len(similarity_values)>5:
            similarity_values = similarity_values[:5]
        else:
            continue
        similarity_feature_dict[news_id]=similarity_values
    np.save('gossipcop_image_text_similarity_feature.npy',similarity_feature_dict)

if __name__ == "__main__":
    model = VGG16(weights='imagenet', include_top=True)
    print(model.summary())
    im_path = 'data/gossipcop_images/'
    image_feature_extraction(model, im_path)
    image_text_similarity_extraction(model,im_path)