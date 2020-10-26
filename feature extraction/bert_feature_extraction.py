#tensorflow==2.0

import tensorflow_hub as hub
import tensorflow as tf
import bert

from tensorflow.keras.models import Model       # Keras is the new high level API for TensorFlow


max_seq_length = 64
def get_masks(tokens, max_seq_length):
    """Mask for padding"""
    if len(tokens)>max_seq_length:
        raise IndexError("Token length more than max seq length!")
    return [1]*len(tokens) + [0] * (max_seq_length - len(tokens))


def get_segments(tokens, max_seq_length):
    """Segments: 0 for the first sequence, 1 for the second"""
    if len(tokens)>max_seq_length:
        raise IndexError("Token length more than max seq length!")
    segments = []
    current_segment_id = 0
    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            current_segment_id = 1
    return segments + [0] * (max_seq_length - len(tokens))


def get_ids(tokens, tokenizer, max_seq_length):
    """Token ids from Tokenizer vocab"""
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = token_ids + [0] * (max_seq_length-len(token_ids))
    return input_ids

def model_bulid():
    input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                           name="input_word_ids")
    input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                       name="input_mask")
    segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                        name="segment_ids")
    bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1", trainable=False)
    pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=[pooled_output, sequence_output])
    return bert_layer, model

if __name__ == "__main__":
    max_seq_length = 64
    bert_layer, model = model_bulid()

    FullTokenizer = bert.bert_tokenization.FullTokenizer
    vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = FullTokenizer(vocabulary_file, to_lower_case)

    test_file = 'gossipcop_news_file.txt'
    data = open(test_file,'r',encoding='utf8')
    feature_dict={}
    for line in data.readlines():
        id = line.split('\t')[0]
        label = line.split('\t')[0].split('_')[0]
        title = line.split('\t')[1]
        stokens = tokenizer.tokenize(title)
        stokens = ["[CLS]"] + stokens + ["[SEP]"]

        input_ids = get_ids(stokens, tokenizer, max_seq_length)
        input_masks = get_masks(stokens, max_seq_length)
        input_segments = get_segments(stokens, max_seq_length)

        pool_embs, all_embs = model.predict([[input_ids],[input_masks],[input_segments]])

        feature_dict[id]=pool_embs
    np.save('bert_feature.npy',feature_dict)
