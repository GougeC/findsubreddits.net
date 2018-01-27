import pandas as pd
import numpy as np
import nltk
import pymongo
import re
from collections import Counter
import string
from nltk import word_tokenize
import tensorflow as tf
import time
import random
from multiprocessing import Pool
import multiprocessing
from keras.preprocessing.sequence import pad_sequences,make_sampling_table,skipgrams
from keras.layers import Embedding, Dense, Reshape, merge
from keras.models import Model
import pickle
import word2vec_preperation_functions as w2vp

def get_featurized_word(word,word_map,feature_mat):
    if word in word_map:
        return feature_map[word_map[word],:]
    else:
        return feature_map[word_map['UNK'],:]
def get_features_from_num(num,reverse_dictionary,feature_mat):
    if num in reverse_dictionary:
        return feature_mat[num,:]
    else:
        return feature_mat[0,:]
def make_input_datapoints_multi(datapoints,sub_labels,vocab_size):
    number_cores = multiprocessing.cpu_count()
    data_labels = zip(datapoints,sub_labels)
    global sampling_table
    sampling_table = sampling_table = make_sampling_table(vocab_size+1)
    proc_pool = Pool(number_cores)
    input_datapoints = proc_pool.map(create_input_point,data_labels)
    return input_datapoints

def create_input_point(point):
    global sampling_table
    datapoint , sub = point
    couples,labels = skipgrams(datapoint,vocab_size, window_size=3, sampling_table=sampling_table)
    word_target, word_context = zip(*couples)
    word_target = np.array(word_target,dtype = "int32")
    word_context = np.array(word_context,dtype = "int32")
    input_datapoint = {'word_target':word_target,
                            'word_context':word_context,
                            'labels': labels}
    return input_datapoint

if __name__ == '__main__':
    with open('keys/mongoconnect.txt') as f:
        s = f.read()
    s = s[:-1]
    client = pymongo.MongoClient(s)
    db = client.get_database('capstone_db')
    vocab_size = 10000

    #create mapped data from web text
    datapoints, sub_labels, word_mapping = w2vp.prepare_for_word2vec(db,vocab_size,True)
    validating = False
    reverse_dictionary = dict(zip(word_mapping.values(), word_mapping.keys()))
    word_mapping['NONCE'] = vocab_size
    reverse_dictionary[vocab_size] = 'NONCE'

    vocab_size = len(word_mapping)
    window_size = 3
    vector_dimension = 300
    epochs = 1000
    validation_size = 16
    validation_window = 100
    validation_examples = np.random.choice(validation_window, validation_size, replace=False)
#    with open('wordmapping1.pkl','wb') as f:
#        pickle.dump(word_mapping,f)
#    with open('sub_labels.pkl','wb')as f:
#        pickle.dump(sub_labels,f)
    with open('datapoints.pkl','wb')as f:
        pickle.dump(datapoints,f)


    num_samples = len(datapoints)
    print(len(datapoints),"number datapoints")
    print(datapoints[0])
    print(len(datapoints[0]),"number inside one datapoint")
    max_length = 100
    print(vocab_size,"vocab size")
    print(list(word_mapping.keys())[-1], "last key in the wordmap")
    datapoints = pad_sequences(datapoints, maxlen = max_length, dtype = 'int32',
                                     padding = 'post', truncating = 'post', value = vocab_size)
    print(len(datapoints),"number of datapoints")
    print(datapoints[0].shape)
    # making skipgram training pairs to train the word embedding
    t1 = time.time()
    input_datapoints = make_input_datapoints_multi(datapoints,sub_labels,vocab_size)
    t2 = time.time()
    print("input data took {} minutes".format((t2-t1)/60))
    print("making input data worked!")
    with open('input_datapoints.pkl','wb') as f:
        pickle.dump(input_datapoints,f)
    #creating the wordembedding network
    input_target = keras.Input((1,))
    input_context = keras.Input((1,))
    embedding = Embedding(vocab_size,vector_dimension,input_length =1, name = 'word_embedding')
    target = embedding(input_target)
    target = Reshape((vector_dimension,1))(target)
    context = embedding(input_context)
    context = Reshape((vector_dimension,1))(target)

    dot_product = merge([target, context], mode='dot', dot_axes=1)
    dot_product = Reshape((1,))(dot_product)
    output = Dense(1,activation = 'sigmoid')(dot_product)
    similarity = merge([target, context], mode='cos', dot_axes=0)

    model = Model(input=[input_target, input_context], output=output)
    model.compile(loss='binary_crossentropy', optimizer='adam')

    validation_model = Model(input = [input_target,input_context], output=similarity)
    # a class for call back for validating mid training
    class SimilarityCallback:
        def run_sim(self):
            for i in range(valid_size):
                validation_word = reverse_dictionary[valid_examples[i]]
                sim = self._get_sim(datapoints[validation_example[j]])
                top_k = 5 #number of nearest to display
                nearest = (-sim).argsort()[1:top_k+1]
                log_str = "Nearest to {}: ".format(validation_word)
                for k in range(top_k):
                    close = reverse_dictionary[nearest[k]]
                    log_str+= "{}, ".format(close)
                print(log_str)
            def _get_sim(v_w_idx):
                sim = np.zeros((vocab_size,))
                v_w = np.zeros((1,))
                t_w = np.zeros((1,))
                for i in range(vocab_size):
                    v_w = v_w_idx
                    t_w = i
                    out = validation_model.predict_on_batch([v_w, t_w])
                    sim[i] = out
                return sim

    sim_cb = SimilarityCallback()
    #training the network
    #this makes sure to train the network on every sample
    w_target = np.zeros((1,))
    w_context = np.zeros((1,))
    lbls = np.zeros((1,))
    loops = 1000
    cnt = 0
    for i in range(loops):
        for s in range(num_samples):
            sample = input_datapoints[s]
            idx = np.random.randint(0,len(sample['labels']))
            w_target[0,] = sample['word_target'][idx]
            w_context[0,] = sample['word_context'][idx]
            lbls[0,] = sample['labels'][idx]
            loss = model.train_on_batch([w_target,w_context],lbls)
            cnt+=1
            if cnt%100 == 0:
                print("Interation {}, loss = {}".format(cnt,loss))
            if validating == True:
                if cnt%1000 == 0:
                    sim_sib.run_sim()

    featurized = model.layers[0].get_weights()
    print(featurized.shape)
    np.save('vectorized_words.npy',featurized)
    with open('wordmapping.pkl','wb') as f:
        pickle.dump(word_mapping,f)
    with open('vector_words.pkl','wb')as f:
        pickle.dump(featurized,f)



    X = []
    for point in datapoints:
        x_point = []
        for num in datapoints:
            x_point.append(get_features_from_num(num,reverse_dictionary,featurized))
        x_point = np.array(x_point)
        x_point = np.mean(x_point,axis = 0)
        X.append(x_point)
    Xdf = pd.DataFrame(X)
    ydf = pd.Series(sub_labels)
    training_data = Xdf
    training_data['labels'] = ydf
    training_data.to_csv('training_data.csv',index = False)
