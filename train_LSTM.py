import keras
import pandas as pd
import numpy as np
import time
import pickle

from keras.models import Sequential
from keras.layers import LSTM, Dense ,Dropout,GRU,Recurrent

import train_convnet as tc
import project_utils as utils

if __name__ == '__main__':
    import datetime
    now = datetime.datetime.now()
    test_num = 2
    datestr = 'models/RNN'+str(test_num)+'_'+str(now.month) +'_' + str(now.day)
    sub_list = pd.read_csv('sub_list.csv',header=None)[0].values.tolist()
    print("fitting {} subs".format(len(sub_list)))
    X,y,sub_dict = tc.create_x_y(sub_list)

    with open(datestr+'m_1_dict.pkl','wb') as f:
        pickle.dump(sub_dict,f)
    with open('subreddit_class_weights.pkl','rb') as f:
        sub_weights = pickle.load(f)
    sub_to_ind = {v:k for k,v in sub_dict.items()}
    class_weights = {}
    for sub,weight in sub_weights.items():
        class_weights[sub_to_ind[sub]] = weight
    t1 = time.time()

    word_index, X_train,X_val,y_train,y_val = utils.create_word_index_train_val(X,y,
                                                                             MAX_SEQUENCE_LENGTH = 100,
                                                                             MAX_WORDS=20000,
                                                                             test_size = 100000)

    embedding_dict = utils.create_embedding_dict(sub_list,
                                            size = 300,
                                            epochs = 15,
                                            use_GloVe = True)


    embedding_layer = utils.create_embedding_layer(word_index,embedding_dict,300,100)
    t2 = time.time()

    print("prepping to fit model took: {} minutes".format((t2-t1)/60))
    t1 = time.time()

    model = Sequential()
    model.add(embedding_layer)
    model.add(LSTM(1000, return_sequences=False, activation='softmax'))
    model.add(Dense(len(sub_list), activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer = 'rmsprop', metrics = ['acc'])
    model.fit(X_train, y_train, epochs = 20,  class_weight=class_weights,batch_size= 1000,validation_data=(X_val,y_val))

    cf = utils.create_confusion_matrix(y_val, model.predict(X_val),sub_dict)

    print(cf)
    t2 = time.time()

    print("Time to train network with GloVe embeddings: {} minutes".format((t2-t1)/60))
    #evaluate model


    model.save(datestr+'m_1_model.HDF5')
    with open(datestr+'m_1_index.pkl','wb') as f:
        pickle.dump(word_index,f)
    with open(datestr+'m_1_subdict.pkl','wb') as f:
        pickle.dump(sub_dict,f)

    t1 = time.time()

    model = Sequential()
    model.add(embedding_layer)
    model.add(LSTM(1000, return_sequences=False, activation='softmax'))
    model.add(Dense(1000,activation = 'relu'))
    model.add(Dense(len(sub_list), activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer = 'rmsprop', metrics = ['acc'])
    model.fit(X_train, y_train, epochs = 20, class_weight=class_weights,batch_size= 1000,validation_data=(X_val,y_val))

    cf = tc.create_confusion_matrix(y_val, model.predict(X_val),sub_dict)
    print(cf)
    t2 = time.time()

    print("Time to train network with GloVe embeddings: {} minutes".format((t2-t1)/60))
    #evaluate model


    model.save(datestr+'m_2_model.HDF5')
    with open(datestr+'m_2_index.pkl','wb') as f:
        pickle.dump(word_index,f)
    with open(datestr+'m_2_subdict.pkl','wb') as f:
        pickle.dump(sub_dict,f)

    t1 = time.time()

    model = Sequential()
    model.add(embedding_layer)
    model.add(GRU(1000, return_sequences=False, activation='softmax'))
    model.add(Dense(len(sub_list), activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer = 'rmsprop', metrics = ['acc'])
    model.fit(X_train, y_train, epochs = 20, class_weight=class_weights,batch_size= 1000,validation_data=(X_val,y_val))

    cf = tc.create_confusion_matrix(y_val, model.predict(X_val),sub_dict)
    print(cf)
    t2 = time.time()

    print("Time to train network with GloVe embeddings: {} minutes".format((t2-t1)/60))
    #evaluate model


    model.save(datestr+'m_3_model.HDF5')
    with open(datestr+'m_3_index.pkl','wb') as f:
        pickle.dump(word_index,f)
    with open(datestr+'m_3_subdict.pkl','wb') as f:
        pickle.dump(sub_dict,f)
