import numpy as np
import pandas as pd
import pymongo
import pickle
import time

from keras.layers import Embedding, Conv1D, MaxPooling1D ,Flatten, Dense, GlobalMaxPooling1D, Dropout, merge
from keras.preprocessing.text import Tokenizer
from keras.models import Model
from keras import Input


import tensorflow as tf
from keras import backend as K
from keras.layers.core import Lambda
from keras.layers import Input, merge
from keras.utils import multi_gpu_model
from keras import optimizers

from sklearn.metrics import f1_score,roc_auc_score

import train_word2vec
from project_utils import create_x_y, create_word_index_train_val, create_embedding_dict, create_embedding_layer, connect_to_mongo

def create_model(word_index,embedding_dict,EMBEDDING_DIM,MAX_SEQUENCE_LENGTH,NUM_CLASSES):
    print("1 conv layer no max pooling w dropout")
    embedding_layer = create_embedding_layer(word_index,embedding_dict,EMBEDDING_DIM,MAX_SEQUENCE_LENGTH)
    input_sequence = Input(shape = (MAX_SEQUENCE_LENGTH,),dtype = 'int32')
    embedded_sequences = embedding_layer(input_sequence)

    x = Conv1D(128, 5, activation='relu',name = "cv1")(embedded_sequences)
    x = Flatten()(x)
    x = Dropout(.3)(x)
    x = Dense(2048, activation='relu')(x)
    output = Dense(NUM_CLASSES, activation='softmax')(x)
    rmsop = optimizers.RMSprop(lr=0.005, rho=0.9, epsilon=None, decay=0.000002)
    model = Model(input_sequence,output)
    #model = multi_gpu_model(model,2)
    model.compile(loss='categorical_crossentropy',
                  optimizer=rmsop,
                  metrics=['acc'])
    return model

def create_model2(word_index,embedding_dict,EMBEDDING_DIM,MAX_SEQUENCE_LENGTH,NUM_CLASSES):
    print("two convolutional layers max pooling 5 inbetween w dropout")
    embedding_layer = create_embedding_layer(word_index,embedding_dict,EMBEDDING_DIM,MAX_SEQUENCE_LENGTH)
    input_sequence = Input(shape = (MAX_SEQUENCE_LENGTH,),dtype = 'int32')
    embedded_sequences = embedding_layer(input_sequence)
    x = Conv1D(128, 5, activation='relu',name = "cv1")(embedded_sequences)
    x = Dropout(.3)(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu',name = "cv2")(x)
    x = Flatten()(x)
    x = Dense(2048, activation='relu')(x)
    output = Dense(NUM_CLASSES, activation='softmax')(x)
    rmsop = optimizers.RMSprop(lr=0.005, rho=0.9, epsilon=None, decay=0.000002)
    model = Model(input_sequence,output)
    #model = multi_gpu_model(model,2)

    model.compile(loss='categorical_crossentropy',
                  optimizer=rmsop,
                  metrics=['acc'])
    return model

def create_model3(word_index,embedding_dict,EMBEDDING_DIM,MAX_SEQUENCE_LENGTH,NUM_CLASSES):
    print("two convolutional layers no max pooling inbetween w dropout")
    embedding_layer = create_embedding_layer(word_index,embedding_dict,EMBEDDING_DIM,MAX_SEQUENCE_LENGTH)
    input_sequence = Input(shape = (MAX_SEQUENCE_LENGTH,),dtype = 'int32')
    embedded_sequences = embedding_layer(input_sequence)
    x = Conv1D(128, 5, activation='relu',name = "cv1")(embedded_sequences)
    x = Dropout(.3)(x)
    x = Conv1D(128, 5, activation='relu',name = "cv2")(x)
    x = Flatten()(x)
    x = Dropout(.3)(x)
    x = Dense(2048, activation='relu')(x)
    output = Dense(NUM_CLASSES, activation='softmax')(x)
    rmsop = optimizers.RMSprop(lr=0.005, rho=0.9, epsilon=None, decay=0.000002)
    model = Model(input_sequence,output)
    #model = multi_gpu_model(model,2)
    model.compile(loss='categorical_crossentropy',
                  optimizer=rmsop,
                  metrics=['acc'])
    return model


def create_modelcurrent(word_index,embedding_dict,EMBEDDING_DIM,MAX_SEQUENCE_LENGTH,NUM_CLASSES):
    print("one conv layer with global max pooling")
    embedding_layer = create_embedding_layer(word_index,embedding_dict,EMBEDDING_DIM,MAX_SEQUENCE_LENGTH)
    input_sequence = Input(shape = (MAX_SEQUENCE_LENGTH,),dtype = 'int32')
    embedded_sequences = embedding_layer(input_sequence)
    x = Conv1D(128, 5, activation='relu',name = "cv1")(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(2048, activation='relu')(x)
    output = Dense(NUM_CLASSES, activation='softmax')(x)
    rmsop = optimizers.RMSprop(lr=0.005, rho=0.9, epsilon=None, decay=0.000002)
    model = Model(input_sequence,output)
    #model = multi_gpu_model(model,2)

    model.compile(loss='categorical_crossentropy',
                  optimizer=rmsop,
                  metrics=['acc'])
    return model


def create_confusion_matrix(y_true,predictions,sub_mapping):
    '''
    returns a pandas matrix of the confusion matrix for the parameters given
    '''
    prediction_classes = np.argmax(predictions,axis = 1)
    true_classes = np.argmax(y_true,axis = 1)
    num_classes = len(y_true[0])
    con_matrix = np.zeros((num_classes,num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            val = np.sum((prediction_classes == i) & (true_classes ==j))
            con_matrix[i,j] = val
    index = [sub_mapping[x] for x in range(num_classes)]
    con_table = pd.DataFrame(con_matrix,index = index,columns = index)
    row_summations = con_table.T.sum()
    print("columns are actual, rows are predicted")
    con_table['totals'] = row_summations
    col_sums = con_table.sum()
    con_table =con_table.T
    con_table['totals'] = col_sums
    print("f1 score, macro: ",f1_score(true_classes,prediction_classes,average = 'macro'))
    print("f1 score, weighted: ",f1_score(true_classes,prediction_classes,average='weighted'))
    print("f1 score, micro: ",f1_score(true_classes,prediction_classes,average = 'micro'))
    return con_table.T

def to_multi_gpu(model,n_gpus = 2):
    '''
    Takes a keras model and returns a model that can be parallellized on multiple gpus
    '''
    #code from user jonilaserson on github
    with tf.device('/cpu:0'):
        x = Input(model.input_shape[1:], name=model.input_names[0])
    towers = []
    for g in range(n_gpus):
        with tf.device('/gpu:' + str(g)):
            slice_g = Lambda(slice_batch, lambda shape: shape,
                                arguments={'n_gpus':n_gpus, 'part':g})(x)
            towers.append(model(slice_g))
    with tf.device('/cpu:0'):
        merged = merge(towers, mode='concat', concat_axis=0)
    return Model(input=[x], output=merged)


def slice_batch(x,n_gpus,part):
    #code from user jonilaserson on github
    '''
    helper function for multi gpu
    '''
    sh = K.shape(x)
    L = sh[0] / n_gpus
    if part == n_gpus - 1:
        return x[part*L:]
    return x[part*L:(part+1)*L]



if __name__ =='__main__':
    import datetime
    now = datetime.datetime.now()
    test_num = 1
    datestr = 'models/'+str(test_num)+'_'+str(now.month) +'_' + str(now.day)
    t1 = time.time()

    #get data for x and y from the given sub_list
    db = connect_to_mongo('reddit_capstone425')

    sub_list = pd.read_csv('sub_list.csv',header=None)[0].values.tolist()

    print("getting data from db")
    X,y,sub_dict = create_x_y(sub_list)

    with open(datestr+'m_1_dict.pkl','wb') as f:
        pickle.dump(sub_dict,f)

    with open('subreddit_class_weights.pkl','rb') as f:
        sub_weights = pickle.load(f)
    sub_to_ind = {v: k for k, v in sub_dict.items()}

    class_weights = {}
    for sub,weight in sub_weights.items():
        class_weights[sub_to_ind[sub]] = weight
    #create word index and training/validation data

    print("creating word index")
    word_index, X_train,X_val,y_train,y_val = create_word_index_train_val(X, y,
                                                                          MAX_SEQUENCE_LENGTH = 100,
                                                                          MAX_WORDS = 20000 )
    #gets embedding dict trains one if not use_GloVe
    #note glove always returns 300 len embedding atm
    print("creating embedding dict")
    embedding_dict = create_embedding_dict(sub_list,
                                           size = 300,
                                           epochs = 15,
                                          use_GloVe = True)

    #creates keras model for training
    print("creating model")

    modelcurrent = create_modelcurrent(word_index = word_index,
                          embedding_dict= embedding_dict,
                          EMBEDDING_DIM= 300,
                          MAX_SEQUENCE_LENGTH = 100,
                         NUM_CLASSES = len(y_train[0]))
    #fitting model

    modelcurrent.fit(X_train,y_train,batch_size=5000,epochs = 3,validation_data=(X_val,y_val),class_weight = class_weights)

    print("trying to pickle model")
    model2.save(datestr+'model.HDF5')

    with open(datestr+'subdict.pkl','wb') as f:
        pickle.dump(sub_dict,f)
    with open(datestr+'index.pkl','wb') as f:
            pickle.dump(word_index,f)
