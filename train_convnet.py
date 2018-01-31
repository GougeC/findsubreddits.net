import numpy as np
import pandas as pd
import pymongo
from keras.layers import Embedding,Conv1D,MaxPooling1D,Flatten,Dense,GlobalMaxPooling1D
from keras.preprocessing.text import Tokenizer,text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Sequential
from keras import Input
from keras.utils import to_categorical
import w2vutils

import train_word2vec

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,roc_auc_score

import time

def create_x_y(sublist):
    '''
    takes in a list of subreddits and gets the raw data from the subs. It then adds them all
    to the same array and adds the subreddit as a numerical label.
    Returns: a list of strings, a list of numbers and a dictionary to map those numbers to which subreddit
    they represent
    '''
    sub_dict = {}
    ind = 0
    X = []
    y = []
    for sub in sublist:
        data = w2vutils.get_sub_raw(sub)
        for point in data:
            X.append(point)
            y.append(ind)
        sub_dict[ind] = sub
        ind+=1
    return X,y,sub_dict

def create_word_index_train_val(X,y,MAX_WORDS,MAX_SEQUENCE_LENGTH):
    '''
    RETURNS : word_index, X_train,X_val,y_train,y_val
    takes in an X and y from the make_x_y function and creates a word_index dictionary and padds each X value
    to the MAX_WORDS parameter (or truncates it) it then splits the set into train test sets
    '''
    tokenizer = Tokenizer(num_words = MAX_WORDS,
                     filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                     lower = True,
                     split = " ",
                     char_level = False)
    print("fitting tokenizer")
    tokenizer.fit_on_texts(X)
    s = tokenizer.texts_to_sequences(X)
    print("padding sequences")
    word_index = tokenizer.word_index
    data = pad_sequences(sequences = s, maxlen = MAX_SEQUENCE_LENGTH)
    print("processing data and splitting into train_test")
    labels = to_categorical(np.array(y))
    inds = np.random.shuffle(np.arange(len(data)))
    test_inds = inds[:10000]
    train_inds = inds[10000:]
    X_train, X_val, y_train,y_val = data[train_inds],data[test_inds],labels[train_inds],labels[test_inds]
    return word_index, X_train,X_val,y_train,y_val

def create_embedding_matrix(word_index,embedding_dict,EMBEDDING_DIM):
    '''
    given a word index, embedding dict and the length of the embeddings this makes a numpy matrix
    that can be used to make a keras embedding layer
    '''
    embedding_matrix = np.zeros((len(word_index)+1,EMBEDDING_DIM))
    for word, ind in word_index.items():
        if word in embedding_dict:
            embedding_vector = embedding_dict[word]
            embedding_matrix[ind] = embedding_vector
    return embedding_matrix


def create_embedding_layer(word_index,embedding_dict,EMBEDDING_DIM,MAX_SEQUENCE_LENGTH):
    '''
    creates an embedding layer for a keras model
    '''
    embedding_matrix = create_embedding_matrix(word_index,embedding_dict,EMBEDDING_DIM)
    embedding_layer = Embedding(len(word_index)+1,EMBEDDING_DIM,
                            weights = [embedding_matrix],
                           input_length = MAX_SEQUENCE_LENGTH,
                           trainable = False)
    return embedding_layer

def create_embedding_dict(sublist,size,epochs,use_GloVe = False):
    '''
    trains a word embedding with gensim and returns a dictionary
    if use_GloVe it returns the 300 length GloVe embeddings
    '''
    if use_GloVe:
        embedding_dict = w2vutils.process_embeddings('embeddings/glove.6B.300d.txt')
        return embedding_dict
    w2vmodel = train_word2vec.train_word2vec(sub_list,size = 100, epochs = 15)
    embedding_dict = {}
    for word in w2vmodel.wv.vocab:
        embedding_dict[word] = w2vmodel.wv.word_vec(word)
    return embedding_dict


def create_model(word_index,embedding_dict,EMBEDDING_DIM,MAX_SEQUENCE_LENGTH,NUM_CLASSES):

    embedding_layer = create_embedding_layer(word_index,embedding_dict,EMBEDDING_DIM,MAX_SEQUENCE_LENGTH)
    input_sequence = Input(shape = (MAX_SEQUENCE_LENGTH,),dtype = 'int32')
    embedded_sequences = embedding_layer(input_sequence)
    x = Conv1D(128, 3, activation='relu',name = "cv1")(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 3, activation='relu',name = "cv2")(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 3, activation='relu',name = "cv3")(x)
    x = GlobalMaxPooling1D()(x)

    x = Dense(128, activation='relu')(x)
    output = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(input_sequence,output)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
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


if __name__ =='__main__':
    t1 = time.time()
    #get data for x and y from the given sub_list
    db = w2vutils.connect_to_mongo()
    sub_list = db.posts.distinct('subreddit')
    print("getting data from db")

    X,y,sub_dict = create_x_y(sub_list)

    #create word index and training/validation data
    print("creating word index")
    word_index, X_train,X_val,y_train,y_val = create_word_index_train_val(X,y,
                                                                          MAX_SEQUENCE_LENGTH = 100,
                                                                          MAX_WORDS=10000)
    #gets embedding dict trains one if not use_GloVe
    #note glove always returns 300 len embedding atm
    print("creating embedding dict")
    embedding_dict = create_embedding_dict(sub_list,
                                           size = 300,
                                           epochs = 15,
                                          use_GloVe = True)

    #creates keras model for training
    print("creating model")
    model = create_model(word_index = word_index,
                          embedding_dict= embedding_dict,
                          EMBEDDING_DIM= 300,
                          MAX_SEQUENCE_LENGTH = 100,
                         NUM_CLASSES = len(y_train[0]))

    #fitting model
    model.fit(X_train,y_train,batch_size=100,epochs = 10,validation_data=(X_val,y_val))

    t2 = time.time()

    print("Time to train network with GloVe embeddings: {} minutes".format((t2-t1)/60))
    #evaluate model
    preds = model.predict_on_batch(X_val)
    confusion_matrix = make_confusion_matrix(y_val,preds,sub_indexs)
    #print(confusion_matrix)
    t1 = time.time()
    #get data for x and y from the given sub_list
    print("getting data from the db")
    X,y,sub_dict = create_x_y(sub_list)

    #create word index and training/validation data
    print("creating word index")
    word_index, X_train,X_val,y_train,y_val = create_word_index_train_val(X,y,
                                                                          MAX_SEQUENCE_LENGTH = 100,
                                                                          MAX_WORDS=10000)
    #gets embedding dict trains one if not use_GloVe
    #note glove always returns 300 len embedding atm
    print("creating embedding dict")
    embedding_dict = create_embedding_dict(sub_list,
                                           size = 300,
                                           epochs = 15,
                                          use_GloVe = False)

    #creates keras model for training
    print("creating model")
    model2 = create_model(word_index = word_index,
                          embedding_dict= embedding_dict,
                          EMBEDDING_DIM= 300,
                          MAX_SEQUENCE_LENGTH = 100,
                         NUM_CLASSES = len(y_train[0]))

    #fitting model
    model2.fit(X_train,y_train,batch_size=100,epochs = 10,validation_data=(X_val,y_val))

    t2 = time.time()

    print("Time to train word2vec and network: {} minutes".format((t2-t1)/60))
    #evaluate model
    preds2 = model2.predict_on_batch(X_val)
    confusion_matrix = make_confusion_matrix(y_val,preds2,sub_indexs)
    #print(confusion_matrix)


    print("trying to pickle models")
    import pickle
    with open('gensim_model.pkl','wb') as f:
        pickle.dump(model2,f)
    with open('glove_model.pkl','wb') as f:
        pickle.dump(model,f)
