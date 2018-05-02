import pymongo
import pandas as pd
import numpy as np
import nltk
import string
import re
import gensim
import time

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Embedding


##This is a bunch of helper functions for the Reddit2vec project
DBNAME = "reddit_capstone425"

def connect_to_mongo(dbname = DBNAME):
    '''
    Connects to a mongodb client using pymongo and connects to the 'reddit_capstone' db
    '''
    with open('keys/mongoconnect.txt') as f:
        s = f.read()
    s = s[:-1]
    client = pymongo.MongoClient(s)
    db = client.get_database(dbname)
    return db

def clean_and_tokenize(comment, filter_stopwords = False, cap_at_100 = False):
    '''
    Takes in a string of text and cleans it by removing punctuation and common symbols and then returns a
    list of word tokens
    '''
    if type(comment)==str:
        try:
            c = re.sub('['+string.punctuation+']', '',comment)
        except:
            c = comment
        c = c.replace('@','')
        c = c.replace('#','')
        c = c.replace('\n','')
        c = nltk.word_tokenize(c.lower())
    else:
        return None
    if not filter_stopwords:
        if cap_at_100:
            return c[:100]
        return c
    else:
        from nltk.corpus import stopwords
        stopWords = set(stopwords.words('english'))
        if cap_at_100:
            return [word for word in c if word not in stopWords][:100]
        return [word for word in c if word not in stopWords]


def get_process_comments(subname,filter_stopwords = False):
    '''
    gets the data for the parameter subreddit from my mongodb and then cleans and tokenizes the comments and titles
    for the subreddit.
    Returns a list of comments/titles that have been transformed into a list of word tokens
    '''
    db = connect_to_mongo()
    sub_data = db.posts.find({'subreddit':subname},{'data':1})
    points = []
    for post in sub_data:
        for comment in post['data']['comments']:
            points.append(clean_and_tokenize(comment,filter_stopwords))
        points.append(clean_and_tokenize(post['data']['title'],filter_stopwords))
    return points

def map_and_condense(data,mapping):
    '''
    This function maps a list of datapoints using the map_to_num function and then averages them.
    Returns a list of vectors of the dimension of the word embeddings
    '''
    datapoints = []
    for c in data:
        point = map_to_nums(c,mapping)
        if point is not None:
            point = np.mean(point, axis = 0)
            datapoints.append(point)
    return datapoints

def make_df_for_sub(sub_name,mapping):
    '''
    makes a pandas dataframe for the given sub to be used in modeling
    '''
    data = get_process_comments(sub_name)
    condensed = map_and_condense(data,mapping)
    dataframe = pd.DataFrame(condensed)
    dataframe['label'] = sub_name
    return dataframe

def prep_input_data_mean(data,mapping):
    '''
    maps one comment or text data point into a word embedding vector for modeling condenses the point by a
    simple mean
    '''
    point  = clean_and_tokenize(data)
    point = map_to_nums(point,mapping)
    point = point.mean(axis = 0)
    return point


def map_to_nums(comment,mapping):
    '''
    maps a single input point to a number using a word embedding dictionary from the parameter mapping
    '''
    comm_array = []
    for word in comment:
        if word in mapping:
            comm_array.append(mapping[word])
    if comm_array:
        return np.array(comm_array)
    else:
        return None

def create_subreddit_vector(subreddit,mapping):
    '''
    takes an entire subreddit and maps it to a single vector just using a basic mean.
    '''
    df = make_df_for_sub(subreddit).drop('label',axis = 1)
    mat = df.values
    return mat.mean(axis = 0)

def process_embeddings(path):
    '''
    reads embeddings from GloVe from a text file and saves them as a dictionary
    '''
    mapping = {}
    with open(path) as f:
        for line in f.readlines():
            line = line.split()
            key = line[0]
            val = np.array(list(map(float,line[1:])))
            mapping[key] = val
    return mapping

def map_no_condense(subreddit,mapping):
    '''
    creates a mapping of all the text data in a subreddit without
    condensing it in any way
    '''
    data = get_process_comments(subreddit,filter_stopwords=True)
    mapped = [map_to_nums(x,mapping) for x in data]
    return mapped

def get_sub_raw(subname):
    '''
    gets just the raw text comments and titles from a sub
    '''
    db = connect_to_mongo()
    sub_data = db.posts.find({'subreddit':subname},{'data':1})
    points = []
    for post in sub_data:
        for comment in post['data']['comments']:
            points.append(comment)
        points.append(post['data']['title'])
    return points

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
        data = get_sub_raw(sub)
        for point in data:
            X.append(point)
            y.append(ind)
        sub_dict[ind] = sub
        ind+=1
    return X,y,sub_dict

def create_word_index_train_val(X,y,MAX_WORDS,MAX_SEQUENCE_LENGTH,test_size = 100000):
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
    t1 = time.time()
    tokenizer.fit_on_texts(X)
    t2 = time.time()

    print("done. time : {} seconds".format(t2-t1))
    s = tokenizer.texts_to_sequences(X)

    print("padding sequences")
    word_index = tokenizer.word_index
    data = pad_sequences(sequences = s, maxlen = MAX_SEQUENCE_LENGTH)

    print("processing data and splitting into train_test")
    labels = to_categorical(np.array(y))
    inds = np.arange(len(data))
    np.random.shuffle(inds)

    test_inds = inds[:test_size]
    train_inds = inds[test_size:]

    X_train, X_val, y_train, y_val = data[train_inds], data[test_inds], labels[train_inds], labels[test_inds]
    return word_index, X_train, X_val, y_train, y_val

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
        embedding_dict = process_embeddings('embeddings/glove.6B.300d.txt')
        return embedding_dict
    w2vmodel = train_word2vec.train_word2vec(sub_list,size = 100, epochs = 15)
    embedding_dict = {}
    for word in w2vmodel.wv.vocab:
        embedding_dict[word] = w2vmodel.wv.word_vec(word)
    return embedding_dict
