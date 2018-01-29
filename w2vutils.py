import pymongo
import pandas as pd
import numpy as np
import nltk
import string
import re
import gensim
import time

def connect_to_mongo():
    '''
    Connects to a mongodb client using pymongo and connects to the 'reddit_capstone' db
    '''
    with open('keys/mongoconnect.txt') as f:
        s = f.read()
    s = s[:-1]
    client = pymongo.MongoClient(s)
    db = client.get_database('reddit_capstone')
    return db

def clean_and_tokenize(comment):
    '''
    Takes in a string of text and cleans it by removing punctuation and common symbols and then returns a
    list of word tokens
    '''
    c = re.sub('['+string.punctuation+']', '',comment)
    c = c.replace('@','')
    c = c.replace('#','')
    c = c.replace('\n','')
    c = nltk.word_tokenize(c.lower())
    return c


def get_process_comments(subname):
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
            points.append(clean_and_tokenize(comment))
        points.append(clean_and_tokenize(post['data']['title']))
    return points

def map_and_condense(data):
    '''
    This function maps a list of datapoints using the map_to_num function and then averages them.
    Returns a list of vectors of the dimension of the word embeddings
    '''
    datapoints = []
    for c in data:
        point = map_to_nums(c)
        if point is not None:
            point = np.mean(point, axis = 0)
            datapoints.append(point)
    return datapoints

def make_df_for_sub(sub_name):
    '''
    makes a pandas dataframe for the given sub to be used in modeling
    '''
    data = get_process_comments(sub_name)
    condensed = map_and_condense(data)
    dataframe = pd.DataFrame(condensed)
    dataframe['label'] = sub_name
    return dataframe

def prep_input_data_mean(data):
    '''
    maps one comment or text data point into a word embedding vector for modeling condenses the point by a
    simple mean
    '''
    point  = clean_and_tokenize(data)
    point = map_to_nums(point)
    point = point.mean(axis = 0)
    return point


def map_to_nums(comment):
    '''
    maps a single input point to a number using a word embedding named mapping. This assumes that the mapping
    emmbedding dictionary is global
    '''
    comm_array = []
    for word in comment:
        if word in vocab:
            comm_array.append(mapping[word])
    if comm_array:
        return np.array(comm_array)
    else:
        return None

def create_subreddit_vector(subreddit):
    '''
    takes an entire subreddit and maps it to a single vector just using a basic mean.
    '''
    df = make_df_for_sub(subreddit).drop('label',axis = 1)
    mat = df.values
    return mat.mean(axis = 0)
