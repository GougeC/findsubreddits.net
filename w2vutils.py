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
        if cap_at_25:
            return c[:100]
        return c
    else:
        from nltk.corpus import stopwords
        stopWords = set(stopwords.words('english'))
        if cap_at_25:
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
