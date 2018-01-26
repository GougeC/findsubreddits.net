import pandas as pd
import numpy as np
import nltk
import pymongo
import re
from collections import Counter
import string
from nltk.probability import FreqDist
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import re
from nltk.stem.porter import PorterStemmer
import pymongo
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
import tensorflow as tf
import time
import multiprocessing
import pickle
from multiprocessing import Process


def get_sub_term_freq_for_word2vec(subreddit,db):
    '''
    sub_term_freq except altered to include titles and selftext for the
    word2vec implementation. This leaves in stopwords
    '''
    #assigning stopwords from nltk.stopwords
    #getting posts from the database
    t1 = time.time()
    print('getting r/{}'.format(subreddit))
    posts = db.posts.find({'subreddit':subreddit},{'data.comments':1,'data.title':1,'data.selftext':1})
    sub_counter = Counter()
    #looping over the posts
    for p in posts:
        comments = p['data']['comments']
        title = p['data']['title']
        selftext = p['data']['selftext']
        #tokenizing each comment then counting the terms same for titles and selftext
        for com in comments:
            com = com.lower()
            com = re.sub('['+string.punctuation+']', '', com)
            for word in word_tokenize(com):
                sub_counter[word]+=1
        title = title.lower()
        title = re.sub('['+string.punctuation+']', '', title)
        for word in word_tokenize(title):
            sub_counter[word]+=1
        if selftext:
            selftext = selftext.lower()
            selftext = re.sub('['+string.punctuation+']', '', selftext)
            for word in word_tokenize(selftext):
                sub_counter[word]+=1
    t2 = time.time()
    print("finished r/{} in {} s".format(subreddit,t2-t1))
    return sub_counter
def clean_comment(com):
    com = com.lower()
    com = re.sub('['+string.punctuation+']', '', com)
    tokenized = word_tokenize(com)
    return tokenized
def map_to_numbers(db,subreddit,mapping):
    '''
    Takes in a mongodb instance, a subreddit name and a mapping and creates numpy vectors that
    can be used in a word2vec implementation by mapping each word to an integer, and words not in
    the vocabulary to "UNK"
    '''
    #gets data from mongodb
    posts = db.posts.find({'subreddit':subreddit},{'data.comments':1,'data.title':1,'data.selftext':1})
    datapoints = []
    #iterates over the comments title and selftext of every post in the subreddit in the database
    for p in posts:
        comments = p['data']['comments']
        title = p['data']['title']
        selftext = p['data']['selftext']
        comments.append(title)
        if selftext:
            comments.append(selftext)
        for com in comments:
            tokenized = clean_comment(com)
            vect = []
            #maps the tokenized vector to a numeric vector mapping words not in mapping to UNK
            for t in tokenized:
                if t in mapping:
                    vect.append(mapping[t])
                else:
                    vect.append(mapping['UNK'])
            datapoints.append(np.array(vect))

    return (subreddit,datapoints)

def map_one_sub(subreddit,mapping):
    db = connect_to_mongo()
    return map_to_numbers(db,subreddit,mapping)
def map_sub_list(lst,mapping,i,res_dict):
    db = connect_to_mongo()
    res = []
    for s in lst:
        res.append(map_to_numbers(db,s,mapping))
    res_dict[i] = res

def connect_to_mongo():
    client = pymongo.MongoClient('mongodb://ec2-54-214-228-72.us-west-2.compute.amazonaws.com:27017/')
    db = client.get_database('capstone_db')
    return db

def count_one_sub(sub):
    '''
    helper function to connect to db before calling get_sub_term_freq_for_word2vec
    '''
    db = connect_to_mongo()
    return get_sub_term_freq_for_word2vec(sub,db)

def label_datapoints(data):
    '''
    returns an list of numpy arrays of mapped datapoints and an array of matching labels, in order
    '''
    datapoints = []
    labels = []
    for sub, points in data:
        for p in points:
            datapoints.append(np.array(p))
            labels.append(sub)
    return datapoints,labels

    #creating the word mapping of the (number of words) most common words

def create_mapping(frequencies,number_of_words):
    '''
    Creates a word mapping dictionary based on the frequencies
    that are passed in. Creates a mapping for as many words as passed
    into number_of_words
    '''
    counts = [('UNK',-1)]
    counts.extend(frequencies.most_common(number_of_words - 1))
    word_mapping = {}
    print("creating word mapping")
    for word, c in counts:
        word_mapping[word] = len(word_mapping)

    return word_mapping

def map_subreddits_pool(all_subs,word_mapping):
    '''
    maps every subreddit using the sublist and the mapping passed in
    '''
    datapoints = []
    labels = []
    print("Beginning to label and transform subreddits")
    t1 = time.time()
    pool = multiprocessing.Pool(4)
    results = pool.map(map_one_sub,args = (all_subs,word_mapping))
    #turning these into numpy arrays
    t2 = time.time()
    print("mapping subs took {} seconds".format(t2-t1))
    return results
def map_subreddits_multiproc(all_subs,word_mapping):
    N = len(all_subs)
    k = N//4
    sub_lists = [all_subs[:k],all_subs[k:2*k],all_subs[k:3*k],all_subs[3*k:]]
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    procs = []
    for i in range(4):
        p = Process(target = map_sub_list,args = (sub_lists[i],word_mapping,i,return_dict))
        procs.append(p)
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    results = []
    for i in range(4):
        results.append(return_dict[i])
    return results
def prepare_for_word2vec(db, number_of_words,map_done = True):
    '''
    Prepares data from my database for word2vec. Takes in the database of subreddits and the
    number of words to include in the vocabulary for the mapping
    and returns a training set of numbered vectors and the mapping used to translate
    them to integers. Additionally this returns vector of labels to use to train a classifier
    on after training word2vec
    '''
    # of words to include in vocabulary for the word2vec implementation
    all_subs = db.posts.distinct('subreddit')
    if not map_done:
        t1 = time.time()
        N_subs = len(all_subs)
        p_counters = {}
        #counting all the words in the corpus
        pool = multiprocessing.Pool(4)
        results = pool.map(count_one_sub,all_subs)
        final_counter = Counter()
        for r in results:
            final_counter += r
        t2 = time.time()
        print("counting subs took {} seconds".format(t2-t1))
        print("creating word map...")
        word_mapping = create_mapping(final_counter, number_of_words)
        with open('wordmapping.pkl','wb') as f:
            pickle.dump(word_mapping,f)
        map
    if map_done:
        with open('wordmapping.pkl','rb') as f:
            word_mapping = pickle.load(f)
    data = map_subreddits_multiproc(all_subs,word_mapping)
    datapoints, labels = label_datapoints(data)
    return datapoints, labels, word_mapping
