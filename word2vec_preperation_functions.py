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

def get_sub_term_freq_for_word2vec(subreddit,db):
    '''
    sub_term_freq except altered to include titles and selftext for the
    word2vec implementation. This leaves in stopwords
    '''
    #assigning stopwords from nltk.stopwords
    #getting posts from the database

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
    return sub_counter

def map_to_numbers(db,subreddit, mapping):
    '''
    Takes in a mongodb instance a subreddit name and a mapping and creates numpy vectors that
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
            com = com.lower()
            com = re.sub('['+string.punctuation+']', '', com)
            tokenized = word_tokenize(com)
            vect = []
            #maps the tokenized vector to a numeric vector mapping words not in mapping to UNK
            for t in tokenized:
                if t in mapping:
                    vect.append(mapping[t])
                else:
                    vect.append(mapping['UNK'])
            datapoints.append(np.array(vect))

    return datapoints


def prepare_for_word2vec(db):
    '''
    Prepares data from my database for word2vec. Takes in the database of subreddits
    and returns a training set of numbered vectors and the mapping used to translate
    them to integers. Additionally this returns vector of labels to use to train a classifier
    on after training word2vec
    '''
    # of words to include in vocabulary for the word2vec implementation
    number_of_words = 50000

    all_subs = db.posts.distinct('subreddit')
    N_subs = len(all_subs)
    p_counters = {}
    #counting all the words in the corpus
    i = 0
    num_processes = 4
    n_per_p = N_subs//num_processes
    result_q = multiprocessing.Queue()
    jobs = []
    t1 = time.time()
    def count_list_of_subs(list_of_subs,result_queue):
        client = pymongo.MongoClient('mongodb://ec2-54-214-228-72.us-west-2.compute.amazonaws.com:27017/')
        db = client.get_database('capstone_db')
        total_freqs = Counter()
        for sub in list_of_subs:
            cntr = get_sub_term_freq_for_word2vec(sub,db)
            total_freqs+=cntr
            print("Finished Counting for {}".format(sub))
        result_queue.put(total_freqs)

    for i in range(num_processes - 1):
        proc_list = all_subs[i*n_per_p:(i+1)*n_per_p]
        p = multiprocessing.Process(target = count_list_of_subs(proc_list,result_q))
        jobs.append(p)

    proc_list = all_subs[(num_processes-1)*n_per_p:]
    p = multiprocessing.Process(target = count_list_of_subs(proc_list,result_q))
    jobs.append(p)

    for job in jobs: job.start()
    for job in jobs: job.join()

    final_counter = Counter()
    while not result_q.empty():
        final_counter+=result_q.get()
    total_freqs = final_counter

    t2 = time.time()
    print("counting subs took {} seconds".format(t2-t1))

    #creating the word mapping of the (number of words) most common words
    counts = [('UNK',-1)]
    counts.extend(total_freqs.most_common(number_of_words - 1))
    word_mapping = {}
    print("creating word mapping")
    for word, c in counts:
        word_mapping[word] = len(word_mapping)

    #labeling and mapping the training data
    datapoints = []
    labels = []
    print("Beginning to label and transform subreddits")
    i=0
    t3 = time.time()
    for sub in all_subs:
        i+=1
        if i%20==0:
            print("mapping and labeling r/{}. {}/618 subs".format(sub,i))
            t2 = time.time()
            print("total time elapsed: {}".format(t2 - t1))
        s_data = map_to_numbers(db,sub,word_mapping)
        for d in s_data:
            datapoints.append(d)
            labels.append(sub)
    #turning these into numpy arrays
    t2 = time.time()
    print("mapping subs took {} seconds".format(t2-t3))
    datapoints = np.array(datapoints)
    labels = np.array(labels)
    print("total time elapsed: {}".format(t2 - t1))
    return datapoints, labels, word_mapping
