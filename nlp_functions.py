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

def get_sub_term_freq(subreddit,db):
    '''
    Returns a counter of all the terms in a given subreddit's comments.
    Everything is lowercased and punctuation is removed
    returns a Counter object
    takes in a string sub name and a database to pull info from
    '''
    #assigning stopwords from nltk.stopwords
    stopWords = set(stopwords.words('english'))
    #getting posts from the database
    posts = db.posts.find({'subreddit':subreddit},{'data.comments':1})
    sub_counter = Counter()
    #looping over the posts
    for p in posts:
        comments = p['data']['comments']
        #tokenizing each comment then counting the terms
        for com in comments:
            com = com.lower()
            com = re.sub('['+string.punctuation+']', '', com)
            for word in word_tokenize(com):
                if word not in stopWords:
                    sub_counter[word]+=1
    return sub_counter


def get_tf_idf_subreddits(db):
    '''
    This returns tf-idf vectors for each subreddit, treating all of
    the comments from each subreddit as one document.
    takes in a link to the mongodb server on the ec2 that
    I have pulling reddit data
    '''
    #declaring variables and pulling data from the subreddit
    all_subs = db.posts.distinct('subreddit')
    N_subs = len(all_subs)
    sub_counters = {}
    total_freqs = Counter()

    for sub in all_subs:
        cntr = get_sub_term_freq(sub,db)
        total_freqs+=cntr
        sub_counters[sub] = cntr

    n_terms = len(total_freqs)
    tf_vector = np.zeros(n_terms)
    word_lookup = {}
    index_lookup = {}
    total_words = sum(total_freqs.values())

    #create total corpus frequency vector
    for index, element in enumerate(total_freqs.most_common()):
        frequency = counter[element]
        tf_vector[index] = frequency
        word_lookup[element] = index
        index_lookup[index] = element

    subreddit_vectors = {}
    sublookup = {}
    #create a matrix of zeros to update as we loop over subs
    raw_count_matrix = np.zeros((N_subs,n_terms))
    #looping over each subreddit and updating zeromatrix to its raw count
    for index,sub,scnter in enumerate(sub_counters.items()):
        raw_count_vec = np.zeros(n_words)
        sublookup[sub] = index
        for term,freq in scnter.most_common():
            ind = word_lookup[word]
            raw_count_vec[ind] = freq
            raw_count_matrix[index,ind] = freq
        subreddit_vectors[sub]= {'raw_count':raw_count_vec}
    #convert raw count matrix into tf-idf matrix
    n,m = raw_count_matrix.shape
    tf_idf_matrix = np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            #commented out code is for tf where tf = doc_count/length_doc
                #tf = (raw_count_matrix[i,j])/(np.sum(raw_count_matrix(i,:)))

            #instead I will use 1 + log(raw_count) as the tf term for now
            #this will be set to zero if the word is not in the document
            rc = raw_count_matrix[i,j]
            tf = 0 if rc == 0 else (1 + np.log(rc))
            n_t = np.sum(raw_count_matrix[:,j] > 0)
            #idf here is defined as "smooth" idf with the 1 to avoid
            #division by zero
            idf = np.log(1 + (N_subs/n_t))
            #assigning the tf_idf to its entry in the matrix
            tf_idf_matrix[i,j] = tf*idf

    #assigning information to dictionaries so that it can be found easily
    for sub in all_subs:
        subreddit_vectors[sub]['tfidf'] = tf_idf_matrix[sublookup[sub,:]]

    return_info = {'tfidf_matrix':tf_idf_matrix,
                   'subreddit_lookup_d':sublookup,
                   'word_to_index_d': word_lookup,
                   'index_to_word_d':index_lookup,
                   'raw_count_matrix':raw_count_matrix,
                   'subreddit_to_vec':subreddit_vectors}

    return return_info
