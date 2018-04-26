import pymongo
import pandas as pd
import numpy as np
import nltk
import string
import re
import gensim
import time
import multiprocessing

DBNAME = 'reddit_capstone425'

def connect_to_mongo(dbname = DBNAME):
    with open('keys/mongoconnect.txt') as f:
        s = f.read()
    s = s[:-1]
    client = pymongo.MongoClient(s)
    db = client.get_database(dbname)
    return db

def yield_sentences(comment):
    sentence_tokens = nltk.sent_tokenize(comment)
    for s in sentence_tokens:
        s = re.sub('['+string.punctuation+']', '',s)
        s = s.replace('@','')
        s = s.replace('#','')
        s = s.replace('\n','')
        yield(nltk.word_tokenize(s.lower()))

def subreddit_sentences(db, subname):
    sub = db.posts.find({'subreddit':subname},{"data":1})
    for post in sub:
        for comment in post['data']['comments']:
            for s in yield_sentences(comment):
                yield s
        for s in yield_sentences(post['data']['title']):
            yield s

class Sub_Iterator():
    def __init__(self, list_of_subs):
        self.sub_list = list_of_subs
        self.db = connect_to_mongo()
    def __iter__(self):
        for sub in self.sub_list:
            sub_gen = subreddit_sentences(self.db,sub)
            for s in sub_gen:
                yield s

def train_word2vec(subs, size = 300, epochs = 10, min_count = 10):
    n_cores = multiprocessing.cpu_count()
    print("Training word2vec on {} subreddits".format(len(subs)))
    t1 = time.time()
    sub_iter = Sub_Iterator(subs)
    model = gensim.models.Word2Vec(sub_iter, size = size, window = 3, min_count = min_count, workers = n_cores)
    sub_iter = Sub_Iterator(subs)
    t_words = model.corpus_count
    model.train(sub_iter,total_words = t_words,epochs = epochs)
    t2 = time.time()
    print("training completed, elapsed time: {}".format(t2 - t1))
    return model

if __name__ == '__main__':
    db = connect_to_mongo()
    subs = db.posts.distinct('subreddit')

    model = train_word2vec(subs)
    model.save("w2vmodel")
