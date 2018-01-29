import pymongo
import pandas as pd
import numpy as np
import nltk
import string
import re
import gensim
import time
import multiprocessing

def connect_to_mongo():
    with open('keys/mongoconnect.txt') as f:
        s = f.read()
    s = s[:-1]
    client = pymongo.MongoClient(s)
    db = client.get_database('reddit_capstone')
    return db

def yield_sentences(comment):
    sentence_tokens = nltk.sent_tokenize(comment)
    for s in sentence_tokens:
        s = re.sub('['+string.punctuation+']', '',s)
        s = s.replace('@','')
        s = s.replace('#','')
        s = s.replace('\n','')
        yield(nltk.word_tokenize(s.lower()))

def subreddit_sentences(db,subname):
    sub = db.posts.find({'subreddit':subname},{"data":1})
    for post in sub:
        for comment in post['data']['comments']:
            for s in yield_sentences(comment):
                yield s
        for s in yield_sentences(post['data']['title']):
            yield s

class Sub_Iterator():
    def __init__(self,list_of_subs):
        self.sub_list = list_of_subs
        self.db = connect_to_mongo()
    def __iter__(self):
        for sub in self.sub_list:
            sub_gen = subreddit_sentences(self.db,sub)
            for s in sub_gen:
                yield s

if __name__ == '__main__':
    db = connect_to_mongo()
    subs = db.posts.distinct('subreddit')
    n_cores = multiprocessing.cpu_count()
    print("Training word2vec on {} subreddits".format(len(subs)))
    t1 = time.time()
    sub_iter = Sub_Iterator(subs)
    model = gensim.models.Word2Vec(sub_iter,size = 300,window = 3, min_count = 10,workers = n_cores)
    sub_iter = Sub_Iterator(subs)
    t_words = model.corpus_count
    model.train(sub_iter,total_words = t_words,epochs = 10)
    model.save("w2vmodel")
    t2 = time.time()
    print("training completed, elapsed time: {}".format(t2 - t1))
