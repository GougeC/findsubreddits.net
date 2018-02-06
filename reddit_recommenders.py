import pandas as pd
import numpy as np

from scipy.spatial.distance import cdist
from collections import Counter

import keras
import simple_but_tough as sbt
import w2vutils as wvu
import pickle

import requests
from bs4 import BeautifulSoup
import re


from keras.preprocessing.text import Tokenizer,text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Sequential
from keras import Input
from keras.utils import to_categorical

## This class
class CNN_reddit_recommender():
    def __init__(self,model_path,sub_map_path,word_ind_path):
        """
        creates the model from a model file, a sub_maping and a word_index
        """
        self.cnn = keras.models.load_model(model_path)

        self.sub_mapping = pickle.load(open(sub_map_path,'rb'))

        self.word_index = pickle.load(open(word_ind_path,'rb'))

    def predict_on_text(self,text,num_pred=5,as_link = True):
        '''
        gets prediction based on some text copy pasted in
        '''
        prepped = self.prep_for_model(text)
        pred = self.cnn.predict(prepped)
        results = {}
        for i,sub in self.sub_mapping.items():
            results[sub] = np.sum(pred[:,i])
        rdf = pd.DataFrame.from_dict(results,orient='index')

        recs = rdf.sort_values(by =0,ascending = False).index[:num_pred]
        if as_link:
            return as_urls(recs)
        return recs

    def predict_on_list_handles(self,handle_list,num_pred = 5):
        '''
        takes in a python list of twitter handles and gets num_pred predictions for the last 25 tweets
        of each of the accouts (puts all of the text together and makes a recommendation)
        '''
        handle_list = handle_list.split(',')
        handle_list = [a.strip() for a in handle_list]
        handle_list = [a for a in handle_list if a != '']
        data = []
        for handle in handle_list:
            if handle[0] == '@':
                data.extend(get_tweets(handle[1:],25))
            else:
                data.extend(get_tweets(handle,25))
        texts = '\n'.join(data)

        return self.predict_on_text(texts,num_pred)
    def predict_on_twitter(self,twitter_handle,num_pred = 5):
        """
        gets prediction with data from a twitter handle
        """
        return self.predict_on_text("\n".join(get_tweets(twitter_handle,25)), num_pred)

    def prep_for_model(self,text):
        """
        prepares text data for the cnn model
        """
        paragraphs = text.split('\n')
        sequences = [text_to_word_sequence(p,
                                      filters='!"#$%&()“*’’+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                      lower=True, split=" ") for p in paragraphs]
        sequences = [x for x in sequences if x != []]
        mapped_sequences = []
        for point in sequences:
            point = [self.word_index[word] for word in point if word in self.word_index]
            mapped_sequences.append(point)
        padded_sequence = pad_sequences(mapped_sequences,100)
        return padded_sequence

## This was the first draft recommender I built which is based on
## an implementation of the following paper: https://openreview.net/pdf?id=SyK00v5xx
## It is slower and has more variance in recommendations than the subsequent CNN models
class SBT_Recommender():

    def __init__(self, corpus_count_path, sub_info_path):
        """
        for now this object is initialized given paths to the files
        for a sub_info_dict and a corpus counter object
        """
        corpus_counts = pickle.load(open(corpus_count_path,'rb'))
        self.frequencies = sbt.Frequency_Map()
        self.frequencies.update_with_counter(corpus_counts)
        self.sub_info_dict = pickle.load(open(sub_info_path,'rb'))
        self.subtoind = {}
        self.indtosub = {}
        self.sub_matrix = []
        for s, data in self.sub_info_dict.items():
            self.subtoind[s] = data['subind']
            self.indtosub[data['subind']] = s
            self.sub_matrix.append(data['sub_vector'])
        self.sub_matrix = np.array(self.sub_matrix)

    def recommend_from_text(self,text,number_recommendations = 5,link = True):
        '''
        input: a string of text(copy pasted article etc) and an integer number of recommendations that
        you want.
        output: a list of subreddits in string form or if link = True links to these subreddits
        '''
        if type(text) != str or text == '':
            return 'this function takes a string'
        #splits the text into paragraphs based on line skips
        #(this is temporary and can be improved)
        paragraphs = [x for x in text.split('\n') if x != '']
        vectors = sbt.sentence_to_vec(paragraphs,self.frequencies)
        vector = np.array(vectors).mean(axis = 0).reshape(1,-1)
        closest = cdist(vector,self.sub_matrix,metric = 'cosine').argsort()
        results = closest[0][0:number_recommendations]
        results = [self.indtosub[x] for x in results]
        if link:
            return as_urls(results)
        return results




def clean_twitter_data(text_data):
    """
    input: a list of strings that are presumably tweets
    returns: those same strings with urls @s and hashtags removed
    """
    clean_tweets = []
    try:
    # UCS-4
        emoj = re.compile(u'([\U00002600-\U000027BF])|([\U0001f300-\U0001f64F])|([\U0001f680-\U0001f6FF])')
    except re.error:
    # UCS-2
        emoj = re.compile(u'([\u2600-\u27BF])|([\uD83C][\uDF00-\uDFFF])|([\uD83D][\uDC00-\uDE4F])|([\uD83D][\uDE80-\uDEFF])')

    for text in text_data:
        tweet = re.sub('(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?'
                      , '', text)
        tweet = re.sub(emoj, "",tweet)
        tweet = re.sub(r"/[U0001F601-U0001F64F]/u", "",tweet)

        tweet = tweet.lower()
        tweet = tweet.replace('@','')
        tweet = tweet.replace('#','')
        tweet = tweet.replace('\n','')
        tweet = re.sub(r"http\S+", "",tweet)
        tweet = re.sub(r"pic.twitter.com\S+", "",tweet)
        tweet = re.sub(r"/[U0001F601-U0001F64F]/u", "",tweet)
        clean_tweets.append(tweet)
    return clean_tweets

def scrape_user_tweets(screen_name,num_tweets = 10):
    '''
    get up to num_tweets tweets from the provided users handle

    '''
    req = requests.get("https://twitter.com/"+screen_name)
    soup = BeautifulSoup(req.text,"lxml")
    timeline = soup.find_all("div",id = 'timeline')[0]
    tweets = []
    for i in range(num_tweets):

        try:
            tweet = timeline.find_all(class_="TweetTextSize")[i].text
        except:
            break
        tweets.append(tweet)
    return tweets

def get_tweets(screen_name,num_tweets = 10):
    '''
    combines two other functions for ease of use
    '''
    tweets = scrape_user_tweets(screen_name,num_tweets)
    return clean_twitter_data(tweets)

def as_urls(subs):
    '''
    converts sub names to links to the actual subreddits
    '''
    results = ['r/'+a for a in subs]
    return results
