import numpy as np
from sklearn.decomposition import PCA
from collections import Counter
#helper function
from w2vutils import clean_and_tokenize, process_embeddings

## this code is an adapted version of https://github.com/peter3125/sentence2vec/blob/master/sentence2vec.py
## by peter3125 on github (Peter de Vocht)
## this is an implementation of the following paper: https://openreview.net/pdf?id=SyK00v5xx

#object for a word with associated vector
class Word:
    def __init__(self, text, vector):
        self.text = text
        self.vector = vector
#object for a sentence, a list of words
class Sentence:
    def __init__(self, word_list):
        self.word_list = word_list

    # return the length of a sentence
    def len(self):
        return len(self.word_list)

class Frequency_Map:
    """
    This class is essentially a counter but it returns the frequency relative to the total word count instead of a count.
    It can also be set to always return 0 for ease of coding in the sentence_to_vec function
    """
    ## this assumes the corpus is a list of of strings
    def __init__(self):
        self.counts = Counter()
        self.total_words = 0
        self.disabled = False

    def update_map(self,corpus):
        for string in corpus:
            for word in clean_and_tokenize(string,False):
                self.counts[word]+=1
                self.total_words+=1
    #returns the frequency of a word
    def get_word_frequency(self,word_text):
        if not self.disabled:
            return self.counts[word_text]/self.total_words
        else:
            return 0.0
    def update_with_counter(self,counter):
        self.counts += counter
        self.total_words += sum(counter.values())

    def return_only_ones(self):
        #this will make the counter return 1 for everything
        self.disabled = True

def prep_text_for_stv(comments):
    '''
    Prepares text as objects for the sentence_to_vec function
    '''
    mapping = process_embeddings('embeddings/glove.6B.100d.txt')
    ##this assumes all the text is in a list of strings, and in the interest of my project
    ## this defines a "sentence" as an entire comment
    #frequencies = Frequency_Map()
    #frequencies.update_map(comments)
    object_comments = []
    for comment in comments:
        com_list = []
        for word in clean_and_tokenize(comment,True,True):
            if word in mapping:
                com_list.append(Word(word,mapping[word]))
        if com_list:
            object_comments.append(Sentence(com_list))
    return object_comments



# A SIMPLE BUT TOUGH TO BEAT BASELINE FOR SENTENCE EMBEDDINGS
# Sanjeev Arora, Yingyu Liang, Tengyu Ma
# Princeton University
def sentence_to_vec(comments,frequencies,embedding_size = 100, a =1e-3,use_frequencies = True):
    """
    This sums the word embedding vectors in a sentence (or the block of tokens specified as a sentence) and then subtracts the first principal component
    of the sentence vector to create a vector that can be used for cosine similarities between sentences
    """
    sentence_list = prep_text_for_stv(comments)
    if not use_frequencies:
        frequencies.return_only_ones()
    sentence_set = []
    for sentence in sentence_list:
        vs = np.zeros(embedding_size)  # add all word2vec values into one vector for the sentence
        sentence_length = sentence.len()
        for word in sentence.word_list:
            a_value = a / (a + frequencies.get_word_frequency(word.text))  # smooth inverse frequency, SIF
            vs = np.add(vs, np.multiply(a_value, word.vector))  # vs += sif * word_vector

        vs = np.divide(vs, sentence_length)  # weighted average
        sentence_set.append(vs)  # add to our existing re-calculated set of sentences

    # calculate PCA of this sentence set
    pca = PCA(n_components=embedding_size)
    pca.fit(np.array(sentence_set))
    u = pca.components_[0]  # the PCA vector
    u = np.multiply(u, np.transpose(u))  # u x uT

    # pad the vector?  (occurs if we have less sentences than embeddings_size)
    if len(u) < embedding_size:
        for i in range(embedding_size - len(u)):
            u = np.append(u, 0)  # add needed extension for multiplication below

    # resulting sentence vectors, vs = vs -u x uT x vs
    sentence_vecs = []
    for vs in sentence_set:
        sub = np.multiply(u,vs)
        sentence_vecs.append(np.subtract(vs, sub))

    return sentence_vecs
