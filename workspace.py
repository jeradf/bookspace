from pprint import pprint
import gzip
import cPickle as pickle
import os
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec, Phrases
from gensim.utils import simple_preprocess
# import app_settings as settings
# root = settings.ROOT_PATH.replace('latest','corpra/')
root = '/Users/jerad/model/corpra/'
import numpy as np
# from numpy import linalg as LA
# %matplotlib inline
import codecs
import matplotlib.pyplot as plt
# import seaborn as sns
from numpy import dot,prod,array
from gensim import matutils
import logging
import pandas as pd
from gensim import utils
from random import shuffle
from pandas import notnull,isnull

logging.root.handlers = []  # Jupyter messes up logging so needs a reset
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

fn = root+'books_complete_metadata_and_wcount.p'
nw = pd.read_pickle(fn)

nw = nw[notnull(nw['title'])]
nw.sort_values('rank',inplace=True)
q = '(rank<2000000 and N>4 and nwords>2500)'
q +=' or (N>20 and nwords>2500)'
q+= ' or (nwords>5000 and N>10)'

nw2 = nw.query(q).copy(deep=True)
print len(nw2)

from __future__ import division


# del nw2['t_len']
nw2['wcount'] = [m.docvecs.doctags[asin].word_count if asin in m.docvecs else np.nan 
                for asin in nw2.asin.tolist()]
nw2 = nw2[notnull(nw2.wcount)]
nw2 = nw2.query('wcount>=2000');print len(nw2)

round1 = lambda val: int(val*10)/10
round1(3.33)
a2r = {asin:round1(rating) for asin,rating in nw2['asin overall'.split()].values}
# a2wcount = {a:wcount for a,wcount in nw2['asin wcount'.split()].values}
title2asin = {title:asin for title,asin in nw2['title asin'.split()].values}
pickle.dump(title2asin, open('/Users/jerad/Dropbox/my_app/model/title2asin.p','wb'))
pickle.dump(title2asin, open('/Users/jerad/Dropbox/my_app/model/title2asin.p','wb'))


nw2.to_pickle(root+'select_books_metadata_and_wcount.p')

# bg = Phrases.load(root+'amz_bigram_aug6_130kbooks.p')





class IterDocs(object):
    def __init__(self, fnames, dirname):
        self.dirname = dirname
        self.file_names = fnames
    
    def __iter__(self):
        for fname in self.file_names:
            with codecs.open(self.dirname+fname,'rb',encoding='utf8') as infile:
                words = infile.read().split()
            
            if len(words)<2000:
                continue

            words = words[:min(8000,len(words)-1)]
            tag = utils.to_unicode(fname[:-4])

            yield LabeledSentence(words,[tag])
                        

    def to_array(self):
        self.sentences = []        
        for fname in self.file_names:
            with codecs.open(self.dirname+fname,'rb',encoding='utf8') as infile:
                words = infile.read().split()
            words = words[:min(8000,len(words)-1)]
            tag = utils.to_unicode(fname[:-4])             
            self.sentences.append( LabeledSentence(words,[tag]) )
        return self.sentences

    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences


dirname = root+"review_files3_bg/clean_bg/"
train_asins = set([asin for asin in nw2.asin.tolist() if a2r[asin]>=3.5 ])

fnames = [fname for fname in os.listdir(dirname) 
           if fname.endswith('.txt') and fname[:-4] in train_asins]
shuffle(fnames)
print len(fnames)

# print len(train_asins)
docs = IterDocs(fnames, dirname)



m = Doc2Vec(dm=1, min_count=100, 
            window=4, size=400, workers=16,
            sample=1e-5)

# m.build_vocab(docs)
m.train(docs)

def s(matches):
    asins = zip(*matches)[0]
    return [a2t[asin] for asin in asins]

# v = [m.docvecs[t2a[book]]]
# print bg[words.split()]
# v = [m[w] for w in bg[words.split()] if w in m]
def book_words(title, model):    
    print "Query: %s"%title
    v = model.docvecs[t2a[title]]
    pprint(model.most_similar(positive=[v],topn=10));print
    book_matches = [(a2t[x[0]], a2r[x[0]],x[1]) for x in 
                     model.docvecs.most_similar(positive=[v],topn=30)]
    pprint(book_matches)
    

def word_books(word, model): 
    print "Query: %s"%word
    if type(word)!=list:
        word = bg[word.split()]
    v = array([model[w] for w in word]).sum(axis=0)
    pprint(model.most_similar(positive=[v],topn=10));print
    book_matches = [(a2t[x[0]],a2r[x[0]], x[1]) for x in 
                     model.docvecs.most_similar(positive=[v],topn=30)]
    pprint(book_matches)
# nw2.wcount.hist()
from numpy.linalg import norm
nw2['vlen'] = [norm(m.docvecs[asin]) for asin in nw2.asin.tolist()]
# search('secret history',2)
# b = 'The Goldfinch: A Novel (Pulitzer Prize for Fiction)'
# b = 'The Goldfinch: A Novel (Pulitzer Prize for Fiction)'
# bg['machine learning'.split()]
# book_words(b,m)
# word_books(' scifi',m2)





