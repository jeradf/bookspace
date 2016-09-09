from __future__ import division
from pprint import pprint
import cPickle as pickle
import os
import app_settings as settings
import sys

import time
from collections import defaultdict, Counter, OrderedDict
import string 
import re
import numpy as np
from numpy import prod, dot
from gensim.models import Doc2Vec, Phrases
root = settings.root_path
big_file_dir = os.path.expanduser('~')+'/model/corpra/'
if sys.platform=='darwin':
    root = root.replace(os.path.expanduser('~'),
                        os.path.expanduser('~')+'/Dropbox')

########################################################################
# Find nearest neighbors in product space
#######################################################################
model = Doc2Vec.load(root+"model/movie_space/idf_reddit")
bigram = Phrases.load(big_file_dir+'movies_bigram_large.p','rb')
book_data = pickle.load( open(root+"model/movie_space/book_meta_data.p", "rb" ) )
title2asin = pickle.load( open(root+"model/movie_space/title2asin.p", "rb" ) )

def get_similar(query_book, pos_words, neg_words, topn=100):
    try:
        pos_vecs = []
        all_query_words = []
        for book in query_book:
            if book in title2asin:
                print "\tFound book: ", title2asin[book]            
                all_query_words.append(title2asin[book])
                pos_vecs.append(model.docvecs[title2asin[book]])

        for word in bigram[pos_words.replace(',', ' ').lower().split()]:
            if word in model:
                print "\tPositive word: %s"%word
                all_query_words.append( word )
                pos_vecs.append( model[word] )

        neg_vecs = []
        for word in bigram[neg_words.replace(',', ' ').lower().split()]:
            if word in model:
                print "\tNegative word: %s"%word
                all_query_words.append( word )
                neg_vecs.append( model[word] )

        if not pos_vecs:
            print "No positive vecs found. Book: %s\nPos_words: %s"%(query_book, pos_words)
            return OrderedDict()
        else:
            print "\tGot a total of %d pos_vecs and %d neg_vecs"%(len(pos_vecs),len(neg_vecs))

        print "\tAll query words:\n\t %s"%('\n\t'.join(all_query_words))
        asins, distances = zip(*model.docvecs.most_similar(positive=pos_vecs, 
                                                           negative=neg_vecs,
                                                           topn=300))
        asins = filter(lambda asin: asin not in all_query_words and asin in book_data ,
                        asins)

        items = get_book_data(asins[:min(100,len(asins))],
                              topn=100) if asins else OrderedDict()
        print "got %d items"%len(items)
    except Exception as e:
        print "Exception in sim:\n\t%s"%e
        return None
    return items

def movie_string(data):
    try:
        s = u'Title: {title}\nIMDB Rating: {rating:.1f}\n'    
        s += u'Cast: {actors}\nPlot: {plot:.120s}...'
        s = s.format(**data).replace('\n','<br>')
    except Exception as e:
        print "exception at in movie_string: %s"%e
        return ""
    return s

def get_book_data(asins,topn):
    items = OrderedDict()
    for i, asin in enumerate(asins):
        if asin in book_data:
            book = book_data[asin]
            if book['rating']>6.5:
                item = {'id': 'book_img' +str(i),
                        'img_link': 'posters/'+asin+".jpg",
                        'buy_link': book['buy_link'],
                        'description': movie_string(book)}
                words = book['description'].split()
                # item['description'] = ' '.join( words[:min(60, len(words))]) +  " ... "
                items[i] = item
            if len(items)>=topn:
                break
    return items

########################################################################
# Movie title autocomplete functions
########################################################################

title_data = pickle.load( open(root+"model/movie_space/title_data.p", "rb" ) )
titles = sorted(title_data.keys())

RE_PUNCT_ALL = re.compile('([%s])+' % re.escape(string.punctuation ), re.UNICODE)

def t2w(t):
    return RE_PUNCT_ALL.sub(" ", t.lower() ).split()

def make_word2title_hash(titles):
    w2t = defaultdict(list)
    for title in titles:
        for word in t2w(title):
            w2t[word].append(title)
    return {w:list(set(t)) for w,t in w2t.iteritems()}

w2t = make_word2title_hash(titles)

def match_score(words_in_query, title, title_data=title_data):
    words_in_title = title_data[title]['title_words']
    common_words = len(words_in_query.intersection(words_in_title))
    total_words = len(words_in_title.union(words_in_query))
    n_reviews = title_data[title]['num_reviews']
    score = n_reviews * int(100*(common_words/total_words))/100
    return score

def title_search(query):
    words_in_query = set(t2w(query))
    matches = []
    for word in words_in_query:
        if word in w2t:
            matches.extend(w2t[word])

    if not matches:
        return None
    
    matches = Counter(matches)    
    N = max(matches.values())
    matched_titles = [title for title, count in matches.iteritems()
                      if count==N]                
    # Score each candidate by fraction of common words in all
    # words times the number of reviews
    scores = [match_score(words_in_query,title) 
              for title in matched_titles]                
    
    scored_titles = sorted( zip(matched_titles, scores),
                           key=lambda item: item[1],
                           reverse=True);        
    
    scored_titles = scored_titles[:min(len(scored_titles),20)]
    suggestions = list(zip(*scored_titles)[0])
    return suggestions
