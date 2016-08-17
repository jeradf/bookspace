from flask import Flask, render_template, request, jsonify
from gensim import matutils
import time
from collections import OrderedDict
import numpy as np
from numpy import prod, dot
import json
from gensim.models import Doc2Vec, Phrases
from gensim.matutils import unitvec
import cPickle as pickle
from bisect import bisect_left
import os
import sys
from book_search import title_search
from gensim.utils import any2unicode
import app_settings as settings
root = settings.root_path
home_dir = os.path.expanduser('~')
if sys.platform=='darwin':
    root = root.replace(home_dir, home_dir+'/Dropbox')

model = Doc2Vec.load(home_dir+"/model/corpra/1M_d2v_trained_with_review_docs_and_related_docs_2")
bigram = Phrases.load(root+'model/amz_bigram_aug6_130kbooks.p','rb')
book_data = pickle.load( open(root+"model/book_meta_data.p", "rb" ) )
title2asin = pickle.load( open(root+"model/title2asin_select.p", "rb" ) )

def get_book_data(asins):
    items = OrderedDict()
    for i, asin in enumerate(asins):
        if asin in book_data:
            book = book_data[asin]
            item = {'id': 'book_img' +str(i),
                    'img_link': book['imUrl'],
                    'buy_link': book['buy_link'],
                    'description': '' }        
            words = book['description'].split()
            item['description'] = ' '.join( words[:min(60, len(words))]) +  " ... "
            items[i] = item
            if len(items)>=42:
                break
    return items

def sim(query_book, pos_words, neg_words, topn=20):
    pos_vecs = []
    all_query_words = []
    if query_book in title2asin:
        all_query_words.append(query_book)
        all_query_words.append(title2asin[query_book])
        pos_vecs.append(model.docvecs[title2asin[query_book]])

    for word in bigram[pos_words.replace(',', ' ').lower().split()]:
        if word in model:            
            all_query_words.append( word )
            pos_vecs.append( model[word] )

    neg_vecs = []
    for word in bigram[neg_words.replace(',', ' ').lower().split()]:
        if word in model:
            all_query_words.append( word )
            neg_vecs.append( model[word] )

    if not pos_vecs:
        print "No positive vecs found. Book: %s\nPos_words: %s"%(query_book, pos_words)
        return None

    print "All query words: \n\t%s" %'\n\t'.join(all_query_words)
    asins, distances = zip(*model.docvecs.most_similar(positive=pos_vecs, 
                                                       negative=neg_vecs,
                                                       topn=100))
    asins = filter(lambda asin: asin not in all_query_words, asins)    
    return asins[:min(topn,len(asins)) ]


app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html', items={}, query={})

@app.route('/about/')
def about():
    return render_template('about.html')    

@app.route('/search/', methods=['GET'])
def get_similar():
    book = request.args.get('query_book', '', type=str)
    print 'Query book: "%s"'%book
    if book and book not in title2asin:
        print "Book not found"
        query_params, items = {}, {}
        return render_template('index.html', items=items, query=query_params)
    
    pos_words = request.args.get('plus', '', type=str)    
    neg_words = request.args.get('minus', '', type=str)    
    
    if len(pos_words)>1000 or len(neg_words)>1000:
        query_params,items = {}, {}
        return render_template('index.html', items=items, query=query_params)

    query_params = {'book':book, 'pos':pos_words, 'neg':neg_words}
    
    try:
        asins = sim(book, pos_words, neg_words, topn=52)
        items = get_book_data(asins) if asins else OrderedDict()
    except Exception as e:
        print "Exception in get_similar %s"%e
        query_params = {}
    return render_template('index.html', items=items, query=query_params)

@app.route('/suggest',methods=['GET'])
def suggest():
    query_string = str(request.args.get('term')).lower()
    matching_titles = title_search(query_string)
    if matching_titles:
        title_dict = [{'label':title} for title in matching_titles]
    else:
        title_dict = [{}]
    return json.dumps( title_dict )


if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=int("8000"),
        debug=True
    )

