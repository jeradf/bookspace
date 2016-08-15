from flask import Flask, render_template, request, jsonify
from gensim import matutils
import time
from collections import OrderedDict
import numpy as np
from numpy import prod, dot
import json
from gensim.models import Doc2Vec, Phrases
from gensim.matutils import unitvec
from amazon.api import AmazonAPI
import cPickle as pickle
from bisect import bisect_left
import os
import app_settings as settings
from flask import Markup
from book_search import search as title_search
from book_search import make_word2title_hash
import HTMLParser

def fix_html(s):
    return HTMLParser.HTMLParser().unescape(s)

root = settings.root_path

fn = os.path.join(os.path.expanduser('~'),
     "model/corpra/1M_d2v_trained_with_review_docs_and_related_docs_2")
model = Doc2Vec.load(fn)

title2asin = {fix_html(title):asin for title, asin in 
              pickle.load( open(root+"model/almost_all_title2asin.p", "rb" ) ).iteritems()
              if asin in model.docvecs}

asin2title = {asin:title for title, asin in title2asin.iteritems()}
titles  = sorted(title2asin.keys())
w2t = make_word2title_hash(titles)

# put lowered keys in too
for title in titles:
    title2asin[title.lower()] = title2asin[title]

titles_lowered = [t.lower() for t in titles]
titles_lowered_to_titles_upper = dict(zip(titles_lowered, titles))

amazon = AmazonAPI(settings.az_key_id, settings.az_pw_key, 'bookspace0d-20')

bigram = Phrases.load(root+'model/amz_bigram_aug6_130kbooks.p','rb')

def parse_amazon_item_info(product,i):
    try:
        item = {'id': 'book_img' +str(i),
                'img_link': product.large_image_url,
                'buy_link': product.offer_url,
                'description': '' }                
        if product.editorial_review and len(product.editorial_review)>1:
            text = product.editorial_review
            nwords = len(text.split())
            text = ' '.join(text.split()[:min(100, nwords)]) +  " ... "
            item['description'] = text
    
    except Exception as e:
        print "Exception while parsing amazon product: %s"%e
        return {}    
    return item

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
                                                       topn=5000))
    asins = filter(lambda asin: asin not in all_query_words, asins)    
    return asins[:min(topn,len(asins)) ]

# Initialize the Flask application
app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html', items={}, query={})


@app.route('/search/', methods=['GET'])
def get_similar():
    book = request.args.get('query_book', '', type=str)
    print 'Query book: "%s"'%book
    if book and book not in title2asin:
        print "Book not found"
        query_params,items = {},{}
        return render_template('index.html', items=items, query=query_params)
    
    pos_words = request.args.get('plus', '', type=str)
    neg_words = request.args.get('minus', '', type=str)    
    query_params = {'book':book, 'pos':pos_words, 'neg':neg_words}    
    items = OrderedDict()
    try:
        asins = sim(book, pos_words, neg_words, topn=42)
        if asins:
            for i in xrange(0, len(asins), 10):
                # Get amazon book details
                print "fetching product info; batch %d"%i
                amz_products = amazon.lookup(ItemId=','.join(asins[i:i+10]))                
                # Parse book details into dictionary 
                for ix, product in enumerate(amz_products):
                    items[i+ix] = parse_amazon_item_info(product, ix+i)

    except Exception as e:
        print "Exception in get_similar %s"%e
        query_params = {}
    return render_template('index.html', items=items, query=query_params)


@app.route('/suggest',methods=['GET'])
def suggest():
    search = str(request.args.get('term')).lower()
    ix = bisect_left(titles_lowered, search)

    suggested_titles = titles[ix:ix+10]

    suggested_titles += title_search(search,w2t)

    title_dict = []
    
    for word in suggested_titles:
        title_dict.append({'label':word })
    
    return json.dumps( title_dict )


if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=int("8000"),
        debug=True
    )

