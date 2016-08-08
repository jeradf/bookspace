from flask import Flask, render_template, request, jsonify
from gensim import matutils
import time
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
from book_search import fuzzy_search
root = '/home/ubuntu/bookspace'
## Load Vocab list, i.e. all words and all book titles

title2asin = pickle.load( open(root+"/model/title2asin.p", "rb" ) )
asin2title = {asin:title for title,asin in title2asin.iteritems()}

titles  = sorted(title2asin.keys())

# put lowered keys in too
for title in titles:
    title2asin[title.lower()] = title2asin[title]


titles_lowered = [t.lower() for t in titles]
titles_lowered_to_titles_upper = dict(zip(titles_lowered, titles))

amazon = AmazonAPI(settings.az_key_id,
                    settings.az_pw_key,
                    'bookspace0d-20')


# model = Doc2Vec.load(root+'/model/amz_130k_aug7')
model = Doc2Vec.load(root+'/model/amz_130k_aug8_dbow+w,d300,n3,w8,mc15,s1e-06,t16')

bigram = Phrases.load(root+'/model/amz_bigram_aug6_130kbooks.p','rb')


def get_vecs(words):
    v = []
    for word in words:
        if word in model.docvecs:
            v.append(model.docvecs[word])
        else:
            if isinstance(word,list):
                word = word[0]
            w = bigram[word.lower().split()]
            for item in w:
                if item in model.vocab:
                    v.append(model[item])
    return v

def sim(pos=[''],neg=[''], model=model,  topn=20):
    if isinstance(pos, unicode) or isinstance(pos, str):
        pos = [pos]
    all_words = pos+neg
    pos_vecs = get_vecs(pos)
    neg_vecs = get_vecs(neg)
    # pos_dists = [((1 + dot(model.docvecs.doctag_syn0, term)) / 2) for term in pos_vecs]
    # neg_dists = [((1 + dot(model.docvecs.doctag_syn0, term)) / 2) for term in neg_vecs]

    # dists = prod(pos_dists, axis=0) / (prod(neg_dists, axis=0) + 0.000001)
    # best = matutils.argsort(dists, topn=topn + len(all_words), reverse=True)
    # if best.any():
    #     print "best",best[0]
    
    # matches = [(model.docvecs.offset2doctag[sim], float(dists[sim])) for sim in best][:max(0,topn)]
    
    matches = model.docvecs.most_similar(positive=pos_vecs, 
                                         negative=neg_vecs,
                                         topn=20)
    matches = [x[0] for x in matches if x[0] not in all_words]
    if matches:
        print 'mathces', matches[0]
    return matches


# Initialize the Flask application
app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/_add_numbers')
def get_similar():
    a = request.args.get('a', type=str)
    b = request.args.get('b', type=str)
    c = request.args.get('c', type=str)
    
    print 'Query book: "%s"'%a
    
    try:
        a = title2asin[a] if a in title2asin else ''
        print 'Query book mapped to: "%s"'%a
        
        if len(b)==0:
            b = ''
        else:
            b = b.replace(',',' ')
            print "Positive words: %s"%b

        if len(c)==0:
            c = ''
        else:
            c = c.replace(',',' ')
            print "Negative words: %s"%c

        
        asins = sim(pos=[a, b], neg=[c], topn=20)
        # asins = [tag2asin[tag[0]] for tag in matches
        #         if tag[0] in tag2asin]
        
        batches = [ asins[:10], asins[10:] ]
        
        prods = []
        for batch in batches:
            prods += amazon.lookup(ItemId=','.join(batch))
            time.sleep(.1)        
        print len(prods)

        results = [{'result': product.title,
                    'img_link': product.large_image_url,
                    'buy_link':product.offer_url,
                    'description': product.editorial_review[:200] if product.editorial_review and len(product.editorial_review)>0 else ''}
                   for product in prods]
    except Exception as e:
        print "Exception in get_similar %s"%e
        return 'bye'
    return json.dumps(results)



@app.route('/suggest',methods=['GET'])
def suggest():
    search = str(request.args.get('term')).lower()
    ix = bisect_left(titles_lowered, search)

    suggested_titles = titles[ix:ix+10]

    suggested_titles += fuzzy_search(search)

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

