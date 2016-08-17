from pprint import pprint
import gzip
import cPickle as pickle
import os
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec, Phrases
from gensim.utils import simple_preprocess
from gensim.utils import any2unicode
import app_settings as settings
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



root = settings.root_path

fn = os.path.join(os.path.expanduser('~'),
     "model/corpra/1M_d2v_trained_with_review_docs_and_related_docs_2")
model = Doc2Vec.load(fn)


root = "/Users/jerad/model/corpra"
import HTMLParser

def fix_html(s):
    return HTMLParser.HTMLParser().unescape(s)

# fn = root+'/books_complete_metadata_and_wcount.p'
# nw = pd.read_pickle(fn)

fn = root+'/books_meta_data_df.p'
df = pd.read_pickle(fn)

fn = root+"/books_complete_metadata_and_wcount.p"
df2 = pd.read_pickle(fn)

root = settings.root_path.replace('jerad','jerad/Dropbox')
# t2a = {fix_html(title):asin for title, asin in 
#               pickle.load( open(root+"model/almost_all_title2asin.p", "rb" ) ).iteritems()
#               if asin in model.docvecs}

# asins = t2a.values()
asins = model.docvecs.doctags.keys()
ix = df.asin.isin(asins)

print len(df[ix])
df = df[ix]

asins = model.docvecs.doctags.keys()
ix = df2.asin.isin(asins)
print len(df2[ix])
df2 = df2[ix]


# Make data structures for title search 
asins = model.docvecs.doctags.keys()
df2 = df2.set_index('asin').loc[asins].query('N>=10')
asins = set(asins)
a2t = df2['title'].to_dict();print len(a2t)
# a2t = {a:t for t,a in df2[keep_df]['title asin'.split()].values};print len(a2t)
t2a = {t:a for t,a in 
    df2[pd.notnull(df2.title)].query('N>=10').reset_index()['title asin'.split()].values}

pickle.dump(title_data, open(root+'model/title_data.p','wb'))

title2numreviews = {t:n for t,n in 
                    df2[pd.notnull(df2.title)].query('N>=10')['title N'.split()].values }
title_data = {}
for title in titles:
    title_data[title] = {'num_reviews': title2numreviews[title],
                         'asin': t2a[title],
                         'title_words': set(t2w(title))}


title_data = {fix_html(t):d for t, d in title_data.iteritems()}

from book_search import title_search
title_search

query = 'atlas shrugged'
query = 'pride and prejudice zombies'
query = '1984'
query = 'python'
query = 'on the road'
query = 'war and peace'
query = 'the communist manifesto'
query = 'communist manifesto'
query = 'selfish gene'
query = "Freakonomics"
query = "harry "
query = 'fifty'
# query = "swallows and Amazons"
t = title_search(query); pprint(t[:10])




# df[ix].head()
t2a = {t:a for t,a in df['title asin'.split()].values}
a2t = {a:t for t,a in df['title asin'.split()].values}

df['description'] = [fix_html(s)
                    if type(s) in (str,unicode) else ""
                    for s in df.description.tolist() ]




book_meta_data = df.set_index('asin')['imUrl title description'.split()].T.to_dict()


for asin in book_meta_data.keys():
    book_meta_data[asin]['buy_link'] = "https://www.amazon.com/dp/%s/?tag=bookspace0d-20"%asin



pickle.dump(book_meta_data,open('model/book_meta_data.p','wb'))


book_data = pickle.load( open(root+"model/book_meta_data.p", "rb" ) )


cols = list(df.columns.values)
cols.remove('description')
cols.remove('categories')
cols.remove('imUrl')
cols.remove('price')
cols.remove('brand')


fn = root+'/books_meta_data_df_without_descriptions.p'
df[cols].to_pickle(fn)

# df = df[notnull(df.description)]

# t2a = pickle.load( open(os.getcwd()+"/model/title2asin.p", "rb" ) )
# df = df[df.asin.isin(t2a.values())]
cols = df.columns.values
df[[]]


import HTMLParser
def fix(s):
    return HTMLParser.HTMLParser().unescape(s)

df['description'] = [simple_preprocess(fix(s)) 
                    if type(s) in (str,unicode) else np.nan
                    for s in df.description.tolist() ]

df = df[notnull(df.description)];print len(df)

df['desc_len'] = [len(d) for d in df.description.tolist()]

df = df[notnull(df.desc_len)].query('desc_len>50');print len(df)

df.sort_values('desc_len',ascending=False, inplace=True)
df.head()


from __future__ import division

asins = df.query('desc_len>50').asin.tolist()


fn = root+'/book_description_corpus_200word_min.txt'
with codecs.open(fn,'wb',encoding='utf8') as f:
    for d,a in df.query('desc_len>200')['description asin'.split()].values:
        s = u"%s\t%s\n"%(a, u' '.join(d))
        f.write(s)

bg = Phrases(df.query('desc_len>50').description.tolist())
bg.save(root+'/book_description_bg.p')

docs = [LabeledSentence(bg[d],[any2unicode(a)]) 
        for d, a in df.query('desc_len>50')['description asin'.split()].values]

# Docs composed of related books
docs2 = []
n = []
a2r = {}
# for asin, r in df[notnull(df.related)]['asin related'.split()].values:
for asin, r in df[notnull(df.related)]['asin related'.split()].values:
    if type(r)!=dict:
        continue
    doc = []
    for vals in r.values():
        doc+=vals
    
    if len(doc)>=1:
        a2r[asin] = list(set(doc))
        docs2.append(LabeledSentence(doc, [asin]))



m2 = Doc2Vec(dm=1, min_count=3, 
            window=4, size=200, 
            workers=16, negative=3, sample=1e-2,
             iter=10);

from gensim.models import Word2Vec


a2desc = {doc.tags[0]:doc.words for doc in docs}

docs1 = []
for asin, desc in a2desc.iteritems():     
    document = []
    i = 0
    desc.reverse()
    while len(desc)>0:
        if i%5==0:
            document.append(asin)
        else:
            document.append(desc.pop())
    docs1.append(document)


for asin,related in a2r.iteritems():
    dd = related
    if asin in a2desc:
        dd += a2desc[asin]
        shu
    docs2.append( )


m2 = Word2Vec(sg=1, min_count=3, 
              window=4, size=300, 
              workers=16, negative=3, sample=1e-2,
              iter=2);

m2.build_vocab(docs2)
shuffle(docs2)
m2.train(docs2)


m = Doc2Vec(dm=1, min_count=25, 
            window=8, size=200, 
            workers=16, negative=2,
            sample=1e-4, iter=10)

m.build_vocab(docs)
m.train(docs)


t2a = {title:asin for title,asin in df['title asin'.split()].values}
a2t = {asin:title for asin,title in df['asin title'.split()].values}

def s(matches):
    return [(a2t[x[0]],x[1]) for x in matches if x[0] in a2t]
    

def alike(p=[], n=[], topn=30, m=m):
    if type(p)!=list:
        p=[p]
    if type(n)!=list:
        n=[n]    
    p = [m.docvecs[t2a[item]] if item in t2a else m[item] 
         for item in p]
    n = [m.docvecs[t2a[item]] if item in t2a else m[item] 
         for item in n]
    word_matches = m.most_similar(positive=p,negative=n,topn=5)
    if len(word_matches[0][0])==10:
        pprint(s(word_matches));print
    else:
        pprint(word_matches);print
    pprint(s(m.docvecs.most_similar(positive=p,negative=n,topn=topn)))

def book_words(title, model):
    print "Query: %s"%title
    v = model.docvecs[t2a[title]]
    pprint(model.most_similar(positive=[v],topn=10));print
    book_matches = [(a2t[x[0]], x[1]) for x in 
                     model.docvecs.most_similar(positive=[v],topn=10)]
    pprint(book_matches)
    
def word_books(word, model):
    print "Query: %s"%word
    if type(word)!=list:
        word = word.split()
    v = array([model[w] for w in word]).sum(axis=0)
    pprint(model.most_similar(positive=[v],topn=10));print
    book_matches = [(a2t[x[0]], x[1]) for x in 
                     model.docvecs.most_similar(positive=[v],topn=10)]
    pprint(book_matches)

# alike(p=b,m=m2)
p = 'The Nearest Exit (Milo Weaver)'
# p = ['The_Picture_of_Dorian_Gray.txt']
# p = 'wittgenstein'
p = 'Incompleteness: The Proof and Paradox of Kurt Godel (Great Discoveries)'
# p = ['new_york']
# p = search('on the road',1)[0].replace(' ','_')
p = "The Picture of Dorian Gray (Norton Critical Edition)"
p ='The Goldfinch: A Novel (Pulitzer Prize for Fiction)' ;
p='The Up and Comer'
alike(p=p,m=m,topn=10)


def alike2(p=[], n=[], topn=30):
    if type(p)!=list:
        p=[p]
    if type(n)!=list:
        n=[n]    
    p = [t2a[x] for x in p if x in t2a];
    n = [t2a[x] for x in n if x in t2a]

    p = [m2[item] for item in p
        if item in m2 ];

    n = [m2[item] for item in n
        if item in m2 ]

    word_matches = m2.most_similar_cosmul(positive=p, 
                                  negative=n,
                                  topn=topn)
    pprint(s(word_matches));print

def am(title,a2r=a2r,topn=10):
    if title in t2a and t2a[title] in a2r:
        related = a2r[t2a[title]]
        tt = [a2t[a] for a in related  
                if a in a2t]
        pprint(tt[:min(topn,len(tt))])

p = 'All the Light We Cannot See: A Novel'
p = 'The Girl with the Dragon Tattoo Book 1 (Millennium Trilogy)'
p ='Pattern Recognition and Machine Learning (Information Science and Statistics)'
p = 'Fifty Shades of Grey: Book One of the Fifty Shades Trilogy'
p = 'Gone Girl: A Novel'

p = 'Syntactic Structures (2nd Edition)'
p ='A Foreign Country'
p = 'The Nearest Exit (Milo Weaver)'
p = 'The Bourne Identity (Bourne Trilogy No.1)'
p = "Dead Eye (A Gray Man Novel)"
p = 'The Gray Man (A Gray Man Novel)'
am(p)
alike2(p=p, n=[],topn=50)


# from wtforms import Form, StringField, validators
# class UsernameForm(Form):
#     username = StringField('Username', [validators.Length(min=5)], default=u'test')

# form = UsernameForm()
# form['username']


env = Environment(loader=PackageLoader(root,
                         root+'/templates'))
from jinja2 import Template
import codecs






results = [ {'id': 'book_img' +str(i),
            'img_link': product.medium_image_url,
            'author': product.author,
            'rating': product.rating
            'buy_link': product.offer_url,
            'description': product.editorial_review[:200] 
            if product.editorial_review 
            and len(product.editorial_review)>0 else ''}
            for i, product in enumerate(prods) ]
items = OrderedDict(zip(range(len(results)),results))


common_books = set(m.docvecs.doctags.keys()).intersection(set(m2.vocab.keys()))
print len(common_books)

cb = [b for b in m2.index2word[:664000] if b in common_books]
print len(cb)

from sklearn.linear_model import LinearRegression
clf = LinearRegression()


from gensim.matutils import unitvec

# clf.fit( array( [ unitvec(m.docvecs[b]) for b in cb]),
#          array( [ unitvec(m2[b]) for b in cb]))

clf.fit( array( [ unitvec(m2[b]) for b in cb]),
        array( [ unitvec(m.docvecs[b]) for b in cb]))


def wsearch(words):
    vecs = []
    for word in words:
        if word in m:
            v = unitvec(m[word])
            v = clf.predict(v.reshape(1,-1))
            vecs.append(v[0])
    vecs = array(vecs).sum(axis=0)    
    pprint(s(m2.most_similar([ v[0]])))


def map_word(word):
    if word in m:
        v = unitvec(m[word])
        return clf.predict(v.reshape(1,-1))[0]

def map_vec(vec):    
    return clf.predict(unitvec(vec).reshape(1,-1))[0]


mat = []
for tag in m.docvecs.doctags:
    # get m2's vector and map to m 
    if tag in m2:
        v = map_vec(m2[tag])
    else:
        v = np.zeros(v.shape)
    mat.append(v)
array(mat)

        

mat = array(mat).astype(np.float32)
m.docvecs.doctag_syn0 = mat


wsearch(['oscar_wilde'])
wsearch(['jane_austen'])
wsearch(['charles_dickens'])
wsearch(['marx'])
wsearch(['economics'])
wsearch(['math'])

dot(
    unitvec(m.docvecs[t2a[p]]),
            unitvec(m2[t2a[p]]) 
            )





nw = nw[notnull(nw['title'])]
nw.sort_values('rank',inplace=True)
q = '(rank<2000000 and N>4 and nwords>2500)'
q +=' or (N>20 and nwords>2500)'
q+= ' or (nwords>5000 and N>10)'

nw2 = nw.query(q).copy(deep=True)
print len(nw2)


def pca_transform_vecs(vecs,n=50):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n)
    pca.fit(vecs)
    return pca.transform(vecs)    



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



m = Doc2Vec(dm=1, min_count=15, 
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





