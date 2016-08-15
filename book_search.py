from collections import defaultdict, Counter
import string 
import re
import cPickle as pickle
import os
remove_punc = string.punctuation 
DIGIT_RE = re.compile(r'\d',re.UNICODE)
RE_PUNCT_ALL = re.compile('([%s])+' % re.escape(remove_punc), re.UNICODE)

def t2w(t):
    return RE_PUNCT_ALL.sub(" ", t.lower() ).split()

def make_word2title_hash(titles):
    w2t = defaultdict(list)
    for title in titles:
        for word in t2w(title):
            w2t[word].append(title)
    
    for key in w2t.keys():
        w2t[key] = list(set(w2t[key]))
    return w2t

def search(query,w2t):
    matches = []
    for word in query.lower().split():
        matches.extend(w2t[word])
    print len(matches)
    matches = Counter(matches)
    if matches:
        return list(zip(*matches.most_common(8))[0])
    else:
        return []

