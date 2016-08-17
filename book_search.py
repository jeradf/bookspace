from __future__ import division
from collections import defaultdict, Counter
import string 
import re
import cPickle as pickle
import os
import app_settings as settings
import sys
root = settings.root_path
if sys.platform=='darwin':
    root = root.replace(os.path.expanduser('~'), os.path.expanduser('~')+'/Dropbox')

title_data = pickle.load( open(root+"model/title_data.p", "rb" ) )
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
    
