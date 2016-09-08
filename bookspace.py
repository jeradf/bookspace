from pprint import pprint
from flask import Flask, render_template, request, jsonify
import json
from bisect import bisect_left
import app_settings as settings
from movie_space import title_search, get_similar, title2asin
import os


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html', items={}, query={})

@app.route('/about/')
def about():
    return render_template('about.html')    

@app.route('/search/', methods=['GET'])
def search():
    books = request.args.get('query_book', '', type=str)    
    print 'Query book: "%s"'%books.split(',')
    
    book_query = []
    for book in books.split(','):
        if book.strip() in title2asin:
            book_query.append(book.strip())
    
    if books and not book_query:
        print "Book not found"
        query_params, items = {}, {}
        return render_template('index.html', items={}, query={})
    
    pos_words = request.args.get('plus', '', type=str)    
    neg_words = request.args.get('minus', '', type=str)    
    
    if len(pos_words)>1000 or len(neg_words)>1000:
        query_params,items = {}, {}
        return render_template('index.html', items=items, query=query_params)

    query_params = {'book': ', '.join(book_query), 'pos':pos_words, 'neg':neg_words}
    
    try:
        items = get_similar(book_query, pos_words, neg_words, topn=100)        
        
    except Exception as e:
        print "Exception in get_similar %s"%e.message
        print "\tReason: %s"%e.reason
        print "\tStart: %s, End: %s"%(e.start,e.end)
        return render_template('index.html', items={}, query={})
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
    app.run(host='0.0.0.0',
            port=int("8000"),
            debug=0)
