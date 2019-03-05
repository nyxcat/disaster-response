import json
import plotly
import pandas as pd
import numpy as np
import sys
import re
import os
# check if skmulitlearn is installed, if not install it.
import os
try:
    import skmultilearn
except ImportError:
    os.system('pip install scikit-multilearn')

from skmultilearn.problem_transform import LabelPowerset
from sklearn.base import BaseEstimator, TransformerMixin
from string import punctuation
import nltk
nltk.download(['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger'])
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)
r = re.compile(r'[\s{}]+'.format(re.escape(punctuation)))

class PunctuationExtractor(BaseEstimator, TransformerMixin):
    '''
    This class extracts the total number of 
    punctuations in the documents.
    '''
    def fit(self, X, y=None):
        return self
    
    def get_punctuation(self, x):
        x_no_space = ''.join(x.split())
        return len(r.findall(x_no_space))
        
    def transform(self, X):
        X = pd.Series(X)
        return np.reshape(X.apply(self.get_punctuation).values, (X.shape[0],1))
    
class VerbExtractor(BaseEstimator, TransformerMixin):
    '''
    This class extracts the total number of 
    verbs in the documents.
    '''
    def fit(self, X, y=None):
        return self
    
    def get_verb(self, x):
        verb_count = 0
        for word, tag in pos_tag(x.split()):
            if tag.startswith('V'):
                verb_count += 1
        return verb_count
    
    def transform(self, X):
        X = pd.Series(X)
        return np.reshape(X.apply(self.get_verb).values, (X.shape[0],1))
    
class WordLenExtractor(BaseEstimator, TransformerMixin):
    '''
    This class extracts the total number of 
    punctuations in the documents.
    '''
    def fit(self, X, y=None):
        return self
    
    def get_wordlen(self, x):
        ave_len = 0
        for word in x.split():
            ave_len += len(word)
        if len(x.split()) == 0:
            ave_len = 0
        else:
            ave_len = ave_len / float(len(x.split()))
        return ave_len
        
    def transform(self, X):
        X = pd.Series(X)
        return np.reshape(X.apply(self.get_wordlen).values, (X.shape[0],1))


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
print('Loading Database...')
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('msg_cat_merged', engine)

# load model
print('Loading Model...')
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
#    genre_relate_counts = df.groupby(['genre', 'related']).count()['message']
   # print(df.head())
    y0_related = list()
    y1_related = list()
    for k, g in df.groupby(['genre', 'related']):
        if 0 in k:
            y0_related.append(g.count()['message'])
        if 1 in k:
            y1_related.append(g.count()['message'])

    cat = df.iloc[:, 3:].columns.values
    cat_count = df.iloc[:, 3:].sum().values

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                {
                    "x":  genre_names,
                    "y": y0_related,
                    "type": "bar",
                    "name": "Not related"
                },
                {
                    "x":  genre_names,
                    "y": y1_related,
                    "type": "bar",
                    "name": "Related"
                }
            ],

            'layout': {
                'title': 'Number Of Related Events In Each Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                },
                'barmode': 'stack'
            }
        },
        {
            'data': [
                {
                    "x":  cat,
                    "y": cat_count,
                    "type": "bar",
                    "name": "Message Counts"
                }
            ],

            'layout': {
                'title': 'Number Of Messages In Each Category',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 
    # use model to predict classification for query
    classification_labels = model.predict([query])[0].toarray()[0]
    classification_results = dict(zip(df.columns[3:], classification_labels))
    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
    