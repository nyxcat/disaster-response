# import libraries
import sys
import sqlite3
import pandas as pd
from sklearn.externals import joblib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import re
from string import punctuation
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score

import nltk
nltk.download(['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger'])

from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag

# add other features: Verb counts and Puncutation counts 
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
        X=pd.Series(X)
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
        X=pd.Series(X)
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
        X=pd.Series(X)
        return np.reshape(X.apply(self.get_wordlen).values, (X.shape[0],1))



def load_data(database_filepath):
    '''
    This function reads a database into numpy arrays, 
    and split data into train and test sets.
    Args:
        database_filepath(str): path to database file
    Returns:
        X_train(numpy arrays): training input
        X_test(numpy arrays): test input
        Y_train(numpy arrays): training output
        Y_test(numpy arrays): test output
        label_name(list): list of label names
    '''
    # database_filepath = 'disaster.db'
    conn = sqlite3.connect(database_filepath)
    df = pd.read_sql('SELECT * FROM msg_cat_merged', conn)
    X = df.iloc[:, 1].values
    Y = df.iloc[:, 3:].values

    cur = conn.execute('SELECT * FROM msg_cat_merged')
    column_names = [col[0] for col in cur.description]

    label_names = column_names[3:]

    #X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)
    return X, Y, label_names



def get_wordnet_tag(treebanktag):
    '''
    This function converts pos_tag to wordnet tag.
    Args:
        treebanktag: position tag of word
    Returns:
        wordnet position
    '''
    if treebanktag.startswith('N'):
        return wordnet.NOUN
    if treebanktag.startswith('J'):
        return wordnet.ADJ
    if treebanktag.startswith('V'):
        return wordnet.VERB
    if treebanktag.startswith('R'):
        return wordnet.ADV
    
def tokenize(text):
    '''
    This function tokenizes sentences to words.
    Args:
        text(str)
    Returns:
        clean_tokens(list): list of lemmatized words
    '''
    tokens = word_tokenize(text)
    
    lemmatizer = WordNetLemmatizer()
    clean_tokens = list()
    
    for vals in pos_tag(tokens):
        t, tag = vals
        tag = get_wordnet_tag(tag)
        clean_t = t
        if tag:
            clean_t = lemmatizer.lemmatize(t, pos=tag).lower().strip()
        clean_tokens.append(clean_t)
    
    return clean_tokens



def build_model(tokenize):
    '''
    This function uses pipeline to build a randomforest model
    Args:
        tokenize: the tokenize function
    Returns:
        cv: best model    
    '''
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('nlp_pipeline', Pipeline([
                ('counter', CountVectorizer(tokenizer=tokenize, max_df=0.3, ngram_range=(1,1))),
                ('tfidf', TfidfTransformer(norm=None, smooth_idf=True))
            ])),
            ('verb_count', VerbExtractor()),
            ('punc_count', PunctuationExtractor()),
            ('word_len', WordLenExtractor())
        ])),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=50)))
    ])

    parameters = {
        'clf__estimator__min_samples_split': [2]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    # cv.fit(X_train, Y_train)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    This function report the f1 score, precision and recall 
    for each output category of the dataset.
    Args: 
        model
        X_test(numpy arrays): test input
        Y_train(numpy arrays): training output
        Y_test(numpy arrays): test output
        category_names(list): list of label names
    '''
    Y_pred = model.predict(X_test)
    for i in range(Y_test.shape[1]):
        print('lable {}'.format(category_names[i]))
        print(classification_report(Y_test[:, i], Y_pred[:, i]))


def save_model(model, model_filepath):
    joblib.dump(model.best_estimator_,  model_filepath, compress=3)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model(tokenize=tokenize)
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()