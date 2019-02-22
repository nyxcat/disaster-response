# Disaster Response Pipeline Project


### Introduction
This project analyze disaster data from [Figure Eight](https://www.figure-eight.com) to build a model for an API that classifies disaster messages into 36 categories. A machine learning pipeline is created to categorize these events so that these messages can be sent to an appropriate disaster relief agency.
There are 2 datasets used in this project:
- messages.csv, containing real messages that were sent during disaster events
- categories.csv, containing multi-class output of each message
The project  includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. 

### Data Processing 
1. Natural language processing using nltk package -- tokenize, lemmetize words in messages
2. Construct new features:
    a. PunctuationExtractor, extract the number of punctuation in each message
    b. VerbExtractor, get the number of verbs used in sentence
    c. WordLenExtractor, extract the average word length of a message

### Requirement
json
plotly
pandas
numpy
skmultilearn
nltk (punkt, stopwords, wordnet, averaged_perceptron_tagger)
flash
sqlalchemy
sqlite3

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run the web app.
    `python run.py`


