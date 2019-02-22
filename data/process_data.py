import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3

def load_data(messages_filepath, categories_filepath):
    '''
    This function loads the messages and categories datasets, 
    merge two datasets using the common id.
    Args:
        messages_filepath (str): path to messages.csv file
        categories_filepath (str): path to categories.csv file
    Returns:
        df (pandas DataFrame): a merged dataframe with messages 
        and categories datasets using the common id
    '''
    messages = pd.read_csv(messages_filepath, index_col=0)
    categories = pd.read_csv(categories_filepath, index_col=0)
    df = pd.concat([messages, categories], axis=1, join='inner')
    return df


def clean_data(df):
    '''
    This function cleans the dataframe in following steps:
        1. Expand the categories column into 36 columns and retain only the int value
        2. Rename the expanded categories columns with category names, and drop the categories column
        3. concatenate the original dataframe with the new 'categories' dataframe
        4. drop the 'original' column, which contains redundant information to 'messages' column
        5. remove duplicates
        6. remove rows where 'related'=2
    Args:
        df (pandas DataFrame): a dataframe to clean
    Returns:
        df (pandas DataFrame): a cleaned dataframe
    '''
    # create a dataframe of the 36 individual category columns
    new_categories = df['categories'].str.split(';', expand=True)
    
    # extract a list of new column names for categories.
    row = new_categories.iloc[0]
    extract_str = lambda x: x[:-2]
    category_colnames = row.apply(extract_str)
    new_categories.columns = category_colnames

    # Convert category values to just numbers 0 or 1
    for column in new_categories:
        new_categories[column] = new_categories[column].str[-1]
        new_categories[column] = new_categories[column].astype(int)

    df.drop(['categories'], axis=1, inplace=True)

    df = pd.concat([df, new_categories], axis=1, join='inner')
    df.drop(['original'], axis=1, inplace=True)
    df.drop_duplicates(inplace=True)
    df.drop(df[df.related==2].index.tolist(), axis=0, inplace=True)

    return df

def save_data(df, database_filename):
    '''
    This function saves the clean dataset into an sqlite database.
    Args:
        df (pandas dataframe): a clean dataset
        database_filename (str): path to the sqlite database
    Returns:
        None
    '''
    conn = sqlite3.connect(database_filename)
    df.to_sql('msg_cat_merged', conn)
    conn.commit()
    conn.close()


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()