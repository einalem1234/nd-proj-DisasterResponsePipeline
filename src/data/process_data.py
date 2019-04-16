import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import sys


def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    assert messages.shape[0] == categories.shape[0], "Something is wrong with the data, the two files don't have the same number of rows"
    df = messages.merge(right = categories, how='outer', on=['id'])
    return df


def clean_data(df):
    # Read category column names
    first_row = df['categories'].iloc[0]
    category_colnames = [elem[:-2] for elem in first_row.split(';')]
    #print(category_colnames)
    
    # extract values for categories
    new_categories = pd.DataFrame(data=0, index=np.arange(df.shape[0]), columns=category_colnames)
    print('Converting all categories for all messages, this needs some time!')
    for idx, value in df['categories'].iteritems():
        tmp = value.split('-1')
        # the last slice does not contain any information
        tmp = tmp[:-1]
        for item in tmp:        
            # extract which category is 1
            if ';' in item:
                #print(item.split(';')[-1])
                new_categories.loc[idx, item.split(';')[-1]] = 1
                #print('TODO')
            else:
                new_categories.loc[idx, item] = 1
        if idx > 10: # for testing
            break
        
    #replace categories with new columns
    df = df.drop('categories', axis=1)
    df = pd.concat([df, new_categories], axis = 1)
    
    #remove duplicate entries
    df = df.drop_duplicates(keep = 'first')
    
    return df


def save_data(df, database_filename):
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('messages', engine, index=False, if_exists = 'replace')  


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