from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
import pickle
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine
import sys

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

def load_data(database_filepath):
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('messages', engine)

    #TODO - SHould not be needed
    df = df.dropna(axis = 0, how = 'any') 

    X = df.message
    Y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)
    category_names = Y.columns

    return  X, Y, category_names


def tokenize(text):
    # Might be not needed
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    # TODO - add optimized parameters
    forest = RandomForestClassifier(n_estimators=100, random_state=1, criterion = 'entropy')
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize, stop_words = 'english')),
        ('tfidf', TfidfTransformer(use_idf = False)),
        ('classifier', MultiOutputClassifier(estimator = forest))
    ])
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    
    Y_pred = pd.DataFrame(Y_pred, columns = category_names)
    assert isinstance(Y_test, pd.DataFrame), 'Wrong Datatype: Y_test'
    assert isinstance(Y_pred, pd.DataFrame), 'Wrong Datatype: Y_pred'
  
    for column in category_names:
        print(column.upper())
        print(classification_report(y_true = Y_test[column].astype('int32').tolist(), y_pred = Y_pred[column].astype('int32').tolist()))


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
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