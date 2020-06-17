import sys

from sqlalchemy import create_engine
import pandas as pd

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
nltk.download(['punkt', 'wordnet'])

from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors   import KNeighborsClassifier
from sklearn.ensemble    import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

import pickle


def load_data(database_filepath):
    """Loads messages from database and creates a DataFrame"""

    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table("messages", engine)
    # features = ['message'], 'original', 'genre']
    category_names  = ['related', 'request', 'offer', 'aid_related', 'medical_help',
                        'medical_products', 'search_and_rescue', 'security', 'military',
                        'child_alone', 'water', 'food', 'shelter',  'clothing', 'money',
                        'missing_people', 'refugees', 'death', 'other_aid',
                        'infrastructure_related', 'transport', 'buildings', 'electricity',
                        'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
                        'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
                        'other_weather', 'direct_report']
    X = df.message.values
    y = df[category_names].values
    return X, y, category_names


def tokenize(text):
    """Performs NLP transformations"""

    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower()) # normalization
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    return Pipeline([
        ('tokenize', CountVectorizer(tokenizer=tokenize)),
        ('tdf',      TfidfTransformer()),
        ('cls',      MultiOutputClassifier(RandomForestClassifier()))
    ])


def evaluate_model(model, X_test, y_test, category_names):
    """Calculates Precision, Recall and F1_score for all targets individually"""

    y_pred = model.predict(X_test)
    for index, y_test_col in enumerate(y_test.transpose()):
        print(category_names[index])
        # print(y_test_col, y_pred.T[index])
        print(classification_report(y_test_col, y_pred.T[index]))


def save_model(model, model_filepath):
    """Stores the model in a pickle file"""

    pkl_filename = model_filepath
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)


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
