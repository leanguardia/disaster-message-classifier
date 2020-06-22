import json
import plotly
import pandas as pd
import numpy as np
from ast import literal_eval

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def _count_words(X):
    """Transforms messages Series the number of words it contains"""
    X = pd.Series(X)
    return X.apply(lambda msg: len(msg.strip().split(' '))).values.reshape(-1, 1)


# load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)
df.categories = df.categories.apply(literal_eval)

# load model
model = joblib.load("models/message-cls.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # Number of words vs number of categories
    heatmap = df.groupby(['words_count', 'category_count']).size().unstack(fill_value=0)
    heatmap = np.flipud(np.rot90(heatmap.values, 1))
    words_vs_categories_viz = {
        'data': [{
            'type': 'heatmap',
            'z': heatmap,
            'colorscale': [[0, '#fef5e7'], [1, '#1f77b4']],
        }],
        'layout': {
            'height': 600,
            'title': 'Number of words by number of categories in messages',
            'xaxis': { 
                'title': "Number of words", 'dtick': 5,
            },
            'yaxis': { 
                'title': 'Number of categories', 'dtick': 2,
            },
        }
    }

    # Distribution of message categories
    category_dist = df.categories.explode().value_counts(dropna=False)
    labels = ['None' if str(cat) == 'nan' else cat for cat in category_dist.keys()]
    print(labels)
    category_dist_viz = {
        'data': [{
            'type': 'bar',
            'x': category_dist.values,
            'y': labels,
            'orientation': 'h',
            'textposition': 'auto',
            'transforms': [{
                'type': 'sort',
                'target': 'y',
                'order': 'descending'
            }],
            'marker': {
                'color': ['#D3D3D3' if label=='None' else '#1f77b4' for label in labels],
            }
        }],
        'layout': {
            'height': 820,
            'title': 'Distribution of Message Categories',
            'xaxis': { 
                'title': "Number of messages",
                'dtick': 1000,
                'range': [0, 20100]
            },
            'yaxis': { 'automargin': True },
        }
    }
    
    # Distribution of message genres
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    genre_dist_viz = {
        'data': [
            Bar(x = genre_names, y = genre_counts)
        ],
        'layout': {
            'title': 'Distribution of Message Genres',
            'yaxis': { 'title': "Count" },
            'xaxis': { 'title': "Genre" }
        }
    }
    graphs = [words_vs_categories_viz, category_dist_viz, genre_dist_viz]
    
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
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[6:], classification_labels))

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
