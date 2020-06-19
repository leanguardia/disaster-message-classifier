import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Loads messages and categories and merges in one DataFrame"""
    
    messages   = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    return messages.merge(categories, on='id')


def transform_data(df):
    """Perform data cleaning and feature augmentation"""
    
    categ_list, categ_cols = _clean_categories(df.categories) # parse categories
    df.categories = categ_list.astype(str)
    df = pd.concat([df, categ_cols], axis=1)

    # remove duplicates
    df.drop_duplicates(subset=['id'], inplace=True)

    # Calculate the number of words in the message 
    df['words_count'] = df.message.str.strip().str.split(' ').apply(lambda words: len(words))
    # Remove messages with less than 3 words and upper outliers
    words_q1 = df.words_count.quantile(.25)
    words_q3 = df.words_count.quantile(.75)
    words_iqr = words_q3 - words_q1
    min_words = 3
    max_words = words_q3 + 1.5 * words_iqr
    df = df[(df.words_count >= min_words ) & (df.words_count <= max_words)]

    return df


def _clean_categories(categories):
    # tidy up list of categories
    categories_list = categories.str.split(';') \
        .apply(lambda cats: [cat[:-2] for cat in cats if cat[-1]=='1'])

    # expand categories into columns
    categories_cols = categories.str.split(';', expand=True)
    # extract the category desciptions from the first row
    categories_names = categories_cols.loc[0].apply(lambda category: category[:-2])
    categories_cols.columns = categories_names

    for column in categories_cols:
        # set each value to be the last character of the string as integer
        categories_cols[column] = categories_cols[column].str[-1]
        categories_cols[column] = categories_cols[column].astype(int)

    return categories_list, categories_cols


def save_data(df, database_filename):
    """Stores data frame in SQLite database"""
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('messages', engine, if_exists='replace', index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = transform_data(df)
        
        print('Saving data...\n    DATABASE: {} {}'.format(database_filepath, df.shape))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
        
        return df
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    df = main()
