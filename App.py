import pickle

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler


def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text.lower())
    words = [lemmatizer.lemmatize(w) for w in words]
    s_words = set(words)
    return s_words


def feature_extraction(description_column):
    vectorizer = TfidfVectorizer(stop_words='english')
    returned_list = []
    for description in description_column:
        words = preprocess_text(description)
        meaningful_words = []
        for word in words:
            if len(word) < 3:
                continue
            synsets = wordnet.synsets(word)
            if synsets:
                meaningful_words.append(word)

        pos_tags = nltk.pos_tag(meaningful_words)
        nouns = [word for word, pos in pos_tags if pos.startswith('NN')]
        nouns = ' '.join(nouns)
        returned_list.append(nouns)
    returned_list = pd.DataFrame({'New': returned_list})
    # print(returned_list)
    features = vectorizer.fit_transform(returned_list['New'])
    # print(features)
    # Calculate the average TF-IDF value for each row
    max_tfidf = features.max(axis=1)
    max_tfidf = max_tfidf.todense().A1
    # Assign the average TF-IDF values to a new column in the data frame
    description_column = max_tfidf
    return description_column


def weight_genres(genres):
    # Create a dictionary to hold the weights
    weights = {}
    # Loop through the genre list and assign weights based on order of appearance
    for i, genre in enumerate(genres):
        weights[genre] = len(genres) - i
    return weights


def calc_sum_of_list(feature):
    feature = feature.apply(lambda x: sum([float(num.strip(',')) for num in str(x).split()]))
    return feature


def fill_nulls(feature, value):
    feature = feature.fillna(value)
    return feature


def remove_first_word(feature):
    feature = list(
        feature.apply(lambda colm: ', '.join(colm.split(', ')[1:] if len(colm.split(', ')) > 1 else colm.split(', '))))
    return feature


def remove_special_chars(data_frame, column_name):
    # Define a pattern to match special characters
    pattern = r'[^a-zA-Z0-9\s.,:\'()\-"\\]'
    # Create a boolean mask to identify rows with special characters in the specified column
    mask = data_frame[column_name].str.contains(pattern)

    # # Print the rows that will be deleted
    # print("Rows to be deleted:")
    # print(df[mask])
    # Drop rows with special characters in the specified column
    data_frame = data_frame[~mask]
    return data_frame


def genres_weight(df):
    # create a set of unique genres
    unique_genres = set()
    for genres in df['Genres']:
        unique_genres.update(str(genres).split(', '))

    # assign a weight to each genre
    genre_weights = {genre: i + 1 for i, genre in enumerate(unique_genres)}

    # create a dictionary of genres and weights for each row
    genre_dicts = []
    for genres in df['Genres']:
        genre_dict = {}
        for genre in str(genres).split(', '):
            genre_dict[genre] = genre_weights[genre]
        genre_dicts.append(genre_dict)

    # create a DataFrame from the genre dictionaries
    genre_df = pd.DataFrame(genre_dicts)

    # sum the weights for each row and assign the result to a new column
    df['Genres'] = genre_df.sum(axis=1)

    # display the result
    df['Genres'] = replace_Nans(df['Genres'])
    # print(df['Genres'])
    return df['Genres']


def replace_Nans(df):
    df = df.fillna(max(df.dropna()) + (sum(df.dropna()) / len(df.dropna())))
    return df


# region Load models and variables
models_list = pickle.load(open('classifiers.pkl', 'rb'))
decisionTree = models_list['decisionTree']
random_forest = models_list['random_forest']
svm_clf = models_list['svm_clf']
svm_clf_poly=models_list['svm_clf_poly']
svm_clf2 = models_list['svm_clf2']
lr_model = models_list['lr_model']
GNB = models_list['GNB']
knn = models_list['knn']
base_clf = models_list['base_clf']
ada = models_list['ada']
bagging = models_list['bagging']

variables_list = pickle.load(open('variables.pkl', 'rb'))
unimportant_columns = variables_list['unimportant_columns']
x_train = variables_list['x_train']
y_train = variables_list['y_train']
top_feature = variables_list['top_features']
standardization = variables_list['standardization']
global_vars = variables_list['global_vars']
dev_encoder = variables_list['dev_encoder']
lang_encoder = variables_list['lang_encoder']
# endregion


df = pd.read_csv('games-classification-dataset.csv')
y_test = df['Rate']
x_test = df.drop('Rate', axis=1)

x_test = x_test.drop(unimportant_columns, axis=1)

x_test['Description'] = feature_extraction(x_test['Description'])

# print(x_test.isnull().sum())
# print(x_test.isnull().sum())

x_test['In-app Purchases'] = calc_sum_of_list(df['In-app Purchases'])

x_test['Size'] = x_test['Size'].astype(int)
x_test['User Rating Count'] = x_test['User Rating Count'].astype(int)
x_test['In-app Purchases'].fillna(0, inplace=True)
x_test['In-app Purchases'] = x_test['In-app Purchases'].astype(int)

# Remove the '+' sign from the 'Age rating' column
x_test['Age Rating'] = x_test['Age Rating'].str.replace('+', '', regex=False)

# Convert the 'Age rating' column to an integer data type
x_test['Age Rating'] = x_test['Age Rating'].astype(int)

# Convert the date columns to datetime format
x_test['Original Release Date'] = pd.to_datetime(x_test['Original Release Date'], format='%d/%m/%Y')
x_test['Current Version Release Date'] = pd.to_datetime(x_test['Current Version Release Date'],
                                                            format='%d/%m/%Y')

# Extract the year features
x_test['Original Release Date'] = x_test['Original Release Date'].dt.year.astype(int)
x_test['Current Version Release Date'] = x_test['Current Version Release Date'].dt.year.astype(int)

for col in x_test.columns:
    if col == 'In-app Purchases' or col == 'Price':
        x_test[col] = fill_nulls(x_test[col], 0)
    else:
        x_test[col].fillna(global_vars[col], inplace=True)

# Remove the primary genre from the "Genres" feature
x_test['Genres'] = x_test['Genres'].apply(lambda x: x.replace(' ', '').split(','))
x_test['Genres'] = x_test['Genres'].apply(weight_genres)
x_test['weighted genres'] = x_test['Genres'].apply(lambda x: sum(x.values()))
x_test['Genres'] = x_test['weighted genres']
x_test.drop('weighted genres', axis=1)
# x_test['Genres'] = genres_weight(x_test)

data = x_test.join(y_test)
data = remove_special_chars(data, 'Developer')
y_test = data['Rate']
x_test = data.drop('Rate', axis=1)
x_test['Developer'] = x_test['Developer'].str.replace(r'\\xe7\\xe3o', ' ', regex=True)

# Encode categorical columns (Developer, Languages and Primary Genre)
x_test['Developer'] = dev_encoder.transform(x_test['Developer'])
x_test['Languages'] = lang_encoder.transform(x_test['Languages'])

# Feature selection using spearman method
data = x_test.join(y_test)

mapping = {'Low': 0, 'Intermediate': 1, 'High': 2}
data['Rate'] = data['Rate'].map(mapping)

game_data = data.iloc[:, :]
corr = game_data.corr(method='kendall')
# Top Correlated training features with the Value

x_data = game_data[top_feature]


# Standardize the data

game_data = standardization.transform(x_data)
game_data = pd.DataFrame(game_data, columns=top_feature)
y_test = x_data['Rate']
x_test = x_data.drop('Rate', axis=1)

# get the order of columns in the training data
train_columns = list(x_train.columns)

# reorder the columns in the test data to match the order in the training data
x_test = x_test[train_columns]
