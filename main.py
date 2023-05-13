import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler

from CustomLabelEncoder import CustomLabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
# Imports for the plot
import seaborn as sns
from matplotlib import pyplot as plt
# import required libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


lemmatizer = WordNetLemmatizer()
vectorizer = TfidfVectorizer()
def fill_nulls_with_mode(feature):
    mode = feature.mode().iloc[0]
    feature = feature.fillna(mode)
    return feature
def calc_sum_of_list(feature):
    feature = feature.apply(lambda x: sum([float(num.strip(',')) for num in str(x).split()]))
    return feature
def drop_columns(df, columns_names):
    for col in columns_names:
        df = df.drop(col, axis=1)
    return df
def remove_first_word(feature):
    feature = list(
        feature.apply(lambda colm: ', '.join(colm.split(', ')[1:] if len(colm.split(', ')) > 1 else colm.split(', '))))
    return feature
def remove_special_chars(data_frame, column_name):
    # Define a pattern to match special characters
    # pattern = r'[^a-zA-Z0-9\s]'
    pattern = r'[^a-zA-Z0-9\s.,:\'()\-"\\]'
    # Create a boolean mask to identify rows with special characters in the specified column
    mask = data_frame[column_name].str.contains(pattern)

    # # Print the rows that will be deleted
    # print("Rows to be deleted:")
    # print(df[mask])
    # Drop rows with special characters in the specified column
    data_frame = data_frame[~mask]
    return data_frame
# def weight_genres(genres):
#     # Create a dictionary to hold the weights
#     weights = {}
#     # Loop through the genres list and assign weights based on order of appearance
#     for i, genre in enumerate(genres):
#         weights[genre] = len(genres) - i
#     return weights
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
    print(df['Genres'])
    return df['Genres']

def replace_Nans(df):
    df = df.fillna(max(df.dropna()) + (sum(df.dropna()) / len(df.dropna())))
    return df

def replace_genres_missing_vals(row):
    if not row['Primary Genre'] == '' and row['Genres'] == '':
        row['Genres'] = row['Primary Genre']
    elif not row['Genres'] == '' and row['Primary Genre'] == '':
        row['Primary Genre'] = row['Genres'].iloc[0]
    # elif row['Primary Genre'] == '' and row['Genres'] == '':
    #     # row['Genres'] = global_vars['Genres']
    #     # row['Primary Genre'] = global_vars['Primary Genre']
    return row
df = pd.read_csv('games-classification-dataset.csv')

# Split data frame to X and Y
Y = df['Rate']
X = df.drop('Rate', axis=1)

# Split the X and the Y to training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, shuffle=True, random_state=0)


# drop any unnecessary columns
unimportant_columns=[ 'ID','Name','URL','Subtitle','Icon URL','Description','Languages','Primary Genre']    #'Price','Developer','Age Rating','Genres'
x_train = x_train.drop(unimportant_columns, axis=1)

print(x_train.isnull().sum())
x_train['In-app Purchases'].fillna(0, inplace=True)
print(x_train.isnull().sum())

x_train['In-app Purchases'] = calc_sum_of_list(df['In-app Purchases'])
x_train['In-app Purchases'].fillna(0, inplace=True)

# Remove the '+' sign from the 'Age rating' column
x_train['Age Rating'] = x_train['Age Rating'].str.replace('+', '', regex=False)
# Convert the 'Age rating' column to an integer data type
x_train['Age Rating'] = x_train['Age Rating'].astype(int)

x_train['Size'] = x_train['Size'].astype(int)
x_train['User Rating Count'] = x_train['User Rating Count'].astype(int)
x_train['In-app Purchases'] = x_train['In-app Purchases'].astype(int)
x_train['Price'] = x_train['Price'].astype(int)

# Convert the date columns to datetime format
x_train['Original Release Date'] = pd.to_datetime(x_train['Original Release Date'], format='%d/%m/%Y')
x_train['Current Version Release Date'] = pd.to_datetime(x_train['Current Version Release Date'], format='%d/%m/%Y')

# Extract the year features
x_train['Original Release Date'] = x_train['Original Release Date'].dt.year.astype(float)
x_train['Current Version Release Date'] = x_train['Current Version Release Date'].dt.year.astype(float)


# Remove the primary genre from the "Genres" feature
x_train['Genres'] = remove_first_word(x_train['Genres'])
x_train['Genres'] = x_train['Genres'].apply(lambda x: x.replace(' ', '').split(','))

data = x_train.join(y_train)
data = remove_special_chars(data, 'Developer')
y_train = data['Rate']
x_train = data.drop('Rate', axis=1)
x_train['Developer'] = x_train['Developer'].str.replace(r'\\xe7\\xe3o', ' ', regex=True)

# Encode categorical columns (Developer, Languages and Primary Genre)
dev_encoder = CustomLabelEncoder()
lang_encoder = CustomLabelEncoder()
primary_genre_encoder = CustomLabelEncoder()
x_train['Developer'] = dev_encoder.fit_transform(x_train['Developer'])

developer_col= x_train['Developer']
price_col = x_train['Price']
# x_train['Languages'] = lang_encoder.fit_transform(x_train['Languages'])

# x_train['Primary Genre'] = primary_genre_encoder.fit_transform(x_train['Primary Genre'])

# Apply the weight_genres function to the genres column and store the results in a new column called genre_weights
# x_train['Genres'] = x_train['Genres'].apply(weight_genres)
# x_train['Genres'] = x_train['Genres'].apply(sum)
# print(x_train['Genres'].shape)


x_train['Genres'] = genres_col = genres_weight(x_train)
# print(x_train['Genres'].shape)

# x_train['Genres'].fillna(, inplace=True)


print(x_train['Genres'])
# genres_col = x_train['Genres']
# Create a list of all unique genres in the dataset
# unique_genres = list(set([genre for genres in x_train['Genres'] for genre in genres]))

# Create a binary column for each unique genre and set the value to the weight assigned by the weight_genres function
# for genre in unique_genres:
#     x_train[genre] = x_train['genre_weights'].map(lambda x: x.get(genre, 0))

# Drop the genre_weights column since it is no longer needed
# x_train.drop('genre_weights', axis=1, inplace=True)

# Feature selection using spearman method
data = x_train.join(y_train)

mapping = {'Low': 0, 'Intermediate': 1, 'High': 2}
data['Rate'] = data['Rate'].map(mapping)

game_data = data.iloc[:, :]
corr = game_data.corr(method='spearman')
# Top 50% Correlation training features with the Value
top_feature = corr.index[abs(corr['Rate']) > 0.03]
print(top_feature)
x_data = game_data[top_feature]

# Correlation plot
# plt.subplots(figsize=(12, 8))
# top_corr = game_data[top_feature].corr(method='spearman')
# sns.heatmap(top_corr, annot=True)
# plt.show()
# print(x_data.columns)

# Standardize the data
# standardization = StandardScaler()
# game_data = standardization.fit_transform(x_data)
# game_data = pd.DataFrame(game_data, columns=top_feature)
y_train = x_data['Rate']
x_train = x_data.drop('Rate', axis=1)

x_train['Price'] = price_col
x_train['Genres'] = genres_col
# Scale features using MinMaxScaler
# scaler = MinMaxScaler()
# x_train[['User Rating Count', 'Price','In-app Purchases','Age Rating','Size']] = scaler.fit_transform(x_train[['User Rating Count', 'Price','In-app Purchases','Age Rating','Size']])

x_train['Developer'] = developer_col


print(x_train)
# print("SHAPE BEFORE -----------------------------------------------")
# print(x_train.shape)
# x_train.dropna()
# print("SHAPE AFTER -----------------------------------------------")
# print(x_train.shape)
# ---------------------------------Testing Preprocessing-----------------------------------


x_test = x_test.drop(unimportant_columns, axis=1)


print(x_test.isnull().sum())
x_test['In-app Purchases'].fillna(0, inplace=True)
print(x_test.isnull().sum())

x_test['In-app Purchases'] = calc_sum_of_list(df['In-app Purchases'])
x_test['In-app Purchases'].fillna(0, inplace=True)

# Remove the '+' sign from the 'Age rating' column
x_test['Age Rating'] = x_test['Age Rating'].str.replace('+', '', regex=False)
# Convert the 'Age rating' column to an integer data type
x_test['Age Rating'] = x_test['Age Rating'].astype(int)

x_test['Size'] = x_test['Size'].astype(int)
x_test['User Rating Count'] = x_test['User Rating Count'].astype(int)
x_test['In-app Purchases'] = x_test['In-app Purchases'].astype(int)
x_test['Price'] = x_test['Price'].astype(int)

# Convert the date columns to datetime format
x_test['Original Release Date'] = pd.to_datetime(x_test['Original Release Date'], format='%d/%m/%Y')
x_test['Current Version Release Date'] = pd.to_datetime(x_test['Current Version Release Date'], format='%d/%m/%Y')

# Extract the year features
x_test['Original Release Date'] = x_test['Original Release Date'].dt.year.astype(float)
x_test['Current Version Release Date'] = x_test['Current Version Release Date'].dt.year.astype(float)


# Remove the primary genre from the "Genres" feature
x_test['Genres'] = remove_first_word(x_test['Genres'])
x_test['Genres'] = x_test['Genres'].apply(lambda x: x.replace(' ', '').split(','))

data = x_test.join(y_test)
data = remove_special_chars(data, 'Developer')
y_test = data['Rate']
x_test = data.drop('Rate', axis=1)
x_test['Developer'] = x_test['Developer'].str.replace(r'\\xe7\\xe3o', ' ', regex=True)

# Encode categorical columns (Developer, Languages and Primary Genre)
dev_encoder = CustomLabelEncoder()
lang_encoder = CustomLabelEncoder()
primary_genre_encoder = CustomLabelEncoder()
x_test['Developer'] = dev_encoder.fit_transform(x_test['Developer'])
# x_test['Languages'] = lang_encoder.fit_transform(x_test['Languages'])
# x_test['Primary Genre'] = primary_genre_encoder.fit_transform(x_test['Primary Genre'])
#
# # Apply the weight_genres function to the genres column and store the results in a new column called genre_weights
# x_test['Genres'] = x_test['Genres'].apply(weight_genres)

# genres_col = x_test['Genres']

x_test['Genres'] = genres_col = genres_weight(x_test)

# Create a list of all unique genres in the dataset
# unique_genres = list(set([genre for genres in x_test['Genres'] for genre in genres]))

# Create a binary column for each unique genre and set the value to the weight assigned by the weight_genres function
# for genre in unique_genres:
#     x_test[genre] = x_test['genre_weights'].map(lambda x: x.get(genre, 0))

# Drop the genre_weights column since it is no longer needed
# x_test.drop('genre_weights', axis=1, inplace=True)

# Feature selection using spearman method
data = x_test.join(y_test)

mapping = {'Low': 0, 'Intermediate': 1, 'High': 2}
data['Rate'] = data['Rate'].map(mapping)

game_data = data.iloc[:, :]
corr = game_data.corr(method='spearman')
# Top 50% Correlation training features with the Value
top_feature = corr.index[abs(corr['Rate']) > 0.03]
print(top_feature)
x_data = game_data[top_feature]

# Correlation plot
# plt.subplots(figsize=(12, 8))
# top_corr = game_data[top_feature].corr(method='spearman')
# sns.heatmap(top_corr, annot=True)
# plt.show()
# print(x_data.columns)

# Standardize the data
# standardization = StandardScaler()
# game_data = standardization.fit_transform(x_data)
# game_data = pd.DataFrame(game_data, columns=top_feature)
y_test = x_data['Rate']
x_test = x_data.drop('Rate', axis=1)

x_test['Genres'] = genres_col
# Scale features using MinMaxScaler
# scaler = MinMaxScaler()
# x_test[['User Rating Count', 'Price','In-app Purchases','Age Rating','Size']] = scaler.fit_transform(x_test[['User Rating Count', 'Price','In-app Purchases','Age Rating','Size']])


# get the order of columns in the training data
train_columns = list(x_train.columns)

# reorder the columns in the test data to match the order in the training data
x_test = x_test[train_columns]

print(x_test)
# ----------------------------------------------------------------------------------------------------------------------
print("Decision Tree")
# Define the decision tree model with default parameters
model = DecisionTreeClassifier()

# Fit the model to the training data
model.fit(x_train, y_train)

# Predict the labels of the test data
y_pred = model.predict(x_test)

# Evaluate the model's accuracy on the test data
accuracy = model.score(x_test, y_test)
print("Accuracy: ", accuracy)
print()
print("----------------------------------------------------------------------------------------------------------------")
#-----------------------------------------------------------------------------------------------------------------------
print("Random Forest")

# Create a Random Forest classifier with 100 trees
rf = RandomForestClassifier(n_estimators=100)
# fit the model to your training data
rf.fit(x_train, y_train)
# Predict on the test data
y_pred = rf.predict(x_test)

# Evaluate the model performance
accuracy = rf.score(x_test, y_test)
print(f"Accuracy: {accuracy}")
print()
print("----------------------------------------------------------------------------------------------------------------")
#-----------------------------------------------------------------------------------------------------------------------
print("SVM")

svm_clf = svm.SVC(kernel="linear", max_iter=30)

svm_clf.fit(x_train, y_train)

y_pred = svm_clf.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print()
print("----------------------------------------------------------------------------------------------------------------")
# ----------------------------------------------------------------------------------------------------------------------
print("LogisticRegression")

# create a logistic regression model
lr_model = LogisticRegression()

# train the model on the training set
lr_model.fit(x_train, y_train)

# make predictions on the testing set
y_pred = lr_model.predict(x_test)

# evaluate the model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print()
print("----------------------------------------------------------------------------------------------------------------")
# ----------------------------------------------------------------------------------------------------------------------