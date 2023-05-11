import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from CustomLabelEncoder import CustomLabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
# Imports for the plot
import seaborn as sns
from matplotlib import pyplot as plt

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
def weight_genres(genres):
    # Create a dictionary to hold the weights
    weights = {}
    # Loop through the genres list and assign weights based on order of appearance
    for i, genre in enumerate(genres):
        weights[genre] = len(genres) - i
    return weights
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
unimportant_columns=['URL','Name','Subtitle','Icon URL','Price','Description','Developer','Age Rating','Languages']
x_train = drop_columns(x_train, unimportant_columns)

# fill any missing values in the 'Price' and 'In-app Purchases' columns with 0
# x_train['Price'].fillna(0, inplace=True)
x_train['In-app Purchases'].fillna(0, inplace=True)

x_train['In-app Purchases'] = calc_sum_of_list(df['In-app Purchases'])
x_train['In-app Purchases'].fillna(0, inplace=True)

# x_train['Languages'] = fill_nulls_with_mode(x_train['Languages'])

# change datatypes from object
# x_train = x_train.convert_dtypes()

# Remove the '+' sign from the 'Age rating' column
# x_train['Age Rating'] = x_train['Age Rating'].str.replace('+', '', regex=False)
# # Convert the 'Age rating' column to an integer data type
# x_train['Age Rating'] = x_train['Age Rating'].astype(int)

# Remove the primary genre from the "Genres" feature
x_train['Genres'] = remove_first_word(x_train['Genres'])
x_train['Genres'] = x_train['Genres'].apply(lambda x: x.replace(' ', '').split(','))

data = x_train.join(y_train)
# data = remove_special_chars(data, 'Developer')
y_train = data['Rate']
x_train = data.drop('Rate', axis=1)
# x_train['Developer'] = x_train['Developer'].str.replace(r'\\xe7\\xe3o', ' ', regex=True)

# Encode categorical columns (Developer, Languages and Primary Genre)
dev_encoder = CustomLabelEncoder()
lang_encoder = CustomLabelEncoder()
primary_genre_encoder = CustomLabelEncoder()
# x_train['Developer'] = dev_encoder.fit_transform(x_train['Developer'])
# x_train['Languages'] = lang_encoder.fit_transform(x_train['Languages'])
x_train['Primary Genre'] = primary_genre_encoder.fit_transform(x_train['Primary Genre'])

# Extract feature (Difference in days) from 'Original Release Date' and 'Current Release Date'
x_train['Original Release Date'] = pd.to_datetime(x_train['Original Release Date'], errors='coerce', format='%d/%m/%Y')
x_train['Current Version Release Date'] = pd.to_datetime(x_train['Current Version Release Date'], errors='coerce',
                                                         format='%d/%m/%Y')
x_train['Difference in Days'] = (x_train['Current Version Release Date'] - x_train['Original Release Date']).dt.days

# Drop both Original Release Data and Current Version Release Date
x_train.drop(['Original Release Date', 'Current Version Release Date'], axis=1, inplace=True)

# Apply the weight_genres function to the genres column and store the results in a new column called genre_weights
x_train['genre_weights'] = x_train['Genres'].apply(weight_genres)

# Create a list of all unique genres in the dataset
unique_genres = list(set([genre for genres in x_train['Genres'] for genre in genres]))

# Create a binary column for each unique genre and set the value to the weight assigned by the weight_genres function
for genre in unique_genres:
    x_train[genre] = x_train['genre_weights'].map(lambda x: x.get(genre, 0))

# Drop the genre_weights column since it is no longer needed
x_train.drop('genre_weights', axis=1, inplace=True)

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

print(x_train)
# print("SHAPE BEFORE -----------------------------------------------")
# print(x_train.shape)
# x_train.dropna()
# print("SHAPE AFTER -----------------------------------------------")
# print(x_train.shape)
# ---------------------------------Testing Preprocessing-----------------------------------

# drop any unnecessary columns
unimportant_columns=['URL','Name','Subtitle','Icon URL','Price','Description','Developer','Age Rating','Languages']
x_test = drop_columns(x_test, unimportant_columns)
#
# # fill any missing values in the 'Price' and 'In-app Purchases' columns with 0
# # x_test['Price'].fillna(0, inplace=True)
# x_test['In-app Purchases'].fillna(0, inplace=True)
#
# x_test['In-app Purchases'] = calc_sum_of_list(df['In-app Purchases'])
# x_test['In-app Purchases'].fillna(0, inplace=True)
#
# # x_test['Languages'] = fill_nulls_with_mode(x_test['Languages'])
#
# # change datatypes from object
# # x_test = x_test.convert_dtypes()
#
# # Remove the '+' sign from the 'Age rating' column
# # x_test['Age Rating'] = x_test['Age Rating'].str.replace('+', '', regex=False)
# # # Convert the 'Age rating' column to an integer data type
# # x_test['Age Rating'] = x_test['Age Rating'].astype(int)
#
# # Remove the primary genre from the "Genres" feature
# x_test['Genres'] = remove_first_word(x_test['Genres'])
# x_test['Genres'] = x_test['Genres'].apply(lambda x: x.replace(' ', '').split(','))
#
# data = x_test.join(y_test)
# # data = remove_special_chars(data, 'Developer')
# y_test = data['Rate']
# x_test = data.drop('Rate', axis=1)
# # x_test['Developer'] = x_test['Developer'].str.replace(r'\\xe7\\xe3o', ' ', regex=True)
#
# # Encode categorical columns (Developer, Languages and Primary Genre)
# dev_encoder = CustomLabelEncoder()
# lang_encoder = CustomLabelEncoder()
# primary_genre_encoder = CustomLabelEncoder()
# # x_test['Developer'] = dev_encoder.fit_transform(x_test['Developer'])
# # x_test['Languages'] = lang_encoder.fit_transform(x_test['Languages'])
# x_test['Primary Genre'] = primary_genre_encoder.fit_transform(x_test['Primary Genre'])
#
# # Extract feature (Difference in days) from 'Original Release Date' and 'Current Release Date'
# x_test['Original Release Date'] = pd.to_datetime(x_test['Original Release Date'], errors='coerce', format='%d/%m/%Y')
# x_test['Current Version Release Date'] = pd.to_datetime(x_test['Current Version Release Date'], errors='coerce',
#                                                          format='%d/%m/%Y')
# x_test['Difference in Days'] = (x_test['Current Version Release Date'] - x_test['Original Release Date']).dt.days
#
# # Drop both Original Release Data and Current Version Release Date
# x_test.drop(['Original Release Date', 'Current Version Release Date'], axis=1, inplace=True)
#
# # Apply the weight_genres function to the genres column and store the results in a new column called genre_weights
# x_test['genre_weights'] = x_test['Genres'].apply(weight_genres)
#
# # Create a list of all unique genres in the dataset
# unique_genres = list(set([genre for genres in x_test['Genres'] for genre in genres]))
#
# # Create a binary column for each unique genre and set the value to the weight assigned by the weight_genres function
# for genre in unique_genres:
#     x_test[genre] = x_test['genre_weights'].map(lambda x: x.get(genre, 0))
#
# # Drop the genre_weights column since it is no longer needed
# x_test.drop('genre_weights', axis=1, inplace=True)
#
# # Feature selection using spearman method
# data = x_test.join(y_test)
#
# mapping = {'Low': 0, 'Intermediate': 1, 'High': 2}
# data['Rate'] = data['Rate'].map(mapping)
#
# game_data = data.iloc[:, :]
# corr = game_data.corr(method='spearman')
# # Top 50% Correlation testing features with the Value
# top_feature = corr.index[abs(corr['Rate']) > 0.03]
# print(top_feature)
# x_data = game_data[top_feature]
#
# # Correlation plot
# # plt.subplots(figsize=(12, 8))
# # top_corr = game_data[top_feature].corr(method='spearman')
# # sns.heatmap(top_corr, annot=True)
# # plt.show()
# # print(x_data.columns)
#
# # Standardize the data
# # standardization = StandardScaler()
# # game_data = standardization.fit_transform(x_data)
# # game_data = pd.DataFrame(game_data, columns=top_feature)
# y_test = x_data['Rate']
# x_test = x_data.drop('Rate', axis=1)

# Replace the list in 'In-app Purchases' column with the sum of the list in each entry
x_test['In-app Purchases'] = calc_sum_of_list(x_test['In-app Purchases'])

# Change data type to "datetime" in the 'Original Release Date' and 'Current Version Release Date' columns
x_test['Original Release Date'] = pd.to_datetime(x_test['Original Release Date'], errors='coerce',
                                                 format='%d/%m/%Y')
x_test['Current Version Release Date'] = pd.to_datetime(x_test['Current Version Release Date'], errors='coerce',
                                                        format='%d/%m/%Y')

# Remove the '+' sign from the 'Age rating' column
# x_test['Age Rating'] = x_test['Age Rating'].str.replace('+', '', regex=False)
#
# # Convert the 'Age rating' column to an integer data type
# x_test['Age Rating'] = x_test['Age Rating'].astype(int)

# x_test['Languages'] = fill_nulls_with_mode(x_train['Languages'])
# print(x_train.dtypes)
# global_vars['Genres'] = global_vars['Primary Genre']

x_test = x_test.apply(replace_genres_missing_vals, axis=1)

x_test['In-app Purchases'].fillna(0, inplace=True)
# Fill missing values
# for col in x_test.columns:
#     if col == 'In-app Purchases':
#         x_test[col] = fill_nulls(x_test[col], 0)
#     else:
#         x_test[col].fillna(global_vars[col], inplace=True)

# Extract feature (Difference in days) from 'Original Release Date' and 'Current Release Date'
x_test['Difference in Days'] = (x_test['Current Version Release Date'] - x_test['Original Release Date']).dt.days

# Drop both Original Release Data and Current Version Release Date
x_test.drop(['Original Release Date', 'Current Version Release Date'], axis=1, inplace=True)

test_data = x_test.join(y_test)
# test_data = remove_special_chars(test_data, 'Developer')

mapping = {'Low': 0, 'Intermediate': 1, 'High': 2}
data['Rate'] = data['Rate'].map(mapping)

y_test = test_data['Rate']
x_test = test_data.drop('Rate', axis=1)

# x_test['Developer'] = dev_encoder.transform(x_test['Developer'])
# x_test['Languages'] = lang_encoder.transform(x_test['Languages'])
x_test['Primary Genre'] = primary_genre_encoder.transform(x_test['Primary Genre'])

# change datatypes from object
x_test = x_test.convert_dtypes()

# Remove the primary genre from the "Genres" feature
x_test['Genres'] = remove_first_word(x_test['Genres'])

# Change "Genres" values from string to list of strings
x_test['Genres'] = x_test['Genres'].apply(lambda x: x.replace(' ', '').split(','))

# Apply the weight_genres function to the genres column and store the results in a new column called genre_weights
x_test['genre_weights'] = x_test['Genres'].apply(weight_genres)

# Apply one-hot encoding to the 'Genres' column
one_hot_test = x_test['Genres'].str.get_dummies(',')

# Add missing columns to the one-hot encoded test data
missing_cols = set(x_train.columns) - set(one_hot_test.columns)
for col in missing_cols:
    one_hot_test[col] = 0

# Sort the columns in the test data in the same order as in the training data
one_hot_test = one_hot_test[x_train.columns]

# Apply the weighted one-hot encoding to the test data
for genre in unique_genres:
    x_test[genre] = x_test['genre_weights'].map(lambda x: x.get(genre, 0))

x_test.drop('genre_weights', axis=1, inplace=True)
x_test_data = x_test.join(y_test)
col_test = x_test_data.columns
# x_test_data = standardization.transform(x_test_data[top_feature])
x_test_data = pd.DataFrame(x_test_data, columns=top_feature)
y_test = x_test_data['Rate']
x_test = x_test_data.drop('Rate', axis=1)

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
#-----------------------------------------------------------------------------------------------------------------------