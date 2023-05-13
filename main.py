import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from CustomLabelEncoder import CustomLabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
def calc_sum_of_list(feature):
    feature = feature.apply(lambda x: sum([float(num.strip(',')) for num in str(x).split()]))
    return feature
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

# ----------------------------------------------------------------------------------------------------------------------

df = pd.read_csv('games-classification-dataset.csv')

# Split data frame to X and Y
Y = df['Rate']
X = df.drop('Rate', axis=1)

# Split the X and the Y to training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, shuffle=True, random_state=0)

# drop any unnecessary columns
unimportant_columns=[ 'ID','Name','URL','Subtitle','Icon URL','Description','Languages','Age Rating','Primary Genre','Price']

x_train = x_train.drop(unimportant_columns, axis=1)

# print(x_train.isnull().sum())
x_train['In-app Purchases'].fillna(0, inplace=True)
# print(x_train.isnull().sum())

x_train['In-app Purchases'] = calc_sum_of_list(df['In-app Purchases'])
x_train['In-app Purchases'].fillna(0, inplace=True)

x_train['Size'] = x_train['Size'].astype(int)
x_train['User Rating Count'] = x_train['User Rating Count'].astype(int)
x_train['In-app Purchases'] = x_train['In-app Purchases'].astype(int)

# Convert the date columns to datetime format
x_train['Original Release Date'] = pd.to_datetime(x_train['Original Release Date'], format='%d/%m/%Y')
x_train['Current Version Release Date'] = pd.to_datetime(x_train['Current Version Release Date'], format='%d/%m/%Y')

# Extract the year features
x_train['Original Release Date'] = x_train['Original Release Date'].dt.year.astype(float)
x_train['Current Version Release Date'] = x_train['Current Version Release Date'].dt.year.astype(float)

# Remove the primary genre from the "Genres" feature
x_train['Genres'] = x_train['Genres'].apply(lambda x: x.replace(' ', '').split(','))
x_train['Genres'] = genres_weight(x_train)

data = x_train.join(y_train)
data = remove_special_chars(data, 'Developer')
y_train = data['Rate']
x_train = data.drop('Rate', axis=1)
x_train['Developer'] = x_train['Developer'].str.replace(r'\\xe7\\xe3o', ' ', regex=True)

# Encode categorical columns (Developer, Languages and Primary Genre)
dev_encoder = CustomLabelEncoder()
x_train['Developer'] = dev_encoder.fit_transform(x_train['Developer'])
developer_col= x_train['Developer']

# Feature selection using spearman method
data = x_train.join(y_train)

mapping = {'Low': 0, 'Intermediate': 1, 'High': 2}
data['Rate'] = data['Rate'].map(mapping)

game_data = data.iloc[:, :]
corr = game_data.corr(method='kendall')
# Top 50% Correlation training features with the Value
top_feature = corr.index[abs(corr['Rate']) > 0.03]

x_data = game_data[top_feature]
rate =  x_data['Rate']
x_data = x_data.drop('Rate', axis=1)
x_data['Genres'] = game_data['Genres']
x_data['Developer'] = game_data['Developer']
x_data['Rate'] = rate

# Correlation plot
# plt.subplots(figsize=(12, 8))
# top_corr = game_data[top_feature].corr(method='spearman')
# sns.heatmap(top_corr, annot=True)
# plt.show()

# drop last column
top_feature = top_feature[:-1]

# Add additional features to top_feature list
additional_features = ['Genres','Developer','Rate']
top_feature = pd.Index(np.concatenate([top_feature.values, additional_features]))


print(top_feature)

# Standardize the data
standardization = StandardScaler()
game_data = standardization.fit_transform(x_data)
game_data = pd.DataFrame(game_data, columns=top_feature)
y_train = x_data['Rate']
x_train = x_data.drop('Rate', axis=1)

# print(x_train)

# ---------------------------------Testing Preprocessing-----------------------------------

x_test = x_test.drop(unimportant_columns, axis=1)

# print(x_test.isnull().sum())
x_test['In-app Purchases'].fillna(0, inplace=True)
# print(x_test.isnull().sum())

x_test['In-app Purchases'] = calc_sum_of_list(df['In-app Purchases'])
x_test['In-app Purchases'].fillna(0, inplace=True)

x_test['Size'] = x_test['Size'].astype(int)
x_test['User Rating Count'] = x_test['User Rating Count'].astype(int)
x_test['In-app Purchases'] = x_test['In-app Purchases'].astype(int)

# Convert the date columns to datetime format
x_test['Original Release Date'] = pd.to_datetime(x_test['Original Release Date'], format='%d/%m/%Y')
x_test['Current Version Release Date'] = pd.to_datetime(x_test['Current Version Release Date'], format='%d/%m/%Y')

# Extract the year features
x_test['Original Release Date'] = x_test['Original Release Date'].dt.year.astype(float)
x_test['Current Version Release Date'] = x_test['Current Version Release Date'].dt.year.astype(float)

# Remove the primary genre from the "Genres" feature
x_test['Genres'] = x_test['Genres'].apply(lambda x: x.replace(' ', '').split(','))
x_test['Genres'] = genres_weight(x_test)

data = x_test.join(y_test)
data = remove_special_chars(data, 'Developer')
y_test = data['Rate']
x_test = data.drop('Rate', axis=1)
x_test['Developer'] = x_test['Developer'].str.replace(r'\\xe7\\xe3o', ' ', regex=True)

# Encode categorical columns (Developer, Languages and Primary Genre)
dev_encoder = CustomLabelEncoder()
x_test['Developer'] = dev_encoder.fit_transform(x_test['Developer'])

# Feature selection using spearman method
data = x_test.join(y_test)

mapping = {'Low': 0, 'Intermediate': 1, 'High': 2}
data['Rate'] = data['Rate'].map(mapping)

game_data = data.iloc[:, :]
corr = game_data.corr(method='kendall')
# Top 50% Correlation training features with the Value
top_feature = corr.index[abs(corr['Rate']) > 0.03]

x_data = game_data[top_feature]
rate =  x_data['Rate']
x_data = x_data.drop('Rate', axis=1)
x_data['Genres'] = game_data['Genres']
x_data['Rate'] = rate

# Correlation plot
# plt.subplots(figsize=(12, 8))
# top_corr = game_data[top_feature].corr(method='spearman')
# sns.heatmap(top_corr, annot=True)
# plt.show()

# drop last column
top_feature = top_feature[:-1]

# Add additional features to top_feature list
additional_features = ['Genres','Rate']
top_feature = pd.Index(np.concatenate([top_feature.values, additional_features]))

print(top_feature)

# Standardize the data
standardization = StandardScaler()
game_data = standardization.fit_transform(x_data)
game_data = pd.DataFrame(game_data, columns=top_feature)
y_test = x_data['Rate']
x_test = x_data.drop('Rate', axis=1)

# get the order of columns in the training data
train_columns = list(x_train.columns)

# reorder the columns in the test data to match the order in the training data
x_test = x_test[train_columns]

# print(x_test)
print()
print("----------------------------------------------------------------------------------------------------------------")
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
print("Naive Bayes classifier")
# Create a Gaussian Naive Bayes classifier
model = GaussianNB()

# Train the model using the training data
model.fit(x_train, y_train)

# Make predictions on the test data
y_pred = model.predict(x_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print()
print("----------------------------------------------------------------------------------------------------------------")
# ----------------------------------------------------------------------------------------------------------------------