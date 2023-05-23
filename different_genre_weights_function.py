import pickle
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
from mlxtend.classifier import StackingCVClassifier
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn import model_selection
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from CustomLabelEncoder import CustomLabelEncoder
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

lemmatizer = WordNetLemmatizer()
vectorizer = TfidfVectorizer()


def preprocess_text(text):
    words = word_tokenize(text.lower())
    words = [lemmatizer.lemmatize(w) for w in words]
    s_words = set(words)
    return s_words


def feature_extraction(description_column):
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


# ----------------------------------------------------------------------------------------------------------------------
# Global dictionary that will store the mean/mode of each feature to use it in validate
global_vars = {}

df = pd.read_csv('games-classification-dataset.csv')

# Split data frame to X and Y
Y = df['Rate']
X = df.drop('Rate', axis=1)

# Split the X and the Y to training and validate sets
x_train, x_validate, y_train, y_validate = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=42)

# drop any unnecessary columns
unimportant_columns = ['Name', 'URL', 'Subtitle', 'Icon URL',
                       'Primary Genre', 'ID']  # ,'ID','Price','Age Rating','Languages'
global_vars['Languages'] = x_train['Languages'].mode().iloc[0]
global_vars['In-app Purchase'] = 0
global_vars['Age Rating'] = x_train['Age Rating'].mode().iloc[0]
global_vars['Primary Genre'] = x_train['Primary Genre'].mode().iloc[0]
x_train = x_train.drop(unimportant_columns, axis=1)
x_train['Description'] = feature_extraction(x_train['Description'])
# print(x_train.isnull().sum())
x_train['In-app Purchases'].fillna(0, inplace=True)
# print(x_train.isnull().sum())

x_train['In-app Purchases'] = calc_sum_of_list(df['In-app Purchases'])
x_train['In-app Purchases'].fillna(0, inplace=True)

x_train['Size'] = x_train['Size'].astype(int)
x_train['User Rating Count'] = x_train['User Rating Count'].astype(int)
x_train['In-app Purchases'] = x_train['In-app Purchases'].astype(int)

# Remove the '+' sign from the 'Age rating' column
x_train['Age Rating'] = x_train['Age Rating'].str.replace('+', '', regex=False)
# Convert the 'Age rating' column to an integer data type
x_train['Age Rating'] = x_train['Age Rating'].astype(int)

# Convert the date columns to datetime format
x_train['Original Release Date'] = pd.to_datetime(x_train['Original Release Date'], format='%d/%m/%Y')
x_train['Current Version Release Date'] = pd.to_datetime(x_train['Current Version Release Date'], format='%d/%m/%Y')

# Extract the year features
x_train['Original Release Date'] = x_train['Original Release Date'].dt.year.astype(int)
x_train['Current Version Release Date'] = x_train['Current Version Release Date'].dt.year.astype(int)

global_vars['Original Release Date'] = int(datetime.now().year)
global_vars['Current Version Release Date'] = int(datetime.now().year)

# Remove the primary genre from the "Genres" feature
x_train['Genres'] = remove_first_word(x_train['Genres'])
x_train['Genres'] = x_train['Genres'].apply(lambda x: x.replace(' ', '').split(','))
x_train['Genres'] = x_train['Genres'].apply(weight_genres)
x_train['weighted genres'] = x_train['Genres'].apply(lambda x: sum(x.values()))
x_train['Genres'] = x_train['weighted genres']
x_train.drop('weighted genres', axis=1)
# Create a list of all unique genres in the dataset
# x_train['Genres'] = genres_weight(x_train)

data = x_train.join(y_train)
data = remove_special_chars(data, 'Developer')
y_train = data['Rate']
x_train = data.drop('Rate', axis=1)
x_train['Developer'] = x_train['Developer'].str.replace(r'\\xe7\\xe3o', ' ', regex=True)

# filling Developer with a default value "Unknown"
global_vars['Developer'] = 'Unknown'
global_vars['User Rating Count'] = x_train['User Rating Count'].mean()
global_vars['Size'] = x_train['Size'].mean()
global_vars['Original Release Date'] = datetime.now()
global_vars['Current Version Release Date'] = datetime.now()
global_vars['Genres'] = global_vars['Primary Genre']
global_vars['Description'] = 'No Description'

# Encode categorical columns (Developer, Languages and Primary Genre)
dev_encoder = CustomLabelEncoder()
lang_encoder = CustomLabelEncoder()
x_train['Languages'] = lang_encoder.fit_transform(x_train['Languages'])
x_train['Developer'] = dev_encoder.fit_transform(x_train['Developer'])

# Feature selection using kendall method
data = x_train.join(y_train)

mapping = {'Low': 0, 'Intermediate': 1, 'High': 2}
data['Rate'] = data['Rate'].map(mapping)

game_data = data.iloc[:, :]
corr = game_data.corr(method='kendall')
# Top 1% Correlation training features with the Value
top_feature = corr.index[abs(corr['Rate']) > 0.01]

x_data = game_data[top_feature]

print("Top featuressssss:",top_feature)

# Standardize the data
standardization = StandardScaler()
game_data = standardization.fit_transform(x_data)
game_data = pd.DataFrame(game_data, columns=top_feature)
y_train = x_data['Rate']
x_train = x_data.drop('Rate', axis=1)

# print(x_train)
# ---------------------------------validate data Preprocessing-----------------------------------

x_validate = x_validate.drop(unimportant_columns, axis=1)

x_validate['Description'] = feature_extraction(x_validate['Description'])

# print(x_validate.isnull().sum())
# print(x_validate.isnull().sum())

x_validate['In-app Purchases'] = calc_sum_of_list(df['In-app Purchases'])

x_validate['Size'] = x_validate['Size'].astype(int)
x_validate['User Rating Count'] = x_validate['User Rating Count'].astype(int)
x_validate['In-app Purchases'].fillna(0, inplace=True)
x_validate['In-app Purchases'] = x_validate['In-app Purchases'].astype(int)

# Remove the '+' sign from the 'Age rating' column
x_validate['Age Rating'] = x_validate['Age Rating'].str.replace('+', '', regex=False)

# Convert the 'Age rating' column to an integer data type
x_validate['Age Rating'] = x_validate['Age Rating'].astype(int)

# Convert the date columns to datetime format
x_validate['Original Release Date'] = pd.to_datetime(x_validate['Original Release Date'], format='%d/%m/%Y')
x_validate['Current Version Release Date'] = pd.to_datetime(x_validate['Current Version Release Date'],
                                                            format='%d/%m/%Y')

# Extract the year features
x_validate['Original Release Date'] = x_validate['Original Release Date'].dt.year.astype(int)
x_validate['Current Version Release Date'] = x_validate['Current Version Release Date'].dt.year.astype(int)

for col in x_validate.columns:
    if col == 'In-app Purchases' or col == 'Price':
        x_validate[col] = fill_nulls(x_validate[col], 0)
    else:
        x_validate[col].fillna(global_vars[col], inplace=True)

# Remove the primary genre from the "Genres" feature
x_validate['Genres'] = x_validate['Genres'].apply(lambda x: x.replace(' ', '').split(','))
x_validate['Genres'] = x_validate['Genres'].apply(weight_genres)
x_validate['weighted genres'] = x_validate['Genres'].apply(lambda x: sum(x.values()))
x_validate['Genres'] = x_validate['weighted genres']
x_validate.drop('weighted genres', axis=1)
# x_validate['Genres'] = genres_weight(x_validate)

data = x_validate.join(y_validate)
data = remove_special_chars(data, 'Developer')
y_validate = data['Rate']
x_validate = data.drop('Rate', axis=1)
x_validate['Developer'] = x_validate['Developer'].str.replace(r'\\xe7\\xe3o', ' ', regex=True)

# Encode categorical columns (Developer, Languages and Primary Genre)
x_validate['Developer'] = dev_encoder.transform(x_validate['Developer'])
x_validate['Languages'] = lang_encoder.transform(x_validate['Languages'])

# Feature selection using spearman method
data = x_validate.join(y_validate)

mapping = {'Low': 0, 'Intermediate': 1, 'High': 2}
data['Rate'] = data['Rate'].map(mapping)

game_data = data.iloc[:, :]
corr = game_data.corr(method='kendall')
# Top Correlated training features with the Value

x_data = game_data[top_feature]


# Standardize the data

game_data = standardization.transform(x_data)
game_data = pd.DataFrame(game_data, columns=top_feature)
y_validate = x_data['Rate']
x_validate = x_data.drop('Rate', axis=1)

# get the order of columns in the training data
train_columns = list(x_train.columns)

# reorder the columns in the validate data to match the order in the training data
x_validate = x_validate[train_columns]

# print(x_validate)
print()
print(
    "----------------------------------------------------------------------------------------------------------------")
# ----------------------------------------------------------------------------------------------------------------------
print("Decision Tree")

# Define the decision tree model with default parameters
decisionTree = DecisionTreeClassifier(max_depth=5, random_state=42)
# Fit the model to the training data using cross-validation
scores = cross_val_score(decisionTree, x_train, y_train, cv=5)

# Print the cross-validation scores
print("Cross-validation scores:", scores)
print("Mean accuracy:", scores.mean())

# Fit the model to the training data without cross-validation
decisionTree.fit(x_train, y_train)

# Predict the labels of the training data
y_pred_train = decisionTree.predict(x_train)

# Predict the labels of the validate data
y_pred_validate = decisionTree.predict(x_validate)

# plt.scatter(y_train, y_pred_train)
# plt.xlabel('y_train ', fontsize = 20)
# plt.ylabel('y_pred_train', fontsize = 20)
# plt.plot(y_train , y_pred_train, color='red', linewidth = 3)
# plt.show()
#
#
# plt.scatter(y_validate, y_pred_validate)
# plt.xlabel('y_validate ', fontsize = 20)
# plt.ylabel('y_pred_validate', fontsize = 20)
# plt.plot(y_validate, y_pred_validate, color='red', linewidth = 3)
# plt.show()


# Visualize the decision tree
plt.figure(figsize=(20, 10))
plot_tree(decisionTree, filled=True)
plt.show()

# Calculate the accuracy of the model on the training data
accuracy_train = accuracy_score(y_train, y_pred_train)

# Calculate the accuracy of the model on the validate data
accuracy_validate = accuracy_score(y_validate, y_pred_validate)

print("Accuracy on training data:", accuracy_train)
print("Accuracy on validate data:", accuracy_validate)
print()
print(
    "----------------------------------------------------------------------------------------------------------------")
# -----------------------------------------------------------------------------------------------------------------------
print("Random Forest")

# Create a Random Forest classifier with 100 trees
random_forest = RandomForestClassifier(n_estimators=6, max_depth=6)

# Perform 5-fold cross-validation on the training data
scores = cross_val_score(random_forest, x_train, y_train, cv=5)

# Print the cross-validation scores
print("Cross-validation scores:", scores)
print("Mean accuracy:", scores.mean())

# Fit the model to the training data without cross-validation
random_forest.fit(x_train, y_train)

# Predict on the training data
y_pred_train = random_forest.predict(x_train)

# Predict on the validate data
y_pred_validate = random_forest.predict(x_validate)

#
# plt.scatter(y_train, y_pred_train)
# plt.xlabel('y_train ', fontsize = 20)
# plt.ylabel('y_pred_train', fontsize = 20)
# plt.plot(y_train , y_pred_train, color='red', linewidth = 3)
# plt.show()
#
#
# plt.scatter(y_validate, y_pred_validate)
# plt.xlabel('y_validate ', fontsize = 20)
# plt.ylabel('y_pred_validate', fontsize = 20)
# plt.plot(y_validate, y_pred_validate, color='red', linewidth = 3)
# plt.show()

plt.figure(figsize=(20, 10))
plot_tree(random_forest.estimators_[0], feature_names=x_train.columns)
plt.show()

# Calculate the accuracy of the model on the training data
accuracy_train = accuracy_score(y_train, y_pred_train)

# Calculate the accuracy of the model on the validate data
accuracy_validate = accuracy_score(y_validate, y_pred_validate)

print("Accuracy on training data:", accuracy_train)
print("Accuracy on validate data:", accuracy_validate)
print()
print(
    "----------------------------------------------------------------------------------------------------------------")
# -----------------------------------------------------------------------------------------------------------------------
C = 0.1  # SVM regularization parameter

print("SVM WITHOUT kernal")

# Create an SVM classifier with a polynomial kernel of degree 5
svm_clf = svm.SVC(C=C)

# Perform 5-fold cross-validation on the training data
scores = cross_val_score(svm_clf, x_train, y_train, cv=5)

# Print the cross-validation scores
print("Cross-validation scores:", scores)
print("Mean accuracy:", scores.mean())

# Fit the model to the training data without cross-validation
svm_clf.fit(x_train, y_train)

# Predict on the training data
y_pred_train = svm_clf.predict(x_train)

# Predict on the validate data
y_pred_validate = svm_clf.predict(x_validate)

#
# plt.scatter(y_train, y_pred_train)
# plt.xlabel('y_train ', fontsize = 20)
# plt.ylabel('y_pred_train', fontsize = 20)
# plt.plot(y_train , y_pred_train, color='red', linewidth = 3)
# plt.show()
#
#
# plt.scatter(y_validate, y_pred_validate)
# plt.xlabel('y_validate ', fontsize = 20)
# plt.ylabel('y_pred_validate', fontsize = 20)
# plt.plot(y_validate, y_pred_validate, color='red', linewidth = 3)
# plt.show()
#
# import matplotlib.pyplot as plt
# from mlxtend.plotting import plot_decision_regions
#
# plot_decision_regions(x_train.values, y_train.values, clf=svm_clf, legend=2, feature_index=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
#
# # Adding axes annotations
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('SVM poly')
#
# plt.show()


# Calculate the accuracy of the model on the training data
accuracy_train = accuracy_score(y_train, y_pred_train)

# Calculate the accuracy of the model on the validate data
accuracy_validate = accuracy_score(y_validate, y_pred_validate)

print("Accuracy on training data:", accuracy_train)
print("Accuracy on validate data:", accuracy_validate)
print()
print(
    "----------------------------------------------------------------------------------------------------------------")
print("SVM poly")

# Create an SVM classifier with a polynomial kernel of degree 5
svm_clf_poly = svm.SVC(kernel='poly', degree=2,C=C)

# Perform 5-fold cross-validation on the training data
scores = cross_val_score(svm_clf_poly, x_train, y_train, cv=5)

# Print the cross-validation scores
print("Cross-validation scores:", scores)
print("Mean accuracy:", scores.mean())

# Fit the model to the training data without cross-validation
svm_clf_poly.fit(x_train, y_train)

# Predict on the training data
y_pred_train = svm_clf_poly.predict(x_train)

# Predict on the validate data
y_pred_validate = svm_clf_poly.predict(x_validate)

#
# plt.scatter(y_train, y_pred_train)
# plt.xlabel('y_train ', fontsize = 20)
# plt.ylabel('y_pred_train', fontsize = 20)
# plt.plot(y_train , y_pred_train, color='red', linewidth = 3)
# plt.show()
#
#
# plt.scatter(y_validate, y_pred_validate)
# plt.xlabel('y_validate ', fontsize = 20)
# plt.ylabel('y_pred_validate', fontsize = 20)
# plt.plot(y_validate, y_pred_validate, color='red', linewidth = 3)
# plt.show()
#
# import matplotlib.pyplot as plt
# from mlxtend.plotting import plot_decision_regions
#
# plot_decision_regions(x_train.values, y_train.values, clf=svm_clf, legend=2, feature_index=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
#
# # Adding axes annotations
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('SVM poly')
#
# plt.show()


# Calculate the accuracy of the model on the training data
accuracy_train = accuracy_score(y_train, y_pred_train)

# Calculate the accuracy of the model on the validate data
accuracy_validate = accuracy_score(y_validate, y_pred_validate)

print("Accuracy on training data:", accuracy_train)
print("Accuracy on validate data:", accuracy_validate)
print()
print(
    "----------------------------------------------------------------------------------------------------------------")
# ----------------------------------------------------------------------------------------------------------------------
print("SVM sigmoid")

# Create an SVM classifier with a sigmoid kernel
svm_clf2 = svm.SVC(kernel='sigmoid',C=C)

# Perform 5-fold cross-validation on the training data
scores = cross_val_score(svm_clf2, x_train, y_train, cv=5)

# Print the cross-validation scores
print("Cross-validation scores:", scores)
print("Mean accuracy:", scores.mean())

# Fit the model to the training data without cross-validation
svm_clf2.fit(x_train, y_train)

# Predict on the training data
y_pred_train = svm_clf2.predict(x_train)

# Predict on the validate data
y_pred_validate = svm_clf2.predict(x_validate)

# plt.scatter(y_train, y_pred_train)
# plt.xlabel('y_train ', fontsize = 20)
# plt.ylabel('y_pred_train', fontsize = 20)
# plt.plot(y_train , y_pred_train, color='red', linewidth = 3)
# plt.show()
#
#
# plt.scatter(y_validate, y_pred_validate)
# plt.xlabel('y_validate ', fontsize = 20)
# plt.ylabel('y_pred_validate', fontsize = 20)
# plt.plot(y_validate, y_pred_validate, color='red', linewidth = 3)
# plt.show()


# Calculate the accuracy of the model on the training data
accuracy_train = accuracy_score(y_train, y_pred_train)

# Calculate the accuracy of the model on the validate data
accuracy_validate = accuracy_score(y_validate, y_pred_validate)

print("Accuracy on training data:", accuracy_train)
print("Accuracy on validate data:", accuracy_validate)
print()
print(
    "----------------------------------------------------------------------------------------------------------------")
# ----------------------------------------------------------------------------------------------------------------------
print("Logistic Regression")

# Create a logistic regression model
lr_model = LogisticRegression()

# Perform 5-fold cross-validation on the training data
scores = cross_val_score(lr_model, x_train, y_train, cv=5)

# Print the cross-validation scores
print("Cross-validation scores:", scores)
print("Mean accuracy:", scores.mean())

# Train the model on the training set without cross-validation
lr_model.fit(x_train, y_train)

# Predict on the training data
y_pred_train = lr_model.predict(x_train)

# Predict on the validate data
y_pred_validate = lr_model.predict(x_validate)

# plt.scatter(y_train, y_pred_train)
# plt.xlabel('y_train ', fontsize = 20)
# plt.ylabel('y_pred_train', fontsize = 20)
# plt.plot(y_train , y_pred_train, color='red', linewidth = 3)
# plt.show()
#
#
# plt.scatter(y_validate, y_pred_validate)
# plt.xlabel('y_validate ', fontsize = 20)
# plt.ylabel('y_pred_validate', fontsize = 20)
# plt.plot(y_validate, y_pred_validate, color='red', linewidth = 3)
# plt.show()
#

x = pd.concat([x_validate, x_train])
y = pd.concat([y_validate, y_train])
df = x.join(y)


sns.regplot(x=x_validate['Size'], y=y_pred_validate, data=df, logistic=True, ci=None)
plt.plot(x_validate, y_pred_validate)
plt.show()

# Calculate the accuracy of the model on the training data
accuracy_train = accuracy_score(y_train, y_pred_train)

# Calculate the accuracy of the model on the validate data
accuracy_validate = accuracy_score(y_validate, y_pred_validate)

print("Accuracy on training data:", accuracy_train)
print("Accuracy on validate data:", accuracy_validate)
print()
print(
    "----------------------------------------------------------------------------------------------------------------")
# ----------------------------------------------------------------------------------------------------------------------
print("Naive Bayes classifier")

# Create a Gaussian Naive Bayes classifier
GNB = GaussianNB()

# Perform 5-fold cross-validation on the training data
scores = cross_val_score(GNB, x_train, y_train, cv=5)

# Print the cross-validation scores
print("Cross-validation scores:", scores)
print("Mean accuracy:", scores.mean())

# Train the model using the training data without cross-validation
GNB.fit(x_train, y_train)

# Predict on the training data
y_pred_train = GNB.predict(x_train)

# Predict on the validate data
y_pred_validate = GNB.predict(x_validate)

#
# plt.scatter(y_train, y_pred_train)
# plt.xlabel('y_train ', fontsize = 20)
# plt.ylabel('y_pred_train', fontsize = 20)
# plt.plot(y_train , y_pred_train, color='red', linewidth = 3)
# plt.show()
#
#
# plt.scatter(y_validate, y_pred_validate)
# plt.xlabel('y_validate ', fontsize = 20)
# plt.ylabel('y_pred_validate', fontsize = 20)
# plt.plot(y_validate, y_pred_validate, color='red', linewidth = 3)
# plt.show()
#
# from mlxtend.plotting import plot_decision_regions
#
# plot_decision_regions(x_train.values, y_train.values, clf=model, legend=2, feature_index=(0, 1, 2, 3, 4, 5, 6, 7,
# 8, 9)) # Adding axes annotations plt.xlabel('X') plt.ylabel('Y') plt.title('Naive')

# plt.show()

# Calculate the accuracy of the model on the training data
accuracy_train = accuracy_score(y_train, y_pred_train)

# Calculate the accuracy of the model on the validate data
accuracy_validate = accuracy_score(y_validate, y_pred_validate)

print("Accuracy on training data:", accuracy_train)
print("Accuracy on validate data:", accuracy_validate)
print()
print(
    "----------------------------------------------------------------------------------------------------------------")
# ----------------------------------------------------------------------------------------------------------------------
print("KNN")

# Define a range of K values to validate
k_values = range(1, 15)

# Initialize empty lists to store the training and validate accuracy scores
train_scores = []
validate_scores = []

# Loop over each K value and fit the KNN model,
# then compute training and validate accuracy scores using cross-validation

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    train_cv_scores = cross_val_score(knn, x_train, y_train, cv=3)
    validate_cv_scores = cross_val_score(knn, x_validate, y_validate, cv=3)
    train_scores.append(train_cv_scores.mean())
    validate_scores.append(validate_cv_scores.mean())

# Plot the training and validate accuracy scores as a function of K
plt.plot(k_values, train_scores, label="Training")
plt.plot(k_values, validate_scores, label="validate")
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.title("KNN Accuracy Scores")
plt.legend()
plt.show()

knn = KNeighborsClassifier(n_neighbors=1)
train_cv_scores = cross_val_score(knn, x_train, y_train, cv=5)
validate_cv_scores = cross_val_score(knn, x_validate, y_validate, cv=5)

# Print the mean and standard deviation of the accuracy scores
print("Training Accuracy:", train_cv_scores.mean())
print("validate Accuracy:", validate_cv_scores.mean())

print()
print(
    "----------------------------------------------------------------------------------------------------------------")
# ----------------------------------------------------------------------------------------------------------------------
print("AdaBoost Model / Boosting Ensemble Learning")

# Initialize the base model
base_clf = DecisionTreeClassifier(max_depth=1)

# Create an adaboost classifer object
abc = AdaBoostClassifier(estimator=base_clf, n_estimators=100, learning_rate=0.5)

base_clf = base_clf.fit(x_train, y_train)
# Perform 5-fold cross-validation on the training data
scores = cross_val_score(abc, x_train, y_train, cv=5)

# Print the cross-validation scores
print("Cross-validation scores:", scores)
print("Mean accuracy:", scores.mean())

# Train Adaboost Classifer on the training data without cross-validation
ada = abc.fit(x_train, y_train)

# Predict the response for training dataset
y_pred_train = ada.predict(x_train)

# Predict the response for validate dataset
y_pred_validate = ada.predict(x_validate)

# Calculate the accuracy of the model on the training data
accuracy_train = metrics.accuracy_score(y_train, y_pred_train)

# Calculate the accuracy of the model on the validate data
accuracy_validate = metrics.accuracy_score(y_validate, y_pred_validate)

# Print the accuracies
print("Accuracy on training data:", accuracy_train)
print("Accuracy on validate data:", accuracy_validate)

# Visualize the decision tree
plt.figure(figsize=(20, 10))
plot_tree(base_clf, filled=True)
plt.show()

print()
print(
    "----------------------------------------------------------------------------------------------------------------")
# ----------------------------------------------------------------------------------------------------------------------
print("Bagging Ensemble Learning")

# Create a base estimator
base_estimator = DecisionTreeClassifier(max_depth=5, random_state=42)

# Create a Bagging classifier with 30 estimators
bagging = BaggingClassifier(estimator=base_estimator, n_estimators=30, max_samples=0.8, max_features=0.8)

# Perform 5-fold cross-validation on the training data
scores = cross_val_score(bagging, x_train, y_train, cv=5)

# Print the cross-validation scores
print("Cross-validation scores:", scores)
print("Mean accuracy:", scores.mean())

# Train the Bagging classifier on the training data without cross-validation
bagging.fit(x_train, y_train)

# Predict the labels for both the training and validate data
y_train_pred = bagging.predict(x_train)
y_validate_pred = bagging.predict(x_validate)

# plt.scatter(y_train, y_pred_train)
# plt.xlabel('y_train ', fontsize = 20)
# plt.ylabel('y_pred_train', fontsize = 20)
# plt.plot(y_train , y_pred_train, color='red', linewidth = 3)
# plt.show()
#
#
# plt.scatter(y_validate, y_pred_validate)
# plt.xlabel('y_validate ', fontsize = 20)
# plt.ylabel('y_pred_validate', fontsize = 20)
# plt.plot(y_validate, y_pred_validate, color='red', linewidth = 3)
# plt.show()
#


plt.plot(x_validate, y_validate_pred)
plt.show()

# Calculate the accuracy of the model on the training and validate data
train_accuracy = accuracy_score(y_train, y_train_pred)
validate_accuracy = accuracy_score(y_validate, y_validate_pred)

# Print the accuracy scores
print("Accuracy on training data:", train_accuracy)
print("Accuracy on validate data:", validate_accuracy)
print()
print(
    "----------------------------------------------------------------------------------------------------------------")
# ----------------------------------------------------------------------------------------------------------------------
print("Stacking Ensemble Learning")

# Initialize the base models
lr = LogisticRegression()
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
# knn = KNeighborsClassifier(n_neighbors=1)
rf = RandomForestClassifier(n_estimators=6, max_depth=6)
bg = BaggingClassifier(estimator=base_estimator, n_estimators=30, max_samples=0.8, max_features=0.8)
ab = AdaBoostClassifier(estimator=base_clf, n_estimators=100, learning_rate=0.5)

# Fit the base models on the train set
lr.fit(x_train, y_train)
dt.fit(x_train, y_train)
# knn.fit(x_train, y_train)
rf.fit(x_train, y_train)
bg.fit(x_train, y_train)
ab.fit(x_train, y_train)

# Make predictions on the train set using the base models
lr_pred_train = lr.predict(x_train)
dt_pred_train = dt.predict(x_train)
# knn_pred_train = knn.predict(x_train)
rf_pred_train = rf.predict(x_train)
bg_pred_train = bg.predict(x_train)
ab_pred_train = ab.predict(x_train)

# Use the predictions from the base models as input features to train the metamodel on the train set
meta_X_train = np.column_stack((lr_pred_train, dt_pred_train, rf_pred_train, bg_pred_train, ab_pred_train))
meta_clf = LogisticRegression()
meta_clf.fit(meta_X_train, y_train)

# Make predictions on the train set using the metamodel
meta_pred_train = meta_clf.predict(meta_X_train)

# Use the predictions from the base models as input features to make predictions on the validate set using the metamodel
lr_pred_validate = lr.predict(x_validate)
dt_pred_validate = dt.predict(x_validate)
# knn_pred_validate = knn.predict(x_validate)
rf_pred_validate = rf.predict(x_validate)
bg_pred_validate = bg.predict(x_validate)
ab_pred_validate = ab.predict(x_validate)

meta_x_validate = np.column_stack(
    (lr_pred_validate, dt_pred_validate, rf_pred_validate, bg_pred_validate, ab_pred_validate))
meta_pred_validate = meta_clf.predict(meta_x_validate)

#
# plt.scatter(y_train, meta_pred_train)
# plt.xlabel('y_train ', fontsize = 20)
# plt.ylabel('meta_pred_train', fontsize = 20)
# plt.plot(y_train , meta_pred_train, color='red', linewidth = 3)
# plt.show()
#
#
# plt.scatter(y_validate, meta_pred_validate)
# plt.xlabel('y_validate ', fontsize = 20)
# plt.ylabel('meta_pred_validate', fontsize = 20)
# plt.plot(y_validate, meta_pred_validate, color='red', linewidth = 3)
# plt.show()

plt.plot(x_validate, meta_pred_validate)
plt.show()

# Evaluate the performance of the metamodel on the train and validate sets
train_accuracy = accuracy_score(y_train, meta_pred_train)
validate_accuracy = accuracy_score(y_validate, meta_pred_validate)

print('Accuracy of stacking ensemble on train set:', train_accuracy)
print('Accuracy of stacking ensemble on validate set:', validate_accuracy)

print()
print(
    "----------------------------------------------------------------------------------------------------------------")
# ----------------------------------------------------------------------------------------------------------------------
print("Simple Stacking CV Classification")

RANDOM_SEED = 42

clf1 = AdaBoostClassifier(estimator=base_clf, n_estimators=100, learning_rate=0.5)
clf2 = RandomForestClassifier(random_state=RANDOM_SEED)
clf3 = GaussianNB()
Lr = LogisticRegression()

# Starting from v0.16.0, StackingCVRegressor supports
# `random_state` to get deterministic result.
sclf = StackingCVClassifier(classifiers=[clf1, clf2, clf3],
                            meta_classifier=Lr,
                            random_state=RANDOM_SEED)

print('3-fold cross validation:\n')

for clf, label in zip([clf1, clf2, clf3, sclf],
                      ['AdaBoost',
                       'Random Forest',
                       'Naive Bayes',
                       'StackingClassifier']):
    scores = model_selection.cross_val_score(clf, x_validate, y_validate, cv=3, scoring='accuracy')
    print("Accuracy: ", scores.mean(), " Model: ", label)

# region Store models and vars in .pkl file
models = {'decisionTree': decisionTree, 'random_forest': random_forest, 'svm_clf': svm_clf, 'svm_clf2': svm_clf2,'svm_clf_poly': svm_clf_poly,
          'lr_model': lr_model, 'GNB': GNB, 'knn': knn, 'base_clf': base_clf, 'ada': ada, 'bagging': bagging, 'lr': lr,
          'dt': dt, 'rf': rf, 'bg': bg, 'ab': ab, 'meta_clf': meta_clf, 'clf1': clf1, 'clf2': clf2, 'clf3': clf3,
          'Lr': Lr,'sclf':sclf}
variables = {'x_train': x_train, 'y_train': y_train, 'unimportant_columns': unimportant_columns,
             'top_features': top_feature,
             'standardization': standardization, 'global_vars': global_vars, 'dev_encoder': dev_encoder,
             'lang_encoder': lang_encoder}
pickle.dump(models, open('classifiers.pkl', 'wb'))
pickle.dump(variables, open('variables.pkl', 'wb'))
# endregion

# import matplotlib.pyplot as plt
# from mlxtend.plotting import plot_decision_regions
# import matplotlib.gridspec as gridspec
# import itertools
#
# gs = gridspec.GridSpec(2, 2)
#
# fig = plt.figure(figsize=(10,8))
#
# for clf, lab, grd in zip([clf1, clf2, clf3, sclf],
#                          ['AdaBoost',
#                           'Random Forest',
#                           'Naive Bayes',
#                           'StackingCVClassifier'],
#                           itertools.product([0, 1], repeat=2)):
#
#     clf.fit(x_validate, y_validate)
#     ax = plt.subplot(gs[grd[0], grd[1]])
#     fig = plot_decision_regions(X=x_validate.values, y=y_validate.values, clf=clf,legend=2  ,filler_feature_values={ 1: 0.5, 0: 0.5,2: 0.5, 3: 0.5, 4: 0.5, 5: 0.5, 6: 0.5, 7: 0.5, 8: 0.5, 9: 0.5, 10: 0.5},)
#     plt.title(lab)
# plt.show()
