import pickle
import seaborn as sns
import nltk
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import plot_tree
from sklearn import metrics, __all__, model_selection

# region functions used
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
# endregion

# region Load models and variables
models_list = pickle.load(open('classifiers.pkl', 'rb'))
decisionTree = models_list['decisionTree']
random_forest = models_list['random_forest']
svm_clf = models_list['svm_clf']
svm_clf_poly = models_list['svm_clf_poly']
svm_clf2 = models_list['svm_clf2']
lr_model = models_list['lr_model']
GNB = models_list['GNB']
knn = models_list['knn']
base_clf = models_list['base_clf']
ada = models_list['ada']
sclf = models_list['sclf']
bagging = models_list['bagging']
lr = models_list['lr']
dt = models_list['dt']
bg = models_list['bg']
ab = models_list['ab']
rf = models_list['rf']
meta_clf = models_list['meta_clf']
clf1 = models_list['clf1']
clf2 = models_list['clf2']
clf3 = models_list['clf3']
Lr = models_list['Lr']

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
print()

# ------------------------------[Decision Tree]---------------------------------------------
print("Decision Tree : ")
# Predict the labels of the test data
y_pred_test = decisionTree.predict(x_test)
# Visualize the decision tree
plt.figure(figsize=(20, 10))
plot_tree(decisionTree, filled=True)
# plt.show()
# Predict the labels of the training data
y_pred_train = decisionTree.predict(x_train)

# Calculate the accuracy of the model on the training data
accuracy_train = accuracy_score(y_train, y_pred_train)

# Calculate the accuracy of the model on the test data
accuracy_test = accuracy_score(y_test, y_pred_test)

print("Accuracy on training data:", accuracy_train)
print("Accuracy on test data:", accuracy_test)
print()

# ------------------------------[Random Forest]---------------------------------------------
print("Random forest : ")
# Predict on the test data
y_pred_test = random_forest.predict(x_test)

plt.figure(figsize=(20, 10))
plot_tree(random_forest.estimators_[0], feature_names=x_train.columns)
plt.show()

# Predict on the training data
y_pred_train = random_forest.predict(x_train)

# Calculate the accuracy of the model on the training data
accuracy_train = accuracy_score(y_train, y_pred_train)

# Calculate the accuracy of the model on the test data
accuracy_test = accuracy_score(y_test, y_pred_test)

print("Accuracy on training data:", accuracy_train)
print("Accuracy on test data:", accuracy_test)
print()

# ------------------------------[SVM Poly]---------------------------------------------
print("SVM poly : ")
# Predict on the training data
y_pred_train = svm_clf.predict(x_train)

# Predict on the test data
y_pred_test = svm_clf.predict(x_test)

# Calculate the accuracy of the model on the training data
accuracy_train = accuracy_score(y_train, y_pred_train)

# Calculate the accuracy of the model on the test data
accuracy_test = accuracy_score(y_test, y_pred_test)

print("Accuracy on training data:", accuracy_train)
print("Accuracy on test data:", accuracy_test)
print()

# ------------------------------[SVM Sigmoid]---------------------------------------------
print("SVM sigmoid")
# Predict on the training data
y_pred_train = svm_clf2.predict(x_train)

# Predict on the test data
y_pred_test = svm_clf2.predict(x_test)

# Calculate the accuracy of the model on the training data
accuracy_train = accuracy_score(y_train, y_pred_train)

# Calculate the accuracy of the model on the test data
accuracy_test = accuracy_score(y_test, y_pred_test)

print("Accuracy on training data:", accuracy_train)
print("Accuracy on test data:", accuracy_test)
print()

# ------------------------------[Logistic regression]---------------------------------------------
print("Logistic Regression: ")

# Predict on the training data
y_pred_train = lr_model.predict(x_train)

# Predict on the test data
y_pred_test = lr_model.predict(x_test)
x = pd.concat([x_test, x_train])
y = pd.concat([y_test, y_train])
df = x.join(y)
sns.regplot(x=x_test['Size'], y=y_pred_test, data=df, logistic=True, ci=None)
plt.plot(x_test, y_pred_test)
# plt.show()

# Calculate the accuracy of the model on the training data
accuracy_train = accuracy_score(y_train, y_pred_train)

# Calculate the accuracy of the model on the test data
accuracy_test = accuracy_score(y_test, y_pred_test)

print("Accuracy on training data:", accuracy_train)
print("Accuracy on test data:", accuracy_test)
print()

# ------------------------------[Naive bayes classifier]---------------------------------------------
print("Naive Bayes classifier: ")
# Predict on the training data
y_pred_train = GNB.predict(x_train)

# Predict on the test data
y_pred_test = GNB.predict(x_test)

# Calculate the accuracy of the model on the training data
accuracy_train = accuracy_score(y_train, y_pred_train)

# Calculate the accuracy of the model on the test data
accuracy_test = accuracy_score(y_test, y_pred_test)

print("Accuracy on training data:", accuracy_train)
print("Accuracy on test data:", accuracy_test)
print()

# ------------------------------[KNN]---------------------------------------------
train_cv_scores = cross_val_score(knn, x_train, y_train, cv=5)
test_cv_scores = cross_val_score(knn, x_test, y_test, cv=5)

# Print the mean and standard deviation of the accuracy scores
print("Training Accuracy:", train_cv_scores.mean())
print("test Accuracy:", test_cv_scores.mean())
print()

# ---------------------------[AdaBoost]------------------------------------
# Predict the response for training dataset
y_pred_train = ada.predict(x_train)

# Predict the response for test dataset
y_pred_test = ada.predict(x_test)

# Calculate the accuracy of the model on the training data
accuracy_train = metrics.accuracy_score(y_train, y_pred_train)

# Calculate the accuracy of the model on the test data
accuracy_test = metrics.accuracy_score(y_test, y_pred_test)

# Print the accuracies
print("Accuracy on training data:", accuracy_train)
print("Accuracy on test data:", accuracy_test)

# Visualize the decision tree
plt.figure(figsize=(20, 10))
plot_tree(base_clf, filled=True)
# plt.show()
print()

# ---------------------------[Bagging Ensemble learning]------------------------------------
print("Bagging Ensemble Learning: ")
# Predict the labels for both the training and test data
y_train_pred = bagging.predict(x_train)
y_test_pred = bagging.predict(x_test)
plt.plot(x_test, y_test_pred)
plt.show()

# Calculate the accuracy of the model on the training and test data
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Print the accuracy scores
print("Accuracy on training data:", train_accuracy)
print("Accuracy on test data:", test_accuracy)
print()

# ---------------------------[Stacking Ensemble learning]------------------------------------
print('Stacking Ensemble learning')
# Make predictions on the train set using the metamodel
lr_pred_test = lr.predict(x_test)
dt_pred_test = dt.predict(x_test)
# knn_pred_test = knn.predict(x_test)
rf_pred_test = rf.predict(x_test)
bg_pred_test = bg.predict(x_test)
ab_pred_test = ab.predict(x_test)
meta_x_test = np.column_stack(
    (lr_pred_test, dt_pred_test, rf_pred_test, bg_pred_test, ab_pred_test))
meta_pred_test = meta_clf.predict(meta_x_test)
plt.plot(x_test, meta_pred_test)
plt.show()
# Evaluate the performance of the metamodel on the test set
test_accuracy = accuracy_score(y_test, meta_pred_test)

print('Accuracy of stacking ensemble on train set:', train_accuracy)
print('Accuracy of stacking ensemble on test set:', test_accuracy)
print()

# ---------------------------[Simple Stacking CV Classification]------------------------------------
print("Simple Stacking CV Classification: ")
for clf, label in zip([clf1, clf2, clf3, sclf],
                      ['AdaBoost',
                       'Random Forest',
                       'Naive Bayes',
                       'StackingClassifier']):
    scores = model_selection.cross_val_score(clf, x_test, y_test, cv=3, scoring='accuracy')
    print("Accuracy: ", scores.mean(), " Model: ", label)

# def predict_ratings(test_data, true_ratings):
#     # Get the order of columns in the training data
#     train_columns = list(x_train.columns)
#
#     # Reorder the columns in the test data to match the order in the training data
#     test_data = test_data[train_columns]
#
#     # Make predictions using the trained models
#     dt_pred = decisionTree.predict(test_data)
#     rf_pred = random_forest.predict(test_data)
#     svm_pred = svm_clf.predict(test_data)
#     svm_poly_pred = svm_clf_poly.predict(test_data)
#     svm2_pred = svm_clf2.predict(test_data)
#     lr_pred = lr_model.predict(test_data)
#     gnb_pred = GNB.predict(test_data)
#     knn_pred = knn.predict(test_data)
#     ada_pred = ada.predict(test_data)
#     bagging_pred = bagging.predict(test_data)
#
# Use the predictions from the base models as input features to make predictions on the test set using the metamodel
#     lr_pred_test = lr.predict(x_test)
#     dt_pred_test = dt.predict(x_test)
#     rf_pred_test = rf.predict(x_test)
#     bg_pred_test = bg.predict(x_test)
#     ab_pred_test = ab.predict(x_test)
#     meta_x_test = np.column_stack(
#         (lr_pred_test, dt_pred_test, rf_pred_test, bg_pred_test, ab_pred_test))
#     meta_pred_test = meta_clf.predict(meta_x_test)
#
#     # Combine the predictions into a DataFrame
#     pred_df = pd.DataFrame({
#         'Decision Tree': dt_pred,
#         'Random Forest': rf_pred,
#         'SVM': svm_pred,
#         'SVM Poly': svm_poly_pred,
#         'SVM2': svm2_pred,
#         'Logistic Regression': lr_pred,
#         'Naive Bayes': gnb_pred,
#         'k-NN': knn_pred,
#         'AdaBoost': ada_pred,
#         'Bagging': bagging_pred
#     })
#
#     # Calculate the accuracy of each model
#     accuracies = []
#     for col in pred_df.columns:
#         accuracy = sum(pred_df[col] == true_ratings) / len(true_ratings)
#         accuracies.append(accuracy)
#         print(f'{col} Accuracy: {accuracy}')
#
#     # Return the mode of the predicted ratings across all models as the final prediction
#     return pred_df.mode(axis=1)[0], accuracies
