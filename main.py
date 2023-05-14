import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from CustomLabelEncoder import CustomLabelEncoder
from sklearn import svm
import pandas as pd
from sklearn import metrics
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
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
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, shuffle=True, random_state=54)

# drop any unnecessary columns
unimportant_columns=[ 'Name','URL','Subtitle','Icon URL','Description','Primary Genre'] #,'ID','Price','Age Rating','Languages'

x_train = x_train.drop(unimportant_columns, axis=1)

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
lang_encoder = CustomLabelEncoder()
x_train['Languages'] = lang_encoder.fit_transform(x_train['Languages'])
x_train['Developer'] = dev_encoder.fit_transform(x_train['Developer'])

# Feature selection using spearman method
data = x_train.join(y_train)

mapping = {'Low': 0, 'Intermediate': 1, 'High': 2}
data['Rate'] = data['Rate'].map(mapping)

game_data = data.iloc[:, :]
corr = game_data.corr(method='kendall')
# Top 50% Correlation training features with the Value
top_feature = corr.index[abs(corr['Rate']) > 0.01]

x_data = game_data[top_feature]

if 'Developer' not in x_data:
    rate = x_data['Rate']
    x_data = x_data.drop('Rate', axis=1)
    x_data['Developer'] = game_data['Developer']
    x_data['Rate'] = rate

    # drop last column
    top_feature = top_feature[:-1]

    # Add additional features to top_feature list
    additional_features = ['Developer', 'Rate']
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

# Remove the '+' sign from the 'Age rating' column
x_test['Age Rating'] = x_test['Age Rating'].str.replace('+', '', regex=False)

# Convert the 'Age rating' column to an integer data type
x_test['Age Rating'] = x_test['Age Rating'].astype(int)

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
x_test['Languages'] = lang_encoder.transform(x_test['Languages'])

# Feature selection using spearman method
data = x_test.join(y_test)

mapping = {'Low': 0, 'Intermediate': 1, 'High': 2}
data['Rate'] = data['Rate'].map(mapping)

game_data = data.iloc[:, :]
corr = game_data.corr(method='kendall')
# Top 50% Correlation training features with the Value
top_feature = corr.index[abs(corr['Rate']) > 0.01]

x_data = game_data[top_feature]
#
if 'Genres' not in x_data:
    rate = x_data['Rate']
    x_data = x_data.drop('Rate', axis=1)
    x_data['Genres'] = game_data['Genres']
    x_data['Rate'] = rate

    # drop last column
    top_feature = top_feature[:-1]

    # Add additional features to top_feature list
    additional_features = ['Genres', 'Rate']
    top_feature = pd.Index(np.concatenate([top_feature.values, additional_features]))#
if 'Price' not in x_data:
    rate = x_data['Rate']
    x_data = x_data.drop('Rate', axis=1)
    x_data['Price'] = game_data['Price']
    x_data['Rate'] = rate

    # drop last column
    top_feature = top_feature[:-1]

    # Add additional features to top_feature list
    additional_features = ['Price', 'Rate']
    top_feature = pd.Index(np.concatenate([top_feature.values, additional_features]))
if 'Languages' not in x_data:
    rate = x_data['Rate']
    x_data = x_data.drop('Rate', axis=1)
    x_data['Languages'] = game_data['Languages']
    x_data['Rate'] = rate

    # drop last column
    top_feature = top_feature[:-1]

    # Add additional features to top_feature list
    additional_features = ['Languages', 'Rate']
    top_feature = pd.Index(np.concatenate([top_feature.values, additional_features]))

if 'Age Rating' not in x_data:
    rate = x_data['Rate']
    x_data = x_data.drop('Rate', axis=1)
    x_data['Age Rating'] = game_data['Age Rating']
    x_data['Rate'] = rate

    # drop last column
    top_feature = top_feature[:-1]

    # Add additional features to top_feature list
    additional_features = ['Age Rating', 'Rate']
    top_feature = pd.Index(np.concatenate([top_feature.values, additional_features]))

if 'Developer' not in x_data:
    rate = x_data['Rate']
    x_data = x_data.drop('Rate', axis=1)
    # x_data['Genres'] = game_data['Genres']
    x_data['Developer'] = game_data['Developer']
    x_data['Rate'] = rate

    # drop last column
    top_feature = top_feature[:-1]

    # Add additional features to top_feature list
    additional_features = ['Developer', 'Rate']
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

# Define the undersampler
undersampler = RandomUnderSampler()

# Undersample the training data
x_train_undersampled, y_train_undersampled = undersampler.fit_resample(x_train, y_train)

# Define the decision tree model with default parameters
model = DecisionTreeClassifier(max_depth=5,random_state=42)

# Fit the model to the undersampled training data
model.fit(x_train_undersampled, y_train_undersampled)

# Predict the labels of the training data
y_pred_train = model.predict(x_train)

# Predict the labels of the test data
y_pred_test = model.predict(x_test)

# Calculate the accuracy of the model on the training data
accuracy_train = accuracy_score(y_train, y_pred_train)

# Calculate the accuracy of the model on the test data
accuracy_test = accuracy_score(y_test, y_pred_test)

print("Accuracy on training data:", accuracy_train)
print("Accuracy on test data:", accuracy_test)

print()
print("----------------------------------------------------------------------------------------------------------------")
#-----------------------------------------------------------------------------------------------------------------------
print("Random Forest")
# Create a Random Forest classifier with 100 trees
rf = RandomForestClassifier(n_estimators=6, max_depth=6)

# fit the model to your training data
rf.fit(x_train, y_train)

# Predict on the training data
y_pred_train = rf.predict(x_train)

# Predict on the test data
y_pred_test = rf.predict(x_test)

# Calculate the accuracy of the model on the training data
accuracy_train = accuracy_score(y_train, y_pred_train)

# Calculate the accuracy of the model on the test data
accuracy_test = accuracy_score(y_test, y_pred_test)

print("Accuracy on training data:", accuracy_train)
print("Accuracy on test data:", accuracy_test)

print()
print("----------------------------------------------------------------------------------------------------------------")
#-----------------------------------------------------------------------------------------------------------------------
print("SVM poly")
# Create an SVM classifier with a linear kernel and max_iter set to 30
svm_clf = svm.SVC(kernel ='poly', degree = 5)

# Fit the model to the training data
svm_clf.fit(x_train, y_train)

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
print("----------------------------------------------------------------------------------------------------------------")
# ----------------------------------------------------------------------------------------------------------------------
print("SVM sigmoid")
# Create an SVM classifier with a linear kernel and max_iter set to 30
svm_clf = svm.SVC(kernel ='sigmoid')

# Fit the model to the training data
svm_clf.fit(x_train, y_train)

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
print("----------------------------------------------------------------------------------------------------------------")
# ----------------------------------------------------------------------------------------------------------------------
print("Logistic Regression")
# Create a logistic regression model
lr_model = LogisticRegression()

# Train the model on the training set
lr_model.fit(x_train, y_train)

# Predict on the training data
y_pred_train = lr_model.predict(x_train)

# Predict on the testing data
y_pred_test = lr_model.predict(x_test)

# Calculate the accuracy of the model on the training data
accuracy_train = accuracy_score(y_train, y_pred_train)

# Calculate the accuracy of the model on the testing data
accuracy_test = accuracy_score(y_test, y_pred_test)

print("Accuracy on training data:", accuracy_train)
print("Accuracy on testing data:", accuracy_test)
print()
print("----------------------------------------------------------------------------------------------------------------")
# ----------------------------------------------------------------------------------------------------------------------
print("Naive Bayes classifier")
# Create a Gaussian Naive Bayes classifier
model = GaussianNB()

# Train the model using the training data
model.fit(x_train, y_train)

# Predict on the training data
y_pred_train = model.predict(x_train)

# Predict on the test data
y_pred_test = model.predict(x_test)

# Calculate the accuracy of the model on the training data
accuracy_train = accuracy_score(y_train, y_pred_train)

# Calculate the accuracy of the model on the test data
accuracy_test = accuracy_score(y_test, y_pred_test)

print("Accuracy on training data:", accuracy_train)
print("Accuracy on test data:", accuracy_test)

print()
print("----------------------------------------------------------------------------------------------------------------")
# ----------------------------------------------------------------------------------------------------------------------
print("KNN")

# Define a range of K values to test
k_values = range(1,15)

# Initialize empty lists to store the training and testing accuracy scores
train_scores = []
test_scores = []

# Loop over each K value and fit the KNN model, then compute training and testing accuracy scores using cross-validation
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    train_cv_scores = cross_val_score(knn, x_train, y_train, cv=20)
    test_cv_scores = cross_val_score(knn, x_test, y_test, cv=20)
    train_scores.append(train_cv_scores.mean())
    test_scores.append(test_cv_scores.mean())

# Plot the training and testing accuracy scores as a function of K
import matplotlib.pyplot as plt

plt.plot(k_values, train_scores, label="Training")
plt.plot(k_values, test_scores, label="Testing")
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.title("KNN Accuracy Scores")
plt.legend()
plt.show()

knn = KNeighborsClassifier(n_neighbors=1)
train_cv_scores = cross_val_score(knn, x_train, y_train, cv=5)
test_cv_scores = cross_val_score(knn, x_test, y_test, cv=5)


# Print the mean and standard deviation of the accuracy scores
print("Training Accuracy:",train_cv_scores.mean())
print("Test Accuracy:" ,test_cv_scores.mean())

print()
print("----------------------------------------------------------------------------------------------------------------")
# ----------------------------------------------------------------------------------------------------------------------
print("AdaBoost Model / Boosting Ensemble Learning")
# Initialize the base model
base_clf = DecisionTreeClassifier(max_depth=1)
# Create adaboost classifer object
abc = AdaBoostClassifier(base_estimator=base_clf, n_estimators=100, learning_rate=0.5)

# Train Adaboost Classifer
model = abc.fit(x_train, y_train)

# Predict the response for training dataset
y_pred_train = model.predict(x_train)

# Predict the response for test dataset
y_pred_test = model.predict(x_test)

# Calculate the accuracy of the model on the training data
accuracy_train = metrics.accuracy_score(y_train, y_pred_train)

# Calculate the accuracy of the model on the test data
accuracy_test = metrics.accuracy_score(y_test, y_pred_test)

# Print the accuracies
print("Accuracy on training data:", accuracy_train)
print("Accuracy on test data:", accuracy_test)

print()
print("----------------------------------------------------------------------------------------------------------------")
# ----------------------------------------------------------------------------------------------------------------------
print("Bagging Ensemble Learning")
# Create a base estimator
base_estimator = DecisionTreeClassifier(max_depth=5,random_state=42)

# Create a Bagging classifier with 50 estimators
bagging = BaggingClassifier(base_estimator=base_estimator, n_estimators=30, max_samples=0.8, max_features=0.8)

# Train the Bagging classifier on the training data
bagging.fit(x_train, y_train)

# Predict the labels for both the training and test data
y_train_pred = bagging.predict(x_train)
y_test_pred = bagging.predict(x_test)

# Calculate the accuracy of the model on the training and test data
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Print the accuracy scores
print("Accuracy on training data:", train_accuracy)
print("Accuracy on test data:", test_accuracy)

print()
print("----------------------------------------------------------------------------------------------------------------")
# ----------------------------------------------------------------------------------------------------------------------
print("Stacking Ensemble Learning")

# Initialize the base models
lr = LogisticRegression()
dt = DecisionTreeClassifier(max_depth=5,random_state=42)
# knn = KNeighborsClassifier(n_neighbors=1)
rf = RandomForestClassifier(n_estimators=6, max_depth=6)
bg =  BaggingClassifier(base_estimator=base_estimator, n_estimators=30, max_samples=0.8, max_features=0.8)
ab = AdaBoostClassifier(base_estimator=base_clf, n_estimators=100, learning_rate=0.5)

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

# Use the predictions from the base models as input features to train the meta-model on the train set
meta_X_train = np.column_stack((lr_pred_train, dt_pred_train, rf_pred_train, bg_pred_train, ab_pred_train))
meta_clf = LogisticRegression()
meta_clf.fit(meta_X_train, y_train)

# Make predictions on the train set using the meta-model
meta_pred_train = meta_clf.predict(meta_X_train)

# Use the predictions from the base models as input features to make predictions on the test set using the meta-model
lr_pred_test = lr.predict(x_test)
dt_pred_test = dt.predict(x_test)
# knn_pred_test = knn.predict(x_test)
rf_pred_test = rf.predict(x_test)
bg_pred_test = bg.predict(x_test)
ab_pred_test = ab.predict(x_test)

meta_x_test = np.column_stack((lr_pred_test, dt_pred_test, rf_pred_test, bg_pred_test, ab_pred_test))
meta_pred_test = meta_clf.predict(meta_x_test)

# Evaluate the performance of the meta-model on the train and test sets
train_accuracy = accuracy_score(y_train, meta_pred_train)
test_accuracy = accuracy_score(y_test, meta_pred_test)

print('Accuracy of stacking ensemble on train set:',train_accuracy)
print('Accuracy of stacking ensemble on test set:',test_accuracy)

print()
print("----------------------------------------------------------------------------------------------------------------")
# ----------------------------------------------------------------------------------------------------------------------
