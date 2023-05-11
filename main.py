import nltk
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from CustomLabelEncoder import CustomLabelEncoder
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

df = pd.read_csv('games-classification-dataset.csv')

# Split data frame to X and Y
Y = df['Rate']
X = df.drop('Rate', axis=1)

# Split the X and the Y to training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, shuffle=True, random_state=0)

# drop any unnecessary columns
unimportant_columns=['URL','Subtitle', 'Icon URL']
x_train = drop_columns(x_train, unimportant_columns)

# fill any missing values in the 'Price' and 'In-app Purchases' columns with 0
x_train['Price'].fillna(0, inplace=True)
x_train['In-app Purchases'].fillna(0, inplace=True)

x_train['In-app Purchases'] = calc_sum_of_list(df['In-app Purchases'])
x_train['In-app Purchases'].fillna(0, inplace=True)

x_train['Languages'] = fill_nulls_with_mode(x_train['Languages'])

# change datatypes from object
x_train = x_train.convert_dtypes()

# Remove the '+' sign from the 'Age rating' column
x_train['Age Rating'] = x_train['Age Rating'].str.replace('+', '', regex=False)
# Convert the 'Age rating' column to an integer data type
x_train['Age Rating'] = x_train['Age Rating'].astype(int)

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
x_train['Languages'] = lang_encoder.fit_transform(x_train['Languages'])
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
top_feature = corr.index[abs(corr['Rate']) > 0.04]
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
game_data = pd.DataFrame(game_data, columns=top_feature)
y_train = game_data['Rate']
x_train = game_data.drop('Rate', axis=1)

print(x_train)



# drop any rows with missing values
# print(df.shape)
# df.dropna(inplace=True)
# print(df.shape)
#
# # convert the 'Original Release Date' and 'Current Version Release Date' columns to datetime format
# df['Original Release Date'] = pd.to_datetime(df['Original Release Date'])
# df['Current Version Release Date'] = pd.to_datetime(df['Current Version Release Date'])
#
# # print(df.dtypes)
#
# # create a new column 'Rating' as the average of 'User Rating Count' and 'Rate'
# df['User Rating Count'] = pd.to_numeric(df['User Rating Count'], errors='coerce')
#
# # print(df['Rate'])
# # print()
#
# mapping = {'Low': 0, 'Intermediate': 1, 'High': 2}
# df['Rate'] = df['Rate'].map(mapping)
#
# df['Rate'] = pd.to_numeric(df['Rate'], errors='coerce')
# # print(df['Rate'])
# # print()
# df['Rating'] = (df['User Rating Count'] + df['Rate']) / 2
#
# # print("AFTER")
# # print(df.dtypes)
#
#
# # drop the 'User Rating Count' and 'Rate' columns
# df.drop(['User Rating Count', 'Rate'], axis=1, inplace=True)
#
# # print the resulting dataframe
# # print(df.head())
# print(df)