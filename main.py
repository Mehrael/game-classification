import pandas as pd

# read the data into a pandas dataframe
df = pd.read_csv('games-classification-dataset.csv')

# drop any unnecessary columns
df.drop(['Subtitle', 'Icon URL'], axis=1, inplace=True)

# fill any missing values in the 'Price' and 'In-app Purchases' columns with 0
df['Price'].fillna(0, inplace=True)
df['In-app Purchases'].fillna(0, inplace=True)

# convert the 'Price' and 'In-app Purchases' columns to numeric values
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
df['In-app Purchases'] = pd.to_numeric(df['In-app Purchases'], errors='coerce')

# drop any rows with missing values
df.dropna(inplace=True)

# convert the 'Original Release Date' and 'Current Version Release Date' columns to datetime format
df['Original Release Date'] = pd.to_datetime(df['Original Release Date'])
df['Current Version Release Date'] = pd.to_datetime(df['Current Version Release Date'])

print(df.dtypes)

# create a new column 'Rating' as the average of 'User Rating Count' and 'Rate'
df['User Rating Count'] = pd.to_numeric(df['User Rating Count'], errors='coerce')
df['Rate'] = pd.to_numeric(df['Rate'], errors='coerce')
df['Rating'] = (df['User Rating Count'] + df['Rate']) / 2

print("AFTER")
print(df.dtypes)


# drop the 'User Rating Count' and 'Rate' columns
df.drop(['User Rating Count', 'Rate'], axis=1, inplace=True)

# print the resulting dataframe
# print(df.head())
print(df)