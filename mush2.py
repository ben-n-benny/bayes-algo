import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# load mushroom dataset from CSV file
data_df = pd.read_csv('MushroomCSV.csv')

# convert dataframe to dictionary
data = data_df.to_dict(orient='list')

# extract categorical columns
cat_cols = data_df.drop('CLASS', axis=1).select_dtypes(include='object').columns

# print categorical columns
print(cat_cols)

# create a new dataframe from the dictionary and convert categorical columns into one-hot encoding
df = pd.DataFrame(data)
df = pd.get_dummies(df, columns=cat_cols)

# set target variable y and input features X
X = df.drop('CLASS', axis=1)
y = df['CLASS']

# split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=30)

# initialize and fit the Gaussian Naive Bayes classifier
clf = GaussianNB()
clf.fit(X_train, y_train)

# predict the target variable on the test set
y_pred = clf.predict(X_test)

# calculate and print the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# create a dictionary to hold the values for the new mushroom sample
columns_str = "CAPSHAPE,SURFACE,COLOR,BRUISES,ODOR,GILL-ATTACHMENT,GILL-SPACING,GILL-SIZE,GILL-COLOR,STALK-SHAPE,STALK-ROOT,STALK-SURFACE-ABOVE-RING,STALK-SURFACE-BELOW-RING,STALK-COLOR ABOVE-RING,STALK-COLOR-BELOW-RING,VEIL-TYPE,VEIL-COLOR,RING-NUMBER,RING-TYPE,SPORE-PRINT-COLOR,POPULATION,HABITAT"
columns_list = columns_str.split(',')
data_dict = {}
for column in columns_list:
    data_dict[column] = []

# add the values for the new mushroom sample to the dictionary
data_dict['CAPSHAPE'].append('concave')
data_dict['SURFACE'].append('smooth')
data_dict['COLOR'].append('brown')
data_dict['BRUISES'].append('yes')
data_dict['ODOR'].append('almond')
data_dict['GILL-ATTACHMENT'].append('free')
data_dict['GILL-SPACING'].append('close')
data_dict['GILL-SIZE'].append('broad')
data_dict['GILL-COLOR'].append('black')
data_dict['STALK-SHAPE'].append('enlarging')
data_dict['STALK-ROOT'].append('bulbous')
data_dict['STALK-SURFACE-ABOVE-RING'].append('smooth')
data_dict['STALK-SURFACE-BELOW-RING'].append('smooth')
data_dict['STALK-COLOR ABOVE-RING'].append('white')
data_dict['STALK-COLOR-BELOW-RING'].append('white')
data_dict['VEIL-TYPE'].append('partial')
data_dict['VEIL-COLOR'].append('white')
data_dict['RING-NUMBER'].append('one')
data_dict['RING-TYPE'].append('pendant')
data_dict['SPORE-PRINT-COLOR'].append('black')
data_dict['POPULATION'].append('scattered')
data_dict['HABITAT'].append('urban')

# Create a new pandas DataFrame using data from a dictionary
new_df = pd.DataFrame(data_dict)

# Convert categorical columns into binary columns using one-hot encoding
new_df = pd.get_dummies(new_df, columns=cat_cols)

# Reorder the columns in the DataFrame to match the columns in the training data,
# and fill any missing columns with 0
new_df = new_df.reindex(columns=X_train.columns, fill_value=0)

# Use a pre-trained machine learning model to predict the target variable based
# on the features in the new DataFrame
prediction = clf.predict(new_df)

# Print the predicted values to the console
print(prediction)