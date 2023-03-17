import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# load mushroom dataset from CSV file
data_df = pd.read_csv('MushroomCSV.csv')



# Ask user whether to visualize a feature or to predict the class of a new mushroom
task = input('Enter "1" to visualize a feature or "2" to predict the class of a new mushroom: ')

if task == "1":
    print("Features:")
    for i, feature in enumerate(data_df.columns):
        print(f"{i+1}. {feature}")
    # Ask user which feature to visualize
    feature_num = int(input('Enter the number of the feature you want to visualize: '))

    if 1 <= feature_num <= len(data_df.columns):
        feature_to_visualize = data_df.columns[feature_num-1]

        # Count the occurrences of each value in the selected feature
        feature_counts = data_df[feature_to_visualize].value_counts()

        # Plot the feature counts
        plt.bar(feature_counts.index, feature_counts.values)
        plt.title('Counts of ' + feature_to_visualize)
        plt.xlabel(feature_to_visualize)
        plt.ylabel('Count')
        plt.show()

    else:
        print('Invalid feature number.')

elif task == "2":
    # convert dataframe to dictionary
    data = data_df.to_dict(orient='list')

    # extract categorical columns
    cat_cols = data_df.drop('CLASS', axis=1).select_dtypes(include='object').columns

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

    # prompt user to input the characteristics of the mushroom by number
    data_dict = {}
    for i, col in enumerate(cat_cols):
        print("Enter the number corresponding to the " + col + " of the mushroom:")
        options = data_df[col].unique()
        for j, option in enumerate(options):
            print(j, option)
        choice = int(input("Input: "))
        data_dict[col] = [options[choice]]

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
    print("Predicted class: ", prediction[0])

else:
    print('Invalid task number')
