from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

import numpy as np

# define training data and labels
train_data = ['leafy', 'orange', 'violet', 'green']
train_labels = ['poisonous', 'poisonous', 'not poisonous', 'not poisonous']

# create vectorizer to convert text into numerical features
vectorizer = CountVectorizer()

# transform training data into numerical features
X_train = vectorizer.fit_transform(train_data)

# train the classifier on the training data
clf = MultinomialNB().fit(X_train, train_labels)

# define test data
test_data = ['leafy green']

# transform test data into numerical features using the same vectorizer
X_test = vectorizer.transform(test_data)

# predict the class labels of the test data using the trained classifier
predicted_labels = clf.predict(X_test)

# data visualization
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# bar chart of training data labels
unique_labels, label_counts = np.unique(train_labels, return_counts=True)
ax[0].bar(unique_labels, label_counts)
ax[0].set_xlabel('Labels')
ax[0].set_ylabel('Frequency')
ax[0].set_title('Training Data Labels')

# pie chart of predicted labels for test data
unique_pred_labels, pred_label_counts = np.unique(predicted_labels, return_counts=True)
ax[1].pie(pred_label_counts, labels=unique_pred_labels, autopct='%1.1f%%')
ax[1].set_title('Predicted Labels for Test Data')

plt.show()
