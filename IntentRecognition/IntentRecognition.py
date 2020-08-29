# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import unicodedata

# Importing the dataset
training_dataset = pd.read_csv('data/IntentRecognition/training.tsv', delimiter = '\t', quoting = 3)
test_dataset = pd.read_csv('data/IntentRecognition/test.tsv', delimiter = '\t', quoting = 3)

# Cleaning the dataset
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()
training_corpus = []
test_corpus = []

for i in range(len(training_dataset)):
    text = re.sub('[^a-zA-Z]',  " ", unicodedata.normalize('NFD', training_dataset['text'][i]).encode('ascii', 'ignore').decode("utf-8"))
    text = text.lower()
    text = text.split()
    text = [wnl.lemmatize(word) for word in text if not word in set(stopwords.words('english'))]
    text = " ".join(text)
    training_corpus.append(text)

for i in range(len(test_dataset)):
    text = re.sub('[^a-zA-Z]',  " ", unicodedata.normalize('NFD', test_dataset['text'][i]).encode('ascii', 'ignore').decode("utf-8"))
    text = text.lower()
    text = text.split()
    text = [wnl.lemmatize(word) for word in text if not word in set(stopwords.words('english'))]
    text = " ".join(text)
    test_corpus.append(text)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5000)
X_train = cv.fit_transform(training_corpus).toarray()
X_test = cv.transform(test_corpus).toarray()
y_train = training_dataset.iloc[:, 1].values
y_test = test_dataset.iloc[:, 1].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y_train = labelencoder_y.fit_transform(y_train)
y_test = labelencoder_y.transform(y_test)

# Dimensionality Reduction
# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 1500, random_state=0)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
# classifier.score(X_train, y_train) Accuracy of 99%
# classifier.score(X_test, y_test) Accuracy of 98.71%

# Saving all Requirements

# Saving the model
import joblib
joblib.dump(classifier, 'IntentClassifier2.sav')
# Saving the PCA algorithm
"""import pickle
pickle.dump(pca, open("pca.pkl", "wb"))"""
# Saving the CountVectorizer algorithm
"""import pickle
pickle.dump(cv, open("cv.pkl", "wb"))"""
# Saving the LabelEncoder algorithm
"""import pickle
pickle.dump(labelencoder_y, open("le.pkl", "wb"))"""

import sys
user_input = ""
classes = ['AddToPlaylist: Adds the specified song to its specified playlist', 'BookRestaurant: Books the mentioned restaurant located in the specified place', 'GetWeather: Gets the weather in a specified place at a specified time', 'PlayMusic: Plays the specified music by artist, genre, name or playlist', 'RateBook: Rates the mentioned book title on Amazon.', 'SearchCreativeWork: Finds the movie, book, work etc. by the specified name.', 'SearchScreeningEvent: Finds the schedules and locations of the mentioned show.']
print("These are the classes the machine learning model has been trained on:\n")
for i in range(len(classes)):
    print("["+str(i)+"] "+classes[i])
print("\nMake a query relating to any one of these and it should predict the right category of the query.")
while user_input != "quit":
    user_input = input("What would you like SHADOW to do? ")
    user_input = re.sub('[^a-zA-Z]',  " ", unicodedata.normalize('NFD', user_input).encode('ascii', 'ignore').decode("utf-8"))
    user_input = user_input.lower()
    user_input = user_input.split()
    user_input = [wnl.lemmatize(word) for word in user_input if not word in set(stopwords.words('english'))]
    user_input = " ".join(user_input)
    user_input = cv.transform([user_input]).toarray()
    output = "".join(labelencoder_y.inverse_transform(classifier.predict(pca.transform(user_input))))
    print(output)
