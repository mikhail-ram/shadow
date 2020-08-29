import joblib, pickle, re, sys, unicodedata
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

classifier = joblib.load('IntentClassifier.sav')
pca = pickle.load(open("pca.pkl", "rb"))
cv = pickle.load(open("cv.pkl", "rb"))
labelencoder_y = pickle.load(open("le.pkl", "rb"))
wnl = WordNetLemmatizer()
user_input = ""
classes = ['AddToPlaylist: Adds the specified song to its specified playlist', 'BookRestaurant: Books the mentioned restaurant located in the specified place', 'GetWeather: Gets the weather in a specified place at a specified time', 'PlayMusic: Plays the specified music by artist, genre, name or playlist', 'RateBook: Rates the mentioned book title on Amazon.', 'SearchCreativeWork: Finds the movie, book, work etc. by the specified name.', 'SearchScreeningEvent: Finds the schedules and locations of the mentioned show.']
print("These are the classes the machine learning model has been trained on:\n")
for i in range(len(classes)):
    print("["+str(i+1)+"] "+classes[i])
print("\nMake a query relating to any one of these and it should predict the right category of the query.")

while True:
    user_input = input("What would you like SHADOW to do? ")
    if(user_input.lower() == "quit"):
        print("I hope I was of service to you. Goodbye.")
        break
    else:
        user_input = re.sub('[^a-zA-Z]',  " ", unicodedata.normalize('NFD', user_input).encode('ascii', 'ignore').decode("utf-8"))
        user_input = user_input.lower()
        user_input = user_input.split()
        user_input = [wnl.lemmatize(word) for word in user_input if not word in set(stopwords.words('english'))]
        user_input = " ".join(user_input)
        user_input = cv.transform([user_input]).toarray()
        output = "".join(labelencoder_y.inverse_transform(classifier.predict(pca.transform(user_input))))
        print(output)
