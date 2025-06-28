# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib
# import libaries required for nlp
import nltk # Netural language toolkit
import re

# downloadd stopword
from nltk.corpus import stopwords
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

#Loading the data
df = pd.read_csv("IMDB Dataset.csv")
df.info()
df["sentiment"].value_counts()

#mapping the sentiment to some numerical values
df["sentiment"] = df["sentiment"].map({
    "positive" : 1,
    "negative" : 0
})

#Clean the text
def clean_text(text):
  text = re.sub(r"[^a-zA-Z]"," ",text).lower()
  tokens = text.split()
  tokens = [word for word in tokens if word not in stop_words]
  return " ".join(tokens)

#Apply the clean_text function on review 
df["cleaned_review"] = df["review"].apply(clean_text)

# Feature extraction
vectorizer = CountVectorizer(max_features = 5000)
x = vectorizer.fit_transform(df["cleaned_review"])
y = df["sentiment"]

# Divide the dataset into train test and split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 1)

# Train the Model
model = MultinomialNB()
model.fit(x_train,y_train)

# Make the prediction
y_pred = model.predict(x_test)

# calculate the performance metrics
accuracy = accuracy_score(y_pred,y_test)
precision = precision_score(y_pred,y_test)
recall = recall_score(y_pred, y_test)
f1 = f1_score(y_pred, y_test)
cm = confusion_matrix(y_pred, y_test)
cr = classification_report(y_pred, y_test)

# print all the performance metrics parameters
print("The accuracy is :",accuracy)
print("the precision is : ",precision)
print("The recall is : ",recall)
print("The f1_score is : ",f1)
print("-------confustion metrics----------")
print(cm)
print("-----------classification report--------------")
print(cr)

joblib.dump(model,"sentiment_model.pkl")
joblib.dump(vectorizer,"vectorizer.pkl")
print("model and vectorizer is saved")