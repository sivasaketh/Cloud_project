# Importing necessary Packages
import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

# Converting the .csv file to pandas dataframe
df=pd.read_csv('C:/Users/saket/Cloud_Project/news.csv')
df.shape
df.head()

labels=df.label
labels.head()

# dividing the dataset into train and test dataset
x_train,x_test,y_train,y_test=train_test_split(df['text'], labels, test_size=0.2, random_state=7)

# Creating the tfidf vectorizer to convert the news 
# articles to vectors for mathematical calculations and deriving relation
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)

tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)

#Passive aggressive classifier model created and trained based on the training vectors created above
pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)


# Predicting the output of the test data and calculating the accuracy
y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(y_pred)
print("Accuracy:", round(score*100,2),"%")


confusion_matrix = confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])
print(confusion_matrix)

