# Covid-Twitter-Sentiment-Detection
## Attaining and Re-editing the data
For this COVID-19 Twitter sentiment detection project, the dataset used is originally from the "Coronavirus tweets NLP - Text Classification" dataset by Aman Miglani on Kaggle.com. The major difference is that the train set and test set are merged together so that we can do train_test_split by any proportion we want. The dataset we used for our analysis is "CoronaTwitters.csv".
https://www.kaggle.com/datatattle/covid-19-nlp-text-classification?select=Corona_NLP_test.csv

## Libraries and Modules Used
```python
import torch
import os
import glob
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import TensorDataset, DataLoader
import re
import string as st
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
```

## Data Preparation (For Random Forest and Fully Connected Feedforward Network)
Reading the csv file "CoronaTwitters.csv" is the first task we did here. After that, two columns of data were extract from the dataset, which are "Original data" and "Sentiment". The tweets were originally labeled with five types of sentiments: extremely positive, positive, neutral, negative, extremely negative. We use a dictionary to simplify those five types of sentiments, 2 representing positive (both "extermely positive" and "positive" fall into this catagory), 1 representing neutral and 0 representing negative (both "extermely negative" and "negative" fall into this catagory).
```python
#Read Data and create labels
texts = df[['OriginalTweet', 'Sentiment']]
sentiments = {'Extremely Positive':2, 'Positive':2, 'Neutral':1, 'Negative':0, 'Extremely Negative':0}
texts['labels'] = texts['Sentiment'].str.strip().map(sentiments)
texts.head(5)
```
Next, Natural Language Processing (NLP) is used to clean the text data. Things like unnecessary punctuations(including hashtag#), tagging someone in tweets (@username), website links, the line delimiter(\n) or other special characters that may hinders word count. After that, those processed tweets will replaced the "Original data" column.
```python
#Clean the data in Original Tweet
tweets = texts['OriginalTweet']
lowercase = tweets.map(lambda t : t.lower())
noa = lowercase.str.replace('â', '', n=-1)
nohttp = noa.str.replace('[http://][t.co][/][/D+]', '', n=-1)
nopunc = nohttp.str.replace('[,().?!-":#]', '', n=-1)
noat = nopunc.str.replace('[@]\w+', '', n=-1)
non = noat.str.replace('\n', '', n=-1)
new_tweets = non.str.replace('/', ' ', n=-1)
texts['OriginalTweet'] = new_tweets
```

Lastly, we use TF-IDF to represent each record of those tweet. Based on the text of tweets, unigram model was used to construct a 500-individual-word vocabulary, and each record is represented by a vector with 500 features. Each feature is representing a word in the vocabulary. We stored the entire 2d TF-IDF array as X, and the labels as Y.
```python
#Get the data
twitters = new_tweets.values
labels = texts.labels.values
#Using TF-IDF to represent each twitter record, denoted as X;
#Create a label vector Y
vectorizer = TfidfVectorizer(stop_words='english', max_features=500, ngram_range=(1,1)) 
instances = vectorizer.fit_transform(twitters)
X = instances.toarray()
Y = labels

print('The shape of X is:', X.shape)
print('The shape of Y is:', Y.shape)
```
Our train_test_split remain the same throughout those three models we tried to build, with 80% of the original dataset as the train test and 20% as the test set.

## Random Forest
