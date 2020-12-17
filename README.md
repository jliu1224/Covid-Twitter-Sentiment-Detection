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
Our train_test_split remain the same throughout those three models we tried to build (Random Forest, Fully Connected Feedforward Network, Recurrent Neural Network）, with 80% of the original dataset as the train test and 20% as the test set. 

## Random Forest
For the random forest, we set entropy as the criterion to quantify the impurity of each node, the maximum depth of each tree as 9, and the number of trees in the forest as 20. After we fit the training data with the random forest classifier, we are able to print out the accuracy on the train set and the validating set.
```python
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,
                                                   random_state = 1997)
rf_model = RandomForestClassifier(criterion='entropy', max_depth=9, random_state=1997, n_estimators=20)
rf_model.fit(X_train, y_train)

print('Random Forest - Accuracy on training set: {:.4f}'.format(rf_model.score(X_train, y_train)))
print('Random Forest - Accuracy on test set: {:.4f}'.format(rf_model.score(X_test, y_test)))
```

## Fully connected Feedforward Network
Parameters we define beforehand:\
epochs = 5\
lr = 2e-3\
indim = X.shape[1] = 500\
outdim = 3\
drate = 0.8\
batch_size = 500\
\
To prepare the dataset for fully connected feedforward network, we tranform X and Y from numpy array into tensors, use the train_size and test_size we define earlier to create train_dataset and val_dataset, and eventually create train_loader and val_loader using both datasets.
```python
X_tensor = torch.from_numpy(X)
Y_tensor = torch.from_numpy(Y)

dataset = TensorDataset(X_tensor, Y_tensor)
train_size = int(0.8*len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(150))

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
val_loader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size)
```

The fully connected feedforward network we contruct contains one input layer of 500 neurons, 4 hidden layers that contain 375, 244, 100 and 12 neurons respectively, and one output layer that contains 3 neurons. A softmax function will also be applied to the output layer for the convenience of interpretation.
```python
class SentimentNetwork(nn.Module):
  def __init__(self, input_dim, output_dim, dropout_rate):
    
    super(SentimentNetwork,self).__init__()
    
    self.fc1 = nn.Linear(input_dim, 375)
    self.fc2 = nn.Linear(375, 244)
    self.fc3 = nn.Linear(244, 100)
    self.fc4 = nn.Linear(100, 12)
    self.fc5 = nn.Linear(12, output_dim)
    self.dropout = nn.Dropout(p=drate)

  def forward(self,x):
    x = self.dropout(F.relu(self.fc1(x))) 
    x = self.dropout(F.relu(self.fc2(x)))
    x = self.dropout(F.relu(self.fc3(x)))
    x = self.dropout(F.relu(self.fc4(x)))
    x = F.softmax(self.fc5(x))
   
    return x
```

For training and validating purpose, we basically define a training function and a validating function. With the help of these two functions, we will be able to perform loss computation and backpropogation within the training process, making predictions within the validating process, and meanwhile export the training accuracy, training loss, validating accuracy and validating loss for each epoch. Here we use the Adam algorithm as the optimizer, and we set the cross entropy loss as the criterion for loss calculation.

```python
def train(model, train_loader, optimizer, criterion):
  
  epoch_loss_total, epoch_acc_total = 0.0,0.0 # the loss and accuracy for each epoch

  model.train()

  for batch_idx, (data, target) in enumerate(train_loader):  

    optimizer.zero_grad() 
    predictions = model(data.float()) 
    loss = criterion(predictions, target) 
    pred = predictions.data.max(1)[1] # get the index of the max log-probability
    acc = pred.eq(target.data).sum()
    
    #backpropagate
    loss.backward() 
    optimizer.step() 
     
    epoch_loss_total += loss.item()
    epoch_acc_total += acc

  #calculate the average epoch_loss and epoch_acc
  epoch_loss = epoch_loss_total/len(train_loader.dataset)
  epoch_acc = epoch_acc_total/len(train_loader.dataset)

  return epoch_loss, epoch_acc
  
  def evaluate(model, val_loader, criterion):
  
  epoch_loss_total, epoch_acc_total = 0.0,0.0 # the loss and accuracy for each epoch

  model.eval()
    
  with torch.no_grad():
    for data, target in val_loader: 
      predictions = model(data.float())
      loss = criterion(predictions, target) 
      pred = predictions.data.max(1)[1]
      acc = pred.eq(target.data).sum()

      epoch_loss_total += loss.item()
      epoch_acc_total += acc
    #calculate the average epoch_loss and epoch_acc
    epoch_loss = epoch_loss_total/len(val_loader.dataset)
    epoch_acc = epoch_acc_total/len(val_loader.dataset)   

    return epoch_loss, epoch_acc
    
for epoch in range(epochs):
  train_loss, train_acc = train(model, train_loader, optimizer, criterion)
  valid_loss, valid_acc = evaluate(model, val_loader, criterion)
```

## Recurrent Neural Network
To build RNN, we need to re-prepare our data since RNN requires a different form of inputs.
First of all, we split the dataframe 'texts' into three dataframes (positive_tweets, neutral_tweets and negative_tweets) based on the labels we assigned to each instance earlier. Then, we will use three for loops in order to concatenate thos instances back together as a corpus, and meanwhile assign a new label to every tweet record, with [1, 0, 0] representing positive, [0, 1, 0] representing neutral and [0, 0, 1] representing negative. 
After all mentioned above has been done, we re-apply TF-IDF vectorizer with the same set of parameters to the corpus again. This time we get X1 with a shape of (44955, 500) and y1 containing the new labels with a shape of (44955, 30.
```python
positive_tweets = texts[texts['labels']==2]
neutral_tweets = texts[texts['labels']==1]
negative_tweets = texts[texts['labels']==0]

corpus = []
labels = []
for doc in positive_tweets['OriginalTweet']:
  corpus.append(doc.replace('\n', ' '))
  labels.append([1, 0, 0])
for doc in neutral_tweets['OriginalTweet']:
  corpus.append(doc.replace('\n', ' '))
  labels.append([0, 1, 0])
for doc in negative_tweets['OriginalTweet']:
  corpus.append(doc.replace('\n', ' '))
  labels.append([0, 0, 1])
  
vectorizer_r = TfidfVectorizer(max_features=500, stop_words='english')
X1 = vectorizer_r.fit_transform(corpus)
y1 = np.array(labels)
print(X1.shape, y1.shape)
```

Next up, the word_tokenizer and the vocabulary from vectorizer_r plays an important part in tranforming current data into a 3d_array and sequence padding. 

First thing we do here is to get rid of the words in each tweet that are not included in our vectorizer vocabulary. After we tokenize the processed texts, we use indices i and j together to access each word in each record within the nested for loop. Once we access that word, we check if that word is in the vocabulary. If yes, then the word will be appended into the list "terms" we created for each record. If not, we ignore it. We did the same task for every record in the data.

After finish checking each record, if the length of the list "terms" is greater than the seq_length (we initialize it as -1), then seq_length will be equal to the length of "terms". This way, after finishing the entire loop, the sequence length will be equal to the length of the longest list "terms".

```python
doc_terms_list_train = [word_tokenizer(doc_str) for doc_str in corpus]
docs = []
for i in range(len(doc_terms_list_train)):
  terms = []
  for j in range(len(doc_terms_list_train[i])):
    w = doc_terms_list_train[i][j]
    if w in vocab:
      terms.append(w)
  if len(terms) > seq_length:
    seq_length = len(terms)
  docs.append(terms)
```
Next, we create a 3d-array consist of 44955 records of 28 zero-vectors with a size of 500. We use the predefined sequence length and the length of each document to calculate the number of paddings we need. Then, we use indices i and j to access each word in each document, and use the word as the key of the vocabulary to get another index (idx). With indices i and the new index, we are able to get the TF-IDF value in X1. Lastly, using indices i, j+n_padding and the new index (idx), we are able to replace the corresponding 0 within the correponding word vector in the matrix full of zeros (in this case, its name is 'datasets').
```python
max_features = 500
datasets = np.zeros((X1.shape[0], seq_length, max_features))

for i in range(len(docs)):
  n_padding = seq_length - len(docs[i])

  for j in range(len(docs[i])):
    w = docs[i][j]
    idx = vocab[w]
    tfidf_val = X1[i, idx]
    datasets[i, j+n_padding, idx] = tfidf_val

datasets = datasets.astype(np.float32)
y1 = y1.astype(np.float32)
```

Using the train_test_split to datasets and y1, we basically get X1_train with a shape of (35964, 28, 500), X1_val with a shape of (8991, 28, 500), y1_train with a shape of (35964, 3) and y1_val with a shape of (8991, 3). Now the datasets are in satisfying shapes. After that we just use the same methods as we did in fully connected feedforward network to generate train_loader_r and val_loader_r, only with a different batch_size of 250.

The Network that we created contains an input layer of 500 neurons, a RNN network contains three hidden layers of 400 neurons each. After that, only the last time step of each rnn output will be sent to the next layer of 25 neurons, and then an output layer with 3 neurons, which a softmax function will also be applied to.

```python
#Parameters
input_size = 500
hidden_size = 400
n_layers = 3
output_size = 25

class Model(nn.Module):

  def __init__(self, input_size, output_size, hidden_size, n_layers):
    super().__init__()
    self.hidden_size = hidden_size
    self.n_layers = n_layers
    self.rnn = nn.RNN(input_size,hidden_size,n_layers,batch_first=True) # rnn layer
    self.fc1 = nn.Linear(hidden_size,output_size) # rnn output (y_t) --> output (y'_t)
    self.fc2 = nn.Linear(output_size,3) #the output from the last time period ->sentiment prediction

  def forward(self,x, hidden):
    batch_size = x.size()[0]
    hidden = self.init_hidden(batch_size)
    
    rnn_out,hidden = self.rnn(x,hidden)
    rnn_out = self.fc1(rnn_out)
    last_out = rnn_out[:,-1,:].view(batch_size,-1)
    out = F.softmax(self.fc2(last_out))
    return out,hidden 

  def init_hidden(self,batch_size):
    hidden = torch.zeros(self.n_layers, batch_size, self.hidden_size).cuda()
    return hidden

RNNmodel = Model(input_size, output_size, hidden_size, n_layers)
```

For training and validation, we also use cross entropy loss as the criterion and Adam algorithm as the optimizer, only this time we set number of epochs to be 6 and the learning rate to be 1e-4. We also use clip_grad_norm to help prevent the exploding gradient problem in RNN. We use a nested for loop to print out train accuracy, train loss, validating accuracy and validating loss for every 10 batches in each epoch.
```python
#Define hyperparameters
n_epochs = 6
lr = 1e-4
counter = 0
clip = 5

#Define loss and optimizer
criterion_r = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(RNNmodel.parameters(), lr=lr)

RNNmodel.train()

for epochs in range(n_epochs):
  #initiate hidden state
  h = RNNmodel.init_hidden(batch_size_r)

  #batch_loop
  for inputs, labels in train_loader_r:
    inputs, labels = inputs.to(device), labels.to(device)
    counter += 1
    
    RNNmodel.zero_grad()

    outputs, h = RNNmodel(inputs, h)

    loss = criterion_r(outputs, torch.max(labels, 1)[1])
    loss.backward()
    pred = torch.max(outputs, 1)[1]
    acc = pred.eq(torch.max(labels, 1)[1]).sum()

    #Clip_grad_norm to help prevent the exploding gradient problem in RNNs
    nn.utils.clip_grad_norm(RNNmodel.parameters(), clip)
    optimizer.step()

    ##Validation Loss
    if counter % 10 == 0:
      val_h = RNNmodel.init_hidden(batch_size_r)
      val_losses = []


      RNNmodel.eval()

      for inputs, labels in val_loader_r:
        inputs, labels = inputs.to(device), labels.to(device)
        val_outputs, val_h = RNNmodel(inputs, val_h)
        val_loss = criterion_r(val_outputs, torch.max(labels, 1)[1])
        val_losses.append(val_loss.item())
        pred = val_outputs.data.max(1)[1]
        val_acc = pred.eq(labels.data.max(1)[1]).sum()

      RNNmodel.train()

      print('Epoch:{}/{}'.format(epochs+1, n_epochs),
            'Batch:{}'.format(counter),
            'Train Accuracy:{:.5f}'.format(acc/batch_size_r),
            'Train Loss:{:.5f}'.format(loss.item()),
            'Val Accuracy:{:.5f}'.format(val_acc/batch_size_r),
            'Val Loss:{:.5f}'.format(np.mean(val_losses)))
```  
