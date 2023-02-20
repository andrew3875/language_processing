#!/usr/bin/env python
# coding: utf-8

# In[1]:


#For future QA classification. Copy of classification.ipynb
import pandas as pd
import os


# In[2]:


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import pickle


# In[3]:


train1 = pd.read_csv('/home/andrei/Work/data/past_plans.csv') #1
train2 = pd.read_csv('/home/andrei/Work/data/records.csv') #0
df = train1.append(train2) 


# In[4]:


train1


# In[5]:


df["text_lower"]  = [text.lower() for text in df["text"]]


# In[6]:


df


# In[7]:


#vectorizer = TfidfVectorizer(min_df= 3, stop_words="english", sublinear_tf=True, norm='l2', ngram_range=(1, 2))
vectorizer = CountVectorizer(min_df= 1, ngram_range=(1, 1))
final_features = vectorizer.fit_transform(df['text_lower']).toarray()
final_features.shape
len(sorted(list(vectorizer.vocabulary_)))


# In[8]:


X = df['text_lower']
Y = df['class']


# In[9]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)


# In[10]:


pipeline = Pipeline([('vect', vectorizer),
                     ('chi',  SelectKBest(chi2, k='all')), #1200
                     ('clf', LogisticRegression(random_state=0))])


# In[11]:


model = pipeline.fit(X_train, y_train) #build model
#print(os.getcwd())


# In[12]:


saved_model = os.getcwd() + "/" + "model3.pkl"
with open(saved_model, 'wb') as file:
    pickle.dump(model, file)
#with open(saved_model, 'rb') as file:
#    model = pickle.load(file)
ytest = np.array(y_test)


# In[13]:


print(classification_report(ytest, model.predict(X_test)))


# In[14]:


print(confusion_matrix(ytest, model.predict(X_test)))


# In[15]:


len(df)


# In[17]:


test = ['What records did I make yesterday?']
#test = ['Did I do anything last September?']
model.predict(test)
#model.predict_proba(test) #time 1 loc 0


# In[217]:


test = [line.strip("\n") for line in open(os.getcwd()+"/"+"plans4_test") if line!='\n']
test


# In[19]:


#test = ["when is the meeting" , "when is smth"]

#model.predict(test)
model.predict_proba(test) #time 1 loc 0


# In[19]:


final_features


# In[20]:


len(final_features)


# In[21]:


final_features[1]


# In[22]:


final_features[2]


# In[ ]:




