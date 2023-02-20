#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pickle


# In[2]:


saved_model = os.getcwd() + "/" + "model3.pkl" #load model stored locally


# In[3]:


def date_deadline(message):
    with open(saved_model, 'rb') as file:
        model = pickle.load(file)
    if list(model.predict([message])) == [0]:
        return "date"
    if list(model.predict([message])) == [1]:
        return "deadline"


# In[2]:


saved_model = os.getcwd() + "/" + "model.pkl" #load model stored locally


# In[3]:


def get_type(message):
    with open(saved_model, 'rb') as file:
        model = pickle.load(file)
    if list(model.predict([message])) == [0]:
        return "location"
    if list(model.predict([message])) == [1]:
        return "time"


# In[2]:


saved_model = os.getcwd() + "/" + "model2.pkl" #load model stored locally


# In[12]:


def get_type_question_answer(message):
    with open(saved_model, 'rb') as file:
        model = pickle.load(file)
    if list(model.predict([message])) == [0]:
        return "question"
    if list(model.predict([message])) == [1]:
        return "answer"


# In[6]:


import dateparser
from dateparser.search import search_dates


# In[7]:


def get_date(message): #returns date in M/D/Y 00:00:00 format
    res = search_dates(message, languages = ['en'])
    if not res:
        return(None)
    else:
        date = res[0][1].strftime('%x %X')
        return(date)


# In[8]:


get_date("call mom tomorrow")


# In[3]:


import spacy
nlp = spacy.load("en_core_web_trf")


# In[31]:


def get_subject(message):
    result = [chunk.text for chunk in nlp(message).noun_chunks if chunk.root.dep_=='nsubj' or chunk.root.dep_=='conj']
    return(result)


# In[32]:


get_subject("Where are my important documents and blue jeans?")


# In[15]:


import datetime, dateparser, re
def get_time(message):
    result = ['','']
    doc = nlp(message) 
    entities = [ent.text for ent in doc.ents if ent.label_=='DATE' or ent.label_=='TIME']
    if entities:
        parsed_date = dateparser.parse(" ".join(entities)) 
        if parsed_date: # date was recognized by dateparser
            for ent in doc.ents:
                if ent.label_ == 'DATE' or ent.label_==' TIME' and ent.start != 0:
                    prev_token = doc[ent.start - 1]
                    if prev_token.text == "before":
                        result = ['', parsed_date.strftime('%Y-%m-%d %H:%M:%S')]
                    elif prev_token.text == "after":
                        result = [parsed_date.strftime('%Y-%m-%d %H:%M:%S'), '']
                    else:
                        result = [parsed_date.strftime('%Y-%m-%d %H:%M:%S'), '']
                    break
    if result == ['','']: # between dates
        if entities:     
            match = re.match(".*(between)\s+(.*)\s+(and)\s+(.*)", entities[0])
            if match:
                if match.group(1) == "between" and match.group(3) == "and":
                    date1 = match.group(2)
                    date2 = match.group(4)
                    parsed_date1 = dateparser.parse(date1)
                    parsed_date2 = dateparser.parse(date2)
                    conv_date1 = parsed_date1.strftime('%Y-%m-%d %H:%M:%S')
                    date2_fix = parsed_date2 + datetime.timedelta(days=1)
                    date2_fix = date2_fix - datetime.timedelta(seconds=1)
                    conv_date2 = date2_fix.strftime('%Y-%m-%d %H:%M:%S')
                    result = [conv_date1, conv_date2]
    return(result)


# In[21]:


get_time("Find events between 11th November  and 13th November")


# In[19]:


get_time("Find events before 11th November")


# In[20]:


get_time("Find events after 11th November")


# In[4]:


date_deadline('What records did I make yesterday?')


# In[34]:


date_deadline('Did I do anything last September?')


# In[39]:


get_type_question_answer('i need to go  tomorrow')


# In[2]:


saved_model = os.getcwd() + "/" + "model_get_type.pkl" #load model stored locally


# In[5]:


def get_type(message):
    with open(saved_model, 'rb') as file:
        model = pickle.load(file)
    if list(model.predict([message.lower()])) == [0]:
        return "location"
    if list(model.predict([message.lower()])) == [1]:
        return "time"
    if list(model.predict([message.lower()])) == [2]:
        return "note"


# In[7]:


get_type('I have to buy tomatoes')


# In[8]:


get_type('I have to buy tomatoes at 5 PM')


# In[9]:


get_type('tomatoes are on the shelf')


# In[ ]:




