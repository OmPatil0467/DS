#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Assignment - A7  |  Name : Pratik Pingale  |  Roll No : 19CO056


# ### Sample Sentences

# In[2]:


sentence1 = "I will walk 500 miles and I would walk 500 more. Just to be the man who walks " +             "a thousand miles to fall down at your door!"
sentence2 = "I played the play playfully as the players were playing in the play with playfullness"


# ### Tokenization

# In[3]:


from nltk import word_tokenize, sent_tokenize

print('Tokenized words:', word_tokenize(sentence1))
print('\nTokenized sentences:', sent_tokenize(sentence1))


# ### POS Tagging

# In[4]:


from nltk import pos_tag

token = word_tokenize(sentence1) + word_tokenize(sentence2)
tagged = pos_tag(token)                 

print("Tagging Parts of Speech:", tagged)


# ### Stop-Words Removal

# In[5]:


from nltk.corpus import stopwords

stop_words = stopwords.words('english')

token = word_tokenize(sentence1)
cleaned_token = []

for word in token:
    if word not in stop_words:
        cleaned_token.append(word)

print('Unclean version:', token)
print('\nCleaned version:', cleaned_token)


# ### Stemming

# In[6]:


from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

token = word_tokenize(sentence2)

stemmed = [stemmer.stem(word) for word in token]
print(" ".join(stemmed))


# ### Lemmatization

# In[7]:


from nltk.stem import WordNetLemmatizer 

lemmatizer = WordNetLemmatizer()

token = word_tokenize(sentence2)

lemmatized_output = [lemmatizer.lemmatize(word) for word in token]
print(" ".join(lemmatized_output))


# In[1]:


import math
from collections import Counter


# In[2]:


# Sample corpus of documents
corpus = [
'The quick brown fox jumps over the lazy dog',
'The brown fox is quick',
'The lazy dog is sleeping'
]


# In[3]:


# Tokenize the documents
tokenized_docs = [doc.lower().split() for doc in corpus]


# In[4]:


# Count the term frequency for each document
tf_docs = [Counter(tokens) for tokens in tokenized_docs]


# In[6]:


n_docs = len(corpus)
idf = {}
for tokens in tokenized_docs:
    for token in set(tokens):
        idf[token] = idf.get(token, 0) + 1
for token in idf:
    idf[token] = math.log(n_docs / idf[token])


# In[7]:


tfidf_docs = []
for tf_doc in tf_docs:
    tfidf_doc = {}
    for token, freq in tf_doc.items():
        tfidf_doc[token] = freq * idf[token]
    tfidf_docs.append(tfidf_doc)


# In[9]:


# Print the resulting TF-IDF representation for each document
for i, tfidf_doc in enumerate(tfidf_docs):
    print(f"Document {i+1}: {tfidf_doc}")


# In[ ]:




