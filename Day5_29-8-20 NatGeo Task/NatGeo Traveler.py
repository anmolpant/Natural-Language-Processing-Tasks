#!/usr/bin/env python
# coding: utf-8

# ## NLP hands on activity
# 
# ### Anmol Pant
# ### 18BCE0283

# ### import libraries

# In[1]:


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# ### document extraction and stopword removal

# In[2]:


def extract_doc(doc):
    f = open(doc,"r", encoding='utf-8')
    data = f.read()
    word_tokens = word_tokenize(data)
    stop_words = list(set(stopwords.words('english')))
    filtered_sentence = []
    stopwords_list = []
    for w in word_tokens: 
        if w not in stop_words: 
            filtered_sentence.append(w)
        else:
            stopwords_list.append(w)
    
    return filtered_sentence


# In[3]:


d1 = extract_doc(r"C:\Users\anmol\Downloads\natgeo1.txt")
d2 = extract_doc(r"C:\Users\anmol\Downloads\natgeo2.txt")
d3 = extract_doc(r"C:\Users\anmol\Downloads\natgeo3.txt")
d4 = extract_doc(r"C:\Users\anmol\Downloads\natgeo4.txt")
d5 = extract_doc(r"C:\Users\anmol\Downloads\sample.txt")


# In[4]:


distinct_words = list(set(d1+d2+d3+d4+d5))


# In[5]:


#list of distinct words
distinct_words


# ### bag of words representation

# In[6]:


import pandas as pd

def Bag_of_words(word_list):
    word_l = []
    for word in distinct_words:
        if (word in word_list) == True:
            word_l.append(word_list.count(word))
        else:
            word_l.append(0)
        
    return word_l

df = pd.DataFrame()
df["words"] = distinct_words
df["d1"] = Bag_of_words(d1)
df["d2"] = Bag_of_words(d2)
df["d3"] = Bag_of_words(d3)
df["d4"] = Bag_of_words(d4)
df["d5"] = Bag_of_words(d5)

    
print(df.head(100))


# ## TF-IDF vectorizer

# ### tf

# In[7]:


import math

def TF(word_list):
    word_l = []
    for word in distinct_words:
        if (word in word_list) == True:
            word_l.append(1+math.log(1+math.log(word_list.count(word))))
        else:
            word_l.append(0)
        
    return word_l

tf = pd.DataFrame()
tf["words"] = distinct_words
tf["d1"] = TF(d1)
tf["d2"] = TF(d2)
tf["d3"] = TF(d3)
tf["d4"] = TF(d4)
tf["d5"] = TF(d5)
    
print(tf)


# ### idf

# In[8]:


idf = pd.DataFrame()

def IDF():
    l = []
    D = [d1,d2,d3,d4,d5]
    for i in distinct_words:
        d = 0
        for j in D:
            if i in j:
                d+=1
                
        l.append(math.log(1+5/d))
        
    return l
        
idf["word"] = distinct_words
idf["relevance"] = IDF()

        
print(idf)


# ### tf-idf

# In[9]:


import numpy as np
tfidf = pd.DataFrame()
tfidf["words"] = distinct_words

n1 = np.array(tf.d1)
n2 = np.array(tf.d2)
n3 = np.array(tf.d3)
n4 = np.array(tf.d4)
n5 = np.array(tf.d5)
x = np.array(idf.relevance)

tfidf["d1"] = n1*x
tfidf["d2"] = n2*x
tfidf["d3"] = n3*x
tfidf["d4"] = n4*x
tfidf["d5"] = n5*x

print(tfidf)


# In[10]:


#normalized tf-idf for docs
tfidf.d1/=len(distinct_words)
tfidf.d2/=len(distinct_words)
tfidf.d3/=len(distinct_words)
tfidf.d4/=len(distinct_words)
tfidf.d5/=len(distinct_words)
print(tfidf)


# ## Cosine Similarity

# In[11]:


#cosine-similarity
import numpy as np
import math

cs = {}
q = 0
for i in np.array(tfidf.d5):
    q+=(i**2)
    
# print(q)

def cosine_sim(doc):
    d = 0
    for i in doc:
        d+=(i**2) 
        
#     print(d)
    a = doc.dot(np.array(tfidf.d5))
#     print(a)
#     return (a/(math.sqrt(q*d)))
    return('{0:.10f}'.format((a/(math.sqrt(q*d)))))

cs["d1"] = float(cosine_sim(np.array(tfidf.d1)))
cs["d2"] = float(cosine_sim(np.array(tfidf.d2)))
cs["d3"] = float(cosine_sim(np.array(tfidf.d3)))
cs["d4"] = float(cosine_sim(np.array(tfidf.d4)))
print(cs)


# In[12]:


#document ranking
def rank_doc(array):
    l = array.copy()
    valid = True
    while valid:
#         print(l)
        if(len(np.where(array == max(l))[0]) > 1):
            for i in range(len(np.where(array == max(l))[0])):
                print((np.where(array == max(l))[0][i]+1),)
            l = np.delete(l,np.where(l == max(l)),0)
        else:
            print(np.where(array == max(l))[0][0]+1)
            l = np.delete(l,np.where(l == max(l))[0],0)
            
        if(len(l) == 0):
            valid = False
            

rank_doc(np.array(list(cs.values())))


# ### Eucledian Distance and Document Ranking

# In[15]:


ed = {}
def euc_d(doc,query):
    e = 0
    for i in range(len(doc)):
        e+=(doc[i] - query[i])**2
        
    return ('{0:.10f}'.format(math.sqrt(e)))
        
ed["d1"] = float(euc_d(np.array(tfidf.d1), np.array(tfidf.d5)))
ed["d2"] = float(euc_d(np.array(tfidf.d2), np.array(tfidf.d5)))
ed["d3"] = float(euc_d(np.array(tfidf.d3), np.array(tfidf.d5)))
ed["d4"] = float(euc_d(np.array(tfidf.d4), np.array(tfidf.d5)))
print(ed)
print(list(ed.values()))

def rank_doc(array):
    l = array.copy()
    valid = True
    while valid:
        if(len(np.where(array == min(l))[0]) > 1):
            for i in range(len(np.where(array == min(l))[0])):
                print((np.where(array == min(l))[0][i]+1),)
            l = np.delete(l,np.where(l == min(l)),0)
        else:
            print(np.where(array == min(l))[0][0]+1)
            l = np.delete(l,np.where(l == min(l))[0],0)
            
        if(len(l) == 0):
            valid = False

            

rank_doc(np.array(list(ed.values())))


# Final Note - we see that after computing the cosine similarity and Eucledian distance between the sample article and the 4 articles given to us in the form of the corpus, the similarity is almost negligible.
# Hence it is safe to assume that the articles she writes are original and we can hire her for our publication.  

# In[ ]:




