#!/usr/bin/env python
# coding: utf-8

# In[9]:




import numpy as np
import pandas as pd


# In[10]:


movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv') 


# In[11]:


movies.head()


# In[92]:


movies.shape


# In[93]:


credits.head()


# In[6]:


credits.shape


# In[12]:


movies = movies.merge(credits,on='title')


# In[13]:


movies.shape


# In[14]:


movies.head()
# budget
# homepage
# id
# original_language
# original_title
# popularity
# production_comapny
# production_countries
# release-date(not sure)
#here we are selecting useful features


# In[15]:


movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[16]:


movies.head()


# In[17]:


import ast


# In[18]:


movies.isnull().sum()


# In[19]:


movies.dropna(inplace=True)


# In[20]:


#converting data to list of values
def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name']) 
    return L 


# In[21]:


movies['genres'] = movies['genres'].apply(convert)
movies.head()


# In[22]:


movies['keywords'] = movies['keywords'].apply(convert)
movies.head()


# In[106]:


import ast


# In[23]:


def convert3(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
        counter+=1
    return L 


# In[24]:


movies['cast'].apply(convert3)


# In[109]:


movies['cast'] = movies['cast'].apply(convert3)
movies.head()


# In[110]:


#movies['cast'] = movies['cast'].apply(lambda x:x[0:3])


# In[25]:


def fetch_director(text):
    L = []
    #converting string of list to list
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L 


# In[26]:


movies['crew'] = movies['crew'].apply(fetch_director)


# In[113]:


movies.sample(5)


# In[27]:


def collapse(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ",""))
    return L1


# In[28]:


#removing the spaces in atributes
movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)
movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)


# In[61]:


movies.head()


# In[29]:



movies['overview'] = movies['overview'].apply(lambda x:x.split())


# In[30]:


movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']


# In[116]:


movies.head()


# In[31]:


new = movies.drop(columns=['overview','genres','keywords','cast','crew'])
#new.head()


# In[32]:


new['tags'] = new['tags'].apply(lambda x: " ".join(x))



# In[33]:


new.head()


# In[34]:


new['tags']=new['tags'].apply(lambda x:x.lower())


# In[35]:


new.head()


# In[36]:


#vectorisation using bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')


# In[37]:


vector = cv.fit_transform(new['tags']).toarray()


# In[38]:


vector.shape


# In[39]:


from sklearn.metrics.pairwise import cosine_similarity


# In[40]:


similarity = cosine_similarity(vector)


# In[41]:


# to fetch index of movie
new[new['title'] == 'The Lego Movie'].index[0]


# In[43]:


sorted(list(enumerate(similarity[0])),reverse=True,key = lambda x: x[1])


# In[44]:


def recommend(movie):
    index = new[new['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    for i in distances[1:6]:
        print(new.iloc[i[0]].title)
        


# In[45]:


recommend('Batman')


# In[128]:


import pickle


# In[129]:


pickle.dump(new,open('movie_list.pkl','wb'))
pickle.dump(similarity,open('similarity.pkl','wb'))


# In[130]:


#sending dictionary rather than dataframe
pickle.dump(new.to_dict(),open('movie_dict.pkl','wb'))


# In[ ]:




