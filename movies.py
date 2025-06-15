#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install nltk')


# In[2]:


import numpy as np
import pandas as pd


# In[3]:


movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')


# In[ ]:


movies.head()


# In[5]:


# movies = movies.rename(columns={'id':'movie_id'})
# movies.merge(credits, on='movie_id').shape
movies = movies.merge(credits, on='title')


# In[ ]:


movies.info()


# In[6]:


movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]
movies.shape


# In[7]:


movies.isnull().sum()


# In[8]:


movies.dropna(inplace=True)


# In[9]:


movies.duplicated().sum()


# In[10]:


import ast


# In[12]:


print(type(movies.iloc[0].genres))#this is string need to convert it into list
def dect_to_list(obj):
  list1 = []
  for i in ast.literal_eval(obj):
    list1.append(i["name"])
  return list1
dect_to_list(movies.iloc[0].genres)


# In[13]:


movies['genres'] = movies['genres'].apply(dect_to_list)


# In[14]:


movies['keywords'] = movies['keywords'].apply(dect_to_list)


# In[15]:


def eval(obj):
  list1 = []
  cnt = 0
  for i in ast.literal_eval(obj):
    while cnt < 3:
      list1.append(i['name'])
      cnt += 1
  return list1


# In[16]:


movies['cast'] = movies['cast'].apply(eval)


# In[17]:


def director(obj):
  list1 = []
  for i in ast.literal_eval(obj):
    if i['job'] == 'Director':
      list1.append(i['name'])
      break
  return list1


# In[18]:


movies['crew'] = movies['crew'].apply(director)


# In[19]:


movies['overview'].apply(lambda x:x.split())[0]


# In[20]:


movies['overview'] = movies['overview'].apply(lambda x:x.split())


# In[21]:


movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])


# In[22]:


movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']


# In[23]:


new_df = movies[['movie_id','title','tags']]


# In[24]:


new_df.loc[:, 'tags'] = new_df['tags'].apply(lambda x: " ".join(x))


# In[25]:


import nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[26]:


def stem(text):
  y = []
  for i in text.split():
    y.append(ps.stem(i))
  return " ".join(y)


# In[27]:


new_df['tags'] = new_df['tags'].apply(stem)


# In[28]:


from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors1 = cv.fit_transform(new_df['tags']).toarray()


# In[29]:


cv.get_feature_names_out()
len(cv.get_feature_names_out())


# In[30]:


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
vectors2 = tfidf.fit_transform(new_df['tags']).toarray()


# In[31]:


tfidf.get_feature_names_out()
len(tfidf.get_feature_names_out())


# In[32]:


from sklearn.metrics.pairwise import cosine_similarity


# In[33]:


similarity1 = cosine_similarity(vectors1)
similarity2 = cosine_similarity(vectors2)


# In[34]:


sorted(list(enumerate(similarity2[0])),reverse=True,key=lambda x:x[1])


# In[35]:


def recommend1(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity1[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    for i in movies_list:
        print(new_df.iloc[i[0]].title)


# In[36]:


def recommend2(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity2[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    for i in movies_list:
        print(new_df.iloc[i[0]].title)


# In[37]:


def recommend_multi(movies):
    # Get indices for each input movie
    movie_indices = [new_df[new_df['title'] == movie].index[0] for movie in movies if movie in new_df['title'].values]
    
    if not movie_indices:
        print("None of the movies were found.")
        return

    # Sum their similarity vectors
    combined_similarity = sum(similarity2[i] for i in movie_indices) / len(movie_indices)

    # Enumerate and sort (excluding input movies)
    movies_list = sorted(
        list(enumerate(combined_similarity)),
        reverse=True,
        key=lambda x: x[1]
    )

    # Remove the input movies from the recommendations
    recommended = []
    for i in movies_list:
        if i[0] not in movie_indices:
            recommended.append(new_df.iloc[i[0]].title)
        if len(recommended) == 5:
            break

    # Print recommendations
    for title in recommended:
        print(title)
