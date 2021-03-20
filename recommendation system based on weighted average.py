#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


credits=pd.read_csv("E:\\Data Set\\tmdb_5000_credits.csv")
movies_df=pd.read_csv("E:\\Data Set\\tmdb_5000_movies.csv")


# In[3]:


credits.head()


# In[5]:


movies_df.head()


# In[6]:


credits.shape


# In[7]:


movies_df.shape


# # data cleaning and Transformation

# In[9]:


#combining the movie dataset with credits dataset by the similar column ie:movie_id and id
credits_column_renamed= credits.rename(index=str,columns={"movie_id":"id"})
movies_df_merge= movies_df.merge(credits_column_renamed,on='id')
movies_df_merge.head()


# In[12]:


print(movies_df_merge.columns)


# In[13]:


# eradicating unnecessary columns for better machine learning performance
movies_cleaned_df=movies_df_merge.drop(columns=['homepage','title_x','title_y','status','production_countries'])
movies_cleaned_df.head()


# In[16]:


#checking for missing data and Nan values in dataframe
movies_cleaned_df.info()

# Using Weighted average for each movie's Average Rating

W=Rv+Cm/v+m

where:
W=Weighted Rating
R= average for the movie as a number from 0 to 10(mean)=Rating
v=number of votes for the movie=(votes)
m=minimum votes required to be listed in the top 250(currently 300)
C=the mean vote across the whole report(currently 6.9)

# In[17]:


#Calulating all the components based on the above formula
v=movies_cleaned_df['vote_count']
R=movies_cleaned_df['vote_average']
C=movies_cleaned_df['vote_average'].mean()
m=movies_cleaned_df['vote_count'].quantile(0.70)


# In[20]:


movies_cleaned_df['vote_average'].mean()


# In[26]:


movies_cleaned_df['weighted_average']=((R*v) + (C*m)) /(v+m)


# In[27]:


movies_cleaned_df.head()


# In[30]:


movie_sorted_ranking=movies_cleaned_df.sort_values('weighted_average',ascending=False)
movie_sorted_ranking[['original_title','vote_count','vote_average','weighted_average','popularity']].head()


# In[36]:


import matplotlib.pyplot as plt
import seaborn as sns

weight_average=movie_sorted_ranking.sort_values('weighted_average',ascending=False)
plt.figure(figsize=(12,6))
axis1=sns.barplot(x=weight_average['weighted_average'].head(10),y=weight_average['original_title'].head(10),data=weight_average)
plt.xlim(4,10)
plt.title("best movies by average votes",weight='bold')
plt.xlabel('Weighted average score',weight='bold')
plt.ylabel('Movie Title',weight='bold')
plt.savefig('best_movies.png')


# In[35]:


movies_cleaned_df['popularity'].head()


# In[40]:


popularity=movie_sorted_ranking.sort_values('popularity',ascending=False)
plt.figure(figsize=(12,6))
ax=sns.barplot(x=popularity['popularity'].head(10),y=popularity['original_title'].head(10),data=popularity)
plt.title('most popular by votes',weight='bold')
plt.xlabel('score of popularity ',weight='bold')
plt.ylabel('movie title',weight='bold')
plt.savefig('best popular movies.png')


# #  Recommendation is based on scaled weighted average and popularity score(both have equal priority 50%)

# In[42]:


# weighted average and popularity are having different magintudes which leades to bad fitting in machine learning model ,there scaling is done
from sklearn.preprocessing import MinMaxScaler
scaling=MinMaxScaler()
movie_scaled_df=scaling.fit_transform(movies_cleaned_df[['weighted_average','popularity']])
movie_normalized_df=pd.DataFrame(movie_scaled_df,columns=['weighted_average','popularity'])
movie_normalized_df.head()


# In[44]:


movies_cleaned_df[['normalized_weight_average','normalized_popularity']]=movie_normalized_df


# In[45]:


movies_cleaned_df.head()


# In[48]:


movies_cleaned_df['score']=movies_cleaned_df['normalized_weight_average']*0.5 + movies_cleaned_df['normalized_popularity']*0.5
movies_scored_df=movies_cleaned_df.sort_values(['score'],ascending=False)
movies_scored_df[['original_title','normalized_weight_average','normalized_popularity','score']].head(20)


# In[49]:


scored_df = movies_cleaned_df.sort_values('score', ascending=False)

plt.figure(figsize=(16,6))

ax = sns.barplot(x=scored_df['score'].head(10), y=scored_df['original_title'].head(10), data=scored_df, palette='deep')

#plt.xlim(3.55, 5.25)
plt.title('Best Rated & Most Popular Blend', weight='bold')
plt.xlabel('Score', weight='bold')
plt.ylabel('Movie Title', weight='bold')

plt.savefig('scored_movies.png')


# In[ ]:




