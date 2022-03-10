# -*- coding: utf-8 -*-
"""
@author: M
"""
#vectorizer and reducer class

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer,HashingVectorizer,TfidfVectorizer
import pickle
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


df = pd.read_csv('C:\\Users\\M\\Downloads\\two_clo_dialect_dataset.csv')
print(len(df))


corpus = df['new_text'].astype('str').tolist()
labels = df['label'].tolist()

'''
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
print(len(vectorizer.get_feature_names_out()))
pickle.dump(vectorizer, open('C:\\Users\\M\\Downloads\\vectorizer.pkl','wb'))
'''

'''
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
print(len(vectorizer.get_feature_names_out()))
pickle.dump(vectorizer, open('C:\\Users\\M\\Downloads\\vectorizer2.pkl','wb'))
'''

'''
its dim >> dim of count or tfidvectorizer !
vectorizer = HashingVectorizer()
X = vectorizer.fit_transform(corpus)
pickle.dump(vectorizer, open('C:\\Users\\M\\Downloads\\vectorizer3.pkl','wb'))
vec = pickle.load(open('C:\\Users\\M\\Downloads\\vectorizer3.pkl', 'rb'))
print(vec.transform(corpus).shape)
'''

'''
have the same dim and strings !, so, tfid score isnot usful in the dataset

vec2 = pickle.load(open('C:\\Users\\M\\Downloads\\vectorizer2.pkl', 'rb'))

vec = pickle.load(open('C:\\Users\\M\\Downloads\\vectorizer.pkl', 'rb'))

print(set(vec.get_feature_names_out()) == set(vec2.get_feature_names_out()))
'''

'''
Z = TSNE(init = 'random' , n_components= 2).fit_transform(X)

plt.scatter(Z[:,0] , Z[:,1] , c = labels)
plt.show()
'''
import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD

def get_indexes ():
    freqs = df['label'].value_counts()
    
    start = [0 for _ in range(18)]
    start[0] = 0
    for c in range(1,18):
        start[c] += freqs[c-1] + start[c-1]  
    
    indexes = []# (mini , maxi+1) for all classes
    for c in range(18):
        indexes.append(  (start[c] , start[c] + freqs[c]) )
       
    return indexes

def sample_data (data , label ,intervals, r = 0.2):
  data = np.array(data)
  label = np.array(label)
  ret = np.array([])
  retl = np.array([])
  for i in range(18):
    l = intervals[i][0]
    h = intervals[i][1]
    index = np.random.choice(range(l,h), size = int((h-l) * r) , replace = False )
    ret = np.concatenate((ret , data[index]), axis = 0)
    retl = np.concatenate((retl , label[index]), axis = 0)

  return ret,retl
'''
MemoryError

indexes = get_indexes()
vec = pickle.load(open('C:\\Users\\M\\Downloads\\vectorizer.pkl', 'rb'))
sample_corpus, sample_label = sample_data(corpus , labels, indexes , 0.1)
X = vec.transform(sample_corpus)
print(X.shape)

reducer = TruncatedSVD(n_components= 300)
Z = reducer.fit_transform(X)

pickle.dump(reducer, open('C:\\Users\\M\\Downloads\\reducer.pkl','wb'))

print(Z.shape)
'''