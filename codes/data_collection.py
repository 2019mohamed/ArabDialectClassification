# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import requests
import pandas as pd

df = pd.read_csv('C:\\Users\\M\\Downloads\\dialect_dataset.csv')
N = len(df)
print(N)
loop = int(N /1000)
#rem = N - (loop * 1000) = 197
text = []
for i in range(loop + 1):
    if i == loop:
        sample = df.iloc[i*1000 : i*1000+197 ,  0].astype('str').tolist()
        
    sample = df.iloc[i*1000 : i*1000+1000 ,  0].astype('str').tolist()
    
    response = requests.post('https://recruitment.aimtechnologies.co/ai-tasks' 
                             , json = sample )
    ret = response.json()
    text += [ret[k] for k in ret.keys()]
    #print([ret[k] for k in ret.keys()])

df['text'] = text
print(df.head(20))
print(df.tail(20))

df.to_csv('C:\\Users\\M\\Downloads\\new_dialect_dataset.csv', index = False)

df1 = pd.read_csv('C:\\Users\\M\\Downloads\\new_dialect_dataset.csv')
print(len(df1))

print(df1.head(20))
print(df1.tail(20))



