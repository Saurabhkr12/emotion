import pandas as pd
import numpy as np

import seaborn as sns
import neattext.functions as nfx
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


df=pd.read_csv('emotion_dataset_2.csv')
print(df['Emotion'].value_counts())
sns.countplot(x='Emotion', data=df)
plt.show()

df['clean_text']=df['Text'].apply(nfx.remove_userhandles)
df['clean_text']=df['clean_text'].apply(nfx.remove_stopwords)
df['clean_text']=df['clean_text'].apply(nfx.remove_special_characters)

print(df.head())

Xfeatures=df['clean_text']
ylabels=df['Emotion']

X_train,X_test,Y_train,Y_test=train_test_split(Xfeatures,ylabels,test_size=0.3,random_state=42)

from sklearn.pipeline import Pipeline
pipe_lr=Pipeline(steps=[('cv',CountVectorizer()),('lr',LogisticRegression())])

pipe_lr.fit(X_train,Y_train)

print(pipe_lr)
y_pred=pipe_lr.predict(X_test)

print(str(pipe_lr.score(Y_test,y_pred)*100)+'%')


#ex1='i am so happy to have you'
#print(pipe_lr.predict([ex1]))

while True:

    txt=input('Enter your input here: ')

    print(pipe_lr.predict([txt])[0])
    k=input('any more question y/n:-') 
    if k=='y':
        continue
    elif k=='n':
        break 