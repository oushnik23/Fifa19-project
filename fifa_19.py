# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 10:47:24 2021

@author: Administrator
"""

import os
os.getcwd()
os.chdir("C:/Users/Administrator/Desktop/PYTHON")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

fifa=pd.read_csv("fifa19.csv")
fifa.columns
fifa.dtypes
fifa.shape
fifa.info()

fifa['International Reputation'].value_counts()
fifa['Position'].value_counts()
pl=fifa[['Position','Name','Overall']].head(10)
fifa['Club']
fifa['Club'].value_counts()
fifa['Nationality'].value_counts()

fifa=fifa.drop(['Unnamed: 0','Club Logo','Jersey Number','Photo','Joined','Special','Loaned From','Body Type', 'Release Clause','Weight','Height','Contract Valid Until'],axis=1)

a=fifa[fifa.Nationality=='India']
fifa[fifa.Club=='Manchester United']
fifa[fifa.Club=='FC Barcelona']['Name']

fifa.drop_duplicates(subset="Name", inplace=True)
a=fifa.groupby('Name')['Overall'].sum().sort_values(ascending=False)

fifa.rename(columns={'Preferred Foot':'Preferred_foot'},inplace=True)

fifa['Preferred_foot'].unique()

fifa.dropna(axis='columns')
fifa['Preferred_foot']=fifa.Preferred_foot.fillna(fifa.Preferred_foot.mode()[0])
c=fifa.isnull().sum()
fifa['Club']=fifa.Club.fillna('No Club')
fifa['Position'].unique()
fifa['Position']=fifa.Position.fillna('Unknown')

fifa[['Name','Club']]

fifa1=fifa.dropna(axis=0,how='any')
fifa1.columns

len(fifa)-len(fifa1)

print("Total no of club {0}".format(fifa1['Club'].nunique()))
print("Total no of country {0}".format(fifa1['Nationality'].nunique()))
print("Maximum Overall : " +str(fifa.loc[fifa['Overall'].idxmax()][1]))
print("Maximum Potential : " +str(fifa1.loc[fifa1['Potential'].idxmax()][1]))

col=['Crossing','Finishing','Overall','Stamina']
i=0;
while i<len(col):
    print("Best {0} : {1}".format(col[i],fifa1.loc[fifa[col[i]].idxmax()][1]))
    i+=1

fifa1['Value'].apply(type)
fifa1['Value']

b=fifa1.head().T

def value_and_wage_conversion(Value):
    if isinstance(Value,str):
        out = Value.replace('€', '')
        if 'M' in out:
            out = float(out.replace('M', ''))*1000000
        elif 'K' in Value:
            out = float(out.replace('K', ''))*1000
        return float(out)

fifa1['Value'] = fifa1['Value'].apply(lambda x: value_and_wage_conversion(x))
fifa1['Wage'] = fifa1['Wage'].apply(lambda x: value_and_wage_conversion(x))

print("Most valuable player :" +str(fifa1.loc[fifa1['Wage'].idxmax()][1]))
print("Most valuable player :" +str(fifa1.loc[fifa1['Value'].idxmax()][1]))

#fifa1.to_csv('fifa_clean.csv',index=False)

sns.jointplot(x=fifa1['Age'],y=fifa1['Potential'])
fifa1['Position'].unique()
fifa1.columns
temp=fifa1["Work Rate"].str.split("/ ", n = 1, expand = True)
fifa1['Work_rate1']=temp[0]
fifa1['Work_rate2']=temp[1]

fifa1=fifa1.drop(['Flag'],axis=1)
fifa1=fifa1.drop(['LS','ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW', 'LAM', 'CAM', 'RAM', 'LM','LCM', 'CM', 'RCM', 'RM', 'LWB', 'LDM', 'CDM', 'RDM', 'RWB', 'LB','LCB', 'CB', 'RCB', 'RB'],axis=1)
fifa1=fifa1.drop(['Work Rate'],axis=1)

fifa1['Preferred_foot'].unique()
fifa1['Preferred_foot']=fifa1['Preferred_foot'].replace('Right',1)
fifa1['Preferred_foot']=fifa1['Preferred_foot'].replace('Left',0)

fifa1['']
fifa1['Real Face']=fifa1['Real Face'].replace('Yes',1)
fifa1['Real Face']=fifa1['Real Face'].replace('No',0)


fifa1['Position'].apply(type)
fifa1['Position'].unique()

def simple_position(fifa1):
    if(fifa1['Position']=='GK'):
        return 'GK'
    elif ((fifa1['Position']=='RWB')|(fifa1['Position']=='LWB')|(fifa1['Position']=='CB')|(fifa1['Position']=='LB')|(fifa1['Position']=='RB')|(fifa1['Position']=='RCB')|(fifa1['Position']=='LCB')) :
        return 'CD'
    elif ((fifa1['Position']=='LCM')|(fifa1['Position']=='RCM')|(fifa1['Position']=='CM')|(fifa1['Position']=='LM')|(fifa1['Position']=='RM')):
        return 'CM'
    elif ((fifa1['Position']=='LDM')|(fifa1['Position']=='RDM')|(fifa1['Position']=='CDM')):
        return 'CDM'
    elif ((fifa1['Position']=='CAM')|(fifa1['Position']=='RAM')|(fifa1['Position']=='LAM')):
        return 'AM'
    elif ((fifa1['Position']=='RF')|(fifa1['Position']=='ST')|(fifa1['Position']=='LF')|(fifa1['Position']=='LW')|(fifa1['Position']=='RS')|(fifa1['Position']=='RW')|(fifa1['Position']=='CF')):
        return 'ST'
    else:
        return fifa1.Position
fifa1['Simple_Position']=fifa.apply(simple_position,axis=1)

x=fifa1.drop(['Overall','Name','Nationality','Club','Position'], axis=1)
y=fifa1['Overall']

x=pd.get_dummies(x)

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=.2)

from sklearn.linear_model import LinearRegression
lr_model=LinearRegression()
lr_model.fit(xtrain,ytrain)

prediction=lr_model.predict(xtest)
np.set_printoptions(precision=0)
print(np.concatenate((prediction.reshape(len(prediction),1),ytest.values.reshape(len(ytest),1)),1))

from sklearn.metrics import r2_score 
r2_score(ytest,prediction)*100

from sklearn.model_selection import cross_val_score
score=cross_val_score(X=xtrain,y=ytrain,cv=10,estimator=lr_model)
print("Accuracy {0} :".format(score.mean()*100))
print("Standard deviation {0}".format(score.std()*100))
