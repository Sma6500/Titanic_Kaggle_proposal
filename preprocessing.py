# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 23:03:12 2020

@author: luthe
"""

##################Kaggle compétition introductive, les survivants du Titanic #######################

#%% loading library
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import StandardScaler

#%% preprocessing 

Training_data=pd.read_csv(r'C:\Users\luthe\Documents\AI\Compétitions_kaggle\Titanic_0\Data\train.csv')
Test_data=pd.read_csv(r'C:\Users\luthe\Documents\AI\Compétitions_kaggle\Titanic_0\Data\test.csv')

#%%
def title_list(df):
    #input: Data_frame of titanic_values
    #output: Data_frame with the titles in a new columns, cf ci-dessous pour la liste des titres
    df['Title'] = df['Name'].str.extract('([A-Za-z]+)\.', expand=True)
    df['Title']=df['Title'].replace({'Ms':'Miss','Mlle':'Miss','Mme':'Mrs'})
    df['Title']=df['Title'].replace(['Sir','Don','Dona','Jonkheer','Lady','Countess'], 'Noble')
    df['Title']=df['Title'].replace(['Dr', 'Rev','Col','Major','Capt'], 'Others')
    return df


def Age_cat(df):
    #input : Data_frame of titanic_values
    #output : Data_frame with the AgeCat and the title 'YoungMiss'
    
    #we cut our age in 5 AgeCat
    df['AgeCat']=pd.cut(df['Age'],
                  [0,15,25,40,60, 
                   np.max(df['Age'])], 
                   labels=['Children','Young adult','Adult','Old adult', 'Very old adult'])
    
    #we create a new title for the young miss
    for index,age in df[['PassengerId','Age']][df['Title']=='Miss'].iterrows():
       if math.isnan(age['Age']) and df.loc[:,'Parch'][index]>=1:
           df.loc[:,'AgeCat'][index]='Children'
           df.loc[:,'Title'][index]='YoungMiss'
       if age['Age']<15:
           df.loc[:,'Title'][index]='YoungMiss'
        
    #we assign the master with no age to the cat children   
    for index,age in df[['PassengerId','Age']][df['Title']=='Master'].iterrows():
        if math.isnan(age['Age']):
            df.loc[:,'AgeCat'][index]='Children'
            
    #We assign the other unknwon age to the cat adult that correpson to the median and mean
    for index,age in df[['PassengerId','Age']][df['Title'].isin(['Miss','Mr','Mrs','Noble','Others'])].iterrows():
        if math.isnan(age['Age']):
            df.loc[:,'AgeCat'][index]='Adult'
            #df.loc[df['PassengerId']==index+1, 'AgeCat']='Adult'      
    return df
            
def Age_list(Age_passagers):
    #input: Ages des passagers
    #output: Ages des passagers avec les nan remplacés par la moyenne
    Age_passagers=np.float64(Age_passagers)
    mean_age=0
    count=0
    for i in Age_passagers: #on fait la moyenne
        if not(math.isnan(i)):
            mean_age+=i
            count+=1
        mean_age=mean_age/count
    count=0
    for i in Age_passagers: #on remplace les nan
        if math.isnan(i):
            Age_passagers[count]=mean_age
        count+=1
    return Age_passagers


def Proche_Compagnons_list(df): 
    #input : Data_frame of titanic_values
    #output: Same Data_frame with two new columns : 
            # Compagnons that is the sum of Parch and SibSp
            # Proche that is the size of the family
    df['Compagnons']=df['SibSp']+df['Parch']
    df['Proche']=pd.cut(df['Compagnons'], 
                             [0,1,4,np.max(df['Compagnons'])+1],
                             right=False,
                             labels=['alone','small family', 'big family'])
    return df

def Adjusted_Fare(df):
    #input: Data_frame of titanic_values
    #output: Same Data_frame but with the Fare_adjusted to each personn 
    
    df['Fare_adjusted']=df['Fare'].copy() #on copie nos fare actuels

    for Ticket, grp_same_ticket_info in df[['Ticket', 'Name', 'Pclass', 'Fare', 'PassengerId']].groupby(['Ticket']):
    #on parcours la data en groupant par ticket et après on a 2 variables
    #Ticket c'est le numéro du Ticket, grp_same_ticket_info c'est le data frame réduit aux gens qui ont le même ticket
        if (len(grp_same_ticket_info) != 1):
        #si on a plus d'une personne qui ont le même ticket
            for index, row in grp_same_ticket_info.iterrows():
            #on parcours le dataframe du groupe, on vire l'index et on regarde juste la ligne
                passID = row['PassengerId'] #on récupère l'ID du passager
                df.loc[df['PassengerId'] == passID, 'Fare_adjusted'] = df['Fare'][df['PassengerId'] == passID]/len(grp_same_ticket_info)
    return df



def preprocessing_X(df, standardisation=True, Sieblings=False, Parch=False , Age=False): 
    #input: dataframe of the Titaniv_data
    #we can choose to put the Sieblings or the Parents or not,
    #we can choose to standardize or not
    #same with age filling the missing values by the mean
    #output: X data_frame for training or predict and eventually Y of survived for training
    
    #first we check the labels columns with the survival
    
    X=df.copy()
    Columns=X.columns
    
    X=Proche_Compagnons_list(X)
    X=Adjusted_Fare(X)
    X=title_list(X)
    X=Age_cat(X)
    X['Aile']=X['Cabin'].str.extract('([ABCDEF])', expand=True) 

    
    if not(Sieblings):
        del X['SibSp']
    if not(Parch):
        del X['Parch']
    if Age: 
        X['Age']=Age_list(X['Age'])
        del X['AgeCat']
    if not(Age):
        del X['Age']
        
    if "Survived" in Columns:
        Y=df['Survived']
        del X['Survived']
    else : 
        Y=0    
    

    X = X.drop(['PassengerId','Name','Cabin','Ticket','Fare'],axis=1)

    X=pd.get_dummies(X)
    Columns=X.columns
    scaler=StandardScaler()
    if standardisation:
        X=scaler.fit_transform(X)
        X=pd.DataFrame(X, columns=Columns)
    return X,Y
    

