# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 15:37:13 2020

@author: luthe
"""



##################Kaggle compétition introductive, les survivants du Titanic #######################



import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sb



#%% Data loading


### Loading the files
Training_data=pd.read_csv(r'C:\Users\luthe\Documents\AI\Compétitions_kaggle\Titanic_0\Data\train.csv')
Test_data=pd.read_csv(r'C:\Users\luthe\Documents\AI\Compétitions_kaggle\Titanic_0\Data\test.csv')
#%%
##########################Seaborn visualization############
# Plotting the correlation matrix, we can notice that Fare is the only that is strongly
#correleted with Survived
Corel_matrix=sb.heatmap(Training_data[["Survived","SibSp","Parch","Age","Fare"]].corr(),
                 cmap = "Blues",
                 annot=True)

################ Data visualization : we're gonna explore our data to decide how to preprocess it ########
#%% Pclass and Sex

#plotting nb of each sex, there is adulttwice more men than women
Nb_ofeach_sex = sb.catplot(x="Sex",
                           kind="count",
                           data=Training_data,
                           height=3.5)
#Plotting the survival rate for each class,
#People from the first class are more likely to survive than others                     
Survival_rate_Pclass = sb.catplot(x="Pclass",
                                  y="Survived",
                                  kind="bar",
                                  data=Training_data).set_ylabels("survival probability")
#Plotting the survival rate for each sex according the class
Survival_rate_Sex_Pclass = sb.catplot(x="Sex",
                                       y="Survived",
                                       hue="Pclass",
                                       kind="bar",
                                       data=Training_data).set_ylabels("survival probability")

#ce qu'on apprend ici est assez évident, les femmes ont plus de chances de survivre 
#que les hommes en moyenne, et la richesse influe également, les riches ont plus de chance de survivre
#Nos données sont déjà subdivisées dans ces catégories assez claires pour un algorithme

print( Training_data[["Pclass","Survived"]].groupby(["Pclass"], as_index = False).mean() )
print( Training_data[["Sex","Survived"]].groupby(["Sex"], as_index = False).mean() )
#pas de missing values dans ces données


#%% Age_Cat

#Plotting the age hist of those who survived and the other 
Survival_Age = sb.FacetGrid(Training_data,
                            col='Survived',
                            height=4).map(sb.distplot,
                                            "Age",
                                            kde=False) #it's hard to see, let's make some category
#let's put people's age in 5 categories : 0-->15 Children,
#                                        15-->25 young adult,
#                                        25-->40 adult,
#                                        40-->60 old adult,
#                                        >60 very old adult
Training_data['AgeCat']=pd.cut(Training_data['Age'],
                  [0,15,25,40,60, 
                   np.max(Training_data['Age'])], 
                   labels=['Children','Young adult','Adult','Old adult', 'Very old adult'])

Survival_rate_AgeCat=sb.catplot(x='AgeCat',
                                y='Survived',
                                kind='bar',
                                data=Training_data).set_ylabels("survival probability")

#As expected, young are more likely to survive as old
#this feature engeneering will allow us to try to guess the age category of some 'nan' people
#from other feature

Survival_rate_AgeCat_Pclass=sb.catplot(x='AgeCat',
                                    y='Survived',
                                    hue='Pclass',
                                    kind='bar',
                                    data=Training_data).set_ylabels("survival probability")

#%%Compagnons

print( Training_data[["SibSp","Survived"]].groupby(["SibSp"], as_index = False).mean() )
print( Training_data[["Parch","Survived"]].groupby(["Parch"], as_index = False).mean() )
 
#let's create a feature of proche 

Training_data['Compagnons']=Training_data['SibSp']+Training_data['Parch']
 
print(Training_data[['Compagnons','Survived']].groupby(['Compagnons'], as_index=False).mean())


#When you are alone, you have less chance to survive than when you have a small family
#Let's cut Proche in  3 category : alone, small family, big family>3

Training_data['Proche']=pd.cut(Training_data['Compagnons'], 
                             [0,1,4,np.max(Training_data['Compagnons'])+1],
                             right=False,
                             labels=['alone','small family', 'big family'])

Survival_rate_Proche=sb.catplot(x='Proche',
                                y='Survived',
                                kind='bar',
                                data=Training_data).set_ylabels("survival probability")

#%%Proche and Fare


print(Training_data[['Fare','Name','Ticket','SibSp','Parch']].iloc[Training_data.index[(Training_data['Pclass']==3) & (Training_data['Fare']>60)]])

#Alors en fait ce que je constate en explorant un peu les données, c'est :
#    - Ya des prix de billets qui sont élevés mais quand même des Pclass qui sont à 2 ou 3
#    - Des numéros de Tickets égaux ont exactement les même prix de tickets
#    - Des numéros de Tickets égaux ont soit des noms communs associés (style même famille)ou
#    une sorte de servante car c'est souvent un vieux avec un jeune.
#    - on a des proches mal évalués alors qu'ils ont le même nom et le même ticket 
#    cf ticket numéro 1601
#    
#Donc je vais réajuster les Fare en des prix unitaires par personne en prenant les Tickets égaux
#(on peut considérer que les liens avec une servante agissent pas tant que ça dans la survie)

Training_data['Fare_adjusted']=Training_data['Fare'].copy() #on copie nos fare actuels

for Ticket, grp_same_ticket_info in Training_data[['Ticket', 'Name', 'Pclass', 'Fare', 'PassengerId']].groupby(['Ticket']):
    #on parcours la data en groupant par ticket et après on a 2 variables
    #Ticket c'est le numéro du Ticket, grp_same_ticket_info c'est le data frame réduit aux gens qui ont le même ticket
    if (len(grp_same_ticket_info) != 1):
        #si on a plus d'une personne qui ont le même ticket
        for index, row in grp_same_ticket_info.iterrows():
            #on parcours le dataframe du groupe, on vire l'index et on regarde juste la ligne
            passID = row['PassengerId'] #on récupère l'ID du passager
            Training_data.loc[Training_data['PassengerId'] == passID, 'Fare_adjusted'] = Training_data['Fare'][Training_data['PassengerId'] == passID]/len(grp_same_ticket_info)
                #on remodifie le df original en divisant par la longueur du groupe
                
                
#%%Title 
                
#We can notice that most of the passengers have a title that give clues on their age, 
# social status, may be we can use thoses title in order to complete some of the missing values

# extracting the Title (which always ends with a ".")

Training_data['Title'] = Training_data['Name'].str.extract('([A-Za-z]+)\.', expand=True)

print(Training_data['Title'].value_counts())

#There is some title that means the same thing so let's regroup them

Training_data['Title']=Training_data['Title'].replace({'Ms':'Miss','Mlle':'Miss','Mme':'Mrs'})

#let's put the noble together in a category noble 

Training_data['Title']=Training_data['Title'].replace(['Sir','Don','Dona','Jonkheer','Lady','Countess'], 'Noble')

#And the rest in a category other, but we know that they probably have a higher social standing

Training_data['Title']=Training_data['Title'].replace(['Dr', 'Rev','Col','Major','Capt'], 'Others')
                
#%%let's precise more Title

print(Training_data[['Age','Title']].groupby(['Title']).mean(),
      Training_data['Age'][Training_data['Title']=='Miss'].var())

#So, what we can notice is that master are mostly children and miss has a big standard deviation (=sqrt(var))
#let's create two group, miss and young miss. 
#We can also notice that a young miss is more likely with at least one Parch so if we have not her age 
#but she's a miss and she has a Parch we can set her AgeCat to Children 

for index,age in Training_data[['PassengerId','Age']][Training_data['Title']=='Miss'].iterrows():
    if math.isnan(age['Age']) and Training_data.loc[:,'Parch'][index]>=1:
        Training_data.loc[Training_data['PassengerId']==index+1, 'AgeCat']='Children'
        Training_data.loc[Training_data['PassengerId']==index+1, 'Title']='YoungMiss'
    if age['Age']<15:
        Training_data.loc[Training_data['PassengerId']==index+1, 'Title']='YoungMiss'
    

print(Training_data[['Age','Title']].groupby(['Title']).mean(),
      Training_data['Age'][Training_data['Title']=='Miss'].var())

#On peut observer que l'écart type des miss n'est plus que d'une dizaine d'année et que les young miss
#sont jeunes

#on peut aussi dans le même genre mettre les master dans children

for index,age in Training_data[['PassengerId','Age']][Training_data['Title']=='Master'].iterrows():
    if math.isnan(age['Age']):
        Training_data.loc[Training_data['PassengerId']==index+1, 'AgeCat']='Children'

#enfin pour compléter les ages nan qui manque on les place dans la catégorie adult

for index,age in Training_data[['PassengerId','Age']][Training_data['Title'].isin(['Miss','Mr','Mrs','Noble','Others'])].iterrows():
    if math.isnan(age['Age']):
        Training_data.loc[Training_data['PassengerId']==index+1, 'AgeCat']='Adult'
        
#%%Cabines
        
#From the cabines number we can extract the corresponding boat deck A,B,C,D,E or F, we dont take the others 
#because there is not enough value
        
Training_data['Aile'] = Training_data['Cabin'].str.extract('([ABCDEF])', expand=True)

#let's visualize :

Survival_rate_Aile=sb.catplot('Aile',
                              'Survived',
                              kind='bar',
                              data=Training_data).set_ylabels('Survival rate')
