# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 00:17:42 2020

@author: luthe
"""



##################Kaggle compétition introductive, les survivants du Titanic #######################




import sys
sys.path.append('C:\\Users\\luthe\\Documents\\AI\\Compétitions_kaggle\\Titanic_0') #j'ai pas encore pigé pourquoi je dois faire ça
import preprocessing as ppg
import pandas as pd
import numpy as np
import sklearn
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import Perceptron
import seaborn as sb
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.calibration import calibration_curve


#%% Data loading and preprocessing


### Loading the files
Training_data=pd.read_csv(r'C:\Users\luthe\Documents\AI\Compétitions_kaggle\Titanic_0\Data\train.csv')
Test_data=pd.read_csv(r'C:\Users\luthe\Documents\AI\Compétitions_kaggle\Titanic_0\Data\test.csv')

## preprocessing them
X,Y=ppg.preprocessing_X(Training_data)
X_to_evaluate,y=ppg.preprocessing_X(Test_data)
X_to_evaluate.loc[:,'Fare_adjusted'][152]=np.mean(X['Fare_adjusted']) #c'est le seul passager qui a pas de Fare,
#le nan m'empeche de run correctement les algo mais c'est bizarre parce que ça a marché les
#2 premières fois du coup je laisse ça là au cas où pour que je m'en souvienne


### making train and test, and weight because there is more dead than survived people
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1, random_state=0)


Nombre_de_survivant=np.sum(Y)
class_weight={0:Y.shape[0]-Nombre_de_survivant, 1:Nombre_de_survivant}


#%%Random Forest Exploration

###affichons les scores pour différents paramètres
score_forest=np.zeros((5,5,2,2))
i,j=0,0
for m_f in [5,10,15,20]: #max features
    j=0
    for m_d in [5,10,15,20,25]: #max depth
        forest_titanic=RandomForestClassifier(n_estimators=10000, 
                                                  max_features=m_f,
                                                  max_depth=m_d,
                                                  class_weight=class_weight).fit(X_train,Y_train)
        score_forest[i][j][0][0]=forest_titanic.score(X_train,Y_train)
        score_forest[i][j][0][1]=forest_titanic.score(X_test,Y_test)
        forest_titanic=RandomForestClassifier(n_estimators=10000, 
                                                  max_features=m_f,
                                                  max_depth=m_d,
                                                  criterion='entropy',
                                                  class_weight=class_weight).fit(X_train,Y_train)
        score_forest[i][j][1][0]=forest_titanic.score(X_train,Y_train)
        score_forest[i][j][1][1]=forest_titanic.score(X_test,Y_test)
        j+=1
    i+=1

print(score_forest)


#%%Gradient Boosting Exploration

score_gbrt=np.zeros((4,5,2))
i,j=0,0
for learning_rate in [0.001,0.01,0.1,0.5]: #learning_rate
    j=0
    for m_d in [1,2,3,4,5]: #max depth
        Gbrt_titanic=GradientBoostingClassifier(n_estimators=10000, 
                                                  learning_rate=learning_rate,
                                                  max_depth=m_d).fit(X_train,Y_train)
        score_gbrt[i][j][0]=Gbrt_titanic.score(X_train,Y_train)
        score_gbrt[i][j][1]=Gbrt_titanic.score(X_test,Y_test)
        j+=1
    i+=1
print(score_gbrt)

#%%Perceptron Exploration 

penalty=['l2','l1','elasticnet']
alpha=[10**(-i) for i in range(1,6)]
score_pcpt=np.zeros((3,5,2))

i,j=0,0
for pnlt in penalty:
    j=0
    for a in alpha:
        pcpt_titanic=Perceptron(penalty=pnlt,
                                alpha=a,
                                tol=0.000001,
                                class_weight=class_weight).fit(X_train,Y_train)
        score_pcpt[i][j][0]=pcpt_titanic.score(X_train,Y_train)
        score_pcpt[i][j][1]=pcpt_titanic.score(X_test,Y_test)
        j+=1
    i+=1

print(score_pcpt)

#%% SVM Exploration

param_grid={'C':[0.001,0.01,0.1,1,10,100],
            'gamma':[0.001,0.01,0.1,1,10,100]}
SVC_grid_search=GridSearchCV(SVC(class_weight=class_weight), param_grid=param_grid, cv=5)
SVC_grid_search.fit(X_train,Y_train)
print(SVC_grid_search.score(X_test,Y_test),
      SVC_grid_search.best_params_)



#%% Algorithm training 

######Random Forest####################

#Pour m_f=20 et m_d=10 et criterion='entropy' on a le meilleur résultat sans overfitting 
    
m_f=5
m_d=15

forest_titanic=CalibratedClassifierCV(RandomForestClassifier(n_estimators=10000,
                                      max_depth=m_d,
                                      max_features=m_f,
                                      criterion='gini',
                                      class_weight=class_weight)).fit(X_train,Y_train)


########Gradient Boosting #################

#pour learning_rate=0.01 et m_d=4 on a le meilleur résultat sans overfitting

learning_rate=0.1
m_d_2=1

Gbrt_titanic=CalibratedClassifierCV(GradientBoostingClassifier(n_estimators=10000,
                                        learning_rate=learning_rate,
                                        max_depth=m_d_2)).fit(X_train,Y_train)


##########Perceptron ##################

#on choisit une norme l2 et un alpha=10^-5

alpha=10**(-1)
penality='l1'

pcpt_titanic=CalibratedClassifierCV(Perceptron(penalty=penality,
                        alpha=alpha,
                        tol=10**(-6),
                        class_weight=class_weight)).fit(X_train,Y_train)


###########SVM###############################

C=0.01
gamma=0.001

SVC_titanic=CalibratedClassifierCV(SVC(C=C,
                gamma=gamma,
                class_weight=class_weight, probability=True)).fit(X_train,Y_train)





#%% fonction à déplacer dans un fichier d'analyse de modèle binaire

def Stat_predict_prob_bin_class(modele, test):
    #input : modèle à analyser, data sur laquelle tester
    #output: moyenne et variance des probas prédites
    predict_proba=modele.predict_proba(test)
    res=np.zeros(len(predict_proba))
    for i in range(len(predict_proba)):
        res[i]=max(predict_proba[i]) #on prend la proba de l'élement prédit
    return np.mean(res),np.var(res)

def Stat_decision_function_bin_class(modele, test):
    #input : modèle à analyser, data sur laquelle tester
    #output : moyenne et variance des parties pos puis neg de la fonction de décision
    decision_func=modele.decision_function(test)
    pos=np.array([])
    neg=np.array([])
    for i in decision_func:
        if i>0:
            pos=np.append(pos, i)
        else:
            neg=np.append(neg, i)
    return [np.mean(pos),np.var(pos)],[np.mean(neg),np.var(neg)]

def plot_reliability_curve(X_test, Y_test, modele, with_predict_proba=True, bins=10):
    if with_predict_proba :
        proba=modele.predict_proba(X_test)[:,1]
        fop,mpv=calibration_curve(Y_test,proba, n_bins=bins)
    else :
        decision_function=modele.decision_function(X_test)
        fop,mpv=calibration_curve(Y_test, decision_function, normalize=True, n_bins=bins)
    # plot perfectly calibrated
    plt.plot([0, 1], [0, 1], linestyle='--', color='black')
    #plot the reliability curve
    plt.plot(mpv, fop, marker='.')

#%%Random Forest visualization

print(forest_titanic.score(X_test,Y_test),
      forest_titanic.score(X_train,Y_train),
      Stat_predict_prob_bin_class(forest_titanic, X_test))

plot_reliability_curve(X_test, Y_test, forest_titanic)

#%%Gradient Boosting visualization

print(Gbrt_titanic.score(X_test,Y_test),
      Gbrt_titanic.score(X_train,Y_train),
      Stat_predict_prob_bin_class(Gbrt_titanic, X_test))

plot_reliability_curve(X_test, Y_test, Gbrt_titanic)

#%%Perceptron visualization

print(pcpt_titanic.score(X_test,Y_test),
      pcpt_titanic.score(X_train,Y_train),
      Stat_predict_prob_bin_class(pcpt_titanic, X_test))
     
plot_reliability_curve(X_test, Y_test, pcpt_titanic)

#%%SVM visualization

print(SVC_titanic.score(X_test,Y_test),
      SVC_titanic.score(X_train,Y_train),
      Stat_predict_prob_bin_class(SVC_titanic, X_test))

plot_reliability_curve(X_test,Y_test,SVC_titanic)

#%%Algo combiné exploration 

from sklearn.ensemble import VotingClassifier

n=2000
estimators=[('forest',CalibratedClassifierCV(RandomForestClassifier(n_estimators=n, max_features=m_f,
                                      max_depth=m_d,
                                      criterion='gini',
                                      class_weight=class_weight))),
           ('Gr',CalibratedClassifierCV(GradientBoostingClassifier(n_estimators=n,
                                        learning_rate=learning_rate,
                                        max_depth=m_d_2))),
            ('Pc', CalibratedClassifierCV(Perceptron(penalty=penality,
                        alpha=alpha,
                        tol=10**(-6),
                        class_weight=class_weight))),
            ('SVC', CalibratedClassifierCV(SVC(C=C,
                                               gamma=gamma,
                                               probability=True, 
                                               class_weight=class_weight)))]


grid_weights={'weights':[[40,15,15,15],[15,40,15,15],[15,15,40,15],[15,15,15,40],
                       [25,25,25,25],[20,30,20,30],[20,30,30,20],[20,20,30,30]]}

Algo_combiné_grid_search=GridSearchCV(VotingClassifier(estimators=estimators,
                                                       voting='soft'),
                                    param_grid=grid_weights,
                                    cv=5).fit(X_train,Y_train)
    
print(Algo_combiné_grid_search.score(X_test,Y_test),
      Algo_combiné_grid_search.best_params_)


#%%Submission

weights=[15,40,15,15]
n=10000
Algo_combiné=VotingClassifier(estimators=estimators,
                              voting='soft',
                              weights=weights).fit(X,Y)

result=Algo_combiné.predict(X_to_evaluate)

submission = pd.DataFrame({
    "PassengerId": Test_data["PassengerId"],
    "Survived": result
})
submission.to_csv(r'C:\Users\luthe\Documents\AI\Compétitions_kaggle\Titanic_0\submission_5.csv', index = False)

        
        
        
        