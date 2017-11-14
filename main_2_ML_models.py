import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

import os, sys
lib_path = os.path.abspath(os.path.join('Ml models'))
sys.path.append(lib_path)
from prep_variables_norm import prep_variables_norm
from cv import cv
from plot_scores import plot_scores
from clf_LogisticRegression import clf_LogisticRegression
from clf_KNN import clf_KNN
from prep_variables_no_norm import prep_variables_no_norm
from clf_GradientBoostingClassifier import clf_GradientBoostingClassifier
from clf_RandomForestClassifier import clf_RandomForestClassifier
from clf_MLPClassifier import clf_MLPClassifier




train_original = pd.read_csv('Data/titanic_train_READY.csv')
train_original.set_index('PassengerId',inplace=True)
train_X = train_original.copy()
train_X.drop('Survived', axis=1, inplace=True)
train_Y = train_original.copy()['Survived']

# assigning port of embarkation to 2 passnegers without Embarked value
# S is assigned as the most probable embarkation port for the 1st class passengers
train_X.loc[[62,830],'Embarked'] = 'S'

# Variables pre-processing with normalization
#[train_X_norm, scaler_dict] = prep_variables_norm(train_X, Train=True)


## 1) Logistic regression
clf_LogisticRegression(LogisticRegression,
                       train_X,
                       train_Y,
                       prep_variables_norm,
                       np,
                       pd,
                       cv,
                       plot_scores,
                       test_param=False,
                       c=np.arange(0.1,2,0.1))
## Conclusion: Logistic regression expected accuracy: 0.80
# standard C=1 performance seems to be good, so C=1 is chosen for logistic regression





## 2) Naive Bayes
cv_res = cv(train_X, train_Y, GaussianNB(), prep_variables_norm)
# Conclusion: Gaussian Naive Bayes classifier expected accuracy: 0.74
print('Gaussian Naive Bayes classifier expected accuracy: {0:.2f}'.format(cv_res['Avg_test_score']))
print('')





## 3) KNN
clf_KNN(KNeighborsClassifier,
        train_X,
        train_Y,
        prep_variables_norm,
        np,
        pd,
        cv,
        plot_scores,
        test_param=False,
        n_neighbors=np.arange(3,8,1),
        algorithm='kd_tree')
## Conclusion: KNN classifier expected accuracy: 0.77
# Chosen parameters:
# algorithm = 'kd_tree'
# n_neighbors = 7






## 4) Gradient Boosted Decision Trees
# *this classifier does not require data normalization
#* testing a range of parameters for this method is only possible inside the function so far
clf_GradientBoostingClassifier(GradientBoostingClassifier,
                               train_X,
                               train_Y,
                               prep_variables_no_norm,
                               np,
                               pd,
                               cv,
                               plot_scores,
                               test_param=False)
## Conclusion: Gradient Boosted Decision Trees classifier expected accuracy: 0.82
# Chosen parameters:
# learning rate 0.2,
# n_estimators = 8





## 5) Random forest
# *this classifier does not require data normalization
#* testing a range of parameters for this method is only possible inside the function so far
clf_RandomForestClassifier(RandomForestClassifier,
                           train_X,
                           train_Y,
                           prep_variables_no_norm,
                           np,
                           pd,
                           cv,
                           plot_scores,
                           test_param=False,
                           n_estimators = 10,
                           max_features = 6,
                           max_depth = 3)
## Conclusion: Random forest classifier expected accuracy: 0.81
# Chosen parameters:
# n_estimators = 10
# max_features = 6
# max_depth = 3





## 6) Neural networks
#* testing a range of parameters for this method is only possible inside the function so far
clf_MLPClassifier(MLPClassifier,
                  train_X,
                  train_Y,
                  prep_variables_norm,
                  np,
                  pd,
                  cv,
                  plot_scores,
                  test_param=False,
                  alpha=2,
                  hidden_layer_sizes=[3, 60],
                  activation='relu',
                  solver='lbfgs')

## Conclusion: Neural nets classifier expected accuracy: 0.81
# chosen parameters:
# alpha=5
# hidden_layer_sizes = [3,60]
# activation= 'relu'





print('- Training final model')
print('- Testing final model')

test = pd.read_csv('Data/titanic_test_READY2.csv')
test.set_index('PassengerId',inplace=True)


#clf = GradientBoostingClassifier(n_estimators=8, learning_rate=0.2)
#clf = RandomForestClassifier(n_estimators = 10,
#                             max_features = 6,
#                             max_depth = 3)

clf = MLPClassifier(activation='relu',
                    hidden_layer_sizes=[3, 60],
                    alpha=2,
                    solver='lbfgs')


norm_required = True

if norm_required:
    [X_train, scaler_dict] = prep_variables_norm(train_X, Train=True)
    X_test = prep_variables_norm(test, Train=False, scaler_dict=scaler_dict)
else:
    X_train = prep_variables_no_norm(train_X)
    X_test = prep_variables_no_norm(test)


clf.fit(X_train, train_Y)
y_predicted = clf.predict(X_test)
y_predicted_df = pd.DataFrame(y_predicted,columns={'Survived'}, index=X_test.index)

y_predicted_df.to_csv('Kaggle submissions/titanic_submission5-MLP2.csv', sep=',', encoding='utf-8')