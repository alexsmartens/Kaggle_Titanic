import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


train_original = pd.read_csv('titanic_train_READY.csv')
train_original.set_index('PassengerId',inplace=True)
train_X = train_original.copy()
train_X.drop('Survived', axis=1, inplace=True)
train_Y = train_original.copy()['Survived']

## Removing unnecessary features
# Cabin feature has been used for locating passengers on the ship
# Ticket feature has been used for groping passengers with the same tickets at the same place on the ship
# Name_title feature has been used for age detection
# Name, Name_last, Name_other features are personal identifiers


def prep_variables(X_train):
    train_X.drop(['Cabin',
                  'Ticket',
                  'Name_title',
                  'Name',
                  'Name_last',
                  'Name_other'], axis=1, inplace=True)

    # Continues variables standardization
    X_train.loc[:,'Age'] = preprocessing.scale( X_train.loc[:,'Age'].astype('float64') )
    X_train.loc[:,'Room center longitude(X)'] = preprocessing.scale( X_train.loc[:,'Room center longitude(X)'].astype('float64') )
    X_train.loc[:,'Room center latitude(Y)'] = preprocessing.scale( X_train.loc[:,'Room center latitude(Y)'].astype('float64') )
    X_train.loc[:,'Fare'] = preprocessing.scale( X_train.loc[:,'Fare'].astype('float64') )

    # Integer variables standardization
    X_train.loc[:,'Deck level'] = preprocessing.scale( X_train.loc[:,'Deck level'].astype('float64') )
    X_train.loc[:,'Parch'] = preprocessing.scale( X_train.loc[:,'Parch'].astype('float64') )
    X_train.loc[:,'SibSp'] = preprocessing.scale( X_train.loc[:,'SibSp'].astype('float64') )

    # Categorical variables transformation
    X_train.loc[:,'Sex'] = (X_train.loc[:,'Sex']=='female')*1
    X_train.rename(columns={'Sex':'Sex==female'}, inplace=True)

    X_train = pd.merge(X_train, pd.get_dummies(X_train.loc[:,'Embarked'],'Embarked'), left_index=True, right_index=True)
    X_train.drop('Embarked',axis=1, inplace=True)

    X_train = pd.merge(X_train, pd.get_dummies(X_train.loc[:,'Pclass'],'Pclass'), left_index=True, right_index=True)
    X_train.drop('Pclass', axis=1, inplace=True)
    return X_train

train_X = prep_variables(train_X)



def cv(train_X, train_Y, clf_model):
    scores_train=[]
    scores_test=[]
    clf_list=[]
    for i in range(0,5):
        X_train, X_test, y_train, y_test = train_test_split(train_X, train_Y, test_size=0.2)
        clf = clf_model.fit(X_train, y_train)
        scores_train.append( clf.score(X_train, y_train) )
        scores_test.append(clf.score(X_test, y_test))
        clf_list.append(clf)
    return {'Avg_train_score': np.mean(scores_train), 'Avg_test_score': np.mean(scores_test)}

def plot_c_scores (scores):
    plt.figure()
    train_series = plt.plot(scores.index,
                            scores.loc[:, 'Avg_train_score'],
                            c='black',
                            label='Train')
    test_series = plt.plot(scores.index,
                            scores.loc[:, 'Avg_test_score'],
                            c='green',
                            label='Test')
    plt.xlabel('Model parameter')
    plt.ylabel('Classifier score')
    plt.title('Selection of model parameters')
    plt.legend(title='Legend',
               loc=6)
    plt.show()





## 1) Logistic regression
def log_reg_check_c(c):
    scores = pd.DataFrame(columns={'Avg_train_score', 'Avg_test_score'})
    for i in c:
        cv_res = cv(train_X, train_Y, LogisticRegression(C=i))
        scores.loc[i] = [cv_res['Avg_train_score'], cv_res['Avg_test_score']]
    return scores

# rough selection of C regularization parameter
c_rough = [0.001, 0.01, 0.1, 0.5, 1, 2, 3, 5, 10, 20, 50, 100]

scores = log_reg_check_c(c_rough)
# print(scores)
# plot_c_scores(scores)

c_fine = np.arange(0.1, 2, 0.1)
scores = log_reg_check_c(c_fine)
#print(scores)
#plot_c_scores(scores)

## Conclusion: varince of logisticregression classifier performance with different C is large.
# Standard C=1 performance seems to be good, so C=1 is chosen for logistic regression
# Logistic regression expected accuracy: 0.80
print('Logistic regression expected accuracy: 0.80')





## 2) Naive Bayes

cv_res = cv(train_X, train_Y, GaussianNB())
#print(cv_res)
# Gaussian Naive Bayes classifier expected accuracy: 0.74
print('Gaussian Naive Bayes classifier expected accuracy: 0.74')





## 3) KNN
#print( cv(train_X, train_Y, KNeighborsClassifier()) )

def KNN_neighbor(train_X, train_Y,n_neighbors,algorithm):
    scores = pd.DataFrame(columns={'Avg_train_score', 'Avg_test_score'})
    for i in n_neighbors:
        cv_res = cv(train_X, train_Y, KNeighborsClassifier(n_neighbors=i,algorithm=algorithm))
        scores.loc[i] = [cv_res['Avg_train_score'], cv_res['Avg_test_score']]
    return scores

n_neighbors = np.arange(3,50,1)

algorithm = 'kd_tree'
#scores = KNN_neighbor(train_X, train_Y, n_neighbors,algorithm)
#print(scores)
#plot_c_scores(scores)


## Conclusion: KNN classifier expected accuracy: 0.77
#  chosen parameters:
#algorithm = 'kd_tree'
#n_neighbors = 7
print('KNN classifier expected accuracy: 0.77')







## 4) Gradient Boosted Decision Trees
# *this classifier does not require data normalization
train_X_no_norm = train_original.copy()
train_X_no_norm.drop('Survived', axis=1, inplace=True)

def prep_variables_no_norm(train_X_no_norm):
    train_X_no_norm.drop(['Cabin',
                  'Ticket',
                  'Name_title',
                  'Name',
                  'Name_last',
                  'Name_other'], axis=1, inplace=True)
    train_X_no_norm.loc[:,'Sex'] = (train_X_no_norm.loc[:,'Sex']=='female')*1
    train_X_no_norm.rename(columns={'Sex':'Sex==female'}, inplace=True)
    ## Embarked
    # Departure from
    # - Southampton, UK -> 10 April 1912
    # - Cherbourg, France -> 10 April 1912 (an hour and a half stop)
    # - Queenstown, Ireland (now - Cobh) -> 12 April 1912 (two hours stop)
    embarked_dict = {'S':1, 'C':2, 'Q':3}
    # assigning port of embarkation to 2 passnegers without Embarked value
    # S is assigned as the most probable embarkation port for the 1st class passengers
    train_X_no_norm.loc[:,'Embarked'] = (train_X_no_norm.loc[:,'Embarked']).apply(lambda x: embarked_dict[x])
    return train_X_no_norm

train_X_no_norm.loc[[62,830],'Embarked'] = 'S'

train_X_no_norm = prep_variables_no_norm(train_X_no_norm)

n_estimators_rough = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 300, 500]
n_estimators = np.arange(5, 30, 1)

def gbt_reg_n_estimators(train_X, train_Y, n_estimators):
    scores = pd.DataFrame(columns={'Avg_train_score', 'Avg_test_score'})
    for i in n_estimators:
        cv_res = cv(train_X, train_Y, GradientBoostingClassifier(n_estimators=i))
        scores.loc[i] = [cv_res['Avg_train_score'], cv_res['Avg_test_score']]
    return scores

#scores = gbt_reg_n_estimators(train_X_no_norm, train_Y, n_estimators)
#print(scores)
#plot_c_scores(scores)
# n_estimators = 15 seems to be a reasonably good classifier
# [5,25] is the recommended range for further testing
#n_estimators = 15


learn_rate_rough = [0.001, 0.01, 0.1, 0.5, 1, 2, 3, 5, 10, 20, 50, 100]
learn_rate = np.arange(0.02,0.2,0.01)

def gbt_reg_learn_rate(train_X, train_Y, n_estimators, learn_rate):
    scores = pd.DataFrame(columns={'Avg_train_score', 'Avg_test_score'})
    for i in learn_rate:
        cv_res = cv(train_X, train_Y, GradientBoostingClassifier(n_estimators=n_estimators,learning_rate=i))
        scores.loc[i] = [cv_res['Avg_train_score'], cv_res['Avg_test_score']]
    return scores

#scores = gbt_reg_learn_rate(train_X_no_norm, train_Y, n_estimators, learn_rate)
#print(scores)
#plot_c_scores(scores)
# learning_rate = [0.02, 0.2] seems to be a reasonably good for further investigation

def gbt_reg_learn_n_estimators(train_X, train_Y, n_estimators, learn_rate):
    scores = pd.DataFrame(columns={'n_estimators', 'learn_rate', 'Avg_train_score', 'Avg_test_score'})
    num = 0
    for n in n_estimators:
        for learn in learn_rate:
            cv_res = cv(train_X, train_Y, GradientBoostingClassifier(n_estimators=n,learning_rate=learn))
            num += 1
            scores.loc[num] = pd.Series([n, learn, cv_res['Avg_train_score'], cv_res['Avg_test_score']],
                                        index=['n_estimators', 'learn_rate', 'Avg_train_score', 'Avg_test_score'])
    return scores
learn_rate = np.arange(0.09,0.2,0.01)
n_estimators = np.arange(8,15,1)
#scores = gbt_reg_learn_n_estimators(train_X_no_norm, train_Y, n_estimators, learn_rate)
#print(scores)
#plot_c_scores(scores)
#print(cv(train_X_no_norm, train_Y, GradientBoostingClassifier(n_estimators=8, learning_rate=0.2)))
## Conclusion: chosen parameters: learning rate 0.2, n_estimators = 8
# Gradient Boosted Decision Trees classifier expected accuracy: 0.82
print('Gradient Boosted Decision Trees classifier expected accuracy: 0.82')





## 5) Random forest
# *this classifier does not require data normalization

n_estimators = 10
max_features = int(round(np.sqrt(len(train_X_no_norm.columns))))
max_depth = 3
#print(cv(train_X_no_norm, train_Y, RandomForestClassifier(n_estimators=n_estimators,
#                                                          max_features=max_features,
#                                                          max_depth=max_depth)))
def rf_n_estimator(train_X, train_Y, n_estimators, max_features, max_depth):
    scores = pd.DataFrame(columns={'Avg_train_score', 'Avg_test_score'})
    for i in n_estimators:
        cv_res = cv(train_X, train_Y, RandomForestClassifier(n_estimators=i,
                                                                 max_features=max_features,
                                                                 max_depth=max_depth))
        scores.loc[i] = [cv_res['Avg_train_score'], cv_res['Avg_test_score']]
    return scores

n_estimators = np.arange(8,25,1)
#scores = rf_n_estimator(train_X_no_norm, train_Y, n_estimators, max_features, max_depth)
#print(scores)
#plot_c_scores(scores)
# n_estimators = 12 seems to produce reasonably good results
n_estimators = 12

max_features = [2,3,4,5,6,7,8,9]
def rf_features(train_X, train_Y, n_estimators, max_features, max_depth):
    scores = pd.DataFrame(columns={'Avg_train_score', 'Avg_test_score'})
    for i in max_features:
        cv_res = cv(train_X, train_Y, RandomForestClassifier(n_estimators=n_estimators,
                                                                 max_features=i,
                                                                 max_depth=max_depth))
        scores.loc[i] = [cv_res['Avg_train_score'], cv_res['Avg_test_score']]
    return scores
#scores = rf_features(train_X_no_norm, train_Y, n_estimators, max_features, max_depth)
#print(scores)
#plot_c_scores(scores)
# max_features = 5 seems to produce reasonably good results
max_features = 5


max_features = np.arange(3,7,1)
n_estimators = np.arange(8,16,1)
def rf_n_estimator_features(train_X, train_Y, n_estimators, max_features, max_depth):
    scores = pd.DataFrame(columns={'n_estimators', 'max_features', 'Avg_train_score', 'Avg_test_score'})
    num = 0
    for n in n_estimators:
        for i in max_features:
            cv_res = cv(train_X, train_Y, RandomForestClassifier(n_estimators=n,
                                                                 max_features=i,
                                                                 max_depth=max_depth))
            num += 1
            scores.loc[num] = pd.Series([n, i, cv_res['Avg_train_score'], cv_res['Avg_test_score']],
                                        index=['n_estimators', 'max_features', 'Avg_train_score', 'Avg_test_score'])
    return scores
#scores = rf_n_estimator_features(train_X_no_norm, train_Y, n_estimators, max_features, max_depth)
#print(scores)
#plot_c_scores(scores)

## Conclusion: chosen parameters: learning rate 0.2, n_estimators = 8
#n_estimators = 10
#max_features = 6
#max_depth = 3
#Random forest classifier expected accuracy: 0.81

#print(cv(train_X_no_norm, train_Y, RandomForestClassifier(n_estimators=n_estimators,
#                                                          max_features=max_features,
#                                                          max_depth=max_depth)))
print('Random forest classifier expected accuracy: 0.81')





## 6) Neural networks
activation= 'relu'
hidden_layer_sizes = [5,100]
alpha = 2
solver = 'lbfgs'

#print(cv(train_X, train_Y, MLPClassifier(activation=activation,
#                                         hidden_layer_sizes=hidden_layer_sizes,
#                                         alpha=alpha,
#                                         solver=solver)))

def nn_alpha(train_X_no_norm, train_Y, activation, hidden_layer_sizes, alpha, solver):
    scores = pd.DataFrame(columns={'Avg_train_score', 'Avg_test_score'})
    for i in alpha:
        cv_res = cv(train_X, train_Y, MLPClassifier(activation=activation,
                                         hidden_layer_sizes=hidden_layer_sizes,
                                         alpha=i,
                                         solver=solver))
        scores.loc[i] = [cv_res['Avg_train_score'], cv_res['Avg_test_score']]
    return scores

#alpha_raugh = [0.001, 0.01, 0.1, 0.5, 1, 2, 3, 5, 10, 20, 50, 100]
alpha =np.arange(2,8.5,0.25)
#scores = nn_alpha(train_X, train_Y, activation, hidden_layer_sizes, alpha, solver)
#print(scores)
#plot_c_scores(scores)
# alpha=5 seems to be optimal
alpha=5

def nn_layers(train_X_no_norm, train_Y, activation, hidden_layer_sizes, alpha, solver):
    scores = pd.DataFrame(columns={'layers_num', 'layers_size','Avg_train_score', 'Avg_test_score'})
    num=0
    for i in np.arange(2,5,1):
        for ii in np.arange(49,70,1):
            cv_res = cv(train_X, train_Y, MLPClassifier(activation=activation,
                                             hidden_layer_sizes=[i,ii],
                                             alpha=alpha,
                                             solver=solver))
            num += 1
            scores.loc[num] = pd.Series([i, ii, cv_res['Avg_train_score'], cv_res['Avg_test_score']],
                                        index=['layers_num', 'layers_size', 'Avg_train_score', 'Avg_test_score'])
    return scores
#scores = nn_layers(train_X, train_Y, activation, hidden_layer_sizes, alpha, solver)
#print(scores)
#plot_c_scores(scores)
hidden_layer_sizes = [3,60]

#print(cv(train_X, train_Y, MLPClassifier(activation='relu',
#                                         hidden_layer_sizes=hidden_layer_sizes,
#                                         alpha=alpha,
#                                         solver=solver)))
#
#print(cv(train_X, train_Y, MLPClassifier(activation='logistic',
#                                         hidden_layer_sizes=hidden_layer_sizes,
#                                         alpha=alpha,
#                                         solver=solver)))
#
#print(cv(train_X, train_Y, MLPClassifier(activation='tanh',
#                                         hidden_layer_sizes=hidden_layer_sizes,
#                                         alpha=alpha,
#                                         solver=solver)))


## Conclusion: Neural nets classifier expected accuracy: 0.81
# chosen parameters:
# alpha=5
# hidden_layer_sizes = [3,60]
# activation= 'relu'

print('Conclusion: Neural nets classifier expected accuracy: 0.81')

print('- Training final model')
clf = GradientBoostingClassifier(n_estimators=8, learning_rate=0.2)
clf.fit(train_X_no_norm, train_Y)


print('- Testing final model')
test = pd.read_csv('titanic_test_READY2.csv')
test.set_index('PassengerId',inplace=True)

test_X_no_norm = prep_variables_no_norm(test)
y_predicted = clf.predict(test_X_no_norm)
y_predicted_df = pd.DataFrame(y_predicted,columns={'Survived'}, index=test_X_no_norm.index)

y_predicted_df.to_csv('titanic_submission1.csv', sep=',', encoding='utf-8')