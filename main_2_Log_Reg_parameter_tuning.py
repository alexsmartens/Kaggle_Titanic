import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

lib_path = os.path.abspath(os.path.join('Ml models'))
sys.path.append(lib_path)
from features_processing import features_processing
from grid_search_function import grid_search_function



if __name__ == '__main__':

    train_original = pd.read_csv('Data/titanic_train_READY.csv')
    train_original.set_index('PassengerId',inplace=True)

    train = train_original.copy()
    target = 'Survived'
    [scaler, train] = features_processing(train, target, normalization=True)
    # Choose all predictors except target
    predictors = [x for x in train.columns if x not in [target]]

    # # Oversampling with SMOTE
    # X_res, y_res = SMOTE(kind='regular').fit_sample(train[predictors], train[target])
    #
    # train_res = pd.DataFrame(X_res, columns=train[predictors].columns)
    # train_res.loc[:,'Survived'] = y_res
    # train = train_res

    # Random undersampling
    rus = RandomUnderSampler(random_state=10)
    X_res, y_res = rus.fit_sample(train[predictors], train[target])

    train_res = pd.DataFrame(X_res, columns=train[predictors].columns)
    train_res.loc[:,'Survived'] = y_res
    train = train_res



    param_test1 = {#'C':[0.5, 1, 2, 3, 5, 10, 20, 30, 35, 40, 50, 60],
                   'C': np.arange(0.1,2.1,0.05),
                   'penalty': ['l1', 'l2']
                   }
    gsearch1 = GridSearchCV(estimator = LogisticRegression(
                                                      random_state=10,
                                                      max_iter=1000,
                                                      solver='saga'
                                                    ),
                            param_grid = param_test1,
                            scoring='accuracy',n_jobs=-1,iid=False, cv=5)
    # gsearch1.fit(train[predictors], train[target])
    # clf = gsearch1


    gsearch_chosen = LogisticRegression(
                                   random_state=10,
                                   max_iter=1000,
                                   solver='saga',
                                   C=1.1,
                                   penalty='l1'
                                   )
    # gsearch_chosen.fit(train[predictors], train[target])

    # # Print results
    # print("Best parameters set found on development set:")
    # print(clf.best_params_)
    #
    # print("Grid scores on development set:")
    # means = clf.cv_results_['mean_test_score']
    # stds = clf.cv_results_['std_test_score']
    #
    # for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    #     print("%0.3f (+/-%0.03f) for %r"
    #           % (mean, std * 2, params))





    # Predicting result for submission
    submitting = False
    if submitting:

        # Lading test data
        test = pd.read_csv('Data/titanic_test_READY.csv')
        test.set_index('PassengerId',inplace=True)
        [scaler, test_X] = features_processing(test, target, normalization=True, training=False, scaler=scaler)

        y_predicted = gsearch_chosen.predict(test_X)
        y_predicted_df = pd.DataFrame(y_predicted, columns={'Survived'}, index=test_X.index)
        # y_predicted_df.sort_index(inplace=True)

        y_predicted_df.to_csv('Kaggle submissions/titanic_submission2_accuracy_undersampled_LogR2.csv', sep=',', encoding='utf-8')



