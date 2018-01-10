import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
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
    X_res, y_res = SMOTE(kind='regular').fit_sample(train[predictors], train[target])

    train_res = pd.DataFrame(X_res, columns=train[predictors].columns)
    train_res.loc[:,'Survived'] = y_res
    train = train_res




    # use a full grid over all parameters
    param_grid1 = {"learning_rate": [0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 5, 7.5, 10],
                  'n_estimators':range(20,81,10),
                  "max_depth": [3, 4, 5],
                  "min_samples_leaf": [1, 3, 10]}

    param_grid2 = {'n_estimators': [0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 5, 7.5, 10],
                   'max_depth': np.arange(6, 17, 2),
                   'min_samples_split': np.arange(100, 200, 10),
                   'max_features': range(2, 14, 2),
                   'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9]}
    clf2 = GradientBoostingClassifier(random_state=10,learning_rate=0.1,min_samples_leaf=4)

    #grid_search_function(train[predictors], train[target], clf, param_grid2, cv=5)
    #grid_search_function(train[predictors], train[target], clf2, param_grid2, cv=5)





    # train.to_csv('Data/titanic_train_test_temp__.csv', sep=',', encoding='utf-8')
    def form_layer_unit_combinations(layers_list, units_list):
        combination_l_u_list = []
        for l in layers_list:
            for u in units_list:
                combination_l_u_list = combination_l_u_list + [[l, u]]
        return combination_l_u_list


    param_test1 = {'hidden_layer_sizes': form_layer_unit_combinations( range(2,11,1), range(10, 101, 10) )}
    gsearch1 = GridSearchCV(estimator = MLPClassifier(alpha=2,
                                                      random_state=10,
                                                      activation='relu',
                                                      solver='lbfgs',
                                                      max_iter=500
                                                      ),
                            param_grid = param_test1,
                            scoring='accuracy',n_jobs=3,iid=False, cv=5)
    # gsearch1.fit(train[predictors], train[target])
    # clf = gsearch1



    param_test2 = {'activation': ['identity','logistic','tanh','relu']}
    gsearch2 = GridSearchCV(estimator = MLPClassifier(alpha=2,
                                                      random_state=10,
                                                      activation='relu',
                                                      solver='lbfgs',
                                                      hidden_layer_sizes=[2,11]#[2,20], [3,44]
                                                      ),
                            param_grid = param_test2,
                            scoring='accuracy',n_jobs=4,iid=False, cv=5)
    # gsearch2.fit(train[predictors], train[target])
    # clf = gsearch2


    # param_test3 = {'alpha': np.arange(0.1, 3, 0.4),
    #     'hidden_layer_sizes': form_layer_unit_combinations( range(2,10,1), range(10, 101, 10) ),
    #     'activation': ['identity','logistic','tanh','relu']}
    # gsearch3 = GridSearchCV(estimator = MLPClassifier(alpha=2,
    #                                                   random_state=10,
    #                                                   solver='lbfgs',
    #                                                   ),
    #                         param_grid = param_test3,
    #                         scoring='roc_auc',n_jobs=-1,iid=False, cv=5)
    param_test3 = {'alpha': np.arange(0.1, 16, 0.5),
        'hidden_layer_sizes': form_layer_unit_combinations( range(2,11,1), range(10, 161, 10) ),
        'activation': ['identity','logistic','tanh','relu']}
    gsearch3 = GridSearchCV(estimator = MLPClassifier(
                                                      random_state=10,
                                                      solver='lbfgs'
                                                      ),
                            param_grid = param_test3,
                            scoring='accuracy',n_jobs=-1,iid=False, cv=5)
    # gsearch3.fit(train[predictors], train[target])
    # clf = gsearch3

    # 0.853(+ / -0.104)
    # for {'activation': 'tanh', 'hidden_layer_sizes': [2, 13]}
    # 0.851(+ / -0.104)
    # for {'activation': 'relu', 'hidden_layer_sizes': [2, 11]}
    # 0.852(+ / -0.078)
    # for {'activation': 'logistic', 'hidden_layer_sizes': [3, 23]}
    # 0.852(+ / -0.077)
    # for {'activation': 'logistic', 'hidden_layer_sizes': [3, 36]}



    param_test4_1 = {'alpha': np.arange(0.1, 6.7, 0.25),
        'hidden_layer_sizes': form_layer_unit_combinations( [2], range(10, 170, 5) ),
        'activation': ['relu']}
    gsearch4_1 = GridSearchCV(estimator = MLPClassifier(
                                                      random_state=10,
                                                      solver='lbfgs'
                                                      ),
                            param_grid = param_test4_1,
                            scoring='accuracy',n_jobs=-1,iid=False, cv=5)
    # gsearch4_1.fit(train[predictors], train[target])
    # clf = gsearch4_1


    param_test4_1_2 = {'alpha': np.arange(2, 5.5, 0.1),
        'hidden_layer_sizes': form_layer_unit_combinations( [2], [60,65,75,80,85,90,110,115,120,125,135,140,145,150,155,160]),
        'activation': ['relu']}
    gsearch4_1_2 = GridSearchCV(estimator = MLPClassifier(
                                                      random_state=10,
                                                      solver='lbfgs'
                                                      ),
                            param_grid = param_test4_1_2,
                            scoring='accuracy',n_jobs=-1,iid=False, cv=5)
    # gsearch4_1_2.fit(train[predictors], train[target])
    # clf = gsearch4_1_2

    param_test4_2 = {'alpha': np.arange(0.5, 13, 0.25),
        'hidden_layer_sizes': form_layer_unit_combinations( range(3, 10, 1), range(20, 170, 5) ),
        'activation': ['relu', 'tanh']}
    gsearch4_2 = GridSearchCV(estimator = MLPClassifier(
                                                      random_state=10,
                                                      solver='lbfgs'
                                                      ),
                            param_grid = param_test4_2,
                            scoring='accuracy',n_jobs=-1,iid=False, cv=5)
    # gsearch4_2.fit(train[predictors], train[target])
    # clf = gsearch4_2

    param_test4_2_2 = {'alpha': np.arange(0.5, 8.6, 0.21),
        'hidden_layer_sizes': form_layer_unit_combinations( range(4, 10, 1), range(20, 171, 5) ),
        'activation': ['relu', 'tanh']}
    gsearch4_2_2 = GridSearchCV(estimator = MLPClassifier(
                                                      random_state=10,
                                                      solver='lbfgs'
                                                      ),
                            param_grid = param_test4_2_2,
                            scoring='accuracy',n_jobs=-1,iid=False, cv=5)
    # gsearch4_2_2.fit(train[predictors], train[target])
    # clf = gsearch4_2_2

    param_test4_3 = {'alpha': np.arange(2, 6, 0.1),
                     'hidden_layer_sizes': form_layer_unit_combinations([10], range(50, 90, 10)),
                     'activation': ['tanh']}
    gsearch4_3 = GridSearchCV(estimator=MLPClassifier(
        random_state=10,
        solver='lbfgs'
    ),
        param_grid=param_test4_3,
        scoring='accuracy', n_jobs=-1, iid=False, cv=5)
    # gsearch4_3.fit(train[predictors], train[target])
    # clf = gsearch4_3

    param_test4_3_2 = {'alpha': np.arange(2.5, 7, 0.1),
                     'hidden_layer_sizes': form_layer_unit_combinations([10], range(30, 115, 5)),
                     'activation': ['tanh']}
    gsearch4_3_2 = GridSearchCV(estimator=MLPClassifier(
        random_state=10,
        solver='lbfgs'
    ),
        param_grid=param_test4_3_2,
        scoring='accuracy', n_jobs=-1, iid=False, cv=5)
    # gsearch4_3_2.fit(train[predictors], train[target])
    # clf = gsearch4_3_2




    gsearch_chosen = MLPClassifier(alpha=6.0,
                                   random_state=10,
                                   activation='tanh',
                                   solver='lbfgs',
                                   hidden_layer_sizes=[10, 55])
    gsearch_chosen.fit(train[predictors], train[target])

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

        y_predicted_df.to_csv('Kaggle submissions/titanic_submission2-SMOTE_MLP_new-4_3.csv', sep=',', encoding='utf-8')



