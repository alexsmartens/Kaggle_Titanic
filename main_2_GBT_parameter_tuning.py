import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE

import os
import sys
lib_path2 = os.path.abspath(os.path.join('Ml models'))
sys.path.append(lib_path2)
from features_processing import features_processing
lib_path = os.path.abspath(os.path.join('GBT_tuning_functions'))
sys.path.append(lib_path)
from modelfit import modelfit





if __name__ == '__main__':
    train_original = pd.read_csv('Data/titanic_train_READY.csv')
    train_original.set_index('PassengerId', inplace=True)

    train = train_original.copy()
    target = 'Survived'
    [scaler, train] = features_processing(train, target, normalization=False)
    # Choose all predictors except target
    predictors = [x for x in train.columns if x not in [target]]

    # Oversampling with SMOTE
    X_res, y_res = SMOTE(kind='regular').fit_sample(train[predictors], train[target])

    train_res = pd.DataFrame(X_res, columns=train[predictors].columns)
    train_res.loc[:,'Survived'] = y_res
    train = train_res



    gbm0 = GradientBoostingClassifier(random_state=10)
    #modelfit(gbm0, train, predictors)


    # Choose all predictors except target & IDcols
    predictors = [x for x in train.columns if x not in [target]]
    param_test1 = {'n_estimators':range(20,81,5)}

    gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1,
                                                                   min_samples_split=7,
                                                                   min_samples_leaf=4,
                                                                   max_depth=5,
                                                                   max_features='sqrt',
                                                                   subsample=0.8,
                                                                   random_state=10),
                            param_grid = param_test1,
                            scoring='accuracy',n_jobs=-1,iid=False, cv=5)
    # gsearch1.fit(train[predictors],train[target])
    # clf = gsearch1



    param_test2 = {'max_depth': np.arange(4,21,1),
                   'min_samples_split': np.arange(100,200,2)}

    gsearch2 = GridSearchCV(estimator=GradientBoostingClassifier(learning_rate=0.1,
                                                                 min_samples_leaf=4,
                                                                 max_features='sqrt',
                                                                 subsample=0.8,
                                                                 random_state=10,
                                                                 n_estimators=40),
                            param_grid=param_test2,
                            scoring='accuracy', n_jobs=-1, iid=False, cv=5)
    # gsearch2.fit(train[predictors], train[target])
    # clf = gsearch2



    param_test3 =  {'max_features':range(2,14,1)}

    gsearch3 = GridSearchCV(estimator=GradientBoostingClassifier(learning_rate=0.1,
                                                                 min_samples_leaf=4,
                                                                 subsample=0.8,
                                                                 random_state=10,
                                                                 n_estimators=40,
                                                                 max_depth=10,
                                                                 min_samples_split=124),
                            param_grid=param_test3,
                            scoring='accuracy', n_jobs=-1, iid=False, cv=5)
    # gsearch3.fit(train[predictors], train[target])
    # clf = gsearch3



    param_test4 = {'subsample': [0.6,0.7,0.75,0.8,0.85,0.9,0.95,0.99]}

    gsearch4 = GridSearchCV(estimator=GradientBoostingClassifier(learning_rate=0.1,
                                                                 min_samples_leaf=4,
                                                                 random_state=10,
                                                                 n_estimators=40,
                                                                 max_depth=10,
                                                                 min_samples_split=124,
                                                                 max_features=9),
                            param_grid=param_test4,
                            scoring='accuracy', n_jobs=-1, iid=False, cv=5)
    # gsearch4.fit(train[predictors], train[target])
    # clf = gsearch4



    param_test5 = {'learning_rate': np.arange(0.01,0.5,0.01)}
    gsearch5 = GridSearchCV(estimator=GradientBoostingClassifier(learning_rate=0.17,
                                                                 min_samples_leaf=4,
                                                                 random_state=10,
                                                                 n_estimators=45,
                                                                 max_depth=7,
                                                                 min_samples_split=126,
                                                                 max_features=3,
                                                                 subsample=0.8),
                            param_grid=param_test5,
                            scoring='accuracy', n_jobs=-1, iid=False, cv=5)
    # gsearch5.fit(train[predictors], train[target])
    # clf = gsearch5

    # gsearch_chosen = GradientBoostingClassifier(learning_rate=0.1,
    #                                                              min_samples_leaf=4,
    #                                                              random_state=10,
    #                                                              n_estimators=40,
    #                                                              max_depth=10,
    #                                                              min_samples_split=124,
    #                                                              max_features=9,
    #                                                              subsample=0.8)

    # gsearch_chosen.fit(train[predictors], train[target])
    # modelfit(gsearch_chosen, train, predictors)


    # The first successful parameters
    param_test6 = {'learning_rate': np.arange(0.01,0.5,0.01),
                   'n_estimators': np.arange(2,31,1)
                   }
    gsearch6 = GridSearchCV(estimator=GradientBoostingClassifier(
                                                                 random_state=10,
                                                                 ),
                            param_grid=param_test6,
                            scoring='accuracy', n_jobs=-1, iid=False, cv=5)
    # gsearch6.fit(train[predictors], train[target])
    # clf = gsearch6

    gsearch_chosen = GradientBoostingClassifier(random_state=10,
                                                learning_rate=0.34,
                                                n_estimators=5)

    # gsearch_chosen.fit(train[predictors], train[target])
    # modelfit(gsearch_chosen, train, predictors)




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
        [scaler, test_X] = features_processing(test, target, normalization=False, training=False, scaler=scaler)

        y_predicted = gsearch_chosen.predict(test_X)
        y_predicted_df = pd.DataFrame(y_predicted, columns={'Survived'}, index=test_X.index)
        # y_predicted_df.sort_index(inplace=True)

        y_predicted_df.to_csv('Kaggle submissions/titanic_submission2_accuracy_SMOTE_GBT3_1st_improved.csv', sep=',', encoding='utf-8')