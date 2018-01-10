import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE

import os
import sys

# add the path to the g++ runtime libraries to the os environment path variable
mingw_path = 'C:\\Program Files\\Compiler-mingw-w64\\x86_64-7.2.0-posix-seh-rt_v5-rev1\\mingw64\\bin'
os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']
import xgboost as xgb
from xgboost.sklearn import XGBClassifier


lib_path = os.path.abspath(os.path.join('Xgboost_tuning_functions'))
sys.path.append(lib_path)
lib_path2 = os.path.abspath(os.path.join('Ml models'))
sys.path.append(lib_path2)
from features_processing import features_processing

from modelfit import modelfit



if __name__ == '__main__':

    train_original = pd.read_csv('Data/titanic_train_READY.csv')
    train_original.set_index('PassengerId', inplace=True)



    train_original = pd.read_csv('Data/titanic_train_READY.csv')
    train_original.set_index('PassengerId', inplace=True)

    train = train_original.copy()
    target = 'Survived'
    [scaler, train] = features_processing(train, target, normalization=False)
    # Choose all predictors except target
    predictors = [x for x in train.columns if x not in [target]]

    # Oversampling with SMOTE
    # X_res, y_res = SMOTE(kind='regular').fit_sample(train[predictors], train[target])
    #
    # train_res = pd.DataFrame(X_res, columns=train[predictors].columns)
    # train_res.loc[:,'Survived'] = y_res
    # train = train_res
    #
    # # Oversampling with SMOTE
    # X_res, y_res = SMOTE(kind='regular').fit_sample(train[predictors], train[target])
    #
    # train_res = pd.DataFrame(X_res, columns=train[predictors].columns)
    # train_res.loc[:, 'Survived'] = y_res
    # train = train_res

    # Undersampling
    # rus = RandomUnderSampler(random_state=0)
    # X_res, y_res = rus.fit_sample(train[predictors], train[target])
    #
    # train_res = pd.DataFrame(X_res, columns=train[predictors].columns)
    # train_res.loc[:,'Survived'] = y_res
    # train = train_res





    #Choose all predictors except target
    predictors = [x for x in train.columns if x not in [target]]
    xgb1 = XGBClassifier(
     learning_rate =0.1,
     n_estimators=140,
     max_depth=5,
     min_child_weight=1,
     gamma=0,
     subsample=0.8,
     colsample_bytree=0.8,
     objective= 'binary:logistic',
     nthread=4,
     scale_pos_weight=1,
     seed=27)
    # modelfit(xgb1, train, predictors, target, xgb)


    param_test1 = {
     'max_depth':range(3,10,2),
     'min_child_weight':range(1,6,2)
    }
    gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate=0.09,
                                                       n_estimators=6,
                                                       gamma=0.01,
                                                       subsample=0.7,
                                                       colsample_bytree=0.9,
                                                       objective= 'binary:logistic',
                                                       nthread=12,
                                                       scale_pos_weight=1,
                                                       seed=27),
     param_grid = param_test1, scoring='accuracy',n_jobs=12,iid=False, cv=5)
    # gsearch1.fit(train[predictors],train[target])
    # clf = gsearch1



    param_test2 = {
        'max_depth': range(2,13,1),
        'min_child_weight': range(1,17,1)
    }
    gsearch2 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.09,
                                                       n_estimators=6,
                                                       gamma=0.01,
                                                       subsample=0.7,
                                                       colsample_bytree=0.9,
                                                       objective= 'binary:logistic',
                                                       nthread=12,
                                                       scale_pos_weight=1,
                                                       seed=27),
                            param_grid=param_test2, scoring='accuracy', n_jobs=12, iid=False, cv=5)
    # gsearch2.fit(train[predictors], train[target])
    # clf = gsearch2



    param_test3 = {
        'gamma': np.arange(0.0,0.5,0.01)
    }
    gsearch3 = GridSearchCV(estimator=XGBClassifier(learning_rate =0.1,
                                                       n_estimators=140,
                                                       subsample=0.8,
                                                       colsample_bytree=0.8,
                                                       objective= 'binary:logistic',
                                                       nthread=12,
                                                       scale_pos_weight=1,
                                                       seed=27,
                                                       max_depth=5,
                                                       min_child_weight=3),
                            param_grid=param_test3, scoring='accuracy', n_jobs=12, iid=False, cv=5)
    # gsearch3.fit(train[predictors], train[target])
    # clf = gsearch3



    xgb2 = XGBClassifier(
        learning_rate=0.1,
        n_estimators=140,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        nthread=12,
        scale_pos_weight=1,
        seed=27,
        max_depth=2,
        min_child_weight=5,
        gamma=0.9)
    # modelfit(xgb2, train, predictors, target, xgb)



    param_test4 = {
        'subsample': np.arange(0.6,1,0.05),
        'colsample_bytree': np.arange(0.6,1,0.05)
    }
    gsearch4 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1,
                                                    n_estimators=140,
                                                    objective='binary:logistic',
                                                    nthread=12,
                                                    scale_pos_weight=1,
                                                    seed=27,
                                                    max_depth=5,
                                                    min_child_weight=3,
                                                    gamma=0.38),
                            param_grid=param_test4, scoring='accuracy', n_jobs=12, iid=False, cv=5)
    # gsearch4.fit(train[predictors], train[target])
    # clf = gsearch4



    param_test6 = {
        'reg_alpha':[0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.01, 0.05]
    }
    gsearch6 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1,
                                                    n_estimators=140,
                                                    objective='binary:logistic',
                                                    nthread=12,
                                                    scale_pos_weight=1,
                                                    seed=27,
                                                    max_depth=5,
                                                    min_child_weight=3,
                                                    gamma=0.38,
                                                    subsample=0.65,
                                                    colsample_bytree=0.8),
                            param_grid=param_test6, scoring='accuracy', n_jobs=12, iid=False, cv=5)
    # gsearch6.fit(train[predictors], train[target])
    # clf = gsearch6



    xgb3 = XGBClassifier(
        learning_rate=0.1,
        n_estimators=140,
        objective='binary:logistic',
        nthread=12,
        scale_pos_weight=1,
        seed=27,
        max_depth=2,
        min_child_weight=5,
        gamma=0.9,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.001)
    # modelfit(xgb3, train, predictors, target, xgb)



    param_test7 = {
        'n_estimators':np.arange(5,40,1),
        'learning_rate': np.arange(0.01,0.31,0.01)
    }
    gsearch7 = GridSearchCV(estimator=XGBClassifier(objective='binary:logistic',
                                                    nthread=12,
                                                    scale_pos_weight=1,
                                                    seed=27,
                                                    max_depth=5,
                                                    min_child_weight=3,
                                                    gamma=0.38,
                                                    subsample=0.65,
                                                    colsample_bytree=0.8,
                                                    reg_alpha=0.003),
                            param_grid=param_test7, scoring='accuracy', n_jobs=12, iid=False, cv=5)
    # gsearch7.fit(train[predictors], train[target])
    # clf = gsearch7




    xgb4 = XGBClassifier(
        learning_rate=0.04,
        n_estimators=7,
        objective='binary:logistic',
        nthread=12,
        scale_pos_weight=1,
        seed=27,
        max_depth=5,
        min_child_weight=3,
        gamma=0.38,
        subsample=0.65,
        colsample_bytree=0.8,
        reg_alpha=0.003)
    modelfit(xgb4, train, predictors, target, xgb)




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
    submitting=True
    if submitting:
        # Lading test data
        test = pd.read_csv('Data/titanic_test_READY.csv')
        test.set_index('PassengerId', inplace=True)
        [scaler, test_X] = features_processing(test, target, normalization=False, training=False, scaler=scaler)

        y_predicted = xgb4.predict(test_X)
        y_predicted_df = pd.DataFrame(y_predicted, columns={'Survived'}, index=test_X.index)

        y_predicted_df.to_csv('Kaggle submissions/titanic_submission2_xgb_new_1.csv', sep=',', encoding='utf-8')