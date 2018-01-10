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
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score

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

    # Oversampling with SMOTE
    # X_res, y_res = SMOTE(kind='regular').fit_sample(train[predictors], train[target])
    #
    # train_res = pd.DataFrame(X_res, columns=train[predictors].columns)
    # train_res.loc[:,'Survived'] = y_res
    # train = train_res


    # SMOTE
    clf1 = GradientBoostingClassifier(learning_rate=0.1,
                                      min_samples_leaf=4,
                                      random_state=10,
                                      n_estimators=50,
                                      max_depth=12,
                                      min_samples_split=146,
                                      max_features=5,
                                      subsample=0.7)
    # NO SMOTE (not tested with SMOTE)
    clf2 =  MLPClassifier(alpha=1.7,
                          random_state=10,
                          activation='relu',
                          solver='lbfgs',
                          hidden_layer_sizes=[2, 11],
                          max_iter=500)
    # NO SMOTE (better without SMOTE, real score:0.76 vs 0.73)
    clf3 = LogisticRegression(random_state=10,
                              max_iter=500,
                              solver='saga',
                              C=0.5,
                              penalty='l1'
                              )
    eclf = VotingClassifier(estimators=[('GBT', clf1), ('MLP', clf2), ('LogR', clf3)], voting='soft', weights=[1, 1, 1])
    # eclf = VotingClassifier(estimators=[('GBT', clf1), ('MLP', clf2), ('LogR', clf3)], voting='hard')

    for clf, label in zip([clf1, clf2, clf3, eclf],
                          ['Gradient boosting', 'Neural Network', 'Logistic Regression', 'Ensemble']):
        scores = cross_val_score(clf,train[predictors], train[target], cv=5, scoring='accuracy')
        print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

    param_test1 = {'lr__C': [1.0, 100.0],
                   'rf__n_estimators': [20, 200], }
    gsearch1 = GridSearchCV(estimator = eclf,
                            param_grid = param_test1,
                            scoring='roc_auc',
                            n_jobs=-1,
                            iid=False,
                            cv=5)
    # gsearch1.fit(train[predictors], train[target])
    # clf = gsearch1

    #
    # gsearch_chosen = eclf
    # print(cross_val_score(eclf, train[predictors], train[target], cv=3))
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

        y_predicted_df.to_csv('Kaggle submissions/titanic_submission2-LogR-2_not sorted.csv', sep=',', encoding='utf-8')



