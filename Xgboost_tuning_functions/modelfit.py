def modelfit(alg, dtrain, predictors, target, xgb, useTrainCV=True, cv_folds=5, early_stopping_rounds=50,printFeatureImportance=True):
    import numpy as np
    import pandas as pd

    from sklearn.model_selection import cross_val_score
    from sklearn import metrics
    import matplotlib.pylab as plt

    from matplotlib.pylab import rcParams
    rcParams['figure.figsize'] = 12, 4

    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='auc', early_stopping_rounds=early_stopping_rounds) #, show_progress=False)
        alg.set_params(n_estimators=cvresult.shape[0])

    # Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target], eval_metric='auc')

    # Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:, 1]

    # Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(dtrain[target].values, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain[target], dtrain_predprob))

    if useTrainCV:
        print("Average CV Score (Train) : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cvresult['train-auc-mean']),
                                                                                                 np.mean(cvresult['train-auc-std']),
                                                                                                 np.min(cvresult['train-auc-mean']),
                                                                                                 np.max(cvresult['train-auc-mean'])))
        print("Average CV Score (Train) : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cvresult['test-auc-mean']),
                                                                                                 np.mean(cvresult['test-auc-std']),
                                                                                                 np.min(cvresult['test-auc-mean']),
                                                                                                 np.max(cvresult['test-auc-mean'])))

    if printFeatureImportance:
        # Relative feature importance
        feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False) / \
                   (pd.Series(alg.get_booster().get_fscore()).sum())
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')
        plt.xticks(rotation=30)
        plt.show()