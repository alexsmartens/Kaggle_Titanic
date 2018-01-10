def modelfit(alg, dtrain, predictors, performCV=True, printFeatureImportance=True, cv_folds=5):
    import numpy as np
    import pandas as pd

    from sklearn.model_selection import cross_val_score
    from sklearn import metrics
    import matplotlib.pylab as plt

    from matplotlib.pylab import rcParams
    rcParams['figure.figsize'] = 12, 4

    # Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['Survived'])

    # Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:, 1]

    if performCV:
        cv_score = cross_val_score(alg, dtrain[predictors], dtrain['Survived'], cv=cv_folds,
                                                    scoring='roc_auc')
    # Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(dtrain['Survived'].values, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['Survived'], dtrain_predprob))


    if performCV:
        print("CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score), np.std(cv_score), np.min(cv_score), np.max(cv_score)))

        # Print Feature Importance:
    if printFeatureImportance:
        plt.figure(figsize=(12, 8))
        feat_imp = pd.Series(alg.feature_importances_, predictors).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title='Feature Importance')
        plt.ylabel('Feature Importance Score')
        plt.xticks(rotation=30)
        plt.show()
