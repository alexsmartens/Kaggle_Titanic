def age_prediction_by_one_title(X, y, scoring_method='neg_mean_absolute_error' , message=''):

    import numpy as np
    from sklearn.dummy import DummyRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import Lasso
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.tree import DecisionTreeRegressor

    # Initiating of regressors for model fitting
    reg_dt_d2 = DecisionTreeRegressor(max_depth=2)
    reg_dt_d5 = DecisionTreeRegressor(max_depth=5)
    reg_dt_d10 = DecisionTreeRegressor(max_depth=10)
    reg_gbm = GradientBoostingRegressor()
    reg_lm = LinearRegression()
    reg_lm_lasso = Lasso()
    reg_dum_mean = DummyRegressor(strategy='mean')
    reg_dum_median = DummyRegressor(strategy='median')

    print(message)

    cv_scores_reg_dt_d2 = cross_val_score(reg_dt_d2, X, y, cv=5, scoring=scoring_method)
    print('Cross-validation scores of DecisionTreeRegressor(max_depth=2): ', cv_scores_reg_dt_d2)
    print('Mean cross-validation score of DecisionTreeRegressor(max_depth=2): ', np.mean(cv_scores_reg_dt_d2))

    cv_scores_reg_dt_d5 = cross_val_score(reg_dt_d5, X, y, cv=5, scoring=scoring_method)
    print('Cross-validation scores of DecisionTreeRegressor(max_depth=5): ', cv_scores_reg_dt_d5)
    print('Mean cross-validation score of DecisionTreeRegressor(max_depth=5): ', np.mean(cv_scores_reg_dt_d5))

    cv_scores_reg_dt_d10 = cross_val_score(reg_dt_d10, X, y, cv=5, scoring=scoring_method)
    print('Cross-validation scores of DecisionTreeRegressor(max_depth=10): ', cv_scores_reg_dt_d10)
    print('Mean cross-validation score of DecisionTreeRegressor(max_depth=10): ', np.mean(cv_scores_reg_dt_d10))

    cv_scores_reg_lm = cross_val_score(reg_lm, X, y, cv=5, scoring=scoring_method)
    print('Cross-validation scores of LinearRegression: ', cv_scores_reg_lm)
    print('Mean cross-validation score of LinearRegression: ', np.mean(cv_scores_reg_lm))

    cv_scores_reg_lasso = cross_val_score(reg_lm_lasso, X, y, cv=5, scoring=scoring_method)
    print('Cross-validation scores of Lasso: ', cv_scores_reg_lasso)
    print('Mean cross-validation score of Lasso: ', np.mean(cv_scores_reg_lasso))

    cv_scores_reg_dum_mean = cross_val_score(reg_dt_d2, X, y, cv=5, scoring=scoring_method)
    print('Cross-validation scores of reg_dum_mean: ', cv_scores_reg_dum_mean)
    print('Mean cross-validation score of reg_dum_mean: ', np.mean(cv_scores_reg_dum_mean))

    cv_scores_reg_dum_median = cross_val_score(reg_dum_median, X, y, cv=5, scoring=scoring_method)
    print('Cross-validation scores of reg_dum_median: ', cv_scores_reg_dum_median)
    print('Mean cross-validation score of reg_dum_median: ', np.mean(cv_scores_reg_dum_median))

    print('Mean models fit: ', np.round([np.mean(cv_scores_reg_dt_d2),
                                         np.mean(cv_scores_reg_dt_d5),
                                         np.mean(cv_scores_reg_dt_d10),
                                         np.mean(cv_scores_reg_lm),
                                         np.mean(cv_scores_reg_lasso),
                                         np.mean(cv_scores_reg_dum_mean),
                                         np.mean(cv_scores_reg_dum_median)], 3))
    return None