def clf_RandomForestClassifier(RandomForestClassifier,
                               train_X_no_norm,
                               train_Y,
                               prep_func,
                               np,
                               pd,
                               cv,
                               plot_scores,
                               test_param=False,
                               n_estimators = 10,
                               max_features = 6,
                               max_depth = 3):

    def rf_n_estimator(train_X, train_Y, n_estimators, max_features, max_depth):
        scores = pd.DataFrame(columns={'Avg_train_score', 'Avg_test_score'})
        for i in n_estimators:
            cv_res = cv(train_X, train_Y, RandomForestClassifier(n_estimators=i,
                                                                 max_features=max_features,
                                                                 max_depth=max_depth),prep_func)
            scores.loc[i] = [cv_res['Avg_train_score'], cv_res['Avg_test_score']]
        return scores

    ##n_estimators = np.arange(8,25,1)
    # scores = rf_n_estimator(train_X_no_norm, train_Y, n_estimators, max_features, max_depth)
    # print(scores)
    # plot_scores(scores)
    # n_estimators = 12 seems to produce reasonably good results
    ##n_estimators = 12

    ##max_features = [2,3,4,5,6,7,8,9]
    def rf_features(train_X, train_Y, n_estimators, max_features, max_depth):
        scores = pd.DataFrame(columns={'Avg_train_score', 'Avg_test_score'})
        for i in max_features:
            cv_res = cv(train_X, train_Y, RandomForestClassifier(n_estimators=n_estimators,
                                                                 max_features=i,
                                                                 max_depth=max_depth),prep_func)
            scores.loc[i] = [cv_res['Avg_train_score'], cv_res['Avg_test_score']]
        return scores

    # scores = rf_features(train_X_no_norm, train_Y, n_estimators, max_features, max_depth)
    # print(scores)
    # plot_scores(scores)
    # max_features = 5 seems to produce reasonably good results
    ##max_features = 5


    ##max_features = np.arange(3,7,1)
    ##n_estimators = np.arange(8, 16, 1)

    def rf_n_estimator_features(train_X, train_Y, n_estimators, max_features, max_depth):
        scores = pd.DataFrame(columns={'n_estimators', 'max_features', 'Avg_train_score', 'Avg_test_score'})
        num = 0
        for n in n_estimators:
            for i in max_features:
                cv_res = cv(train_X, train_Y, RandomForestClassifier(n_estimators=n,
                                                                     max_features=i,
                                                                     max_depth=max_depth),prep_func)
                num += 1
                scores.loc[num] = pd.Series([n, i, cv_res['Avg_train_score'], cv_res['Avg_test_score']],
                                            index=['n_estimators', 'max_features', 'Avg_train_score', 'Avg_test_score'])
        return scores

    # scores = rf_n_estimator_features(train_X_no_norm, train_Y, n_estimators, max_features, max_depth)
    # print(scores)
    # plot_scores(scores)

    cv_res = cv(train_X_no_norm, train_Y, RandomForestClassifier(n_estimators=n_estimators,
                                                                 max_features=max_features,
                                                                 max_depth=max_depth),prep_func)
    print('Random forest classifier expected accuracy: {0:.2f}'.format(cv_res['Avg_test_score']))
    print('')
