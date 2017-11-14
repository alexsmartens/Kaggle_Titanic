def clf_GradientBoostingClassifier(GradientBoostingClassifier,
                                   train_X_no_norm,
                                   train_Y,
                                   prep_func,
                                   np,
                                   pd,
                                   cv,
                                   plot_scores,
                                   test_param=False):
    n_estimators_rough = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 300, 500]
    n_estimators = np.arange(5, 30, 1)

    def gbt_reg_n_estimators(train_X, train_Y, n_estimators):
        scores = pd.DataFrame(columns={'Avg_train_score', 'Avg_test_score'})
        for i in n_estimators:
            cv_res = cv(train_X, train_Y, GradientBoostingClassifier(n_estimators=i), prep_func)
            scores.loc[i] = [cv_res['Avg_train_score'], cv_res['Avg_test_score']]
        return scores

    #scores = gbt_reg_n_estimators(train_X_no_norm, train_Y, n_estimators)
    #print(scores)
    #plot_scores(scores)
    # n_estimators = 15 seems to be a reasonably good classifier
    # [5,25] is the recommended range for further testing
    #n_estimators = 15


    learn_rate_rough = [0.001, 0.01, 0.1, 0.5, 1, 2, 3, 5, 10, 20, 50, 100]
    learn_rate = np.arange(0.02,0.2,0.01)

    def gbt_reg_learn_rate(train_X, train_Y, n_estimators, learn_rate):
        scores = pd.DataFrame(columns={'Avg_train_score', 'Avg_test_score'})
        for i in learn_rate:
            cv_res = cv(train_X, train_Y, GradientBoostingClassifier(n_estimators=n_estimators,learning_rate=i), prep_func)
            scores.loc[i] = [cv_res['Avg_train_score'], cv_res['Avg_test_score']]
        return scores

    #scores = gbt_reg_learn_rate(train_X_no_norm, train_Y, n_estimators, learn_rate)
    #print(scores)
    #plot_scores(scores)
    # learning_rate = [0.02, 0.2] seems to be a reasonably good for further investigation

    def gbt_reg_learn_n_estimators(train_X, train_Y, n_estimators, learn_rate):
        scores = pd.DataFrame(columns={'n_estimators', 'learn_rate', 'Avg_train_score', 'Avg_test_score'})
        num = 0
        for n in n_estimators:
            for learn in learn_rate:
                cv_res = cv(train_X, train_Y, GradientBoostingClassifier(n_estimators=n,learning_rate=learn), prep_func)
                num += 1
                scores.loc[num] = pd.Series([n, learn, cv_res['Avg_train_score'], cv_res['Avg_test_score']],
                                            index=['n_estimators', 'learn_rate', 'Avg_train_score', 'Avg_test_score'])
        return scores
    learn_rate = np.arange(0.09,0.2,0.01)
    n_estimators = np.arange(8,15,1)
    #scores = gbt_reg_learn_n_estimators(train_X_no_norm, train_Y, n_estimators, learn_rate)
    #print(scores)
    #plot_scores(scores)

    cv_res = cv(train_X_no_norm, train_Y, GradientBoostingClassifier(n_estimators=8, learning_rate=0.2), prep_func)
    print('Gradient Boosted Decision Trees classifier expected accuracy: {0:.2f}'.format(cv_res['Avg_test_score']))
    print('')