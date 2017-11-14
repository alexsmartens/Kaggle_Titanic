def clf_LogisticRegression(LogisticRegression,
                           train_X,
                           train_Y,
                           prep_func,
                           np,
                           pd,
                           cv,
                           plot_scores,
                           test_param=False,
                           c=None):

    def log_reg_check_c(c):
        scores = pd.DataFrame(columns={'Avg_train_score', 'Avg_test_score'})
        for i in c:
            cv_res = cv(train_X, train_Y, LogisticRegression(C=i), prep_func)
            scores.loc[i] = [cv_res['Avg_train_score'], cv_res['Avg_test_score']]
        return scores

    # rough selection of C regularization parameter
    c_rough = [0.001, 0.01, 0.1, 0.5, 1, 2, 3, 5, 10, 20, 50, 100]
    # fine selection of C regularization parameter

    if test_param:
        scores = log_reg_check_c(c)
        print(scores)
        plot_scores(scores)
    cv_res = cv(train_X, train_Y, LogisticRegression(C=1), prep_func)
    print('Logistic regression expected accuracy: {0:.2f}'.format(cv_res['Avg_test_score']))
    print('')