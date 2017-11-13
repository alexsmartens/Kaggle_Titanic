def clf_LogisticRegression(LogisticRegression,
                           train_X,
                           train_Y,
                           np,
                           pd,
                           cv,
                           plot_scores,
                           testC=False,
                           c = range(1,21)/10):

    def log_reg_check_c(c):
        scores = pd.DataFrame(columns={'Avg_train_score', 'Avg_test_score'})
        for i in c:
            cv_res = cv(train_X, train_Y, LogisticRegression(C=i))
            scores.loc[i] = [cv_res['Avg_train_score'], cv_res['Avg_test_score']]
        return scores

    # rough selection of C regularization parameter
    c_rough = [0.001, 0.01, 0.1, 0.5, 1, 2, 3, 5, 10, 20, 50, 100]
    # fine selection of C regularization parameter

    if testC:
        scores = log_reg_check_c(c)
        print(scores)
        plot_scores(scores)
    print(c)
    ## Conclusion: standard C=1 performance seems to be good, so C=1 is chosen for logistic regression
    # Logistic regression expected accuracy: 0.80
    cv_res = cv(train_X, train_Y, LogisticRegression(C=1))
    print('Logistic regression expected accuracy: {0:.2f}'.format(cv_res['Avg_test_score']))