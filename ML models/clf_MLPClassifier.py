def clf_MLPClassifier(MLPClassifier,
                      train_X,
                      train_Y,
                      np,
                      pd,
                      cv,
                      plot_scores,
                      test_param=False,
                      alpha=2,
                      hidden_layer_sizes=[3, 60],
                      activation='relu',
                      solver='lbfgs'):

    def nn_alpha(train_X_no_norm, train_Y, activation, hidden_layer_sizes, alpha, solver):
        scores = pd.DataFrame(columns={'Avg_train_score', 'Avg_test_score'})
        for i in alpha:
            cv_res = cv(train_X, train_Y, MLPClassifier(activation=activation,
                                                        hidden_layer_sizes=hidden_layer_sizes,
                                                        alpha=i,
                                                        solver=solver))
            scores.loc[i] = [cv_res['Avg_train_score'], cv_res['Avg_test_score']]
        return scores

    # alpha_raugh = [0.001, 0.01, 0.1, 0.5, 1, 2, 3, 5, 10, 20, 50, 100]
    ##alpha = np.arange(2, 8.5, 0.25)
    # scores = nn_alpha(train_X, train_Y, activation, hidden_layer_sizes, alpha, solver)
    # print(scores)
    # plot_scores(scores)
    # alpha=5 seems to be optimal
    ##alpha = 5

    def nn_layers(train_X_no_norm, train_Y, activation, hidden_layer_sizes, alpha, solver):
        scores = pd.DataFrame(columns={'layers_num', 'layers_size', 'Avg_train_score', 'Avg_test_score'})
        num = 0
        for i in np.arange(2, 5, 1):
            for ii in np.arange(49, 70, 1):
                cv_res = cv(train_X, train_Y, MLPClassifier(activation=activation,
                                                            hidden_layer_sizes=[i, ii],
                                                            alpha=alpha,
                                                            solver=solver))
                num += 1
                scores.loc[num] = pd.Series([i, ii, cv_res['Avg_train_score'], cv_res['Avg_test_score']],
                                            index=['layers_num', 'layers_size', 'Avg_train_score', 'Avg_test_score'])
        return scores

    # scores = nn_layers(train_X, train_Y, activation, hidden_layer_sizes, alpha, solver)
    # print(scores)
    # plot_scores(scores)
    ##hidden_layer_sizes = [3, 60]

    # print(cv(train_X, train_Y, MLPClassifier(activation='relu',
    #                                         hidden_layer_sizes=hidden_layer_sizes,
    #                                         alpha=alpha,
    #                                         solver=solver)))
    #
    # print(cv(train_X, train_Y, MLPClassifier(activation='logistic',
    #                                         hidden_layer_sizes=hidden_layer_sizes,
    #                                         alpha=alpha,
    #                                         solver=solver)))
    #
    # print(cv(train_X, train_Y, MLPClassifier(activation='tanh',
    #                                         hidden_layer_sizes=hidden_layer_sizes,
    #                                         alpha=alpha,
    #                                         solver=solver)))

    cv_res = cv(train_X, train_Y, MLPClassifier(activation=activation,
                                             hidden_layer_sizes=hidden_layer_sizes,
                                             alpha=alpha,
                                             solver=solver))
    print('Neural nets classifier expected accuracy: {0:.2f}'.format(cv_res['Avg_test_score']))
    print('')