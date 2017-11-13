def clf_KNN(KNeighborsClassifier,
            train_X,
            train_Y,
            np,
            pd,
            cv,
            plot_scores,
            test_param=False,
            n_neighbors=None,
            algorithm='kd_tree'):

    def KNN_neighbor(train_X, train_Y, n_neighbors, algorithm):
        scores = pd.DataFrame(columns={'Avg_train_score', 'Avg_test_score'})
        for i in n_neighbors:
            cv_res = cv(train_X, train_Y, KNeighborsClassifier(n_neighbors=i, algorithm=algorithm))
            scores.loc[i] = [cv_res['Avg_train_score'], cv_res['Avg_test_score']]
        return scores

    if test_param:
        scores = KNN_neighbor(train_X, train_Y, n_neighbors,algorithm)
        print(scores)
        plot_scores(scores)

    cv_res = cv(train_X, train_Y, KNeighborsClassifier())
    print('KNN classifier expected accuracy: {0:.2f}'.format(cv_res['Avg_test_score']))
    print('')

