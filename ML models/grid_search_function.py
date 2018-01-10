def grid_search_function(df_X, df_Y, clf, param_grid, cv=3):
    from sklearn.model_selection import GridSearchCV
    import numpy as np
    from time import time

    start = time()
    grid_search = GridSearchCV(clf, param_grid=param_grid, n_jobs=-1)

    grid_search.fit(df_X, df_Y)

    def report(results, n_top=3):
        for i in range(1, n_top + 1):
            candidates = np.flatnonzero(results['rank_test_score'] == i)
            for candidate in candidates:
                print("Model with rank: {0}".format(i))
                print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                    results['mean_test_score'][candidate],
                    results['std_test_score'][candidate]))
                print("Parameters: {0}".format(results['params'][candidate]))
                print("")

    print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
          % (time() - start, len(grid_search.cv_results_)))
    report(grid_search.cv_results_)