def cv(train_X, train_Y, clf_model, prep_func):

    import numpy as np
    from sklearn.model_selection import train_test_split

    scores_train=[]
    scores_test=[]
    clf_list=[]

    for i in range(0,5):
        X_train, X_test, y_train, y_test = train_test_split(train_X, train_Y, test_size=0.2)

        if str(prep_func.__name__)=='prep_variables_norm':
            [X_train, scaler_dict] = prep_func(X_train, Train=True)
            X_test = prep_func(X_test, Train=False, scaler_dict=scaler_dict)
        else:
            X_train = prep_func(X_train)
            X_test = prep_func(X_test)

        clf = clf_model.fit(X_train, y_train)
        scores_train.append(clf.score(X_train, y_train))
        scores_test.append(clf.score(X_test, y_test))
        clf_list.append(clf)
    return {'Avg_train_score': np.mean(scores_train), 'Avg_test_score': np.min(scores_test)}
