def prep_variables_norm(X_train, Train, plot_dist=False, scaler_dict={}):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn import preprocessing


    ## Removing unnecessary features
    # Cabin feature has been used for locating passengers on the ship
    # Ticket feature has been used for groping passengers with the same tickets at the same place on the ship
    # Name_title feature has been used for age detection
    # Name, Name_last, Name_other features are personal identifiers

    if plot_dist:
        # plot variables for standardization
        plt.hist(X_train.loc[:,'Age'])
        plt.ylabel('Probability')
        plt.xlabel('Age')
        plt.title('Age distribution')
        plt.show()

        plt.hist(X_train.loc[:, 'Room center longitude(X)'])
        plt.ylabel('Probability')
        plt.xlabel('Room center longitude(X)')
        plt.title('Room center longitude(X) distribution')
        plt.show()

        plt.hist(X_train.loc[:, 'Room center latitude(Y)'])
        plt.ylabel('Probability')
        plt.xlabel('Room center latitude(Y)')
        plt.title('Room center latitude(Y) distribution')
        plt.show()

        plt.hist(X_train.loc[:, 'Fare'],20)
        plt.ylabel('Probability')
        plt.xlabel('Fare')
        plt.title('Fare distribution')
        plt.show()

        plt.hist(X_train.loc[:, 'Deck level'])
        plt.ylabel('Probability')
        plt.xlabel('Deck level')
        plt.title('Deck level distribution')
        plt.show()

        plt.hist(X_train.loc[:, 'Parch'])
        plt.ylabel('Probability')
        plt.xlabel('Parch')
        plt.title('Parch distribution')
        plt.show()

        plt.hist(X_train.loc[:, 'SibSp'])
        plt.ylabel('Probability')
        plt.xlabel('SibSp')
        plt.title('SibSp distribution')
        plt.show()

    ## Variables transformation
    if Train:
        # fitting standard normal distribution to each feature to be normalized
        # and creating a scaler dictionary for transforming train and test sets
        scaler_dict['Age'] = preprocessing.StandardScaler().fit( np.array(X_train.loc[:,'Age']).reshape(-1, 1) )
        scaler_dict['Room center longitude(X)'] = preprocessing.StandardScaler().fit(np.array(X_train.loc[:, 'Room center longitude(X)']).reshape(-1, 1))
        scaler_dict['Room center latitude(Y)'] = preprocessing.StandardScaler().fit(np.array(X_train.loc[:, 'Room center latitude(Y)']).reshape(-1, 1))
        scaler_dict['Fare'] = preprocessing.StandardScaler().fit(np.array(X_train.loc[:, 'Fare']).reshape(-1, 1))

        scaler_dict['Deck level'] = preprocessing.StandardScaler().fit( np.array( X_train.loc[:, 'Deck level'].astype('float64') ).reshape(-1, 1))
        scaler_dict['Parch'] = preprocessing.StandardScaler().fit(np.array( X_train.loc[:, 'Parch'].astype('float64') ).reshape(-1, 1))
        scaler_dict['SibSp'] = preprocessing.StandardScaler().fit(np.array( X_train.loc[:, 'SibSp'].astype('float64') ).reshape(-1, 1))
    else:
        if scaler_dict=={}:
            print("\033[91m {}\033[00m" \
                  .format('!WARNING.2.1. Please provide the dictionary for transforming the TEST data set'))
            X_train = 0

    X_train = X_train.drop(['Cabin',
                            'Ticket',
                            'Name_title',
                            'Name',
                            'Name_last',
                            'Name_other'], axis=1)

    # Categorical variables transformation
    X_train.loc[:, 'Sex'] = (X_train.loc[:, 'Sex'] == 'female') * 1
    X_train.rename(columns={'Sex': 'Sex==female'}, inplace=True)

    X_train = pd.merge(X_train, pd.get_dummies(X_train.loc[:, 'Embarked'], 'Embarked'), left_index=True,
                       right_index=True)
    X_train.drop('Embarked', axis=1, inplace=True)

    X_train = pd.merge(X_train, pd.get_dummies(X_train.loc[:, 'Pclass'], 'Pclass'), left_index=True, right_index=True)
    X_train.drop('Pclass', axis=1, inplace=True)

    # Continues variables standardization
    X_train.loc[:, 'Age'] = scaler_dict['Age'].transform( np.array(X_train.loc[:,'Age']).reshape(-1, 1) )
    X_train.loc[:, 'Room center longitude(X)'] = scaler_dict['Room center longitude(X)'].transform(np.array(X_train.loc[:, 'Room center longitude(X)']).reshape(-1, 1))
    X_train.loc[:, 'Room center latitude(Y)'] = scaler_dict['Room center latitude(Y)'].transform(np.array(X_train.loc[:, 'Room center latitude(Y)']).reshape(-1, 1))
    X_train.loc[:, 'Fare'] = scaler_dict['Fare'].transform(np.array(X_train.loc[:, 'Fare']).reshape(-1, 1))

    # Integer variables standardization
    X_train.loc[:, 'Deck level'] = scaler_dict['Deck level'].transform(np.array( X_train.loc[:, 'Deck level'].astype('float64') ).reshape(-1, 1))
    X_train.loc[:, 'Parch'] = scaler_dict['Parch'].transform(np.array( X_train.loc[:, 'Parch'].astype('float64') ).reshape(-1, 1))
    X_train.loc[:, 'SibSp'] = scaler_dict['SibSp'].transform(np.array( X_train.loc[:, 'SibSp'].astype('float64') ).reshape(-1, 1))

    if Train:

        return [X_train, scaler_dict]
    else:
        return X_train