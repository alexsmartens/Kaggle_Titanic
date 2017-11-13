def prep_variables_norm(X_train,pd,preprocessing):

    ## Removing unnecessary features
    # Cabin feature has been used for locating passengers on the ship
    # Ticket feature has been used for groping passengers with the same tickets at the same place on the ship
    # Name_title feature has been used for age detection
    # Name, Name_last, Name_other features are personal identifiers

    X_train.drop(['Cabin',
                  'Ticket',
                  'Name_title',
                  'Name',
                  'Name_last',
                  'Name_other'], axis=1, inplace=True)

    # Continues variables standardization
    X_train.loc[:,'Age'] = preprocessing.scale( X_train.loc[:,'Age'].astype('float64') )
    X_train.loc[:,'Room center longitude(X)'] = preprocessing.scale( X_train.loc[:,'Room center longitude(X)'].astype('float64') )
    X_train.loc[:,'Room center latitude(Y)'] = preprocessing.scale( X_train.loc[:,'Room center latitude(Y)'].astype('float64') )
    X_train.loc[:,'Fare'] = preprocessing.scale( X_train.loc[:,'Fare'].astype('float64') )

    # Integer variables standardization
    X_train.loc[:,'Deck level'] = preprocessing.scale( X_train.loc[:,'Deck level'].astype('float64') )
    X_train.loc[:,'Parch'] = preprocessing.scale( X_train.loc[:,'Parch'].astype('float64') )
    X_train.loc[:,'SibSp'] = preprocessing.scale( X_train.loc[:,'SibSp'].astype('float64') )

    # Categorical variables transformation
    X_train.loc[:,'Sex'] = (X_train.loc[:,'Sex']=='female')*1
    X_train.rename(columns={'Sex':'Sex==female'}, inplace=True)

    X_train = pd.merge(X_train, pd.get_dummies(X_train.loc[:,'Embarked'],'Embarked'), left_index=True, right_index=True)
    X_train.drop('Embarked',axis=1, inplace=True)

    X_train = pd.merge(X_train, pd.get_dummies(X_train.loc[:,'Pclass'],'Pclass'), left_index=True, right_index=True)
    X_train.drop('Pclass', axis=1, inplace=True)
    return X_train