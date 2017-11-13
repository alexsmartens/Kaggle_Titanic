def age_detection_optional(train,pd, age_prediction_by_one_title):
    # Preparing train and test sets for age detection
    age_features = ['SibSp','Parch','Name_title'] # 'Sex' is not relevant, is it is considered in 'Name_title'

    age_data_known_indicator = pd.notnull(train.copy()['Age'])
    #print(age_data_known_indicator.sum()) #number of people with known age CHECK
    age_data = train.copy()[age_data_known_indicator]

    X_age = age_data[age_features]
    X_age.loc[:,'Name_title'] = X_age.loc[:,'Name_title']==''

    X_age = pd.merge(X_age, pd.get_dummies( X_age.loc[:,'Name_title'], 'Title' ), left_index=True,
                       right_index=True)
    X_age.drop('Name_title', axis=1, inplace=True)
    y_age = age_data['Age']

    age_prediction_by_one_title(X_age,y_age)