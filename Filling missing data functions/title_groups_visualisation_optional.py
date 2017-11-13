def title_groups_visualisation_optional(train,
                                        pd,
                                        plt,
                                        Axes3D,
                                        age_prediction_by_one_title,
                                        preprocessing,
                                        exta_print=False):

    # Preparing train and test sets for age detection
    age_features = ['SibSp', 'Parch', 'Name_title']  # 'Sex' is not relevant, is it is considered in 'Name_title'

    age_data_known_indicator = pd.notnull(train.copy()['Age'])
    # print(age_data_known_indicator.sum()) #number of people with known age CHECK
    age_data = train.copy()[age_data_known_indicator]

    X_age = age_data[age_features]
    y_age = age_data['Age']


    # Training 3D scatter plot: 'Master' Age dependence based on SibSp and Parch
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(X_age[X_age['Name_title']=='Master'].loc[:,'SibSp'],
               X_age[X_age['Name_title']=='Master'].loc[:,'Parch'],
               y_age[X_age['Name_title']=='Master'])
    ax.set_xlabel('SibSp')
    ax.set_ylabel('Parch')
    ax.set_zlabel('Age')
    ax.set_title('Master age plot')
    plt.show()
    # Conclusion(plot): 'Master' gives enough information to make a conclusion about a person age on its own,
    # fitting a distribution to this data chunk is likely to have a lot of variation form logic perspective.
    # That's why I propose using median age of known people for 'Master's age prediction

    if exta_print:
        print('----------- Master -----------')
        X_age_Master = X_age.copy()[X_age['Name_title']=='Master'].loc[:,['SibSp', 'Parch']]
        y_age_Master = y_age.copy()[X_age['Name_title']=='Master']
        print('')
        print(len(X_age_Master))
        age_prediction_by_one_title(X_age_Master, y_age_Master)
        print('')
        age_prediction_by_one_title(X_age_Master, y_age_Master, 'neg_mean_squared_log_error')
        print('')
        age_prediction_by_one_title(preprocessing.scale(X_age_Master),preprocessing.scale(y_age_Master))
        print('')
        print('')
    # Conclusion(model fitting): use MEDIAN age for Masters age prediction. However, MEAN would work fine as well



    # Training 3D scatter plot: 'Mr' Age dependance based on SibSp and Parch
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(X_age[X_age['Name_title']=='Mr'].loc[:,'SibSp'],
               X_age[X_age['Name_title']=='Mr'].loc[:,'Parch'],
               y_age[X_age['Name_title']=='Mr'])
    ax.set_xlabel('SibSp')
    ax.set_ylabel('Parch')
    ax.set_zlabel('Age')
    ax.set_title('Mr age plot')
    plt.show()
    # Conclusion (plot): general trend for 'Mr' is the higher 'Prach' number | the higher 'SibSp' the lower age.
    # Prediction model fitting might be usefull

    if exta_print:
        print('----------- Mr -----------')
        X_age_Mr = X_age.copy()[X_age['Name_title'] == 'Mr'].loc[:, ['SibSp', 'Parch']]
        y_age_Mr = y_age.copy()[X_age['Name_title'] == 'Mr']
        print(len(X_age_Mr))
        age_prediction_by_one_title(X_age_Mr, y_age_Mr)
        print('')
        age_prediction_by_one_title(X_age_Mr, y_age_Mr, 'neg_mean_squared_log_error')
        print('')
        age_prediction_by_one_title(preprocessing.scale(X_age_Mr), preprocessing.scale(y_age_Mr))
        print('')
        print('')
    # Conclusion(model fitting): use MEDIAN age for Mr_s age prediction. However, MEAN would work fine as well




    # Training 3D scatter plot: 'Mrs' Age dependance based on SibSp and Parch
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(X_age[X_age['Name_title']=='Mrs'].loc[:,'SibSp'],
               X_age[X_age['Name_title']=='Mrs'].loc[:,'Parch'],
               y_age[X_age['Name_title']=='Mrs'])
    ax.set_xlabel('SibSp')
    ax.set_ylabel('Parch')
    ax.set_zlabel('Age')
    ax.set_title('Mrs age plot')
    plt.show()
    # Conclusion (plot): general trend for 'Mrs' is
    # the higher 'Parch' number the less SD from the mean, with approximately the same mean age
    # having one 'SibSp' vs zero 'SibSp' decreases distribution mean
    # Prediction model fitting might be useful

    if exta_print:
        print('----------- Mrs -----------')
        X_age_Mrs = X_age.copy()[X_age['Name_title'] == 'Mrs'].loc[:, ['SibSp', 'Parch']]
        y_age_Mrs = y_age.copy()[X_age['Name_title'] == 'Mrs']
        print(len(X_age_Mrs))
        age_prediction_by_one_title(X_age_Mrs, y_age_Mrs)
        print('')
        age_prediction_by_one_title(X_age_Mrs, y_age_Mrs, 'neg_mean_squared_log_error')
        print('')
        age_prediction_by_one_title(preprocessing.scale(X_age_Mrs), preprocessing.scale(y_age_Mrs))
        print('')
        print('')
    # Conclusion(model fitting): use MEAN age for Mrs_s age prediction. However, MEAN would work fine as well



    # Training 3D scatter plot: 'Miss' Age dependance based on SibSp and Parch
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(X_age[X_age['Name_title']=='Miss'].loc[:,'SibSp'],
               X_age[X_age['Name_title']=='Miss'].loc[:,'Parch'],
               y_age[X_age['Name_title']=='Miss'])
    ax.set_xlabel('SibSp')
    ax.set_ylabel('Parch')
    ax.set_zlabel('Age')
    ax.set_title('Miss age plot')
    plt.show()
    # Conclusion (plot): general trend for 'Mrs' is
    # having one or more 'Parch' dramatically change the age distribution mean comparatively to 'Parch' of zero
    # the higher number of 'SibSp' the less Age is
    # Prediction model fitting should be usefull
    if exta_print:
        print('----------- Miss -----------')
        X_age_Miss = X_age.copy()[X_age['Name_title'] == 'Miss'].loc[:, ['SibSp', 'Parch']]
        y_age_Miss = y_age.copy()[X_age['Name_title'] == 'Miss']
        print(len(X_age_Miss))
        age_prediction_by_one_title(X_age_Miss, y_age_Miss)
        print('')
        age_prediction_by_one_title(X_age_Miss, y_age_Miss, 'neg_mean_squared_log_error')
        print('')
        age_prediction_by_one_title(X_age_Miss, y_age_Miss)
        print('')
        age_prediction_by_one_title(preprocessing.scale(X_age_Miss), preprocessing.scale(y_age_Miss))
        print('')
        print('')
    # Conclusion(model fitting): use MEAN age for Misses age prediction

    # Training 3D scatter plot: Age dependance based on Name_title, SibSp and Parch
    colors_dict = {'Master':'blue', 'Mr':'black', 'Mrs':'red', 'Miss':'pink'}
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(X_age['SibSp'],
               X_age['Parch'],
               y_age,
               c=X_age['Name_title'].apply(lambda x: colors_dict[x]))
    ax.set_xlabel('SibSp')
    ax.set_ylabel('Parch')
    ax.set_zlabel('Age')
    ax.set_title('All titles age plot')
    plt.show()

    ## Fit regression model for age prediction on full data set
    # Categorical features transformation

    if exta_print:
        print('----------- All titles -----------')
        X_age_transformed = pd.get_dummies(X_age.copy().select_dtypes(include=[object]))
        X_age_transformed = pd.merge(X_age_transformed, X_age[['SibSp','Parch']],
                                     left_index=True, right_index=True)

        #age_prediction_by_one_title(X_age_transformed, y_age, 'neg_mean_squared_log_error')
        #print('')
        age_prediction_by_one_title(X_age_transformed, y_age)
        print('')
        age_prediction_by_one_title(preprocessing.scale(X_age_transformed), preprocessing.scale(y_age))