def features_processing(df_X, target, normalization, training=True, scaler=None):
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    df_X.loc[:, 'Sex'] = (df_X.loc[:, 'Sex'] == 'female') * 1
    df_X.rename(columns={'Sex': 'Sex==female'}, inplace=True)


    ethnicity_dict = {'AFRICAN': 'OTHER',
                      'ARAB': 'OTHER',
                      'CHINESE': 'OTHER',
                      'DUTCH': 'EUROPEAN',
                      'ENGLISH': 'ENGLISH',
                      'FRENCH': 'EUROPEAN',
                      'GERMAN': 'EUROPEAN',
                      'GREEK': 'EUROPEAN',
                      'HISPANIC': 'OTHER',
                      'INDIAN': 'OTHER',
                      'ISRAELI': 'OTHER',
                      'ITALIAN': 'EUROPEAN',
                      'JAPANESE': 'OTHER',
                      'NORDIC': 'NORDIC',
                      'ROMANIAN': 'EUROPEAN',
                      'SLAV': 'EUROPEAN',
                      'THAI': 'OTHER',
                      'TURKISH': 'OTHER',
                      'VIETNAMESE': 'OTHER'}
    df_X.loc[:, 'Ethnicity_origin'] = (df_X.loc[:, 'Ethnicity_origin']).apply(lambda x: ethnicity_dict[x])
    df_X = pd.merge(df_X, pd.get_dummies(df_X['Ethnicity_origin'], prefix='Ethnicity'),
                    how='left', left_index=True, right_index=True)


    if not normalization:
        print('Normalization is turned OFF')
        embarked_dict = {'S': 1, 'C': 3, 'Q': 2}
        df_X.loc[:, 'Embarked'] = (df_X.loc[:, 'Embarked']).apply(lambda x: embarked_dict[x])

        df_X = df_X.drop(['Cabin',
                          'Ticket',
                          'Name_title',
                          'Name',
                          'Name_first',
                          'Name_last',
                          'Name_other',
                          'Ticket',
                          'Ticket_Series',
                          'Ticket_No',

                          'Ticket_combined',
                          'Ethnicity_origin',
                          'Ethnicity_prob'], axis=1)
        print(df_X.columns)
        return [scaler, df_X]
    else:
        print('Normalization is turned ON')
        df_X = pd.merge(df_X, pd.get_dummies(df_X['Embarked'], prefix='Embarked'),
                        how='left', left_index=True, right_index=True)



        df_X = df_X.drop(['Cabin',
                          'Ticket',
                          'Name_title',
                          'Name',
                          'Name_first',
                          'Name_last',
                          'Name_other',
                          'Ticket',
                          'Ticket_Series',
                          'Ticket_No',

                          'Ticket_combined',
                          'Ethnicity_origin',
                          'Ethnicity_prob',
                          'Embarked'], axis=1)
        predictors = [x for x in df_X.columns if x not in [target]]
        if training:
            # print(df_X.shape)
            scaler = StandardScaler().fit(X=df_X[predictors])
            df_Y = df_X[target]
            df_X_index = df_X.index
            df_X_res = scaler.transform(X=df_X[predictors])
            # print(df_X_res.shape)
            df_X = pd.DataFrame(df_X_res, columns=df_X[predictors].columns)
            df_X.index = df_X_index
            df_X = pd.merge(df_X, pd.DataFrame(df_Y), how='left', left_index=True, right_index=True)
            print(df_X.columns)
            return [scaler, df_X]
        else:
            df_X_index = df_X.index
            df_X_res = scaler.transform(X=df_X[predictors])
            # print(df_X_res.shape)
            df_X = pd.DataFrame(df_X_res, columns=df_X[predictors].columns)
            df_X.index = df_X_index
            print(df_X.columns)
            return [scaler, df_X]



