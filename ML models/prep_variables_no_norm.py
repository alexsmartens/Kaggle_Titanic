def prep_variables_no_norm(train_X_no_norm):
    train_X_no_norm.drop(['Cabin',
                  'Ticket',
                  'Name_title',
                  'Name',
                  'Name_last',
                  'Name_other'], axis=1, inplace=True)
    train_X_no_norm.loc[:,'Sex'] = (train_X_no_norm.loc[:,'Sex']=='female')*1
    train_X_no_norm.rename(columns={'Sex':'Sex==female'}, inplace=True)

    embarked_dict = {'S':1, 'C':2, 'Q':3}

    train_X_no_norm.loc[:,'Embarked'] = (train_X_no_norm.loc[:,'Embarked']).apply(lambda x: embarked_dict[x])

    return train_X_no_norm