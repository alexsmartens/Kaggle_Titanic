def class1_ passengers_stat():

    # Stat for 1st class passengers embarked at S(Southampton) allocation
    train_1st_S = train.copy()[(train['Embarked']=='S') &
                               (train['Pclass']==1) &
                               ( pd.isnull(train['Cabin'])==False) ]
    #print('----- 1st class passengers embarked at S(Southampton) stat')
    train_1st_S_fare_stat = train_1st_S.groupby('Deck level').agg(['count','mean',np.std])
    train_1st_S_fare_stat = train_1st_S_fare_stat.drop(8, axis=0)
    train_1st_S_fare_stat = train_1st_S_fare_stat.loc[:,'Fare']
    #print(train_1st_S_fare_stat)
    # Stat for 1st class passengers embarked at C(Cherbourg) allocation
    train_1st_C = train.copy()[(train['Embarked']=='C') &
                               (train['Pclass']==1) &
                               ( pd.isnull(train['Cabin'])==False) ]
    #print('----- 1st class passengers embarked at C(Cherbourg) stat')
    train_1st_C_fare_stat = train_1st_C.groupby('Deck level').agg(['count','mean',np.std])
    train_1st_C_fare_stat = train_1st_C_fare_stat.loc[:,'Fare']
    #print(train_1st_C_fare_stat)
    # Stat for 1st class passengers not embarked at C(Cherbourg) or at S(Southampton) allocation
    train_1st_notSC = train.copy()[(train['Pclass']==1) &
                                   ( pd.isnull(train['Cabin'])==False) ]
    deck_codes_1st_notSC = {'A':7, 'B':6, 'C':5, 'D':4, 'E':3}
    #print('----- 1st class passengers NOT embarked at S(Southampton) or C(Cherbourg) stat')
    #print('')
    #print('NA')
    train_1st_notSC_fare_stat = train_1st_notSC.groupby('Deck level').agg(['count','mean',np.std])
    train_1st_notSC_fare_stat = train_1st_notSC_fare_stat.loc[:,'Fare']
    train_1st_notSC_fare_stat = train_1st_notSC_fare_stat.drop(8, axis=0)
    # There are no passengers in this group, so no estimation for this group is planned
    #print(train_1st_notSC_fare_stat)