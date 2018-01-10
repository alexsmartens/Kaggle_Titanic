def class1_passengers_stat (df_X, print_stat=False):
    import numpy as np
    import pandas as pd

    # Stat for 1st class passengers embarked at S(Southampton) allocation
    df_X_1st_S = df_X.copy()[(df_X['Embarked']=='S') &
                               (df_X['Pclass']==1) &
                               ( pd.isnull(df_X['Cabin'])==False)]
    df_X_1st_S_fare_stat = df_X_1st_S.groupby('Cabin_deck_level').agg(['count','mean',np.std])
    df_X_1st_S_fare_stat = df_X_1st_S_fare_stat.drop(8, axis=0)
    df_X_1st_S_fare_stat = df_X_1st_S_fare_stat.loc[:,'Fare']
    if print_stat:
        print('----- 1st class passengers embarked at S(Southampton) stat')
        print(df_X_1st_S_fare_stat)

    # Stat for 1st class passengers embarked at C(Cherbourg) allocation
    df_X_1st_C = df_X.copy()[(df_X['Embarked']=='C') &
                               (df_X['Pclass']==1) &
                               ( pd.isnull(df_X['Cabin'])==False) ]
    df_X_1st_C_fare_stat = df_X_1st_C.groupby('Cabin_deck_level').agg(['count','mean',np.std])
    df_X_1st_C_fare_stat = df_X_1st_C_fare_stat.loc[:,'Fare']
    if print_stat:
        print('----- 1st class passengers embarked at C(Cherbourg) stat')
        print(df_X_1st_C_fare_stat)

    # Stat for 1st class passengers not embarked at C(Cherbourg) or at S(Southampton) allocation
    df_X_1st_notSC = df_X.copy()[(df_X['Pclass']==1) &
                                   ( pd.isnull(df_X['Cabin'])==False) ]
    df_X_1st_notSC_fare_stat = df_X_1st_notSC.groupby('Cabin_deck_level').agg(['count','mean',np.std])
    df_X_1st_notSC_fare_stat = df_X_1st_notSC_fare_stat.loc[:,'Fare']
    df_X_1st_notSC_fare_stat = df_X_1st_notSC_fare_stat.drop(8, axis=0)
    # There are no passengers in this group, so no estimation for this group is planned
    if print_stat:
        print('----- 1st class passengers NOT embarked at S(Southampton) or C(Cherbourg) stat')
        print('')
        print('NA')

    return [{'S':df_X_1st_S, 'C':df_X_1st_C} , {'S':df_X_1st_S_fare_stat, 'C':df_X_1st_C_fare_stat}]

