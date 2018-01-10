def cabin_existing_allocation(df_X):
    import numpy as np
    import pandas as pd
    import scipy

    import os, sys
    lib_path = os.path.abspath(os.path.join(''))
    sys.path.append(lib_path)
    from cabin_assign_internal import cabin_assign_internal

    cabin_location_list = pd.read_csv('./Data/titanic_cabin_location.csv')
    cabin_location_list.set_index('Room code', inplace=True)
    cabin_list = df_X.loc[(pd.notnull(df_X.loc[:, 'Cabin'])) & (df_X.loc[:, 'Pclass']==1), 'Cabin'].unique().tolist()

    df_X.loc[:,'Cabin_deck_level'] = ''
    df_X.loc[:,'Cabin_X'] = ''
    df_X.loc[:,'Cabin_Y'] = ''

    # Assigning cabin location coordinates to 1st class passengers with known cabin number
    for cabin_No in cabin_list:
        if cabin_No in cabin_location_list.index.tolist():
            df_X.loc[df_X.loc[:,'Cabin']==cabin_No, 'Cabin_deck_level'] = cabin_location_list.loc[cabin_No,'Deck level']
            df_X.loc[df_X.loc[:,'Cabin']==cabin_No, 'Cabin_X'] = cabin_location_list.loc[cabin_No, 'Room center longitude(X)']
            df_X.loc[df_X.loc[:,'Cabin']==cabin_No, 'Cabin_Y'] = cabin_location_list.loc[cabin_No, 'Room center latitude(Y)']

        elif cabin_No.split(' ')[0] in cabin_location_list.index.tolist():
            df_X.loc[df_X.loc[:,'Cabin']==cabin_No, 'Cabin_deck_level'] = round(cabin_location_list.loc[
                                                                            cabin_location_list.index.isin(cabin_No.split(' '))].
                                                                            loc[:, 'Deck level'].mean(), 2)
            df_X.loc[df_X.loc[:,'Cabin']==cabin_No, 'Cabin_X'] = round(cabin_location_list.loc[
                                                                            cabin_location_list.index.isin(cabin_No.split(' '))].
                                                                            loc[:,'Room center longitude(X)'].mean(),2)
            df_X.loc[df_X.loc[:,'Cabin']==cabin_No, 'Cabin_Y'] = round(cabin_location_list.loc[
                                                                            cabin_location_list.index.isin(cabin_No.split(' '))].
                                                                            loc[:, 'Room center latitude(Y)'].mean(), 2)
        else:
            print("\033[91m {}\033[00m" \
                  .format('!WARNING. cabin_existing_allocation\n'
                          'A cabin is not in the cabin list'))


    # Assigning cabin location coordinates to 1st class passengers with unknown cabin number and all 2nd and 3rd class passengers
    df_X = cabin_assign_internal (df_X)

    return df_X