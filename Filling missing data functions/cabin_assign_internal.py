def cabin_assign_internal(df_X):
    import numpy as np
    import pandas as pd
    import scipy

    import os, sys
    lib_path = os.path.abspath(os.path.join(''))
    sys.path.append(lib_path)
    from class1_passengers_stat import class1_passengers_stat


    ticket_1st_Age_NA = df_X.loc[
        pd.isnull(df_X.loc[:, 'Cabin']) & (df_X.loc[:, 'Pclass'] == 1), 'Ticket_combined'].unique().tolist()
    [dict_passengers_1st_Embarced, dict_passengers_1st_Embarced_stat] = class1_passengers_stat(df_X)

    # number of neighbors for defining cabin X and Y in the 1st class
    num_neighbors = 3

    # Assigning cabin location coordinates to 1st class passengers with unknown cabin number
    for ticket_No in ticket_1st_Age_NA:
        current_Fare = df_X.loc[df_X.loc[:, 'Ticket_combined'] == ticket_No, 'Fare'].mean()
        current_Embarked = df_X.loc[df_X.loc[:, 'Ticket_combined'] == ticket_No, 'Embarked'].tolist()[0]

        current_1st_fare_stat = dict_passengers_1st_Embarced_stat[current_Embarked]
        current_1st_set = dict_passengers_1st_Embarced[current_Embarked]

        # Finding the most probable cabin deck
        new_deck_level = current_1st_fare_stat.copy().index[np.argmax(scipy.stats.norm(
            current_1st_fare_stat['mean'],
            current_1st_fare_stat['std']).pdf(current_Fare))]
        # print(new_deck_level)

        # Finding N closest cabins by price on the chosen deck,
        # and X and Y based on the chosen neighbours eventually
        closest_neighbor_index = abs(current_1st_set.where(current_1st_set['Cabin_deck_level'] == new_deck_level) \
                                     .dropna() \
                                     .loc[:, 'Fare'] - current_Fare) \
                                     .sort_values().index.tolist()[0:num_neighbors]
        new_X = current_1st_set.loc[closest_neighbor_index, 'Cabin_X'].sum() / \
                len(current_1st_set.loc[closest_neighbor_index, 'Cabin_X'])
        new_Y = current_1st_set.loc[closest_neighbor_index, 'Cabin_Y'].sum() / \
                len(current_1st_set.loc[closest_neighbor_index, 'Cabin_Y'])
        # adding a bit of random noise for distinguishing the locations
        # of group of passengers with the same ticket numbers
        new_X = new_X + np.random.uniform(-1.5, 1.5, 1)[0]
        new_Y = new_Y + np.random.uniform(-1.5, 1.5, 1)[0]
        room_code_new = '_1st_' + '_port_S_' + str(ticket_No)

        df_X.loc[df_X.loc[:, 'Ticket_combined'] == ticket_No, ['Cabin', 'Cabin_deck_level', 'Cabin_X', 'Cabin_Y']] = \
            [room_code_new, new_deck_level, round(new_X, 2), round(new_Y, 2)]
        # print([room_code_new, new_deck_level, round(new_X, 2), round(new_Y, 2)])


    # Assigning cabin location coordinates to 2nd and 3rd class passengers
    df_X.loc[df_X.loc[:, 'Pclass'] == 2, ['Cabin', 'Cabin_deck_level', 'Cabin_X', 'Cabin_Y']] = \
        ['2nd_centroid_2cF', 2, -85, 0]
    df_X.loc[df_X.loc[:, 'Pclass'] == 3, ['Cabin', 'Cabin_deck_level', 'Cabin_X', 'Cabin_Y']] = \
        ['2rd_centroid_3cF1_pos', 2, 85, 0]

    return df_X