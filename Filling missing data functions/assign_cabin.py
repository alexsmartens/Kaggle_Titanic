def assign_cabin(row,
                 ticket_list,
                 cabin_loc,
                 cabin_loc_allocation,
                 cabin_loc_availability,
                 train_1st_S,
                 train_1st_C,
                 train_1st_S_fare_stat,
                 train_1st_C_fare_stat,
                 deck_codes_rev,
                 np,
                 pd,
                 scipy):

    # number of neighbors for defining cabin X and Y in the 1st class
    num_neighbors = 3

    if type(row['Cabin']) is not str:
        ## Estimating cabin location if a ticket number has not been seen before
        if str(row['Ticket']) not in ticket_list.index:
            # Estimating cabiln location for  1st class passengers embarked at S(Southampton)
            # & assigning cabin number
            if (row['Pclass']==1) & (row['Embarked']=='S'):
                # Checking stat
                #print('--- S(Southampton)')
                #print(row['Fare'])
                #print(scipy.stats.norm(
                #    train_1st_S_fare_stat['mean'],
                #    train_1st_S_fare_stat['std']).pdf(row['Fare']))

                # Finding the most probable cabin deck
                new_deck_level = train_1st_S_fare_stat.copy().index[np.argmax(scipy.stats.norm(
                    train_1st_S_fare_stat['mean'],
                    train_1st_S_fare_stat['std']).pdf(row['Fare']))]
                #print(new_deck_level)

                # Finding N closest cabins by price on the chosen deck,
                # and X and Y based on the chosen neighbours eventually
                closest_neighbor_index = abs(train_1st_S.where(train_1st_S['Deck level']==new_deck_level)\
                                             .dropna()\
                                             .loc[:,'Fare']-row['Fare'])\
                                             .sort_values().index.tolist()[0:num_neighbors]
                new_X = train_1st_S.loc[closest_neighbor_index,'Room center longitude(X)'].sum()/\
                      len(train_1st_S.loc[closest_neighbor_index,'Room center longitude(X)'])
                new_Y = train_1st_S.loc[closest_neighbor_index, 'Room center latitude(Y)'].sum() / \
                        len(train_1st_S.loc[closest_neighbor_index, 'Room center latitude(Y)'])
                # adding a bit of random noise for distinguishing the locations
                # of group of passengers with the same ticket numbers
                new_X = new_X + np.random.uniform(-1.5, 1.5, 1)[0]
                new_Y = new_Y + np.random.uniform(-1.5, 1.5, 1)[0]

                room_code_new = deck_codes_rev[new_deck_level] + str(row['Pclass']) + '_port_S_' + str(row['Ticket'])
                row.set_value('Cabin', room_code_new)

            # Estimating cabiln location for  1st class passengers embarked at C(Cherbourg)
            # & assigning cabin number
            elif (row['Pclass']==1) & (row['Embarked']=='C'):
                # Checking stat
                #print('--- C(Cherbourg)')
                #print(row['Fare'])
                #print(scipy.stats.norm(
                #    train_1st_C_fare_stat['mean'],
                #    train_1st_C_fare_stat['std']).pdf(row['Fare']))

                # Finding the most probable cabin deck
                new_deck_level = train_1st_C_fare_stat.copy().index[np.argmax(scipy.stats.norm(
                    train_1st_C_fare_stat['mean'],
                    train_1st_C_fare_stat['std']).pdf(row['Fare']))]
                #print(new_deck_level)

                # Finding N closest cabins by price on the chosen deck,
                # and X and Y based on the chosen neighbours eventually
                closest_neighbor_index = abs(train_1st_C.where(train_1st_C['Deck level'] == new_deck_level) \
                                             .dropna() \
                                             .loc[:, 'Fare'] - row['Fare'])\
                                             .sort_values().index.tolist()[0:num_neighbors]
                new_X = train_1st_C.loc[closest_neighbor_index, 'Room center longitude(X)'].sum() / \
                        len(train_1st_C.loc[closest_neighbor_index, 'Room center longitude(X)'])
                new_Y = train_1st_C.loc[closest_neighbor_index, 'Room center latitude(Y)'].sum() / \
                        len(train_1st_C.loc[closest_neighbor_index, 'Room center latitude(Y)'])
                # adding a bit of random noise for distinguishing the locations
                # of group of passengers with the same ticket numbers
                new_X = new_X + np.random.uniform(-1.5, 1.5, 1)[0]
                new_Y = new_Y + np.random.uniform(-1.5, 1.5, 1)[0]

                room_code_new = deck_codes_rev[new_deck_level] + str(row['Pclass']) + '_port_C_' + str(row['Ticket'])
                row.set_value('Cabin', room_code_new)

            # Estimating cabiln location for  1st class passengers NOT embarked at C(Cherbourg) or at at S(Southampton)
            # & assigning cabin number
            elif (row['Pclass']==1) & (row['Embarked']!='S')& (row['Embarked']!='C'):
                print("\033[91m {}\033[00m"\
                        .format('!WARNING.1.3. The model is not designed to estimate cabin location for \n'
                                'passengers NOT embarked at C(Cherbourg) or at at S(Southampton)'))

            # assigning cabin numbers to 2nd and 3rd class passengers with unknown cabin numbers
            else:
                # choosing centroid for cabin allocation
                if row['Pclass']==2:
                    new_deck_level = np.random.randint(2,5,1)[0]
                    centroid_temp_name = (cabin_loc.copy().\
                                          loc[((pd.isnull(cabin_loc['Centroid_code']) == False)\
                                               &(cabin_loc['Class']==row['Pclass'])\
                                               &(cabin_loc['Deck level'] == new_deck_level)),'Centroid_code']).tolist()[0]
                elif row['Pclass']==3:
                    prob = np.random.uniform(0, 1, 1)[0]
                    if prob<0.25:
                        new_deck_level = 3
                        centroid_temp_name = '3cE'
                    elif prob<(0.25+0.1):
                        new_deck_level = 2
                        centroid_temp_name = '3cF2_neg'
                    elif prob < (0.25 + 0.1 + 0.27):
                        new_deck_level = 2
                        centroid_temp_name = '3cF1_pos'
                    elif prob < (0.25 + 0.1 + 0.27 + 0.19):
                        new_deck_level = 1
                        centroid_temp_name = '3cG2_neg'
                    else:
                        new_deck_level = 1
                        centroid_temp_name = '3cG1_pos'
                    #new_deck_level = np.random.randint(1, 2, 1)[0]
                else:
                    print("\033[91m {}\033[00m" \
                          .format('!WARNING.1.5. 1st class passengers are considered to be of 2nd/3rd class'))
                # selecting an appropriate centroid
                centroid_temp = cabin_loc.copy().loc[cabin_loc['Centroid_code']==centroid_temp_name]

                new_X = centroid_temp['Room center longitude(X)'].tolist()[0] \
                        + np.random.uniform(-1, 1, 1)[0] * centroid_temp['Centroid_square_half_length(X)'].tolist()[0]

                new_Y = centroid_temp['Room center latitude(Y)'].tolist()[0] \
                        + np.random.uniform(-1, 1, 1)[0] * centroid_temp['Centroid_square_half_width(Y)'].tolist()[0]
                room_code_new = (centroid_temp['Centroid_code'] + '_' + str(row['Ticket'])).tolist()[0]
                row.set_value('Cabin', room_code_new)


            ## Assigning new cabin code and its location to corresponding data bases
            new_cabin_values = pd.Series({'Deck level': new_deck_level,
                                          'Deck code': deck_codes_rev[new_deck_level],
                                          'Room center longitude(X)': new_X,
                                          'Room center latitude(Y)': new_Y,
                                          'Class': row['Pclass'] })
            cabin_loc.loc[room_code_new] = pd.concat([new_cabin_values, cabin_loc.iloc[2, 5:12]])
            avalability_temp = pd.Series({'Available': True,
                                          'Occupied_by_passengers': '',
                                          'Multiple_tickets ': False,
                                          'Occupies_multiple_cabins': 'False',
                                          'Multiple_units_No': False})
            cabin_loc_availability.loc[room_code_new] = pd.concat([new_cabin_values, avalability_temp])


        # passenger allocation by hes/her new cabin number
        row = cabin_loc_allocation(row,
                    ticket_list=ticket_list,
                    cabin_loc=cabin_loc,
                    cabin_loc_availability=cabin_loc_availability,
                    np=np)


    # Estimating cabiln location for passengers with only cabin level known (2nd and 3rd classes)
    # & assigning cabin number
    elif ((len(row['Cabin'])==1) & (row['Cabin']!='T')):
        # for passengers falling into levels with only one class zone
        if cabin_loc.where(cabin_loc['Class'] == row['Pclass']).dropna() \
                .loc[row['Cabin'], 'Amount_of_centroids'] == 1:
            # selecting an appropriate centorid
            centroid_temp = cabin_loc.where(cabin_loc['Class'] == row['Pclass']).dropna() \
                .loc[row['Cabin']]
            new_deck_level = centroid_temp['Deck level']
            new_X = centroid_temp['Room center longitude(X)'] \
                    + np.random.uniform(-1, 1, 1)[0] * centroid_temp['Centroid_square_half_length(X)']
            new_Y = centroid_temp['Room center latitude(Y)'] \
                    + np.random.uniform(-1, 1, 1)[0] * centroid_temp['Centroid_square_half_width(Y)']
            room_code_new = centroid_temp['Centroid_code'] + '_' + str(row['Ticket'])
            row.set_value('Cabin', room_code_new)
        else:
            print("\033[91m {}\033[00m" \
                  .format('!WARNING.1.4. The model is not designed to estimate cabin location for \n'
                          'passengers with known cabin level assigned to a two centroids location'))

        ## Assigning new cabin code and its location to corresponding data bases
        new_cabin_values = pd.Series({'Deck level': new_deck_level,
                                      'Deck code': deck_codes_rev[new_deck_level],
                                      'Room center longitude(X)': new_X,
                                      'Room center latitude(Y)': new_Y,
                                      'Class': row['Pclass']})
        cabin_loc.loc[room_code_new] = pd.concat([new_cabin_values, cabin_loc.iloc[2, 5:12]])
        avalability_temp = pd.Series({'Available': True,
                                      'Occupied_by_passengers': '',
                                      'Multiple_tickets ': False,
                                      'Occupies_multiple_cabins': 'False',
                                      'Multiple_units_No': False})
        cabin_loc_availability.loc[room_code_new] = pd.concat([new_cabin_values, avalability_temp])
        # passenger allocation by hes/her new cabin number
        row = cabin_loc_allocation(row,
                    ticket_list=ticket_list,
                    cabin_loc=cabin_loc,
                    cabin_loc_availability=cabin_loc_availability,
                    np=np)


    else:
        row = cabin_loc_allocation(row,
                    ticket_list=ticket_list,
                    cabin_loc=cabin_loc,
                    cabin_loc_availability=cabin_loc_availability,
                    np=np)
    return row