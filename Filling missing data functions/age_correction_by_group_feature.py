def age_correction_by_group_feature(df_X, feature_name, titles_stat, print_satat=False):
    import pandas as pd
    import numpy as np

    # tracing amount of passengers with newly determined age
    if print_satat:
        n_passengers_age_assigned = 0
        list_passengers_age_assigned = []
    # Unpacking stat by title information
    (train_df_descripion_by_title,
     titles_common_list_age_distr,
     titles_common_list_age_distr_male) = titles_stat
    # Creating list of unique feature values
    feature_list = df_X.copy().loc[:, feature_name].unique()[pd.notnull(df_X.loc[:, feature_name].unique())].tolist()

    for feature_No in feature_list:
        df_feature_passengers = df_X.loc[df_X[feature_name] == feature_No]
        n_nanny = 0
        # Checking the chosen passengers have feature-mates
        if len(df_feature_passengers) > 1:
            # Checking if there are groups of passengers with known and unknown age who have the same feature
            if (sum((pd.isnull(df_feature_passengers['Age'])) * 1) > 0) & (
                sum((pd.notnull(df_feature_passengers['Age'])) * 1) > 0):
                # control variable to eliminate confusions with different families having the same last name
                proceed = False

                if feature_name == 'Name_last':
                    # finding groups of passengers with similar tickets
                    df_feature_passengers = df_feature_passengers.assign(Ticket_group = '')
                    ticket_group_tolerance = 25
                    for ind_t in df_feature_passengers.index.tolist()[0:-1]:
                        if pd.notnull(df_feature_passengers.loc[ind_t,'Ticket_group']):
                            df_feature_passengers.loc[ind_t, 'Ticket_group'] = ind_t
                            group_ticket_range = range(df_feature_passengers.loc[ind_t, 'Ticket_No'] - ticket_group_tolerance,
                                                       df_feature_passengers.loc[ind_t, 'Ticket_No'] + ticket_group_tolerance )
                            for ind_tt in df_feature_passengers.index[df_feature_passengers.index!=ind_t].tolist():
                                if df_feature_passengers.loc[ind_tt, 'Ticket_No'] in group_ticket_range:
                                    df_feature_passengers.loc[ind_tt, 'Ticket_group'] = ind_t

                    if len(df_feature_passengers.loc[:, 'Ticket_group'].unique())<len(df_feature_passengers.loc[:, 'Ticket'].unique()):
                        if len(df_feature_passengers.loc[:, 'Ticket_group'].unique())==1:
                            proceed = True
                else:
                    proceed = True


                if proceed:
                    # Summarizing feature-mates by SibSp and Parch  feature combination.
                    # SibSp and Parch values allow to identify whether a passenger is a kid or an adult and
                    # and assign more appropriate age
                    passengers_aggregated = df_feature_passengers.groupby(['SibSp', 'Parch']).agg(['mean', 'count'])
                    # Choose passengers with the specified SibSp and Parch values
                    feature_passengers_NA = df_feature_passengers.loc[(pd.isnull(df_feature_passengers['Age']))]
                    feature_passengers_not_NA = df_feature_passengers.loc[(pd.notnull(df_feature_passengers['Age']))]
                    feature_passengers_not_NA_no_nanny = \
                        feature_passengers_not_NA.loc[(pd.notnull(df_feature_passengers['Age']))]
                    for ind in feature_passengers_NA.index.tolist():
                        # Check whether a group of passengers with similar SibSp and Parch has a mean Age
                        if ~np.isnan(passengers_aggregated.loc[(df_X.loc[ind, 'SibSp'], df_X.loc[ind, 'Parch']), ('Age', 'mean')]):
                            # assigning mean age of a group with the same SibSp and Parch if a passenger age is unknown
                            df_X.loc[ind, 'Age'] = \
                                passengers_aggregated.loc[(df_X.loc[ind, 'SibSp'], df_X.loc[ind, 'Parch']), ('Age', 'mean')]
                            # Collecting stat
                            if print_satat:
                                n_passengers_age_assigned += 1
                                list_passengers_age_assigned.append(ind)
                        # checking if a passenger is nanny
                        elif (df_X.loc[ind, 'SibSp']==0) & (df_X.loc[ind, 'Parch']==0):
                            df_X.loc[ind, 'Age'] = \
                                np.round(train_df_descripion_by_title.loc[df_X.loc[ind, 'Name_title'], 'mean'], 1).tolist()[0]
                            n_nanny += 1

                            # Checking a family logical connection problem
                            if n_nanny>1:
                                print("\033[91m {}\033[00m" \
                                      .format('!WARNING. age_correction_by_group_feature\n'
                                              'More than one nanny is detected in a family'))
                            # Collecting stat
                            if print_satat:
                                n_passengers_age_assigned += 1
                                list_passengers_age_assigned.append(ind)
                        else:
                            if feature_passengers_not_NA_no_nanny['Age'].mean()<=14:
                                df_X.loc[ind, 'Age'] = np.round(train_df_descripion_by_title.loc['Mr', 'mean'].tolist()[0], 1)
                                # Collecting stat
                                if print_satat:
                                    n_passengers_age_assigned += 1
                                    list_passengers_age_assigned.append(ind)
                            elif df_X.loc[ind, 'Name_title'] == 'Mrs':
                                df_X.loc[ind, 'Age'] = np.round(train_df_descripion_by_title.loc['Mrs', 'mean'].tolist()[0], 1)
                                # Collecting stat
                                if print_satat:
                                    n_passengers_age_assigned += 1
                                    list_passengers_age_assigned.append(ind)
                            else:
                                df_X.loc[ind, 'Age'] = np.round(train_df_descripion_by_title.loc['Master', 'mean'].tolist()[0], 1)
                                # Collecting stat
                                if print_satat:
                                    n_passengers_age_assigned += 1
                                    list_passengers_age_assigned.append(ind)
                        # Title correction according to newly assigned age
                        for ind in feature_passengers_NA.index.tolist():
                            if ~np.isnan(df_X.loc[ind, 'Age'] <= 14):
                                if (df_X.loc[ind, 'Age'] <= 14) & (df_X.loc[ind, 'Sex'] == 'female'):
                                    df_X.loc[ind, 'Name_title'] = 'Girl'
                                elif (df_X.loc[ind, 'Age'] <= 14) & (df_X.loc[ind, 'Sex'] == 'male'):
                                    df_X.loc[ind, 'Name_title'] = 'Master'
                                elif ((df_X.loc[ind, 'Sex'] == 'female') & ((df_X.loc[ind, 'SibSp'] > 0) | (df_X.loc[ind, 'Parch'] >0)) ):
                                    df_X.loc[ind, 'Name_title'] = 'Mrs'

    # Printing stat
    if print_satat:
        print('Age was assigned to {} passengers by {} feature'.format(n_passengers_age_assigned,feature_name))
        # print(df_X.loc[list_passengers_age_assigned,['Name', 'Age', feature_name]])
    return df_X