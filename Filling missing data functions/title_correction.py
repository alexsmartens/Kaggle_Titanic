def title_correction(df_X, titles_stat, print_satat=False):
    import numpy as np
    import pandas as pd

    from titles_descriptive_stat import titles_descriptive_stat
    from title_standardization_function import title_standardization_function

    # Assumed kid maximum age
    kid_max_age = 14

    # Unpacking stat by title information
    (train_df_descripion_by_title,
     titles_common_list_age_distr,
     titles_common_list_age_distr_male) = titles_stat

    if print_satat:
        print('There are {} of kids in training set'.format(((df_X.loc[:, 'Age'] <= kid_max_age) * 1).sum()))

    # Titles standardization (getting rid from title outliers like 'Col', 'Don', 'Lady' and etc.)
    df_X = df_X.apply(title_standardization_function,
                      titles_common_list_age_distr_male=titles_common_list_age_distr_male,
                      axis=1)

    df_X.loc[ (df_X['Age']<=kid_max_age) & (df_X['Sex']=='male'), 'Name_title' ] = 'Master'
    df_X.loc[ (df_X['Age']<=kid_max_age) & (df_X['Sex']=='female'), 'Name_title' ] = 'Girl'

    df_X.loc[ pd.isnull(df_X['Age']) & (df_X['Name_title']=='Master'), 'Age' ] = round(train_df_descripion_by_title.loc['Master', 'mean'].tolist()[0],2)

    if kid_max_age<=14:
        df_X.loc[pd.isnull(df_X['Age']) & (df_X['Name_title'] == 'Master'), 'Age'] = round(
            train_df_descripion_by_title.loc['Master', 'mean'].tolist()[0], 2)

    ## Adding kid identification feature
    #df_X.loc[:,'Kid'] = (df_X.loc[:,'Age']<=kid_max_age)*1

    return df_X







