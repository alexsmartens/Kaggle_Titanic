def age_correction_assign_mean_by_title(df_X, titles_stat, print_satat=False):
    import numpy as np
    import pandas as pd
    (train_df_descripion_by_title,
     titles_common_list_age_distr,
     titles_common_list_age_distr_male) = titles_stat

    passengers_age_NA = df_X.loc[pd.isnull(df_X.loc[:,'Age'])].index.tolist()
    for ind in passengers_age_NA:
        # Assign age based on weighted local and global passenger age
        df_X.loc[ind, 'Age'] = round((train_df_descripion_by_title.loc[df_X.loc[ind,'Name_title'], 'mean'].tolist()[0] + \
                                      df_X.loc[range(ind-3, ind+3),'Age'].mean())/2, 2)
        if print_satat:
            print(df_X.loc[ind, 'Age'])
    return df_X