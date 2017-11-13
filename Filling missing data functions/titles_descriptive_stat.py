def titles_descriptive_stat(train, np, print_satat=False):
    train_df_descripion_by_title = train.copy().groupby(['Name_title', 'Sex']).agg(['max', 'min','mean', 'median', 'count',np.std])
    train_df_descripion_by_title = train_df_descripion_by_title[[('Age','min'),
                                                                 ('Age','max'),
                                                                 ('Age','median'),
                                                                 ('Age','mean'),
                                                                 ('Age', 'std'),
                                                                 ('Fare','count')]]
    train_df_descripion_by_title.columns = train_df_descripion_by_title.columns.droplevel()

    if print_satat:
        print(np.round(train_df_descripion_by_title, decimals=1))

    # Summarizing the titles to the most common ones for future age prediction
    titles_common_list = ['Master', 'Miss', 'Mr', 'Mrs']
    titles_common_list_age_distr = train_df_descripion_by_title.copy().loc[titles_common_list]
    titles_common_list_age_distr.reset_index(inplace=True)
    titles_common_list_age_distr.set_index('Name_title',inplace=True)
    titles_common_list_age_distr_male = titles_common_list_age_distr.copy()[titles_common_list_age_distr['Sex']=='male']
    return [train_df_descripion_by_title, titles_common_list_age_distr, titles_common_list_age_distr_male]