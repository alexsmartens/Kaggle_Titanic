def age_predict(row ,np, titles_common_list_age_distr):
    if np.isnan(row['Age']):
        row.set_value('Age', titles_common_list_age_distr['mean'].loc[row['Name_title']])
    return row