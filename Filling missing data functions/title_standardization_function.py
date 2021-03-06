def title_standardization_function(row, titles_common_list_age_distr_male):
    import numpy as np
    import scipy.stats

    if row['Name_title'] not in ['Master', 'Miss', 'Mr', 'Mrs']:
        if row['Sex'] == 'female':
            row.set_value('Name_title', 'Miss')
        else:
            row.set_value( 'Name_title', titles_common_list_age_distr_male.index.tolist()[
                np.argmax(
                    scipy.stats.norm(
                        titles_common_list_age_distr_male['mean'],
                        titles_common_list_age_distr_male['std']).pdf(row['Age']))])
    return row
