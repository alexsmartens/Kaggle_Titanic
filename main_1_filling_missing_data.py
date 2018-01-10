import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor

import os, sys
lib_path = os.path.abspath(os.path.join('Filling missing data functions'))
sys.path.append(lib_path)
from split_name import split_name
from titles_descriptive_stat import titles_descriptive_stat
from title_correction import title_correction
from cabin_check_by_ticket_No import cabin_check_by_ticket_No
from age_correction_by_group_feature import age_correction_by_group_feature
from ticket_standardization import ticket_standardization
from age_correction_assign_mean_by_title import age_correction_assign_mean_by_title
from cabin_existing_allocation import cabin_existing_allocation




# Loading TRAIN set
df_original = pd.read_csv('Data/titanic_train.csv')
df_original.set_index('PassengerId',inplace=True)
df_X = df_original.copy()

# df_X.drop('Survived', axis=1, inplace=True)
# df_Y = df_original.copy().loc[:,'Survived']

# Two passengers Embarked feature is corrected to be 'S' because
# - it is the most probable feature for 1st class passengers
# - these passengers ticket number seems to be in range of ticket numbers of 1st class passengers Embarked at 'S'
df_X.loc[[62,830],'Embarked'] = 'S'
# Age correction for one passenger based on the descriptive info
# The age is corrected according to title=Dr, equivalent to
# df_X.loc[767,'Age'] = train_df_descripion_by_title.loc[('Dr','male'),'median']
df_X.loc[767,'Age'] = 44
# A passenger ticket is corrected to explicitly be a part of a family with the same tickets
df_X.loc[873, 'Ticket']= 'PC 17755'
# df_X.loc[873, 'Family_member'] = 'nanny'
# Ticket number of four passengers corrected to be the same type as the rest tickets
df_X.loc[df_X.loc[:, 'Ticket']=='LINE', 'Ticket'] = '9999999'
df_X.loc[422, 'Ticket'] = 'AQ/5 13032'


# Splitting name by title, last name and first (other) name
df_X = df_X.apply(split_name, axis=1)





# Loading TEST set
df_original_test = pd.read_csv('Data/titanic_test.csv')
df_original_test.set_index('PassengerId',inplace=True)
df_X_test = df_original_test.copy()

# Filling a missing value based on a sample mean
df_X_test.loc[1044,'Fare'] = np.mean(df_X_test.loc[ (df_X_test['Pclass']==3) & (df_X_test['Embarked']=='S') ,'Fare'])






# Combining train and test for age/cabin feature values detection
df_X.loc[:,'Data set'] = 'train'
df_X_test.loc[:,'Data set'] = 'test'
df_X_combined = df_X.copy()
df_X_combined = df_X_combined.append(df_X_test.copy())

# Splitting name by title, last name and first (other) name
df_X_combined = df_X_combined.apply(split_name, axis=1)

# Separating Ticket number and series
df_X_combined.loc[:, 'Ticket_Series'] = ''
df_X_combined.loc[:, 'Ticket_No'] = 0
df_X_combined = df_X_combined.apply(ticket_standardization, axis=1)
df_X_combined = df_X_combined.sort_values(['Ticket_Series', 'Ticket_No'], ascending=[True, True])
df_X_combined['Ticket_combined'] = df_X_combined['Ticket_Series'] + ' ' + df_X_combined['Ticket_No'].astype(str)

# Filling age missing values
titles_stat_initial = titles_descriptive_stat(df_X_combined, print_satat=False)
df_X_combined = title_correction(df_X_combined, titles_stat_initial, print_satat=True)
titles_stat = titles_descriptive_stat(df_X_combined, print_satat=False)
# Filling cabin values for passengers having the same tickets
df_X_combined = cabin_check_by_ticket_No(df_X_combined)
df_X_combined = age_correction_by_group_feature(df_X_combined, 'Cabin', titles_stat_initial, print_satat=True)
df_X_combined = age_correction_by_group_feature(df_X_combined, 'Ticket_combined', titles_stat_initial, print_satat=True)
df_X_combined = age_correction_by_group_feature(df_X_combined, 'Name_last', titles_stat_initial, print_satat=True)
df_X_combined = age_correction_assign_mean_by_title(df_X_combined, titles_stat_initial, print_satat=False)
df_X_combined = cabin_existing_allocation(df_X_combined)

# Load the data set with ethnicity (country) pre-assigned and adding this info to the main data set
df_X_combined_w_ethnicity = pd.read_csv('Data/titanic_train_test_w_country.csv')
df_X_combined_w_ethnicity.set_index('PassengerId',inplace=True)
df_X_combined_ethnicity =  df_X_combined_w_ethnicity[['Country_origin','Country_origin_prob']]
df_X_combined_ethnicity = df_X_combined_ethnicity.rename(columns={'Country_origin': 'Ethnicity_origin',
                                                                  'Country_origin_prob': 'Ethnicity_prob'})
df_X_combined = pd.merge(df_X_combined, df_X_combined_ethnicity, how='left', left_index=True, right_index=True)





# Splitting and saving train and test sets
df_X_test_ready = df_X_combined.loc[df_X_combined.loc[:,'Data set']=='test']
df_X_test_ready = df_X_test_ready.drop('Data set', axis=1)
df_X_test_ready.drop('Survived', axis=1, inplace=True)
df_X_test_ready.to_csv('Data/titanic_test_READY.csv', sep=',', encoding='utf-8')

df_X_train_ready = df_X_combined.loc[df_X_combined.loc[:,'Data set']=='train']
df_X_train_ready = df_X_train_ready.drop('Data set', axis=1)
df_X_train_ready.to_csv('Data/titanic_train_READY.csv', sep=',', encoding='utf-8')

#df_X_combined.to_csv('Data/titanic_train_test_temp.csv', sep=',', encoding='utf-8')