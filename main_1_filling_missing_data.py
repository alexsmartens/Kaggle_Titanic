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
from title_standardization import title_standardization
from titles_descriptive_stat import titles_descriptive_stat
from age_detection_optional import age_detection_optional
from title_groups_visualisation_optional import title_groups_visualisation_optional
from age_prediction_by_one_title_optional import age_prediction_by_one_title
from age_prediction import age_predict
from ticket_list_initial_append import ticket_list_initial_append
from cabin_loc_allocation import cabin_loc_allocation
from plot3d_surv_by_loc import plot3d_surv_by_loc
from plot3d_loc_and_price import plot3d_loc_and_price
from class1_passengers_stat import class1_passengers_stat
from assign_cabin import assign_cabin


## Loading TRAIN data
print('0%   Start TRAIN pre-processing')
train_original = pd.read_csv('Data/titanic_train.csv')
# Setting passenger's Id
train_original.set_index('PassengerId',inplace=True)
train = train_original.copy()


# Pre-processing all features one at a time


## Name feature processing

# Splitting name by title, last name and first (other) name
train = train.apply(split_name, axis=1)

# Computing titles descriptive statistics based on passenger age
[train_df_descripion_by_title,
 titles_common_list_age_distr,
 titles_common_list_age_distr_male] = titles_descriptive_stat(train, np, print_satat=False)
## CONCLUSION: majority of the titles do not provide any additional information about the people. However, they might
# be used for more precise age detection
# *Mrs means that the woman in married, and Miss is referred to an unmarried woman.
print('4%   Title descriptive statistics is computed')

# * manual age correction for one instance based on the descriptive info
# The age is corrected according to title=Dr
train.loc[767,'Age'] = train_df_descripion_by_title.loc[('Dr','male'),'median']

# Titles standardization
train = train.apply(title_standardization,
                    titles_common_list_age_distr_male=titles_common_list_age_distr_male,
                    np=np,
                    scipy=scipy,
                    axis=1)
print('9%   Titles are standardized to be in the following range [Master, Miss, Mr, Mrs]')

# Optional Approaching ML models for age prediction
#age_detection_optional(train,pd, age_prediction_by_one_title)

## Optional scatter plots of passenger age by title groups
#title_groups_visualisation_optional(train,
#                                        pd,
#                                        plt,
#                                        Axes3D,
#                                        age_prediction_by_one_title,
#                                        preprocessing)

## Conclusion: assigning MEAN of a corresponding class to a passenger with unknown age
# produced the results comparable to ML models. So, MEAN prediction approach is chosen
# for age prediction

## Age prediction
age_data_known_indicator = pd.notnull(train.copy()['Age'])
age_to_ptredict = train[age_data_known_indicator==False]
train[age_data_known_indicator==False] = train[age_data_known_indicator==False].apply(age_predict,
                                                                                      np=np,
                                                                                      titles_common_list_age_distr=titles_common_list_age_distr,
                                                                                      axis=1)
print('14%  Age of the passengers with unknown age is identified')





## SibSp & Parch feature processing
# These features do not require pre-processing





## Embarked feature mining explanation
# Departure from
# - Southampton, UK -> 10 April 1912
# - Cherbourg, France -> 10 April 1912 (an hour and a half stop)
# - Queenstown, Ireland (now - Cobh) -> 12 April 1912 (two hours stop)





## Cabin feature mining explanation

# Cabin numbers are used for locating on the ship
# *acc to http://www.titanichg.com/newyorkreturn/

# 'T' - Boat Deck
# 'A' - Promenade Deck (Upper Deck)
# 'B' - Bridge Deck
# 'C' - Shelter Deck
# 'D' - Saloon Deck
# 'E' - Upper Deck
# logical continuation of decks codding based on https://www.encyclopedia-titanica.org/titanic-deckplans/profile.html:
# 'F' - Middle Deck
# 'G' - Lower Deck
# 'E' - Orlop Deck





## Ticket
# Tickets are used for grouping the passengers with the same cabin to the same locations





## Estimating passenger locations on the ship

# loading location map of titanic cabins
cabin_loc = pd.read_csv('Data/titanic_cabin_location.csv')
cabin_loc.set_index('Room code', inplace=True)

# creating cabin availability list for passenger accounting by cabin and
# for quick cabin stat generation if needed
cabin_loc_availability = cabin_loc.copy()
cabin_loc_availability['Available'] = True
cabin_loc_availability['Occupied_by_passengers'] = ''
cabin_loc_availability['Multiple_tickets'] = False
cabin_loc_availability['Occupies_multiple_cabins'] = False
cabin_loc_availability['Multiple_units_No'] = ''

# creating ticket list for assigning cabins for the passengers without cabin number known but
# whose cabin number can be found by ticket
ticket_list = pd.DataFrame(columns=['Cabin','Multiple_cabins'])
train.apply(ticket_list_initial_append,ticket_list=ticket_list, axis=1)
print('18%  Ticket list is created based on the passengers with known cabin number')

# Passengers with known unambiguous cabin number allocation by hes/her cabin number
train = train.apply(cabin_loc_allocation,
                    ticket_list=ticket_list,
                    cabin_loc=cabin_loc,
                    cabin_loc_availability=cabin_loc_availability,
                    np=np,
                    axis=1)
print('      {} passengers out of {} are allocated'.format(\
    (train['Room center longitude(X)'].isnull()==False).sum(), len(train) ) )
print('23%  Passengers with known unambiguous cabin number are allocated:')

## Scatterplot of survival by cabin location
#plot3d_surv_by_loc(train,
#                   plt,
#                   Axes3D)
## Conclusion: I do not find any obvious dependence between location and survival

## Scatterplot of fare by cabin location and by port of embarkation scatterplot
plot3d_loc_and_price(train,
                     plt,
                     Axes3D,
                     np)
# Conclusion: 1st class passengers embarked at S(Southampton) and C(Cherbourg) demonstrate different location and price
#  patterns. However, there are only 4 passengers embarked at Queenstown with known cabin codes, which is not enough
# to see any location patten
# **Note: this cabin data is only representative for 1st class passengers embarked at S(Southampton) and C(Cherbourg)

# Titanic passengers' by class location information allows
# approximate XYZ passengers location only based on their class
# look at the corresponding figure for reference



## Allocation of passengers with unknown cabins
deck_codes = {'A':7, 'B':6, 'C':5, 'D':4, 'E':3, 'F':2, 'G':1}
deck_codes_rev = {7:'A', 6:'B', 5:'C', 4:'D', 3:'E', 2:'F', 1:'G'}

[train_1st_S, train_1st_C, train_1st_S_fare_stat, train_1st_C_fare_stat] = class1_passengers_stat (train,np,pd)
print('26%  1st class passengers allocation statistics is computed')

# Allocation of passengers with unknown cabin numbers
train = train.apply(assign_cabin,
                    ticket_list=ticket_list,
                    cabin_loc=cabin_loc,
                    cabin_loc_allocation=cabin_loc_allocation,
                    cabin_loc_availability=cabin_loc_availability,
                    train_1st_S=train_1st_S,
                    train_1st_C=train_1st_C,
                    train_1st_S_fare_stat=train_1st_S_fare_stat,
                    train_1st_C_fare_stat=train_1st_C_fare_stat,
                    deck_codes_rev=deck_codes_rev,
                    np=np,
                    pd=pd,
                    scipy=scipy,
                    axis=1)
print('68%  TRAIN pre-processing is completed')
train.to_csv('Data/titanic_train_READY.csv', sep=',', encoding='utf-8')
plot3d_surv_by_loc(train,
                   plt,
                   Axes3D)










# Start TEST pre-processing
print('69%   Start TEST pre-processing')
test_original = pd.read_csv('Data/titanic_test.csv')
test_original.set_index('PassengerId',inplace=True)
test = test_original.copy()

test = test.apply(split_name, axis=1)
test = test.apply(title_standardization,
                    titles_common_list_age_distr_male=titles_common_list_age_distr_male,
                    np=np,
                    scipy=scipy, axis=1)



print('71%  Titles are standardized to be in the following range [Master, Miss, Mr, Mrs]')

test[pd.notnull(test.copy()['Age'])==False] = test[pd.notnull(test.copy()['Age'])==False].apply(age_predict,
                                                                                                np=np,
                                                                                                titles_common_list_age_distr=titles_common_list_age_distr,
                                                                                                axis=1)
print('73%  Age of the passengers with unknown age is identified')

test.apply(ticket_list_initial_append,ticket_list=ticket_list, axis=1)
print('75%  Ticket list is created based on the passengers with known cabin number')

test = test.apply(cabin_loc_allocation,
                    ticket_list=ticket_list,
                    cabin_loc=cabin_loc,
                    cabin_loc_availability=cabin_loc_availability,
                    np=np,
                    axis=1)
print('77%  Passengers with known unambiguous cabin number are allocated:')
print('      {} passengers out of {} are allocated'.format(\
    (test['Room center longitude(X)'].isnull()==False).sum(), len(test) ) )

test = test.apply(assign_cabin,
                    ticket_list=ticket_list,
                    cabin_loc=cabin_loc,
                    cabin_loc_allocation=cabin_loc_allocation,
                    cabin_loc_availability=cabin_loc_availability,
                    train_1st_S=train_1st_S,
                    train_1st_C=train_1st_C,
                    train_1st_S_fare_stat=train_1st_S_fare_stat,
                    train_1st_C_fare_stat=train_1st_C_fare_stat,
                    deck_codes_rev=deck_codes_rev,
                    np=np,
                    pd=pd,
                    scipy=scipy,
                    axis=1)
test.to_csv('Data/titanic_test_READY.csv', sep=',', encoding='utf-8')
print('100%  TEST pre-processing is completed')