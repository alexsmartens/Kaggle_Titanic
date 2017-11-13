import pandas as pd
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.tree import DecisionTreeRegressor



# Start TRAIN pre-processing
print('0%   Start TRAIN pre-processing')
train_original = pd.read_csv('titanic_train.csv')
# Setting passenger's Id
train_original.set_index('PassengerId',inplace=True)
train = train_original.copy()

## Name feature processing
def split_name(row):
    name_list1 = row['Name'].split(', ')
    row['Name_last'] = name_list1[0]

    name_list2 = name_list1[1].split('. ')
    row['Name_title'] = name_list2[0]
    row['Name_other'] = name_list2[1]
    return row
train = train.apply(split_name, axis=1)

#print(train['Name_title'].unique())
#train.to_csv('titanic_train__names_titles.csv', sep=',', encoding='utf-8')

# Titles descriptive statistics
train_df_descripion_by_title = train.copy().groupby(['Name_title', 'Sex']).agg(['max', 'min','mean', 'median', 'count',np.std])
train_df_descripion_by_title = train_df_descripion_by_title[[('Age','min'),
                                                             ('Age','max'),
                                                             ('Age','median'),
                                                             ('Age','mean'),
                                                             ('Age', 'std'),
                                                             ('Fare','count')]]
train_df_descripion_by_title.columns = train_df_descripion_by_title.columns.droplevel()
#print(np.round(train_df_descripion_by_title, decimals=1))

## CONCLUSION: majority of the titles do not provide any additional information about the people. However, they might
# be used for more precise age detection
# *Mrs means that the woman in married, and Miss is referred to an unmarried woman.



## Age features processing -> missing values -> fitting Naïve Bayes for age prediction

# Summarizing the titles to the most common ones for future age prediction
titles_common_list = ['Master', 'Miss', 'Mr', 'Mrs']
titles_common_list_age_distr = train_df_descripion_by_title.copy().loc[titles_common_list]
titles_common_list_age_distr.reset_index(inplace=True)
titles_common_list_age_distr.set_index('Name_title',inplace=True)
titles_common_list_age_distr_male = titles_common_list_age_distr.copy()[titles_common_list_age_distr['Sex']=='male']
#print(titles_common_list_age_distr)
print('4%   Title descriptive statistics is computed')


# * manual age correction for one instance based on the descriptive info
# The age is corrected according to title=Dr
train.loc[767,'Age'] = train_df_descripion_by_title.loc[('Dr','male'),'median']


# Titles standardization
def titles_standardization(row):
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

train = train.apply(titles_standardization,axis=1)
print('9%   Titles are standardized to be in the following range [Master, Miss, Mr, Mrs]')
#train.to_csv('titanic_train_3_titles_standardization.csv', sep=',', encoding='utf-8')


# Preparing train and test sets for edge detection
age_features = ['SibSp','Parch','Name_title'] # 'Sex' is not relevant, is it is considered in 'Name_title'

age_data_known_indicator = pd.notnull(train.copy()['Age'])
#print(age_data_known_indicator.sum()) #number of people with known age CHECK
age_data = train.copy()[age_data_known_indicator]

X_age = age_data[age_features]
y_age = age_data['Age']

# Initiating of regressors for model fitting
reg_dt_d2 = DecisionTreeRegressor(max_depth=2)
reg_dt_d5 = DecisionTreeRegressor(max_depth=5)
reg_dt_d10 = DecisionTreeRegressor(max_depth=10)
reg_gbm = GradientBoostingRegressor()
reg_lm = LinearRegression()
reg_lm_lasso = Lasso()
reg_dum_mean = DummyRegressor(strategy='mean')
reg_dum_median = DummyRegressor(strategy='median')

def age_prediction_by_one_tytle(X,y,scoring_method='neg_mean_absolute_error'):
    cv_scores_reg_dt_d2 = cross_val_score(reg_dt_d2, X, y, cv=5, scoring=scoring_method)
    print('Cross-validation scores of DecisionTreeRegressor(max_depth=2): ', cv_scores_reg_dt_d2)
    print('Mean cross-validation score of DecisionTreeRegressor(max_depth=2): ', np.mean(cv_scores_reg_dt_d2))

    cv_scores_reg_dt_d5 = cross_val_score(reg_dt_d5, X, y, cv=5, scoring=scoring_method)
    print('Cross-validation scores of DecisionTreeRegressor(max_depth=5): ', cv_scores_reg_dt_d5)
    print('Mean cross-validation score of DecisionTreeRegressor(max_depth=5): ', np.mean(cv_scores_reg_dt_d5))

    cv_scores_reg_dt_d10 = cross_val_score(reg_dt_d10, X, y, cv=5, scoring=scoring_method)
    print('Cross-validation scores of DecisionTreeRegressor(max_depth=10): ', cv_scores_reg_dt_d10)
    print('Mean cross-validation score of DecisionTreeRegressor(max_depth=10): ', np.mean(cv_scores_reg_dt_d10))

    cv_scores_reg_lm = cross_val_score(reg_lm, X, y, cv=5, scoring=scoring_method)
    print('Cross-validation scores of LinearRegression: ', cv_scores_reg_lm)
    print('Mean cross-validation score of LinearRegression: ', np.mean(cv_scores_reg_lm))

    cv_scores_reg_lasso = cross_val_score(reg_lm_lasso, X, y , cv=5, scoring=scoring_method)
    print('Cross-validation scores of Lasso: ', cv_scores_reg_lasso)
    print('Mean cross-validation score of Lasso: ', np.mean(cv_scores_reg_lasso))

    cv_scores_reg_dum_mean = cross_val_score(reg_dt_d2, X, y, cv=5, scoring=scoring_method)
    print('Cross-validation scores of reg_dum_mean: ', cv_scores_reg_dum_mean)
    print('Mean cross-validation score of reg_dum_mean: ', np.mean(cv_scores_reg_dum_mean))

    cv_scores_reg_dum_median = cross_val_score(reg_dum_median, X, y, cv=5, scoring=scoring_method)
    print('Cross-validation scores of reg_dum_median: ', cv_scores_reg_dum_median)
    print('Mean cross-validation score of reg_dum_median: ', np.mean(cv_scores_reg_dum_median))

    print('Mean models fit: ', np.round([np.mean(cv_scores_reg_dt_d2),
                                np.mean(cv_scores_reg_dt_d5),
                                np.mean(cv_scores_reg_dt_d10),
                                np.mean(cv_scores_reg_lm),
                                np.mean(cv_scores_reg_lasso),
                                np.mean(cv_scores_reg_dum_mean),
                                np.mean(cv_scores_reg_dum_median)],3))
    return None



## SCATTER PLOTS of known passengers by title groups

## Training 3D scatter plot: 'Master' Age dependance based on SibSp and Parch
#fig = plt.figure()
#ax = Axes3D(fig)
#ax.scatter(X_age[X_age['Name_title']=='Master'].loc[:,'SibSp'],
#           X_age[X_age['Name_title']=='Master'].loc[:,'Parch'],
#           y_age[X_age['Name_title']=='Master'])
#ax.set_xlabel('SibSp')
#ax.set_ylabel('Parch')
#ax.set_zlabel('Age')
#plt.show()
## Conclusion(plot): 'Master' gives enough information to make a conclusion about a person age on its own,
## fitting a distribution to this data chunk is likely to have a lot of variation form logic perspective.
## That's why I propose using median age of known people for 'Master's age prediction

#X_age_Master = X_age.copy()[X_age['Name_title']=='Master'].loc[:,['SibSp', 'Parch']]
#y_age_Master = y_age.copy()[X_age['Name_title']=='Master']
#print('')
#print(len(X_age_Master))
#age_prediction_by_one_tytle(X_age_Master, y_age_Master)
#print('')
#age_prediction_by_one_tytle(X_age_Master, y_age_Master, 'neg_mean_squared_log_error')
#print('')
#age_prediction_by_one_tytle(preprocessing.scale(X_age_Master),preprocessing.scale(y_age_Master))
#print('')
#print('')
# Conclusion(model fitting): use MEDIAN age for Masters age prediction. However, MEAN would work fine as well



## Training 3D scatter plot: 'Mr' Age dependance based on SibSp and Parch
#fig = plt.figure()
#ax = Axes3D(fig)
#ax.scatter(X_age[X_age['Name_title']=='Mr'].loc[:,'SibSp'],
#           X_age[X_age['Name_title']=='Mr'].loc[:,'Parch'],
#           y_age[X_age['Name_title']=='Mr'])
#ax.set_xlabel('SibSp')
#ax.set_ylabel('Parch')
#ax.set_zlabel('Age')
#plt.show()
## Conclusion (plot): general trend for 'Mr' is the higher 'Prach' number | the higher 'SibSp' the lower age.
## Prediction model fitting might be usefull

#X_age_Mr = X_age.copy()[X_age['Name_title'] == 'Mr'].loc[:, ['SibSp', 'Parch']]
#y_age_Mr = y_age.copy()[X_age['Name_title'] == 'Mr']
#print(len(X_age_Mr))
#age_prediction_by_one_tytle(X_age_Mr, y_age_Mr)
#print('')
#age_prediction_by_one_tytle(X_age_Mr, y_age_Mr, 'neg_mean_squared_log_error')
#print('')
#age_prediction_by_one_tytle(preprocessing.scale(X_age_Mr), preprocessing.scale(y_age_Mr))
#print('')
#print('')
## Conclusion(model fitting): use MEDIAN age for Mr_s age prediction. However, MEAN would work fine as well




## Training 3D scatter plot: 'Mrs' Age dependance based on SibSp and Parch
#fig = plt.figure()
#ax = Axes3D(fig)
#ax.scatter(X_age[X_age['Name_title']=='Mrs'].loc[:,'SibSp'],
#           X_age[X_age['Name_title']=='Mrs'].loc[:,'Parch'],
#           y_age[X_age['Name_title']=='Mrs'])
#ax.set_xlabel('SibSp')
#ax.set_ylabel('Parch')
#ax.set_zlabel('Age')
#plt.show()
## Conclusion (plot): general trend for 'Mrs' is
## the higher 'Parch' number the less SD from the mean, with approximately the same mean age
## having one 'SibSp' vs zero 'SibSp' decreases distribution mean
## Prediction model fitting might be usefull

#X_age_Mrs = X_age.copy()[X_age['Name_title'] == 'Mrs'].loc[:, ['SibSp', 'Parch']]
#y_age_Mrs = y_age.copy()[X_age['Name_title'] == 'Mrs']
#print(len(X_age_Mrs))
#age_prediction_by_one_tytle(X_age_Mrs, y_age_Mrs)
#print('')
#age_prediction_by_one_tytle(X_age_Mrs, y_age_Mrs, 'neg_mean_squared_log_error')
#print('')
#age_prediction_by_one_tytle(preprocessing.scale(X_age_Mrs), preprocessing.scale(y_age_Mrs))
#print('')
#print('')
## Conclusion(model fitting): use MEAN age for Mrs_s age prediction. However, MEAN would work fine as well



# Training 3D scatter plot: 'Miss' Age dependance based on SibSp and Parch
# fig = plt.figure()
# ax = Axes3D(fig)
#ax.scatter(X_age[X_age['Name_title']=='Miss'].loc[:,'SibSp'],
#           X_age[X_age['Name_title']=='Miss'].loc[:,'Parch'],
#           y_age[X_age['Name_title']=='Miss'])
#ax.set_xlabel('SibSp')
#ax.set_ylabel('Parch')
#ax.set_zlabel('Age')
#plt.show()
## Conclusion (plot): general trend for 'Mrs' is
## having one or more 'Parch' dramatically change the age distribution mean comparatively to 'Parch' of zero
## the higher number of 'SibSp' the less Age is
## Prediction model fitting should be usefull

#X_age_Miss = X_age.copy()[X_age['Name_title'] == 'Miss'].loc[:, ['SibSp', 'Parch']]
#y_age_Miss = y_age.copy()[X_age['Name_title'] == 'Miss']
#print(len(X_age_Miss))
#age_prediction_by_one_tytle(X_age_Miss, y_age_Miss)
#print('')
#age_prediction_by_one_tytle(X_age_Miss, y_age_Miss, 'neg_mean_squared_log_error')
#print('')
#age_prediction_by_one_tytle(X_age_Miss, y_age_Miss)
#print('')
#age_prediction_by_one_tytle(preprocessing.scale(X_age_Miss), preprocessing.scale(y_age_Miss))
#print('')
#print('')
## Conclusion(model fitting): use MEAN age for Misses age prediction

#print('---------------- >>>>>>>>> General Age Model Fitting ----------------')

# Training 3D scatter plot: Age dependance based on Name_title, SibSp and Parch
#colors_dict = {'Master':'blue', 'Mr':'black', 'Mrs':'red', 'Miss':'pink'}
#fig = plt.figure()
#ax = Axes3D(fig)
#ax.scatter(X_age['SibSp'],
#           X_age['Parch'],
#           y_age,
#           c=X_age['Name_title'].apply(lambda x: colors_dict[x]))
#ax.set_xlabel('SibSp')
#ax.set_ylabel('Parch')
#ax.set_zlabel('Age')
#plt.show()

## Fit regression model for age prediction on full data set
# Categorical features transformation
#X_age_transformed = pd.get_dummies(X_age.copy().select_dtypes(include=[object]))
#X_age_transformed = pd.merge(X_age_transformed, X_age[['SibSp','Parch']],
#                             left_index=True, right_index=True)
#
#age_prediction_by_one_tytle(X_age_transformed, y_age, 'neg_mean_squared_log_error')
#print('')
#age_prediction_by_one_tytle(X_age_transformed, y_age)
#print('')
#age_prediction_by_one_tytle(preprocessing.scale(X_age_transformed), preprocessing.scale(y_age))

## Conclusion: the best effective mehod so far is to assign MEAN of a corresponding class to
# instances with unknown age

## Age prediction
age_to_ptredict = train[age_data_known_indicator==False]
#print(age_to_ptredict)
def age_predict(row):
    if np.isnan(row['Age']):
        row.set_value('Age', titles_common_list_age_distr['mean'].loc[row['Name_title']])
    return row

train[age_data_known_indicator==False] = train[age_data_known_indicator==False].apply(age_predict, axis=1)
print('14%  Age of the passengers with unknown age is identified')
#train.to_csv('titanic_train_4_age_prediction.csv', sep=',', encoding='utf-8')





## SibSp & Parch feature processing
# These features do not require processing at this point of time





## Embarked
# Departure from
# - Southampton, UK -> 10 April 1912
# - Cherbourg, France -> 10 April 1912 (an hour and a half stop)
# - Queenstown, Ireland (now - Cobh) -> 12 April 1912 (two hours stop)





## Cabin

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
cabin_loc = pd.read_csv('titanic_rooms_location.csv')
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

def ticket_list_initial_append(row):
    # selecting passengers with defined cabin number
    if type(row['Cabin']) is str:
        # discarding ambiguous passengers with known only deck level but not cabin number
        if (len(row['Cabin'])>1) | (row['Cabin']=='T'):
            # appending ticket_list
            if str(row['Ticket']) not in ticket_list.index:
                ticket_list.loc[str(row['Ticket'])] = [row['Cabin'], False]
            else:
                if row['Cabin'] not in ticket_list.loc[str(row['Ticket']), 'Cabin']:
                    ticket_list.loc[str(row['Ticket']), 'Multiple_cabins'] = True
                    ticket_list.loc[str(row['Ticket']), 'Cabin'] = [ticket_list.loc[str(row['Ticket']), 'Cabin'],
                                                                    row['Cabin']]
train.apply(ticket_list_initial_append,axis=1)
print('18%  Ticket list is created based on the passengers with known cabin number')

# Passengers with known unambiguous cabin number allocation by hes/her cabin number
def cabin_loc_allocation(row):
    # assigning cabin numbers by tickets where this is possible
    if type(row['Cabin']) is not str:
        if str(row['Ticket']) in ticket_list.index:
            if isinstance(ticket_list.loc[str(row['Ticket']), 'Cabin'], list):
                row.loc['Cabin'] = ticket_list.loc[row.loc['Ticket'], 'Cabin'][0]
            else:
                row.loc['Cabin'] = ticket_list.loc[row.loc['Ticket'], 'Cabin']

    # selecting passengers with defined cabin number
    if type(row['Cabin']) is str:
        # discarding ambiguous passengers with known only deck level but not cabin number
        if (len(row['Cabin'])>1) | (row['Cabin']=='T'):
            # passengers occupying only one cabin
            if row['Cabin'] in cabin_loc.index.tolist():
                row.set_value('Room center longitude(X)', cabin_loc.loc[row['Cabin'],'Room center longitude(X)'] )
                row.set_value('Room center latitude(Y)', cabin_loc.loc[row['Cabin'], 'Room center latitude(Y)'] )
                row.set_value('Deck level', cabin_loc.loc[row['Cabin'], 'Deck level'] )
                # adding cabin availability info
                if cabin_loc_availability.loc[row['Cabin'],'Available'] == True:
                    cabin_loc_availability.loc[row['Cabin'],'Available'] = False
                    cabin_loc_availability.loc[row['Cabin'],'Occupied_by_passengers'] = [row.name]
                    cabin_loc_availability.loc[row['Cabin'],'Ticket'] = [str(row['Ticket'])]
                else:
                    cabin_loc_availability.loc[row['Cabin'], 'Occupied_by_passengers'].append(row.name)
                    if str(row['Ticket']) not in cabin_loc_availability.loc[row['Cabin'],'Ticket']:
                        cabin_loc_availability.loc[row['Cabin'],'Ticket'].append(str(row['Ticket']))
            else:
                # passengers occupying multiple cabins
                if row['Cabin'].split(' ')[0] in cabin_loc.index.tolist():
                    # assigning coordinates of "multiple cabin" units
                    mean_loc = np.mean(cabin_loc.loc[row['Cabin'].split(' ')])
                    row.set_value('Room center longitude(X)', mean_loc['Room center longitude(X)'])
                    row.set_value('Room center latitude(Y)', mean_loc['Room center latitude(Y)'] )
                    row.set_value('Deck level', mean_loc['Deck level'])
                    # adding cabin availability info for each unit number
                    for cab_no in row['Cabin'].split(' '):
                        if cabin_loc_availability.loc[cab_no, 'Available'] == True:
                            cabin_loc_availability.loc[cab_no, 'Available'] = False
                            cabin_loc_availability.loc[cab_no, 'Occupied_by_passengers'] = [row.name]
                            cabin_loc_availability.loc[cab_no, 'Ticket'] = [str(row['Ticket'])]
                        else:
                            cabin_loc_availability.loc[cab_no, 'Occupied_by_passengers'].append(row.name)
                            if str(row['Ticket']) not in cabin_loc_availability.loc[cab_no, 'Ticket']:
                                cabin_loc_availability.loc[cab_no, 'Ticket'].append(str(row['Ticket']))
                        cabin_loc_availability.loc[cab_no, 'Occupies_multiple_cabins'] = True
                        cabin_loc_availability.loc[cab_no, 'Multiple_units_No'] = row['Cabin']
    return row

train = train.apply(cabin_loc_allocation, axis=1)
print('23%  Passengers with known unambiguous cabin number are allocated:')
print('      {} passengers out of {} are allocated'.format(\
    (train['Room center longitude(X)'].isnull()==False).sum(), len(train) ) )
#train.to_csv('titanic_train_5_cabins_1.csv', sep=',', encoding='utf-8')
#cabin_loc_availability.to_csv('titanic_train_6_cabin_loc_availability.csv', sep=',', encoding='utf-8')
#ticket_list.to_csv('titanic_train_7_ticket_list.csv', sep=',', encoding='utf-8')

## Survival by cabin location scatterplot
def plot3d_surv_by_loc(train):
    fig = plt.figure()
    ax = Axes3D(fig)
    Y_series = ax.scatter(train.loc[train['Survived']==1, 'Room center longitude(X)'],
                          train.loc[train['Survived']==1, 'Room center latitude(Y)'],
                          train.loc[train['Survived']==1, 'Deck level'],
                          c='green')
    N_series = ax.scatter(train.loc[train['Survived'] == 0, 'Room center longitude(X)'],
                          train.loc[train['Survived'] == 0, 'Room center latitude(Y)'],
                          train.loc[train['Survived'] == 0, 'Deck level'],
                          c='red')
    ax.set_xlabel('Room center longitude(X)')
    ax.set_ylabel('Room center latitude(Y)')
    ax.set_zlabel('Deck level')
    ax.set_title('Survival by cabin location')
    plt.legend([Y_series, N_series],
               ['Survived', 'Not survived'],
               title='Legend',
               loc=6)
    plt.show()
#plot3d_surv_by_loc(train)
## Conclusion: I do not find any obvious dependence between location and survival

## Price by cabin location and by port of embarkation scatterplot
def plot3d_loc_and_price(train):
    colors_dict = {'S':'red', 'C':'#FFA500', 'Q':'green',np.nan:'black'}
    fig = plt.figure()
    ax = Axes3D(fig)
    S_index = train.loc[train['Embarked']=='S', 'Room center longitude(X)'].dropna().index
    S_series = ax.scatter(train.loc[S_index, 'Room center longitude(X)'],
               train.loc[S_index, 'Room center latitude(Y)'],
               train.loc[S_index, 'Deck level'],
               c='red',
               s=train.loc[S_index, 'Fare'])
    C_index = train.loc[train['Embarked']=='C', 'Room center longitude(X)'].dropna().index
    C_series = ax.scatter(train.loc[C_index, 'Room center longitude(X)'],
               train.loc[C_index, 'Room center latitude(Y)'],
               train.loc[C_index, 'Deck level'],
               c='#FFA500',
               s=train.loc[C_index, 'Fare'])
    Q_index = train.loc[train['Embarked']=='Q', 'Room center longitude(X)'].dropna().index
    Q_series = ax.scatter(train.loc[Q_index, 'Room center longitude(X)'],
               train.loc[Q_index, 'Room center latitude(Y)'],
               train.loc[Q_index, 'Deck level'],
               c='green',
               s=train.loc[Q_index, 'Fare'])
    ax.set_xlabel('Room center longitude(X)')
    ax.set_ylabel('Room center latitude(Y)')
    ax.set_zlabel('Deck level')
    ax.set_title('Price by cabin location and by port of embarkation')
    plt.legend([S_series, C_series, Q_series],
               ['S - Southampton', 'C - Cherbourg', 'Q - Queenstown'],
               title='Embarked at',
               loc=6)
    plt.show()
#plot3d_loc_and_price(train)

# Conclusion: 1st class passengers embarked at S(Southampton) and C(Cherbourg) demonstrate different location and price
#  patterns. However, there are only 4 passengers embarked at Queenstown with known cabin codes, which is not enough
# to see any location patten
# **Note: this cabin data is only representative for 1st class passengers embarked at S(Southampton) and C(Cherbourg)



## Titanic passengers' by class location information allows
# approximate XYZ passengers location only based on their class
# look at the corresponding figure for reference

## Allocation of passengers with unknown cabins
deck_codes = {'A':7, 'B':6, 'C':5, 'D':4, 'E':3, 'F':2, 'G':1}
deck_codes_rev = {7:'A', 6:'B', 5:'C', 4:'D', 3:'E', 2:'F', 1:'G'}

# Stat for 1st class passengers embarked at S(Southampton) allocation
train_1st_S_unkn = train.copy()[(train['Embarked']=='S') &
                                (train['Pclass']==1) &
                                ( pd.isnull(train['Cabin']) ) ]
train_1st_S = train.copy()[(train['Embarked']=='S') &
                           (train['Pclass']==1) &
                           ( pd.isnull(train['Cabin'])==False) ]
#print('----- 1st class passengers embarked at S(Southampton) stat')
train_1st_S_fare_stat = train_1st_S.groupby('Deck level').agg(['count','mean',np.std])
train_1st_S_fare_stat = train_1st_S_fare_stat.drop(8, axis=0)
train_1st_S_fare_stat = train_1st_S_fare_stat.loc[:,'Fare']
#print(train_1st_S_fare_stat)
# Stat for 1st class passengers embarked at C(Cherbourg) allocation
train_1st_C_unkn = train.copy()[(train['Embarked']=='C') &
                                (train['Pclass']==1) &
                                ( pd.isnull(train['Cabin']) ) ]
train_1st_C = train.copy()[(train['Embarked']=='C') &
                           (train['Pclass']==1) &
                           ( pd.isnull(train['Cabin'])==False) ]
#print('----- 1st class passengers embarked at C(Cherbourg) stat')
train_1st_C_fare_stat = train_1st_C.groupby('Deck level').agg(['count','mean',np.std])
train_1st_C_fare_stat = train_1st_C_fare_stat.loc[:,'Fare']
#print(train_1st_C_fare_stat)
# Stat for 1st class passengers not embarked at C(Cherbourg) or at S(Southampton) allocation
train_1st_notSC_unkn = train.copy()[(train['Embarked']!='S') &
                                    (train['Embarked']!='C') &
                                    (train['Pclass']==1) &
                                    ( pd.isnull(train['Cabin']) ) ]
train_1st_notSC = train.copy()[(train['Pclass']==1) &
                               ( pd.isnull(train['Cabin'])==False) ]
deck_codes_1st_notSC = {'A':7, 'B':6, 'C':5, 'D':4, 'E':3}
#print('----- 1st class passengers NOT embarked at S(Southampton) or C(Cherbourg) stat')
#print('')
#print('NA')
#print('')
train_1st_notSC_fare_stat = train_1st_notSC.groupby('Deck level').agg(['count','mean',np.std])
train_1st_notSC_fare_stat = train_1st_notSC_fare_stat.loc[:,'Fare']
train_1st_notSC_fare_stat = train_1st_notSC_fare_stat.drop(8, axis=0)
# There are no passengers in this group, so no estimation for this group is planned
#print(train_1st_notSC_fare_stat)

# Stat for 1st class passengers embarked at S(Southampton) allocation
print('26%  1st class passengers allocation statistics is computed')

# Allocation of passengers with unknown cabin numbers
def assign_cabin(row):
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
        row = cabin_loc_allocation(row)


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
        row = cabin_loc_allocation(row)


    else:
        row = cabin_loc_allocation(row)
    return row


train = train.apply(assign_cabin, axis=1)
print('68%  TRAIN pre-processing is completed')
train.to_csv('titanic_train_READY.csv', sep=',', encoding='utf-8')
plot3d_surv_by_loc(train)





# Start TEST pre-processing
print('69%   Start TEST pre-processing')
test_original = pd.read_csv('titanic_test.csv')
test_original.set_index('PassengerId',inplace=True)
test = test_original.copy()

test = test.apply(split_name, axis=1)
test = test.apply(titles_standardization,axis=1)
print('71%  Titles are standardized to be in the following range [Master, Miss, Mr, Mrs]')

test[pd.notnull(test.copy()['Age'])==False] = test[pd.notnull(test.copy()['Age'])==False].apply(age_predict, axis=1)
print('73%  Age of the passengers with unknown age is identified')

test.apply(ticket_list_initial_append,axis=1)
print('75%  Ticket list is created based on the passengers with known cabin number')

test = test.apply(cabin_loc_allocation, axis=1)
print('77%  Passengers with known unambiguous cabin number are allocated:')
print('      {} passengers out of {} are allocated'.format(\
    (test['Room center longitude(X)'].isnull()==False).sum(), len(test) ) )

test = test.apply(assign_cabin, axis=1)
test.to_csv('titanic_test_READY.csv', sep=',', encoding='utf-8')
print('100%  TEST pre-processing is completed')