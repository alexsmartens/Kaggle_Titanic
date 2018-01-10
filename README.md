# Kaggle_Titanic

This is an approach to building ML models for "Titanic: Machine Learning from Disaster" Kaggle competition

My current current results
* Accuracy: 0.80861 
* Top 7%


The project consists from two major parts.

1. Filling missing data. This part is implemented in main_1_filling_missing_data.py. The functions used by main_1_filling_missing_data.py are contained in Filling missing data functions folder. Filling missing data is mainly filling two features:
- Age. My approach to filling age: I figured out that Name feature contain title information. General titles are: Master, Miss, Mr, Mrs. Hence, titles is an indicator of age. I computed distributions of each title through the train set and assigned mean title age to passengers with unknown age.
- Cabin. Only small fraction of passengers have cabin numbers, majority of those people are 1st class passengers. However, I found Titanic deck planes at https://www.encyclopedia-titanica.org/titanic-deckplans/ and using these plans and Cabin feature data I figured out that each Pclass (passenger class) had "very special" location which I pictured in /Data/Class_zones_on_deck_plan.png. Finally, I located each passenger with known cabin to its location on the ship, I assigned 1st class passengers with unknown Cabin based on Fare feature and 2nd and 3rd class passengers based on their Class.

2. Fitting ML models. At first, in some cases data has been normalized for further proceeding with some approaches. Logistic Regression, Naive Bayes, KNN, Gradient Boosted Decision Trees, Random Forest and Neural Network classifiers have been fitted to the data. Next, Neural Networks, Gradient Boosting Trees and Random Forest demonstrated higher accuracy score. Hence, their parameters are tuned for the best accuracy score. As the next step, Extreme Gradient Boosted Trees and an ensamble models are added into the play and tuned. Additionally, Synthetic Minority Over-sampling Technique (SMOTE) feature is added for balancing the asymmetrical dataset. This part is implemented in main_2_GBT_parameter_tuning.py, main_2_Log_Reg_parameter_tuning.py, main_2_MLP_parameter_tuning.py, main_2_Xgboost_parameter_tuning.py, main_3_Ensemble_GBT_MLP_LogR.py files.

Results: as of now, the best accuracy score is produced by Gradient Boosted Decision Trees with and without SMOTE oversampling.
