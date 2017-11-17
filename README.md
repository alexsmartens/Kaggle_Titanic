# Kaggle_Titanic

This is an approach to building ML models for "Titanic: Machine Learning from Disaster" Kaggle competition

My current current results
* accuracy: 0.80861 
* Top 8%


The project consists from two major parts so far.

1. Filling missing data. This part is implemented in main_1_filling_missing_data.py. The functions used by main_1_filling_missing_data.py are contained in Filling missing data functions folder. Filling missing data is mainly filling two features:
- Age. My approach to filling age: I figured out that Name feature contain title information. General titles are: Master, Miss, Mr, Mrs. Hence, titles is an indicator of age. I computed distributions of each title through the train set and assigned mean title age to passengers with unknown age.
- Cabin. Only small fraction of passengers have cabin numbers, majority of those people are 1st class passengers. However, I found Titanic deck planes at https://www.encyclopedia-titanica.org/titanic-deckplans/ and using these plans and Cabin feature data I figured out that each Pclass (passenger class) had "very special" location which I pictured in /Data/Class_zones_on_deck_plan.png. Finally, I located each passenger with known cabin to its location on the ship, I assigned 1st class passengers with unknown Cabin based on Fare feature and 2nd and 3rd class passengers based on their Class.

2. Fitting ML models. This part is implemented in main_2_ML_models.py. The functions used by main_2_ML_models.py are contained in ML models data functions folder. At first, in some cases data has been normalized for further proceeding with some approaches. Logistic Regression, Naive Bayes, KNN, Gradient Boosted Decision Trees, Random Forest and Neural Network classifiers have been fitted to the data. Each ML method has been tested with various parameters. The best parameters were chosen for further methods comparing.

Future  plans: all tested ML approaches indicate similar performance in range [0.78, 0.82]. So, I want to find out what is the main source  of error and try to eliminate it.
