# Predicting MLB Home Runs

## Table of Contents

- [Project Overview](#project-overview)
- [Data Sources](#data-sources)
- [Tools](#tools)
- [Data Exploration](#data-exploration)
- [Data Preparation](#data-preparation)
- [Model Evaluation](#model-evaluation)
- [Data Modeling](#data-modeling)

### Project Overview

I started watching Major League Baseball (MLB) last year, and it has quickly become one of my favourite things to do in my free time. While watching, I couldn’t help but notice how many external factors may be impacting the performance of the league’s players. One of the most exciting plays in sports is the home run, and I wanted to analyze how big of a role these external factors may have on the players’ ability to hit home runs. The goal of this model is to predict if an MLB player will hit a home run in a game based on the variables present in our MLB Game Logs dataset.

### Data Sources

MLB Game Logs Data: The primary dataset used for this analysis is the "mlb_game_logs_sample.csv" file, which contains information from individual player's games.

### Tools

- Excel: Data Cleaning
- Python: Data Exploration, Data Visualization, Data Modeling

### Data Exploration

In order to determine which variables could be useful to incorporate in my model for predicting if MLB players will hit a home run, I used chi-square, forward feature selection, and recursive feature elimination. Using these methods I was able to identify dozens of potentially useful features. However, I did not use all of these features in my model. Instead I experimented with various different feature combinations and compared their metrics with one another to determine which was best. The following seven environmental variables were found to be the most significant in predicting whether or not a player would hit a home run: temperature, sky condition, wind speed, wind direction, batting order, the pitcher’s strong hand, and the stadium in which the game is played. Figures 1 – 7 were created in an attempt to find any additional context as to how these seven variables may be impacting the ability of MLB players to hit home runs.

![image](https://github.com/user-attachments/assets/38a992ab-110e-4887-86f6-5aa598970895)
![image](https://github.com/user-attachments/assets/3310764b-ba06-436c-9b56-8aa1acfe493e)
![image](https://github.com/user-attachments/assets/31a6ada7-4747-4f62-9e9a-cf39961d8f63)

By visualizing some of the significant variables impacting MLB players’ ability to hit home runs, I was able to both reinforce some of my assumptions and find unexpected patterns. Figure 1 is a bar chart comparing the average temperature in which home runs were and were not hit. Although the average temperatures were very similar, it appears that more home runs are hit in warmer temperatures. However, since these values are so similar, it is difficult to say for sure. Figure 2 is a bar chart displaying the average number of home runs hit by players under various sky conditions. This chart tells us that players generally hit the most home runs in overcast weather. Perhaps this sky condition is ideal because the sun won’t be in batters’ eyes. It appears that players hit less home runs at night, which could be because the ball is harder to see at night. Figure 3 and Figure 4 illustrate the impact of wind on hitting home runs. As expected, more home runs are hit when the wind speed is slightly slower. On average, more home runs are hit when the wind is blowing out to left field. This was also expected because it is well-known that most batters are right-handed, and their swinging motion is to the left. This could mean that when the wind is blowing out to left field, the ball is carried further by the wind, resulting in more home runs.Figure 5 is a bar chart showing the average number of home runs hit by players in each position in the batting order. It appears that players who are earlier in the batting order hit more home runs on average than those who are later in the batting order. This is likely due to a combination of two main factors: better players usually bat earlier in the order and players who bat earlier in the order often get more at bats than those who are later in the order. The “NA” position in the batting order is for substitute players, so it is expected that they hit the least amount of home runs on average because they have fewer opportunities to do it. Figure 6 is a bar chart that shows more home runs being hit on average vs left-handed pitchers than right-handed pitchers. This isn’t surprising because it is said that batters generally hit better against opposite-handed pitching, and most batters are right-handed. Lastly, Figure 7 shows that the average number of home runs hit differs between ball parks. This is likely because these parks have different sized fields and are located in areas with different weather patterns. Overall, these statistics and visualizations provide us with additional context for the MLB Game Logs dataset, and help us evaluate the potential variables impacting the ability of players to hit home runs.

### Data Preparation

In the data-preparation stage of creating my model, I imputed the “BattingOrder” variable with a value of 10. As I was looking through the dataset, I noticed that a large majority of players missing their batting order value were pinch hitters, which are substitute players. Since these players are not in the 9-man lineup, but come into the game later, I thought a value of 10 was suitable for them. With respect to dummy variables, six were created: “H.A” (home or away), “Precipitation”, “Sky”, “Stadium”, “Wind.Direction”, and “PitchHand”. These dummy variables were created to represent these categorical variables as numbers, so they could be included in my model. However, I did not create any binned variables for my model because variables such as “Temperature” and “Wind.Speed” were already relatively valuable in their original forms. 

I attempted to use each of StandardScaler, MinMaxScaler, and RobustScaler with my model and they all outputted similar results. Each of these scalers reduced the average accuracy of the model from 0.72 to roughly 0.54, increased the average precision from 0.13 to 0.14, increased the average recall from 0.46 to approximately 0.88, and increased the F1-score from 0.21 to 0.24. I settled with StandardScaler because along with nearly doubling the recall value, it had the best precision and F1-score of all the scalers. I also decided that in this case, the decrease in average accuracy from 0.72 to 0.54 was a fair tradeoff for the increase in average recall from 0.46 to 0.88. This is because home runs are relatively rare, so I didn’t want my model to miss a significant number of positive responses due to the low recall.

### Model Evaluation
#### Table 1: Comparing Statistics from Different Models
![image](https://github.com/user-attachments/assets/98136bc0-e3ef-4c83-9998-1bf53b79c48a)

In order to determine what the best model was for predicting if a home run was going to be hit, I compared the average accuracy, precision, recall, and F1 values for four of my best models. Features for Model 1 were selected using Chi-Square, features for Model 2 were selected using Forward Feature Selection, features for Model 3 were selected using Recursive Feature Elimination, and Model 4 was a stacked model. Model 3 returned relatively low values for average accuracy, precision, and F1-score, but it had the highest recall. On the other hand, Model 1 had the second highest average accuracy, with the highest precision and F1-scores, but it had the lowest recall. Model 2’s performance would rank somewhere between Models 1 and 3 because it didn’t have the highest or lowest value for any of these four metrics. Lastly, Model 4 had by far the highest accuracy, but was mediocre with respect to every other metric. Based on these four metrics, I believe Model 1 is the best. Although it is also important to consider the standard deviations of these scores, the differences between them in this case are so small that I don’t believe they should impact the decision of which model is best. Overall, I believe Model 1 is the best because it has the second highest average accuracy with the highest precision and F1-score. Despite having the second lowest recall, Model 1’s recall is still relatively competitive, so I don’t think the lower average recall value is significant enough to crown either Model 2, 3, or 4 as superior.

Although Model 1 is the best of the four options, there are still a number of ways to develop an even better model. The biggest gripe I have with my model is that it doesnot take player’s stats into account, aside from whether or not they hit a home run. Unfortunately, I was unable to find a dataset that included both player statistics and environmental factors, so as a result, I focused on just the environmental factors. However, to truly analyze how environmental factors impact the players’ ability to hit home runs, more player performance statistics are necessary. Another way to potentially develop a better model is to try more feature combinations. Although the four models I referenced above were the best I was able to make, there were so many different features that I most likely missed some feature combinations that would have resulted in an even better model. Lastly, changing some of the decisions I made in the datapreparation stage of this project could potentially help develop a better model. For example, maybe imputing a different value for the missing data in the “BattingOrder” column could have improved the model

### Data Modeling
#### Code for Stacked Model with Cross Fold Validation

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import KFold
from imblearn.over_sampling import SMOTE

# Import data into a DataFrame.
PATH = "C:/Python/DataSets/"
FILE = "mlb_game_logs_sample.csv"
df = pd.read_csv(PATH + FILE)

# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(df.head())  # View a snapshot of the data.
print(df.describe())  # View stats including counts which highlight missing values.

# Start imputing
def imputeNullValues(colName, df):
    # Create two new column names based on original column name.
    indicatorColName = 'm_' + colName  # Tracks whether imputed.
    imputedColName = 'imp_' + colName  # Stores original & imputed data.

    # Impute NA batting order with 10.
    imputedValue = 10

    # Populate new columns with data.
    imputedColumn = []
    indictorColumn = []
    for i in range(len(df)):
        isImputed = False

        # mi_OriginalName column stores imputed & original data.
        if (np.isnan(df.loc[i][colName])):
            isImputed = True
            imputedColumn.append(imputedValue)
        else:
            imputedColumn.append(df.loc[i][colName])

        # mi_OriginalName column tracks if is imputed (1) or not (0).
        if (isImputed):
            indictorColumn.append(1)
        else:
            indictorColumn.append(0)

    # Append new columns to dataframe but always keep original column.
    df[indicatorColName] = indictorColumn
    df[imputedColName] = imputedColumn
    del df[colName]  # Drop column with null values.
    return df

# Use imputeNullValues function
df = imputeNullValues('BattingOrder', df)
print(df.head(10))

# You can include both the indicator 'm' and imputed 'imp' columns in your model.
# Sometimes both columns boost regression performance, but sometimes they do not.
X = df.copy()  # Create separate copy to prevent unwanted tampering of data.
del X['HomeRun']  # Delete target variable.
del X['Game.ID']  # Delete unique identifier which is completely random.
del X['Player']  # Delete player name which is just an identifier.
del X['Position']  # Delete player position which is generally a defensive identifier.
del X['X1B.Ump']  # Delete 1B umpire's name which is just an identifier.
del X['X2B.Ump']  # Delete 2B umpire's name which is just an identifier.
del X['X3B.Ump']  # Delete 3B umpire's name which is just an identifier.
del X['HP.Ump']  # Delete HP umpire's name which is just an identifier.
del X['Team']  # Delete team name which we will consider an identifier.
del X['Opponent']  # Delete opponent name which we will consider an identifier.

# Get Dummy variables
X = pd.get_dummies(X, columns=['H.A', 'Precipitation', 'Sky', 'Stadium', 'Wind.Direction',
                               'PitchHand'], dtype=int)

print("\n Here are all potential X features - no more nulls exist.")
print(X)
print(X.describe())

y = df['HomeRun']

def getUnfitModels():
    models = list()
    models.append(LogisticRegression(max_iter=1000))
    models.append(DecisionTreeClassifier())
    models.append(AdaBoostClassifier())
    models.append(RandomForestClassifier(n_estimators=10))
    return models

def evaluateModel(y_test, predictions, model):
    precision = round(precision_score(y_test, predictions), 2)
    recall = round(recall_score(y_test, predictions), 2)
    f1 = round(f1_score(y_test, predictions), 2)
    accuracy = round(accuracy_score(y_test, predictions), 2)

    print("Precision:" + str(precision) + " Recall:" + str(recall) + \
          " F1:" + str(f1) + " Accuracy:" + str(accuracy) + \
          " " + model.__class__.__name__)

def fitBaseModels(X_train, y_train, X_test, models):
    dfPredictions = pd.DataFrame()

    # Fit base model and store its predictions in dataframe.
    for i in range(0, len(models)):
        models[i].fit(X_train, y_train)
        predictions = models[i].predict(X_test)
        colName = str(i)
        dfPredictions[colName] = predictions
    return dfPredictions, models

def fitStackedModel(X, y):
    model = LogisticRegression()
    model.fit(X, y)
    return model

# define empty base models here and store them in a list.
unfitModels = getUnfitModels()

# define empty stack model here.
stackedModel = LogisticRegression()

# prepare cross-validation with three folds.
kfold = KFold(n_splits=3, shuffle=True, random_state=0)
accuracyList = []
precisionList = []
recallList = []
f1List = []
count = 0

for train_index, temp_index in kfold.split(X):  # split on all of X

    # PARENT LOOP CODE

    # Temp needs to be split into test and val data sets.
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.50)
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.50)

    # define empty X_predictions dataframe here.
    dfPredictions = pd.DataFrame()

    # Another loop is needed here to loop through all base models.
    for model in unfitModels:
        # CHILD LOOP CODE

        # Apply SMOTE to the training set
        smote = SMOTE()
        X_train_SMOTE, y_train_SMOTE = smote.fit_resample(X_train, y_train)

        # The individual models need to be fit with smote train data.
        model.fit(X_train_SMOTE, y_train_SMOTE)

        # Predictions of base models can be made with val data.
        predictions = model.predict(X_val)

        # Add predictions of each model to column of X_predictions dataframe.
        dfPredictions[model.__class__.__name__] = predictions

        # Evaluate each base model with the test data.
        evaluateModel(y_test, predictions, model)

    # Back in PARENT loop
    # Fit another smote object with X_predictions & y_val.
    smote = SMOTE()
    X_pred_SMOTE, y_pred_SMOTE = smote.fit_resample(dfPredictions, y_val)

    # Fit stack with X_predictions_afterSMOTE and y_val_afterSMOTE.
    stackedModel.fit(X_pred_SMOTE, y_pred_SMOTE)

    # Make base model predictions with test data.
    dfPredictions_test = pd.DataFrame()

    for model in unfitModels:
        predictions_test = model.predict(X_test)
        colName = model.__class__.__name__
        dfPredictions_test[colName] = predictions_test
        # Metrics for the base models should be stored in a list here.
        # It is important to track the performance of base models over the long term with unseen data.
        evaluateModel(y_test, predictions_test, model)

    # Make prediction with stack model with X_predictions2 as input.
    stackedPredictions = stackedModel.predict(dfPredictions_test)

    # Store F1, Precision, Recall and Accuracy in list for stack model. (eval with preds & y_test)
    evaluateModel(y_test, stackedPredictions, stackedModel)

    count += 1
    print("\n***K-fold: " + str(count))

    # Calculate accuracy, precision, recall, and F1 scores and add to the list.
    accuracy = metrics.accuracy_score(y_test, stackedPredictions)
    precision = metrics.precision_score(y_test, stackedPredictions)
    recall = metrics.recall_score(y_test, stackedPredictions)
    f1 = metrics.f1_score(y_test, stackedPredictions)

    accuracyList.append(accuracy)
    precisionList.append(precision)
    recallList.append(recall)
    f1List.append(f1)

    print('\nAccuracy: ', accuracy)
    print("\nPrecision: ", precision)
    print("\nRecall: ", recall)
    print("\nf1: ", f1)

# Show averages of scores over multiple runs.
print("\nAccuracy, Precision, Recall, F1, and STD for all Folds:")
print("*********************************************")
print("Average accuracy:  " + str(np.mean(accuracyList)))
print("Accuracy std:      " + str(np.std(accuracyList)))
print("Average precision: " + str(np.mean(precisionList)))
print("Precision std:     " + str(np.std(precisionList)))
print("Average recall:  " + str(np.mean(recallList)))
print("recall std:      " + str(np.std(recallList)))
print("Average f1: " + str(np.mean(f1List)))
print("f1 std:     " + str(np.std(f1List)))
```
