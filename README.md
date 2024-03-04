# AIAP16 Take Home Assignment

## Table of Contents

1. [Name & Email](#Name-&-Email)
2. [Folder Structure Overview](#Folder-Structure-Overview)
3. [Pipeline Execution Instructions](#Pipeline-Execution-Instructions)
4. [Pipeline Flow Description](#Pipeline-Flow-Description)
5. [EDA Key Findings Overview](#EDA-Key-Findings-Overview)
6. [Choice of Models](#Choice-of-Models)
7. [Evaluation of Models](#Evaluation-of-Models)
8. [Other Considerations](#Other-Considerations)
   
## Name & Email

- Name: Lim Chiew Yee Nicholas
- Email: nicklimcy90@gmail.com

## Folder Structure Overview
<img src="https://github.com/Xyfalix/aiap16-lim-chiew-yee-nicholas-743H/assets/129175727/e6472870-471b-4396-b8aa-87803a2d6712" alt="image" width="300"/>

## Pipeline Execution Instructions
- To execute the pipeline, simply run the following command.
- ./run.sh &lt;model&gt;
- Substitute &lt;model&gt; with either decision_tree, naive_bayes, or random_forest

## Pipeline Flow Description
- Import model based on input that user provides
- Run the load_data function from dataloader.py which queries the lung_cancer.db and returns the lung_cancer table within
- Run the preprocess_data function from preprocessor.py which cleans up the raw data, label encodes categorical columns that are ordinal, and one-hot encodes categorical columns that are not ordinal
- Preprocess_data will then generate the X_train, X_test, y_train and y_test splits
- Train the user specified model using the test splits above
- Evaluate the model by comparing training data accuracy vs test data accuracy
   
## EDA Key Findings Overview
- IDs column had duplicates: Removed all duplicated rows
- Age column had negative values: convert all negative values to positive
- COPD history column had null values: replaced all null values with No
- Air Pollution Exposure column had 3 NaN rows: removed all Nan rows
- Created a new column Weight Difference which takes Current Weight - Last Weight
- Start Smoking & Stop Smoking columns had Not Applicable values: converted these values to 0
- Stop Smoking column had Still Smoking values: converted these values to 2024, which is the latest year in the Still Smoking column
- Created a new column Smoking Duration (Years) which returns the result from subtracting Start Smoking from Stop Smoking
- Taken Bronchodilators column had nill values: replaced all null values with No

| Column Name             | Issue                   | Action taken                              |
| :---------------------: | :---------------------: | :---------------------------------------: |
| ID                      | Duplicate rows          | Removed duplicate rows                    |
| Age                     | Negative values         | Converted all negative values to positive |
| COPD History            | Null values             | Replaced null values with "No"            |
| Air Pollution Exposure  | NaN values              | Removed rows with NaN values              |
| Weight Difference       | N.A                     | New column: Current Weight - Last Weight  |
| Start Smoking           | Not Applicable values   | Converted these values to 0               |
| Stop Smoking            | Not Applicable values   | Converted these values to 0               |
| Stop Smoking            | Still Smoking values    | Converted these values to 2024            |
| Smoking Duration (Years)| N.A                     | New column: Stop Smoking - Start Smoking  |
| Taken Bronchodilators   | Null values             | Replaced null values with "No"            |


## Choice of Models
### Brief overview of task
- I have been tasked to develop at least three suitable models that predict whether a patient will be diagnosed with lung cancer, using the data parsed from lung_cancer.db.
- There are a mix of relevant continous and categorical features, and the response is categorical (1 indicating has lung cancer, 0 indicating no lung cancer). This means that this is a classification problem, and the presence of training data means that it is a supervised learning problem.
- Hence, I decided to pick Machine Learning (ML) models that can be used for classification problems.

### Elaboration of model choices
### Decision Tree
1) Decision tree structures are easy to understand and intepret. I was not overwhelmed when learning about how to utilize decision trees in a short amount of time.
2) Feature selection is not necessary because unimportant features are ignored by the tree. I was unsure about including the Smoking Duration column information because there did not appear to be a strong correlation between smoking duration and lung cancer. I decided to include it due to this property.
3) Decision trees work well with small and large datasets. This gave me the confidence that I could use decision trees on the provided data.

### Naive Bayes
1) Naive Bayes assumes that all features are independent of each other, which is an assumption I made when deciding whether to include a feature or not.
2) It is simple and easy to implement. I could understand how the algorithm works on a basic level.
3) It is not sensitive to irrelevant features. This is helpful since I am new to this and I was not sure how to judge relevancy or irrelevancy of a feature.

### Random Forest Ensemble
1) It reduces the overfitting problem in decision trees and improves accuracy. My first impression before testing this out was that it should perform better than a single decision tree.
2) Other than that, it has similar advantages to decision trees.

## Evaluation of Models
### Decision Tree
### 1st iteration (No optimization)
- Training Model Accuracy: 91.2%
- Test Data Accuracy: 63.5%
- From this data, overfitting had likely occurred, and certain parameters (termed as hyperparameters) have to be tuned in order to reduce overfitting. From my understanding, 3 hyperparameters can be tuned to reduce overfitting, namely:
  1) No maximum depth. This causes the model to fit too well to the training data, capturing noise and outliers.
  2) Small minimum samples per leaf which results in leaf nodes too specific to the training data
  3) Too few samples per split which could cause the tree to split when noise is occurred insted of legitimate patterns.
- I used GridSearchCV from scikit-learn model selection to optimize the hyperparameters. After some trial and error and lookup on Google, I arrived at the following.
### 2nd iteration (Optimization done)
- Training Model Accuracy: 72.2%
- Test Data Accuracy: 70.4%
- Optimized parameters are as follows
  1) Max Depth: 8
  2) Min samples per leaf: 9
  3) Min samples per split: 49
- Do note that I only used small ranges for the grid search because it took extremely long to optimize when the ranges were the grid search were set very wide. Test data accuracy increased by ~7% after tuning the hyperparameters

### Naive Bayes
### 1st iteration (No optimization)
- Training Model Accuracy: 63.2%
- Test Data Accuracy: 62.8%
- My understanding of Naive Bayes is that it suffers from 1 critical flaw, which is the 'zero-frequency problem' where zero probability is assigned to a categorical variable whose category in the test data set was not available in the training dataset. Technically, this was very unlikely to happen because the distribution of categorical values were quite even and large. However, I did try out smoothing to see if it would help improve the accuracy of the training model.
### 2nd iteration (Optimization done)
- Training Model Accuracy: 67.8%
- Test Data Accuracy: 67.0%
- Optimized parameters are as follows
  1) var_smoothing: 0.0006579332246575676
- Even though Naive Bayes is easy to use, the lack of hyperparameters to tune does mean that it is difficult to improve the test data accuracy unless the training dataset is modified.

### Random Forest Ensemble
### 1st iteration (No optimization)
- Training Model Accuracy: 91.9%
- Test Data Accuracy: 66.3%
- Similar to decision trees, overfitting had likely occurred since training model accuracy was much higher than test data accuracy. and certain parameters (termed as hyperparameters) have to be tuned in order to reduce overfitting. The hyperparameters that can be tuned are similar to decision trees, but I tried out cross-validation.
- My understanding of cross-validation is as follows. I do not fully understand why it is done, but it seems like cross-validation helps prevent cases where hyperparameter tuning is done too often using test set score accuracy as a reference that it indirectly starts to fit to the characteristics of the test set.
  1) The training set is split into k smaller sets (called folds)
  2) A model is trained using k-1 of the folds as training data.
  3) The resulting model is validated on the fold that was unused.
### 2nd iteration (Optimization done)
- Training Model Accuracy: 74.8%
- Test Data Accuracy: 72.2%
- Optimized parameters are as follows
  1) Max Depth: 10
  2) Min samples per leaf: 10
  3) number of trees in the forest: 10
- Similar to decision trees, I played around with small ranges to optimize. Test data accuracy increased by ~7% after tuning the hyperparameters

## Other Considerations
- Due to time constraints, I did not explore other metrics that are used to evaluate the performance of these ML models.
  1) Precision: The ratio of true positives to the sum of true positives and false positives. It measures the accuracy of the positive predictions.
  2) Recall (Sensitivity or True Positive Rate): The ratio of true positives to the sum of true positives and false negatives. It measures the ability to capture all relevant instances.
  3) F1 Score: The harmonic mean of precision and recall. It balances precision and recall.
  4) Area Under the ROC Curve (AUC-ROC): A measure of the model's ability to distinguish between classes.
