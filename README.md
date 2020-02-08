# Predicting Credit Risk

## Overview
A peer-to-peer lending service wants to utilize machine learning to predict credit risk. The goal is for the algorithm to provide a quicker loan experience and more accurate identification of loan candidates to produce lower default rates. As a part of this team, it will be necessary to experiment with multiple machine learning models to identify the optimal algorithm for the data. Each model will need to be trained and evaluated to examine its performance compared to others.

## Resources
Software:
- imblearn 0.6.1
- numpy 1.18.1
- sklearn 0.22.1
- pandas 1.0.0
- Python 3.7.6

Data Sources:
- `LoanStats_2019Q1.csv`
- `cc_default.csv`
- `diabetes.csv`
- `loans_data.csv`
- `loans.csv`
- `Salary_Data.csv`

## Summary
This project leveraged the power of supervised machine learning algorithms because the datasets all contained labels. Two branches of supervised learning were explored; regression and classification. Broadly speaking, regression is used to predict continuous variables while classification is used for discrete outputs. The most basic form of regression is linear regression which is used to find a linear relationship between a target and one or more predictors. Another form of regression is logistic regression which predicts binary outcomes based on multiple variables. When presented with new data after being trained, the logistic regression model mathematically determines the probability of the samples belonging to a class. Similar to logisitic regression is a support vector machine (SVM) which is also a binary classifier; however, it is a linear classifier. This means the model tries to find a line that seperates the data into two classes and attempts to maximize the margins. Another model type is the decision tree which can be viewed simply as a series of if/else statements. Decision trees can become very complex which leads to the next idea in machine learning which is ensemble learning. This is a relatively basic idea which is that combining multiple simpler models (weak learners) and aggregating their predicitions, their accumulated output will be more accurate and the overall model will be more robust than any single model. This can be applied to decision trees to create a random forest which combines multiple trees all trained on seperate chunks of the training data.   
In order to utilize these algorithms, it may be necessary to perform some pre-processing on the data. For classified data, this may mean encoding the data so that non-numeric variables can be interpreted by the program. Also, if there is a large range in the scales between variables in a dataset, it may be necessary to scale the data so as not to create bias in the model.   
Finally, in order to analyze the performance of the models, there are some key metrics that must be understood. The first is the accuracy of the model. This can be interpreted as the fraction of predictions the model got correct. The next metric is precision which indicates the proportion of correctly identified positive predictions. In other words, given a model's prediction, how likely is it that prediction is correct? And lastly, recall which determines the proportion of correctly identified actual positives. Phrased as a question, given an actual positive, what is the probability it was identified correctly by the model? (F1 score is another metric used for performance analysis which is the harmonic mean of the precision and recall of a model. It attempts to summarize the precision and recall of a model.)

## Challenge
Given a dataset containing loan statistics, the goal is to employ different machine learning techniques to train and evaluate models with unbalanced classes. Then, it will be required to analyze the performance of these models to determine the optimal algorithm for this application. Finally, given the performance metrics, it is necessary to produce a recommendation on which model should be used (if any) to predict credit risk. 

## Challenge Summary
The initial approach to this project was analyze different sampling techniques to account for the class imbalance in the dataset. Two oversampling, one undersampling, and one combination sampling algorithm were created and then combined with a logistic regression model to make predicitions. Based on the performances, the random sampling technique was the best; although, paired with a logistic regression algorithm, it is not reliable enough to be used to predict credit risk. For further analysis, two different types of ensemble learning algorithms were tested (both using random undersampling). Of these two, the Easy Ensemble AdaBoost Classifier managed to perform well enough to be considered for use in the company. However, the model incorrectly labels instances as high-risk frequently which is the largest downfall of the model. In this case, the model was still able to consistently identify relevant data shown by high recall scores so the final recommendation is to explore other ways to improve the precision of the model.

See `Challenge/credit_risk_resampling.ipynb` and `Challenge/credit_risk_ensemble.ipynb` for more details


