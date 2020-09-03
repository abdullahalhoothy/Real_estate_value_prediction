# Real Estate Value Predictor

## Agenda

### 1.Executive Summary
### 2.Motivation
### 3.Introduction
### 4.The Dataset
### 5.Feature Selection and Scaling 
### 6.Preparing the Data
### 7.Choosing the Model
### 8.Hyperparameter Tuning
### 9.Fitting &amp; Testing the Model
### 10.Wrap Up

## Executive Summary

The business objective of a real estate value estimation is essentially finding the **optimal price to place one's real estate listing to maximize its economic benefits**". The economicals burdens that an estimation tool could relive Total at around $45,600  including :

  a- cost of apprisals costs($300-$700)[1]

  b- cost of real-estate agents costs (5%+bonus =~ $21,000)[2]

  c- cost of CMA (Comprehensive market analysis) costs($100)

  d- cost of lost oppertunity (if the sale is delayed by a year because the house was set at a high price, then the price of the home could have been used in that year in other investment potentially yeilding 7% yearly ) with average price of home at $413,507 yields $28,945

  e- undervalued house means that you just lost on the deal say 5% less than you could have with average price of home at 413,507$ thats $20,675 lost


The most important factors in determining the house's price is the size, the year it was built, the size of the garage, and the number of bedrooms and the number of bathrooms.

This project is offering a solution to this problem by introducing the latest AI algorthms to allow estimation and identification of real estate price and the wide 
range of factors and variables to get a sense of the most important combinations as well as thier power to predict house's price. This Approch provided better and more timely prediction 
compared to orgnizations who do not adopt the use of the latest AI technologies


1- https://justo.ca/blog/how-much-will-a-home-appraisal-cost-in-ontario/

2- https://www.getwhatyouwant.ca/real-estate-commission-explained



## Motivation

An real estate value predictor is important not only from the perspective of a buyer, but it also comes in pretty handy to someone who&#39;s planning to sell his or her house but is unsure of what price they should put it up at. Moreover, for someone who likes to play around with different algorithms using a project based approach, this is a great idea to start with, since it covers all the basic principles used in developing an effective machine learning model. This is pretty much the reason why I first came up with the idea of getting my hands dirty with this project.

## Introduction

Ever since the evolution of machine learning, predicting real estate value using it has always been a hot topic. Although several models already do exist on the internet, I found them to be missing some important features which needed to be taken into considering.

In this article, we will cover everything from data cleaning to final model performance and I will be guiding you step by step through all the procedure, so you can have a deep understanding of each step and a firm grasp of the major concepts involved. So, let&#39;s get started without any further ado.

## The Dataset

Exploring the dataset should always be your first step before anything else. While you can also explore the data using pandas, I generally prefer to have a good look at it manually as well to notice any trends which are readily visible.

So, let&#39;s import some libraries needed for the model and have a glimpse of our dataset.

![](https://github.com/abdullahalhoothy/Real_estate_value_prediction/blob/master/images/image1.png)

![](https://github.com/abdullahalhoothy/Real_estate_value_prediction/blob/master/images/image2.png)

![](https://github.com/abdullahalhoothy/Real_estate_value_prediction/blob/master/images/image3.png)

![](https://github.com/abdullahalhoothy/Real_estate_value_prediction/blob/master/images/image4.png)

So, we have over 42 thousand records of real estate prices ,For each house, we capture 18 separate features. The year it was built, how many bedrooms it has. Machine learning algorithms work best when your dataset covers all possible combinations of features in your model. For example, we want our home price dataset to include prices for big houses with lots of bathrooms and no garage and no pool, but also big houses with lots of bathrooms and no garage but with a pool. The more combinations that are represented, the better the model can do at capturing how each of these attributes affects the house's final price. As a minimum, when building machine learning models, a good starting point is to have at least 10 times as many data points in the dataset as the number of features in the model. We have 18 features in our housing dataset so we'd want a bare minimum of 180 houses to work with. This isn't always an absolute requirement.

## Feature Selection and Scaling

In our house price model, if we include the 18 original features, plus the new features that were created by using one hot in coding, we have a total of 63 features. Some of the features, like the size of the house in sq feet, are probably really important to determining the value of the house. Other features, like whether the house has a fireplace, probably matter less when calculating the final price, but how much less? Maybe there are features that don't matter at all, and we can just remove them from our model. With the tree based machine learning algorithm like radiant boosting, we can actually look at the train model and have it tell us how often each feature is used in determining the final price.
To do that, we call model.feature importances ending with an underscore. In scikit-learn, this will give us an array containing the feature importance for each feature. The total of all feature importances will add up to one, so you can think of this as a percentage rating of how often the feature is used in determining a house's value. To make the list of features easier to read, let's sort them from most important to least important. We'll use numpy's argsort function to give the list of array indexes pointing to each element in the array in order. Then we'll use a forward loop to print out each feature name and how important it is. Let's run the program; right click and choose run. Here at the bottom, we can see that these last few features are the most important in the house's price. The most important factors in determining the house's price is the size, the year it was built, the size of the garage, and the number of bedrooms and the number of bathrooms.

![](https://github.com/abdullahalhoothy/Real_estate_value_prediction/blob/master/images/featureselection.png)

Now, as we have explored our dataset, our next step is to apply feature scaling to the dataframe in order to remove any kind of bias from our data which may arise from non-regularized measurements of different fields. Scaling is achieved using pandas as shown below.

![](https://github.com/abdullahalhoothy/Real_estate_value_prediction/blob/master/images/image5.png)

## Preparing the Data

After diving in deeper, I realized there were some features which were either censored or had a lot of missing values from them. Due to this reason, they did not have a great effect on our model so we will remove them in this step, by simply deleting those fields from our dataframe.

![](https://github.com/abdullahalhoothy/Real_estate_value_prediction/blob/master/images/image6.png)

If you noticed while exploring the dataframe, there is some categorical data in our dataset which needs to be taken care of before we train our data. Namely, garage\_type and city. Now, I hope you&#39;re aware that we cannot use categorical data to train our models. So, what do we do?

There is no ordinal relationship between these two fields, so we cannot possibly use integer encoding as it will lead to the algorithm assuming a relationship which doesn&#39;t exist, eventually resulting in unexpected results.

So, we will use One-Hot encoding here to convert our nominal data into numeric form which can be used by the algorithm.

![](https://github.com/abdullahalhoothy/Real_estate_value_prediction/blob/master/images/image7.png)

Now that everything is set, our concluding step in this phase of data preparation is to make arrays which will hold the data for training. We will also split the data for training and testing respectively, to know how our model is performing later. We will use two arrays, one for the features and the other one for the target value. I.e.: price. Moreover, we will reserve 30% of the data for testing while the rest of the data will be used for training.

![](https://github.com/abdullahalhoothy/Real_estate_value_prediction/blob/master/images/image8.png)

## Choosing the Model

After putting in a lot of research, I concluded that the boosting algorithms perform the best in such scenarios. While the process may be computationally expensive, it gives excellent results as it fits the models sequentially various times and improves the accuracy from the previous mistakes it did.

If you have no knowledge about the boosting models, you can dive in further [here](https://en.wikipedia.org/wiki/Boosting_(machine_learning)).

We will be using Gradient Boosting algorithm to train our data here. The model is defined as follows.

![](https://github.com/abdullahalhoothy/Real_estate_value_prediction/blob/master/images/image9.png)

## Hyperparameter Tuning

Finally, we have arrived at our last step before we can actually get the results. While this step is not necessary, configuring the hyperparameters using grid search can greatly improve the performance of our model.

Basically, Grid Search iteratively goes through our data to return the configurations of hyperparameters which are the most suitable according to our dataset and how it fits the model.

I have used Grid Search CV and below mentioned are the parameters which I will be tuning.

- **n\_estimators** : The total number of trees which we will be creating sequentially. Usually, more the trees, better the results. However, due to the possibility of overfitting, we will be tuning it according to the learning rate.

- **max\_depth** : This is the maximum depth of the tree which is created in our model.

- **min\_samples\_leaf** : The minimal samples required for a terminal node or a leaf. This is quite important if you don&#39;t want overfitting to become a hurdle in your results.

- **learning\_rate** : The rate at which subsequent trees learn and hence contribute to the outcome. Higher the learning rate, more the difference it will make to the respective iteration. Lower values are usually preferred due to their robust nature, but they also require a higher number of trees.

- **max\_features** : The maximum number of features to consider while splitting the nodes.

- **loss** : The loss function which is to be minimized in each split.

Now, we will use 5 fold cross-validation for each set of hyperparameters and fit it on our training data to see how it performs. After that, we will be fitting our model with the tuned hyperparameters onto our training data.

![](https://github.com/abdullahalhoothy/Real_estate_value_prediction/blob/master/images/image10.png)

To see how the tuned hyperparameters perform, we will calculate the performance metrics as the Grid Search fits our model.

![](https://github.com/abdullahalhoothy/Real_estate_value_prediction/blob/master/images/image11.png)

**\*output screenshot\***

## Fitting &amp; Testing the Model

So, the time has finally come when we can actually see what we have built. After fitting the model on the training data and testing it using the testing data that we set aside during our train-test split, we will calculate the error rate on both the training and the test data.

The error rate is the performance metric which will let us know how accurate our model is on our training as well as test data.

Moreover, calculating the error on both testing and training data will also let us know if there is some overfitting in our data.

If the training accuracy is unusually high but the testing accuracy isn&#39;t, it will make the case of overfitting apparent. So, let&#39;s see.

![](https://github.com/abdullahalhoothy/Real_estate_value_prediction/blob/master/images/image12.png)

**\*output screenshot\***

## Wrap Up

Throughout this article we have made a complete machine learning model from scratch. Each part of the process was thoroughly explained, and we also got a complete insight of what a model development pipeline looks like.

I hope that you have understood it thoroughly. If you are also interested in making such models, give this model a try to clear up any confusions. Do leave your feedback and a thumbs up if you liked the article!
