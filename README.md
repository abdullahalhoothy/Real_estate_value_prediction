# Real Estate Value Predictor

## Agenda

### 1.Motivation
### 2.Introduction
### 3.The Dataset
### 4.Feature Scaling
### 5.Preparing the Data
### 6.Choosing the Model
### 7.Hyperparameter Tuning
### 8.Fitting &amp; Testing the Model
### 9.Wrap Up

##

## Motivation

An real estate value predictor is important not only from the perspective of a buyer, but it also comes in pretty handy to someone who&#39;s planning to sell his or her house but is unsure of what price they should put it up at. Moreover, for someone who likes to play around with different algorithms using a project based approach, this is a great idea to start with, since it covers all the basic principles used in developing an effective machine learning model. This is pretty much the reason why I first came up with the idea of getting my hands dirty with this project.

## Introduction

Ever since the evolution of machine learning, predicting real estate value using it has always been a hot topic. Although several models already do exist on the internet, I found them to be missing some important features which needed to be taken into considering.

In this article, we will cover everything from data cleaning to final model performance and I will be guiding you step by step through all the procedure, so you can have a deep understanding of each step and a firm grasp of the major concepts involved. So, let&#39;s get started without any further ado.

## The Dataset

Exploring the dataset should always be your first step before anything else. While you can also explore the data using pandas, I generally prefer to have a good look at it manually as well to notice any trends which are readily visible.

So, let&#39;s import some libraries needed for the model and have a glimpse of our dataset.

![](RackMultipart20200821-4-1w9mbnh_html_1761e11783f0bed5.png)

![](RackMultipart20200821-4-1w9mbnh_html_15a85c98d6f44e45.png)

![](RackMultipart20200821-4-1w9mbnh_html_bc6b44765168cfc7.png)

![](RackMultipart20200821-4-1w9mbnh_html_55c5c7eed7dd4515.png)

So, we have over 42 thousand records of real estate prices with each of them consisting of 20 features.

## Feature Scaling

Now, as we have explored our dataset, our next step is to apply feature scaling to the dataframe in order to remove any kind of bias from our data which may arise from non-regularized measurements of different fields. Scaling is achieved using pandas as shown below.

![](RackMultipart20200821-4-1w9mbnh_html_5fed1b0ebc282c18.png)

## Preparing the Data

After diving in deeper, I realized there were some features which were either censored or had a lot of missing values from them. Due to this reason, they did not have a great effect on our model so we will remove them in this step, by simply deleting those fields from our dataframe.

![](RackMultipart20200821-4-1w9mbnh_html_7e181b3c4b1e8b5f.png)

If you noticed while exploring the dataframe, there is some categorical data in our dataset which needs to be taken care of before we train our data. Namely, garage\_type and city. Now, I hope you&#39;re aware that we cannot use categorical data to train our models. So, what do we do?

There is no ordinal relationship between these two fields, so we cannot possibly use integer encoding as it will lead to the algorithm assuming a relationship which doesn&#39;t exist, eventually resulting in unexpected results.

So, we will use One-Hot encoding here to convert our nominal data into numeric form which can be used by the algorithm.

![](RackMultipart20200821-4-1w9mbnh_html_a7dd94bf6b980665.png)

Now that everything is set, our concluding step in this phase of data preparation is to make arrays which will hold the data for training. We will also split the data for training and testing respectively, to know how our model is performing later. We will use two arrays, one for the features and the other one for the target value. I.e.: price. Moreover, we will reserve 30% of the data for testing while the rest of the data will be used for training.

![](RackMultipart20200821-4-1w9mbnh_html_f35a58358e5bad71.png)

## Choosing the Model

After putting in a lot of research, I concluded that the boosting algorithms perform the best in such scenarios. While the process may be computationally expensive, it gives excellent results as it fits the models sequentially various times and improves the accuracy from the previous mistakes it did.

If you have no knowledge about the boosting models, you can dive in further [here](https://en.wikipedia.org/wiki/Boosting_(machine_learning)).

We will be using Gradient Boosting algorithm to train our data here. The model is defined as follows.

![](RackMultipart20200821-4-1w9mbnh_html_394b8faada2becfe.png)

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

![](RackMultipart20200821-4-1w9mbnh_html_b235d862aeccf9e6.png)

To see how the tuned hyperparameters perform, we will calculate the performance metrics as the Grid Search fits our model.

![](RackMultipart20200821-4-1w9mbnh_html_5c6c82dee5d5a27a.png)

**\*output screenshot\***

## Fitting &amp; Testing the Model

So, the time has finally come when we can actually see what we have built. After fitting the model on the training data and testing it using the testing data that we set aside during our train-test split, we will calculate the error rate on both the training and the test data.

The error rate is the performance metric which will let us know how accurate our model is on our training as well as test data.

Moreover, calculating the error on both testing and training data will also let us know if there is some overfitting in our data.

If the training accuracy is unusually high but the testing accuracy isn&#39;t, it will make the case of overfitting apparent. So, let&#39;s see.

![](RackMultipart20200821-4-1w9mbnh_html_77b1d089876b42fb.png)

**\*output screenshot\***

## Wrap Up

Throughout this article we have made a complete machine learning model from scratch. Each part of the process was thoroughly explained, and we also got a complete insight of what a model development pipeline looks like.

I hope that you have understood it thoroughly. If you are also interested in making such models, give this model a try to clear up any confusions. Do leave your feedback and a thumbs up if you liked the article!
