# Complete-Guide-to-Regularization-Techniques-in-Machine-Learning

## Introduction
One of the most common problems every Data Science practitioner faces is **Overfitting**. Have you tackled the situation where your machine learning model performed exceptionally well on the train data but was not able to predict on the unseen data or you were on the top of the competition in the public leaderboard, but your ranking drops by hundreds of places in the final rankings?

***Well β this is the article for you!***

Avoiding overfitting can single-handedly improve our modelβs performance.

In this article, we will understand how regularization helps in overcoming the problem of overfitting and also increases the model interpretability.

This article is written under the assumption that you have a basic understanding of **Regression models** including Simple and Multiple linear regression, etc.

## Table of Contents
- π Why Regularization?
- π What is Regularization?
- π How does Regularization work?
- π Techniques of Regularization
Ridge Regression
Lasso Regression
- π Key differences between Ridge and Lasso Regression
- π Mathematical Formulation of Regularization Techniques
- π What does Regularization Achieve?

## Why Regularization?
Sometimes what happens is that our Machine learning model performs well on the training data but does not perform well on the unseen or test data. It means the model is not able to predict the output or target column for the unseen data by introducing noise in the output, and hence the model is called an **overfitted** model.

Letβs understand the meaning of β**Noise**β in a brief manner:

**By noise we mean those data points in the dataset which donβt really represent the true properties of your data, but only due to a random chance.**

So, to deal with the problem of overfitting we take the help of regularization techniques.

## What is Regularization?
- π It is one of the most important concepts of machine learning. This technique prevents the model from overfitting by adding extra information to it.
- π It is a form of regression that shrinks the coefficient estimates towards zero. In other words, this technique forces us not to learn a more complex or flexible model, to avoid the problem of overfitting.
- π Now, letβs understand the βHow flexibility of a model is represented?β For regression problems, the increase in flexibility of a model is represented by an increase in its coefficients, which are calculated              from the regression line.
- π In simple words, βIn the Regularization technique, we reduce the magnitude of the independent variables by keeping the same number of variablesβ. It maintains accuracy as well as a generalization of the model.

## How does Regularization Work?
Regularization works by adding a penalty or complexity term or shrinkage term with Residual Sum of Squares (RSS) to the complex model.

Letβs consider the **Simple linear regression** equation:

Here Y represents the dependent feature or response which is the learned relation. Then,

Y is approximated to **Ξ²0 + Ξ²1X1 + Ξ²2X2 + β¦+ Ξ²pXp**

Here, **X1, X2, β¦Xp** are the independent features or predictors for Y, and

**Ξ²0, Ξ²1,β¦..Ξ²n** represents the coefficients estimates for different variables or predictors(X), which describes the weights or magnitude attached to the features, respectively.

In simple linear regression, our optimization function or loss function is known as the **residual sum of squares (RSS).**

We choose those set of coefficients, such that the following loss function is minimized:
![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F10796501%2F36b82cecfff87a1eb794621a7a450876%2Freg1.png?generation=1659969293112902&alt=media)
Fig. Cost Function For Simple Linear Regression

Now, this will adjust the coefficient estimates based on the training data. If there is noise present in the training data, then the estimated coefficients wonβt generalize well and are not able to predict the future data.

This is where regularization comes into the picture, which shrinks or regularizes these learned estimates towards zero, by adding a loss function with optimizing parameters to make a model that can predict the accurate value of Y.

## Techniques of Regularization
Mainly, there are two types of regularization techniques, which are given below:

- Ridge Regression
- Lasso Regression

## Ridge Regression
π Ridge regression is one of the types of linear regression in which we introduce a small amount of bias, known as **Ridge regression penalty** so that we can get better long-term predictions.

π In Statistics, it is known as the **L-2 norm**.

π In this technique, the cost function is altered by adding the penalty term (shrinkage term), which multiplies the lambda with the squared weight of each individual feature. Therefore, the optimization function(cost function) becomes:
![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F10796501%2F8c7e28ae380de1048b3f275da292528f%2Freg2.png?generation=1659969556422142&alt=media)
Fig. Cost Function for Ridge Regression

In the above equation, the penalty term regularizes the coefficients of the model, and hence ridge regression reduces the magnitudes of the coefficients that help to decrease the complexity of the model.

π **Usage of Ridge Regression:**

When we have the independent variables which are having high collinearity (problem of multicollinearity) between them, at that time general linear or polynomial regression will fail so to solve such problems, Ridge regression can be used.If we have more parameters than the samples, then Ridge regression helps to solve the problems.
 
π **Limitation of Ridge Regression:**

- **Not helps in Feature Selection**: It decreases the complexity of a model but does not reduce the number of independent variables since it never leads to a coefficient being zero rather only minimizes it. Hence, this technique is not good for feature selection.
- **Model Interpretability**: Its disadvantage is model interpretability since it will shrink the coefficients for least important predictors, very close to zero but it will never make them exactly zero. In other words, the final model will include all the independent variables, also known as predictors.

## Lasso Regression
π Lasso regression is another variant of the regularization technique used to reduce the complexity of the model. It stands for **Least Absolute and Selection Operator**.

π It is similar to the Ridge Regression except that the penalty term includes the absolute weights instead of a square of weights. Therefore, the optimization function becomes:
![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F10796501%2Fe9ee7e0d4cf20a21610ad927b00baf17%2Freg3.png?generation=1659969929884448&alt=media)
Fig. Cost Function for Lasso Regression

π In statistics, it is known as the **L-1 norm**.

π In this technique, the L1 penalty has the eο¬ect of forcing some of the coeο¬cient estimates to be exactly equal to zero which means there is a complete removal of some of the features for model evaluation when the tuning parameter Ξ» is suο¬ciently large. Therefore, the lasso method also performs **Feature selection** and is said to yield **sparse models**.

π **Limitation of Lasso Regression**:

- **Problems with some types of Dataset**: If the number of predictors is greater than the number of data points, Lasso will pick at most n predictors as non-zero, even if all predictors are relevant.
- **Multicollinearity Problem**: If there are two or more highly collinear variables then LASSO regression selects one of them randomly which is not good for the interpretation of our model.

## Key Differences between Ridge and Lasso Regression
π Ridge regression helps us to reduce only the overfitting in the model while keeping all the features present in the model. It reduces the complexity of the model by shrinking the coefficients whereas Lasso regression helps in reducing the problem of overfitting in the model as well as automatic feature selection.

π Lasso Regression tends to make coefficients to absolute zero whereas Ridge regression never sets the value of coefficient to absolute zero.

## Mathematical Formulation of Regularization Techniques
π Now, we are trying to formulate these techniques in mathematical terms. So, these techniques can be understood as solving an equation,

**For ridge regression**, the total sum of squares of coefficients is less than or equal to s and **for Lasso regression**, the total sum of modulus of coefficients is less than or equal to s.

**Here, s is a constant which exists for each value of the shrinkage factor Ξ».**

These equations are also known as constraint functions.

π Letβs take an example to understand the mathematical formulation clearly,

**For Example, Consider there are 2 parameters for a given problem

**Ridge regression:**

According to the above mathematical formulation, the ridge regression is described by **Ξ²1Β² + Ξ²2Β² β€ s.**

This implies that ridge regression coefficients have the smallest RSS (loss function) for all points that lie within the circle given by **Ξ²1Β² + Ξ²2Β² β€ s.**

**Lasso Regression:**

According to the above mathematical formulation, the equation becomes,**|Ξ²1|+|Ξ²2|β€ s.**

This implies that the coefficients for lasso regression have the smallest RSS (loss function) for all points that lie within the diamond given by **|Ξ²1|+|Ξ²2|β€ s.**

The image below describes these equations:
![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F10796501%2F8adc682c28fbcb683e755df78ec98eb0%2Freg4.png?generation=1659970348952760&alt=media)

**Description About Image**: The given image shows the constraint functions(in green areas), for lasso(in left) and ridge regression(in right), along with contours for RSS(red ellipse).

Points on the ellipse describe the value of Residual Sum of Squares (RSS) which is calculated for simple linear regression.

π **For a very large value of s**, the green regions will include the center of the ellipse with itself, which makes the coefficient estimates of both regression techniques equal to the least-squares estimates of simple linear regression. But, the given image shown does not describe this case. In that case, coefficient estimates of lasso and ridge regression are given by the ο¬rst point at which an ellipse interacts with the constraint region.

**Ridge Regression**: Since ridge regression has a circular type constraint region, having no sharp points, so the intersection with the ellipse will not generally occur on the axes, therefore, the ridge regression coeο¬cient estimates will be exclusively non-zero.

**Lasso Regression**: Lasso regression has a diamond type constraint region that has corners at each of the axes, so the ellipse will often intersect the constraint region at axes. When this happens, one of the coeο¬cients (from collinear variables) will be zero and for higher dimensions having parameters greater than 2, many of the coeο¬cient estimates may equal zero simultaneously.

## What does Regularization achieve?
π In simple linear regression, the standard least-squares model tends to have some variance in it, i.e. this model wonβt generalize well for a future data set that is different from its training data.

π Regularization tries to reduce the variance of the model, without a substantial increase in the bias.

π **How Ξ» relates to the principle of βCurse of Dimensionalityβ?**

As the value of Ξ» rises, it significantly reduces the value of coefficient estimates and thus reduces the variance. Till a point, this increase in Ξ» is beneficial for our model as it is only reducing the variance (hence avoiding overfitting), without losing any important properties in the data. But after a certain value of Ξ», the model starts losing some important properties, giving rise to bias in the model and thus underfitting. Therefore, we have to select the value of Ξ» carefully. To select the good value of Ξ», cross-validation comes in handy.

**Important points about Ξ»:**
- Ξ» is the tuning parameter used in regularization that decides how much we want to penalize the flexibility of our model i.e, **controls the impact on bias and variance.**
- When **Ξ» = 0,** the penalty term has no eο¬ect, the equation becomes the cost function of the linear regression model. Hence, for the minimum value of Ξ» i.e, Ξ»=0, the model will resemble the linear regression model. So, the estimates produced by ridge regression will be equal to least squares.
- However, as **Ξ»ββ**(tends to infinity), the impact of the shrinkage penalty increases, and the ridge regression coeο¬cient estimates will approach zero.

## End Notes
Thanks for reading!

If you like this article, you may upvote and follow my [Kaggle](https://www.kaggle.com/bulentorun) account for other articles on Data Science and Machine Learning. Please kindly feel free to contact me on [LinkedIn](https://www.linkedin.com/in/bulentorun/) and [Github](https://github.com/bullor)

Something not mentioned or want to share your thoughts? Feel free to reach out and Iβll get back to you.


