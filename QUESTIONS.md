# Questions
Please try to answer the following questions.
Reference your own code/Notebooks where relevant.

## Your solution
1. What machine learning techniques did you use for your solution? Why?

  I tried an Elastic Net approach, i.e. a combination of ridge regression and lasso regression. The idea is to provide a minimum of insight as to what features are needed for predicting the future consumption. Of course there are more accurate approaches (e.g. boosting or bagging...) but the whole idea was to keep the approach simple and the results easily interpretable.

  the variables initially supplied to the model are the following:
  * half-hourly observations for the past two weeks or so of consumption
  * time (in half-hour intervals)
  * day of the week
  * weekend yes/no
  * month (in the hope of at least partially capturing seasonal holidays)
  * household id

  The categorical variables (all of them with the except of past consumption and weekend flag) have been converted to numeric via 1-hot encoding, and the model trained without intercept.

  The hyper-parameters of the model were assessed via 10-fold cross validation on the training set.

  The model also takes as input the household id, so different predictions will be made for the individual households.

2. What is the error of your prediction?

   a. How did you estimate the error?

   The error is measures as Root Mean Squared Error and is equivalent to 0.18 units - this measure is expressed in the same unit as the y variable.

   b. How do the train and test errors compare?

   there is no label provided for the test set so it is not possible to assess the test error. I expect the test error to be marginally higher than the training error for two reasons:

   * the true error can never be totally assessed via validation or cross-validation, so test error tends to be always a bit higher
   * in building the model, the full consumption history is available. In the test set, the features themselves are output of the prediction model - see next answer.

   c. How will this error change for predicting a further week/month/year into the future?

   Qualitatively, the error will increase the further in the future we try to predict. This is typical of time series models where a given observation is explained by, among other features, current past observations of the same y variable.

   If the time series is sufficiently stationary, predictions could be made long into the future. This statement would likely be in conflict with experience as households tend to modify their lifestyle depending on family circumstances. Weather (which affects consumption) is an example of another variable which does not necessarily repeat itself identically every year.

3. What improvements to your approach would you pursue next? Why?

  This is just a quick exercise. Making predictions for 1 week as per the test set could take up to 30 minutes to finalize on just 10 households (no parallelization implemented in this exercise), so it is obvious that the model does not scale well.

  One option is to cluster households into a few homogeneous groups such that all households in a similar group behave sufficiently similarly to one another, and sufficiently differently from households in other groups.

  Another option of course is using a different model which, applied to a single household or cluster, performs 1-week predictions more efficiently (meaning with less feature engineering involved).

4. Will your approach work for a new household with little/no half-hourly data?
   How would you approach forecasting for a new household?

   A similarity measure could be developed such that, based on the little information available, a household could be identified that behaves similar to the object household, and the model from that identified household could be used instead.

   Little can be done about households with no data other than applying a prediction based on the 'typical' average household consumption and allow for an excess proportionally to what the cost is of buying energy at spot price rather vs buying energy too early.

   Of course, there could be more sophisticated ways for making a prediction for a household with no usage history, e.g. how many components in the household, age, profession, house size, etc.
