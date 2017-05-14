# A Multinomial Retail Classification Problem solved using a two layer stacking ensemble.

This is a Kaggle classification competition[1] posted by the Otto group, which is one of the biggest e-commerce companies mainly based in Germany and France. Our main goal is to categorize data into one of the nine categories. Each target category represents one important product (like fashion, electronics, etc.). This is in short a multi-nominal retail classification problem.


Prediction efficiency will be given more importance than the interpretability, since this is a competition. All the predictors are masked with no names and descriptions and are treated as such, without associating any meaning to them. The test data set provided by the Otto group for this challenge is thrice as big as the train data set, compelling the resulting model to have bias as low as possible. A good score in the private leaderboard is our primary goal.

No predictor labels and description makes the feature selection/extraction mostly automatic using techniques like wrapping and filtering. One of the biggest concerns was the time taken for these tasks to run. Tasks like Recursive Feature Selection, Boruto and in general all the modeling techniques except for some tree based models run for a long time, making fine-tuning of the model cumbersome. To put this into perspective, a 10 fold cross-validated SVM, runs parallel for more than 1 hour in a 7-core machine.

Summary of findings:

 Manual feature engineering is impossible since the predictors are masked. Feature wrapping (Boruta and RFE) and filtering(stepwiseAIC) could not suggest any predictors to be filtered or subsetted. Feature Reduction methods like PCA couldn’t find any interesting features, whereas t – distributed stochastic neighbor embedding (t-SNE) gave good understanding on the data grouping.
 With the assumption that the features could be word counts, Term Frequency – Inverse Document Frequency (TFIDF) is tested and it produced good results. Thus the final model is trained on combination of both the raw data and TFIDF transformation of the data.
 As expected, high performing algorithms like bagging and boosting performed well on the private leaderboard. XGB has the best score of a single model among the lot. Some models like SVM takes unacceptable computational time, making it unfit to include in the pipeline.
 No single model can beat the private leaderboard. Stacking is the best way to get a good score. Our final model, which is a two-layer stacking, got our best score in the competition.


Project Environment:

This is an already completed Kaggle competition. Still, the site allows us to upload our prediction and it can show hypothetically where we would have stood, if we had participated. Our main goal is to climb up the private leaderboard. The project is started with that in mind and hence all the decisions and assumptions are taken under the impression that prediction efficiency will be more preferred than the model interpretability. Some of the simple and poor kaggle performers like Logistic Regression are not included in the model selection process to save time.

Solution Techniques:

Different iterations of stacking are tried. Voting and averaging are simple ensembling techniques that lead to no notable performance increase. Stacking is a type of Ensembling that are designed to boost predictive accuracy by blending the predictions of multiple machine learning models. Linear techniques like Feature Weighted Linear Stacking (FWLS) that incorporates meta-features for improved accuracy, while retaining the well-known virtues of linear regression [1]. There is no clear way to tell which combinations of models and their weights might perform well. There is no silver bullet to make it work. It’s an iterative trial and error process.

Data Preprocessing:

Skewness, Outliers, Missing Values, Correlation and Transformations:
The data is highly skewed in the range [2,37]. There are no missing values in the data set. With no proper problem description and predictor definitions, outlier detection and removal is not done, since there is no way to tell them apart clearly. The correlation matrix shows that the predictors are not highly correlated. The correlations between the predictors are in the range [0.15, 0.77]. We tried log transformation, that worked well for Neural networks and the decision trees. We also tried TFIDF transformation on the data, which is explained in detail in the Feature Engineering.

Feature Engineering:
Wrapping and Filtering:
Automatic Feature wrapping tools like, Baruto and RFE and filtering tools like stepwiseAIC failed to eliminate any of the predictors. The near zero variance are removed from the predictor list using the caret function of the same name. Out of 94 predictors, 13 are removed.

Feature Transformation:
Under the assumption that the variables are word counts, we tried TFIDF (Term Frequency – Inverse Document Frequency), which is a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpus. The tfidf value increases proportionally to the number of times a word appears on the document, but is offset by the frequency of the word in the corpus, which helps to adjust for the fact that some words appear more frequently in general.
In the case of the term frequency tf(t,d), the simplest choice is to use the raw frequency of a term in a document, i.e. the number of times that term t occurs in document d. If we denote the raw frequency of t by ft,d, then the simple tf scheme is tf(t,d) = ft,d.
The inverse document frequency is a measure of how much information the word provides, that is, whether the term is common or rare across all documents. It is the logarithmically scaled inverse fraction of the documents that contain the word, obtained by dividing the total number of documents by the number of documents containing the term, and then taking the logarithm of that quotient [2].
idf (t, D) = log (N/1+|{d ∈D : t∈d }|)
Where the numerator is the total number of documents in the corpus and the denominator is the number of documents where the term t appears.
TF IDF is then given as,
tfidf(t, d, D) = tf(t, d).idf(t, D)
We will train our models on both log transformation of the data as well as the TF IDF transformed data and then combine them using stacking.

Exploratory Data analysis:
Since this is a classification problem, clustering the groups and visualizing them is important. For this we have used t- SNE (t distributed stochastic neighbor embedding) mainly because of its recent popularity. PCA didn’t yield good results with our data set and took 37 PCA’s for a 75 % cumulative variance, making it less fit for visualization. PCA suffers from a disadvantage that its objective function seeks to minimize the Rayleigh quotient, thereby making it susceptible only to the linear relations between the predictors.

But t-SNE uses Kullback- Leibler divergence as the objective function and can find non-linear relations. There seems to be a good separation between the clusters with t –SNE on both raw and TFIDF data [3].

For an initial Model selection and filtering process, the train data set provided by Kaggle was further divided into local training dataset and hold out validation data set with a 75 : 25 ratio 

Once the models are selected the data is then trained on the entire training data set and then the model is used to predict the test data set values and then submitted to the kaggle for evaluation.

Log-loss is the evaluation metric for the competition.
where,
𝑁 = no. of products in dataset
𝑀 = no. of class labels (9 classes)
𝑙𝑜𝑔 = natural log
𝑦𝑖𝑗 = 1 if observation 𝑖 is in class 𝑗 and 0 otherwise
𝑝𝑖𝑗 = predicted probability that observation 𝑖 belongs to class 𝑗

Without getting into the mathematics, the simple intuition is that the log-loss heavily penalizes the predictions that are more confident in their false predictions, thereby selecting the model that not only predicts the correct class but also minimizes the tendency of the model to have high confidence in wrong predictions.

All the models are run using the caret package as the wrapper with 10 fold cross validation. Though caret exposes only few hyper parameters when compared to the original implementation of the package, caret is still chosen to maintain uniformity in code as well as cv and hyper tuning process. All the models are run parallel using the doParallel package on 7 cores.

Variety of Models including Probabilistic Classifier like Naïve Bayes, Clustering Algorithms like KNN, bagged and boosted trees like RF, XGB, GB and neural networks are tried in an attempt to capture the various hidden relationship that might be present in the data.
RF, XGB and GB are the top performing Models in Kaggle competition in general because of their ability to avoid over fitting and neural networks are the best for un-covering non-linear relationship between the predictors. Some Models like (SVM) took impractical computational time thus cannot be included in the modeling.

Single Models performance:
As expected, XGB has the best single model performance on the data set. It can be clearly seen that the top performer in the private leaderboard has a score of 0.38 and our best model has a log loss of 0.522. With a difference of 0.14 there are 1500 ranks, making the ranking very dense as we further move up the leaderboard. The objective is now to decrease the log loss to as low as possible.

Kaggle has a long history of stacking models winning their competitions [5]. Stacking as briefly mentioned before, is an ensembling method where we combine results of multiple models to a single prediction thereby reducing the over fit a single model might have on the data set. The ensembling methods that we tried are linear Stacking and combining submission files from multiple models.

This is the simplest of the ensembling techniques, where we take the results of the multiple models run separately on the test data set and combine them into a single prediction by taking their average.

The first two entries in table 2 are the example of stacking being the black box. The first two entries have performance even lesser than the best performing single model (XGB). The first model is simply the average of all the models. This has an impressive 0.551 log loss considering that it includes models like Naïve Bayes and KNN, which have log loss of 5.17 and 1.37. Since some models perform poorly and some have better performance, the model results are scaled accordingly and then evaluated. In the second entry, the log-loss decreased, but still greater than the best single model performance. For the third model, a correlation is calculated between some of the best performing algorithms to pick models that are least correlated.

The intuition behind this is that the strong models that are least correlated will have the most diverging output but with good accuracy. When we combine them all together and take an average, the error seems to decrease. The third entry in table 2 has the least log loss so far with a value of 0.52008 and a kaggle rank of 1578, making it the best model so far.

Two layer Stacking:

For the two layer stacking, a different data partitioning is used. The data set is split into two equal data sets. The first level models are then trained on the first partition (l1). After training, the models are used to predict the output of the unseen second partition. These outputs are then added as meta features with the second partition. The second layer models are then trained on all these features. This second layer is used to predict the test data set.
The flow chart explains the steps described above
RF, XGB and KNN are chosen for the first layer model. RF and XGB because of their good performance, whereas KNN seems to work best for meta featuring in our internal trials. The second level models are Neural networks and XGB. Ensemble of these two models is then used to form the final prediction.
This model outperforms all other previous results. The log loss of the final predicted value is 0.43380 with a kaggle rank of 397/3514.

Results and Conclusion:

The results of the single model, ensembling submission files and two layer stacking helps us realize that stacking is indeed better in performance. Also part of the credit for good performance needs to be given to the TFIDF features that are later included in the stacking model that finally gave a very good score in the private leaderboard.
For stacking selecting models for the first layer and the second layer is entirely trial and error and hence not all possible combinations are tested out because of costly computational time needed for a thorough examination. Also neural network can be bagged to further decrease the variance and the log loss of the prediction. Thus we have achieved a log loss of 0.43380 and a kaggle rank of 397/3514.

References:
[1] https://www.kaggle.com/c/otto-group-product-classification-challenge
[2] Joseph Sill, Gabor Takacs, Lester Mackey and David Lin. “Feature-Weighted Linear Stacking “. 4th Nov, 2009
[3] Juan Ramos. “Using TF-IDF to Determine Word Relevance in Document Queries”
[4] Laurens van der Maaten, Geoffrey Hinton. “Visualizing Data using t-SNE”. 9th Nov, 2008
[5] https://dnc1994.com/2016/05/rank-10-percent-in-first-kaggle-competition-en
