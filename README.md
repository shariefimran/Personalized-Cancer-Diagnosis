Analysis on Cancer dataset with different Models(Classifiers) and return probability score for each Class

dataset = https://www.kaggle.com/c/msk-redefining-cancer-treatment/data

First I plotted histogram for plotting the different classes with 'Gene' and 'Variation' feature and then featurized the data set with 'Gene'+Variation'+'Text' and then converted 'Text' to Tfidf-Vectorizer using one-hot encoding and then split them in Train, CV and Test data.

Then, we applied different models and plotted Confusion matrix and reported log-loss and made the model interpretable by reporting the probability.

Techniques Used:

Naive Bayes
K-NN
Logistic Regression
Linear SVM
Random Forest
