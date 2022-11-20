# DataCamp-Supervised-Learning-with-scikit-learn

## Chapter 1: Classification

In this chapter, we will learn about classification problems and learn how to solve them using supervised learning techniques. We'll learn how to split data into training and test sets, fit a model, make predictions, and evaluate accuracy. We’ll also discover the relationship between model complexity and performance, applying what we learn to a churn dataset, where we will classify the churn status of a telecom company's customers.

### Requirements:
- No missing values
- Data in numeric format
- Data stored in pandas DataFrame or NumPy array
- Perform Exploratory Data Analysis (EDA) first


### The supervised learning workflow
Scikit-learn offers a repeatable workflow for using supervised learning models to predict the target variable values when presented with new data.

```
from sklearn.module import Model
model = Model()
model.fit(X, y)
predictions = model.predict(X_new)
print(predictions)
```

### 1. k-Nearest Neighbors

The k-nearest neighbors algorithm, also known as KNN or k-NN, is a non-parametric, supervised learning classifier, which uses *proximity* to make classifications or predictions about the grouping of an individual data point.

```
from sklearn.neighbors import KNeighborsClassifier
X = churn_df[["total_day_charge", "total_eve_charge"]].values
y = churn_df["churn"].values
knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(X, y)
X_new
print('Predictions: {}'.format(predictions))

```

## Chapter 2: Regression

## Chapter 3: Fine-Tuning Your Model

## Chapter 4: Preprocessing and Pipelines
