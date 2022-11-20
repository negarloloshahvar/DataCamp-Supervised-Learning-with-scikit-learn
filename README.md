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
predictions = model.predict(X_new)
print('Predictions: {}'.format(predictions))

```

- ```churn_df.values``` returns a numpy array with the underlying data of the DataFrame, *without any index or columns names*.

### Train/testsplit

```
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21, stratify=y)
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, y_train)
print(knn.score(X_test, y_test))
```
```
Modelcomplexityandover/underfitting
train_accuracies = {}
test_accuracies = {}
neighbors = np.arange(1, 26)
for neighbor in neighbors:
    knn = KNeighborsClassifier(n_neighbors=neighbor)
    knn.fit(X_train, y_train)
    train_accuracies[neighbor] = knn.score(X_train, y_train)
    test_accuracies[neighbor] = knn.score(X_test, y_test)
plt.figure(figsize=(8, 6))
plt.title("KNN: Varying Number of Neighbors")
plt.plot(neighbors, train_accuracies.values(), label="Training Accuracy")
plt.plot(neighbors, test_accuracies.values(), label="Testing Accuracy")
plt.legend()
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy")
plt.show()
```

## Chapter 2: Regression

## Chapter 3: Fine-Tuning Your Model

## Chapter 4: Preprocessing and Pipelines
