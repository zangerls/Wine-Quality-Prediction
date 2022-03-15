# Wine Quality Prediction

This aim of this project is to find the optimal regression model to predict wine's quality.

## The data

The data used for the analysis comes from [Kaggle.](https://www.kaggle.com/yasserh/wine-quality-dataset)

## Machine Learning

#### Overview
1. [First look at the data](#1---first-look-at-the-data)
2. [Cleaning the data](#2---cleaning-the-data)
3. [Scaling the data](#3---scaling-the-data)
4. [Feature Selection](#4---feature-selection)
5. [Preparing the data](#5---preparing-the-data)
6. [Training the models](#6---training-the-models)
7. [Evaluating the models](#7---evaluating-the-models)
8. [Comparing the models](#8---comparing-the-models)

### 1. - First look at the data

```py
df = pd.read_csv('WineQT.csv')
df.head()

df.describe()
```

### 2. - Cleaning the data

```py
df.isna().sum().sort_values(ascending=False)

df.dtypes

df.drop('Id', axis=1, inplace=True)

plt.hist(df['quality'])
plt.title('Quality')
plt.show()

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='Blues')
plt.show()

X = df.drop('quality', axis=1)
y = df['quality']
```

### 3. - Scaling the data

```py
X_cols = X.columns

sc = StandardScaler()

sc.fit(X)
array_scaled = sc.transform(X)

X = pd.DataFrame(X, columns=X_cols)
```

### 4. - Feature Selection

```py
lr = LinearRegression()
rfe = RFE(estimator=lr, n_features_to_select=7, step=1)
rfe.fit(X, y)
rfe.ranking_

features_to_keep = rfe.get_support(1)
features_to_keep

X = X[X.columns[features_to_keep]]
X.head()
```

### 5. - Preparing the data

```py
X_test, X_train, y_test, y_train = train_test_split(X, y, test_size=0.2, random_state=44)
```

### 6. - Training the models

#### 6.1. - Linear Regression

```py
lr = LinearRegression()
regressor_lr = lr.fit(X_test, y_test)
y_pred_lr = regressor_lr.predict(X_test)
```

#### 6.2. - K Nearest Neighbors Regression

##### 6.2.1 - Finding best n

```py
AE_lst = []
MSE_lst = []
RMSE_lst = []
neighbors = range(1,11)

for i in neighbors:
    knn = KNeighborsRegressor(n_neighbors=i)
    regressor_loop = knn.fit(X_train, y_train)
    y_pred = regressor_loop.predict(X_test)
    
    MAE_lst.append(mean_absolute_error(y_test, y_pred))
    MSE_lst.append(mean_squared_error(y_test, y_pred))
    RMSE_lst.append(np.sqrt(mean_squared_error(y_test, y_pred)))

d = {'Neighbors': neighbors, 'MAE': MAE_lst, 'MSE': MSE_lst, 'RMSE': RMSE_lst}

errors = pd.DataFrame(data=d)
plt.plot(errors['MAE'])
plt.show()
```

![knn](https://user-images.githubusercontent.com/66888655/158061420-947da41e-c681-4342-8b74-41f10ba686a6.png)


#### 6.3. - Random Forest Regression

```py
knn = KNeighborsRegressor(n_neighbors=4)
regressor_knn = knn.fit(X_train, y_train)
y_pred_knn = regressor_knn.predict(X_test)
```

### 7. - Evaluating the models

#### 7.1. - Linear Regression

##### 7.1.1. - Calculating Errors

```py
mae_lr = mean_absolute_error(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)

print(f'Mean Absolute Error: {mae_lr}')
print(f'Mean Squared Error: {mse_lr}')
print(f'Root Mean Squared Error: {rmse_lr}')
```

##### 7.1.2. - Comparing real quality with predicted quality

```py
entry_X_100 = X.values[100].reshape(1,-1)
entry_y_100 = y.loc[100]
print(X.loc[100], entry_y_100)

y_pred_lr_100 = regressor_lr.predict(entry_X_100)

print(f'Predicted Quality: {y_pred_lr_100[0]}')
print(f'Actual Value: {entry_y_100}')
```

#### 7.2. - K Nearest Neighbor Regression

##### 7.2.1. - Calculating Errors

```py
mae_knn = mean_absolute_error(y_test, y_pred_knn)
mse_knn = mean_squared_error(y_test, y_pred_knn)
rmse_knn = np.sqrt(mse_knn)

print(f'Mean Absolute Error: {mae_knn}')
print(f'Mean Squared Error: {mse_knn}')
print(f'Root Mean Squared Error: {rmse_knn}')
```

##### 7.2.2. - Comparing real quality with predicted quality

```py
entry_X_100 = X.values[100].reshape(1,-1)
entry_y_100 = y.loc[100]
print(X.loc[100], entry_y_100)

y_pred_knn_100 = regressor_knn.predict(entry_X_100)

print(f'Predicted Quality: {y_pred_knn_100[0]}')
print(f'Actual Value: {entry_y_100}')
```

#### 7.3. - Random Forest Regression

##### 7.3.1. - Calculating Errors

```py
mae_rf = mean_absolute_error(y_test, y_pred_lr)
mse_rf = mean_squared_error(y_test, y_pred_lr)
rmse_rf = np.sqrt(mse_rf)

print(f'Mean Absolute Error: {mae_rf}')
print(f'Mean Squared Error: {mse_rf}')
print(f'Root Mean Squared Error: {rmse_rf}')
```

##### 7.1.2. - Comparing real quality with predicted quality

```py
entry_X_100 = X.values[100].reshape(1,-1)
entry_y_100 = y.loc[100]
print(X.loc[100], entry_y_100)

y_pred_rf_100 = regressor_rf.predict(entry_X_100)

print(f'Predicted Quality: {y_pred_rf_100[0]}')
print(f'Actual Value: {entry_y_100}')
```

### 8. - Comparing the models

```py
labels = ['Linear Regression','KNN','Random Forest','Actual']
values = [y_pred_lr_100, y_pred_knn_100, y_pred_rf_100,entry_y_100]

plt.bar(labels, values)
plt.title('Predicted scores vs. Actual Score')
plt.ylabel('Score')
plt.xlabel('Predicting model')
plt.show()
```

![compare 1](https://user-images.githubusercontent.com/66888655/158061400-14ec719a-8106-44cb-88d0-744e553a0159.png)


```py
# np.floats need to be converted to python floats for visualization with matplotlib
errors_lr = [mae_lr, mse_lr, rmse_lr]
errors_lr = [float(x) for x in errors_lr]

errors_knn = [mae_knn, mse_knn, rmse_knn]
errors_knn = [float(x) for x in errors_knn]

errors_rf = [mae_rf, mse_rf, rmse_rf]
errors_rf = [float(x) for x in errors_rf]

cols = ['Mean Absolute Error','Mean Squared Error','Root Mean Squared Error']
x_axis = np.arange(len(cols))

plt.bar(x_axis +0.20, errors_lr, width=0.2, label = 'Linear Regression')
plt.bar(x_axis +0.20*2, errors_knn, width=0.2, label = 'KNN Regression')
plt.bar(x_axis +0.20*3, errors_rf, width=0.2, label = 'Random Forest Regression')

plt.xticks(x_axis, cols)
plt.legend()
plt.title('Errors by Model')
plt.show()
```

![compare 2](https://user-images.githubusercontent.com/66888655/158061406-7ac199ca-c77c-4847-aa71-ee43ca07675d.png)
