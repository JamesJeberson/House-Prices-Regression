# House Prices - Advanced Regression


```python
# Importing the required libraries

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
```


```python
# Importing the 'train.csv' and 'test.csv' as pandas.dataframe

df_1 = pd.read_csv('train.csv')
df_2 = pd.read_csv('test.csv')
if 'SalePrice' not in df_2.columns:
    df_2['SalePrice'] = 0
df = pd.concat([df_1, df_2], axis = 0)
df = df.set_index('Id')
```


```python
df.head()
```


```python
# Creating a DataFrame with all the columns having null values

df_null = df[df.isna().sum()[df.isna().sum()>0].index]
```


```python
df_null.head()
```


```python
# Creating a headmap to visualize the occurance of null values

sns.heatmap(df_null.isna())
```


```python
# Creating a DataFrame to hold the columns which are 'objects' 
df_objects = df[df.select_dtypes(include = ['object']).columns]

# Dropping columns with dtype 'object' and has more than 1100 null instances
df = df.drop(df[df_objects.isna().sum()[df_objects.isna().sum()>1100].index], axis = 1)
```


```python
# Dropping columns wih more than 1100 null instances
df_objects = df_objects.drop(df_objects[df_objects.isna().sum()[df_objects.isna().sum()>1100].index], axis = 1)

# Replacing NaN with 'null'
df_objects = df_objects.fillna('null')

# One hot encoding 'df_objects'
df_objects_encoded = pd.get_dummies(df_objects)

```


```python
df_objects['MSZoning'].value_counts()
```


```python
df_objects_encoded.head()
```


```python
# Dropping all 'null' columns created during one hot encoding (redundant data)

for i in df_objects_encoded.columns:
    if 'null' in i:
        df_objects_encoded = df_objects_encoded.drop(i, axis = 1)
        print(i)
```


```python
# Creating new DataFrame with the encoded columns
new_df = pd.concat([df, df_objects_encoded], axis = 1)

len(df.columns), len(df_objects_encoded.columns), len(new_df.columns)
```


```python
# Dropping columns with dtypr 'object' from new_df
new_df = new_df.drop(df.select_dtypes(include =['object']), axis = 1)

new_df.head()
```


```python
# Checking for NaN values in new_df 
new_df.isna().sum()[new_df.isna().sum()>0]
```


```python
sns.heatmap(new_df.isna())
```


```python
# Defining columns to be filled using mode and mean strategies
mode_columns = ['GarageCars', 'GarageYrBlt', 'BsmtFullBath', 'BsmtHalfBath']
mean_columns = ['LotFrontage', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
                'TotalBsmtSF', 'GarageArea']

# Filling NaN with Mode strategy
for i in mode_columns:
    new_df[i] = new_df[i].fillna(new_df[i].mode()[0])

# Filling NaN with Mean Strategy
for i in mean_columns:
    new_df[i] = new_df[i].fillna(np.round(new_df[i].mean()))
```


```python
# Verifying for NaN values 
new_df.isna().max(axis = 0).max()
```


```python
# Splitting the training data and testing data used (df_1, df_2)
training_data = new_df[0: len(df_1)]
test_data = new_df[len(df_1):]

# Dropping 'SalePrice' from test data
test_data = test_data.drop(columns='SalePrice')
```


```python
# Importing required libraries for machine learning

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error 
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
```


```python
# Creating X_train, X_test, Y_train, Y_test for Regression

X = training_data.drop(columns = 'SalePrice')
y = training_data['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Reshaping from 1D array to 2D array
y_train = np.reshape(y_train, (-1, 1))
y_test = np.reshape(y_test, (-1, 1))

X_train.shape, y_train.shape
```


```python
# Linear Regression

model_1 = LinearRegression().fit(X, y)
y1_pred = model_1.predict(X_test)

mean_squared_error(y_test, y1_pred)
```


```python
# XGBRegressor

model_2 = XGBRegressor(n_estimators=1000, learning_rate=0.1, random_state=42)
model_2.fit(X, y)
y2_pred = model_2.predict(X_test)
mean_squared_error(y_test, y2_pred)
```


```python
# Random Forrest Regressor

model_3 = RandomForestRegressor(n_estimators=1000)
model_3.fit(X, y)
y3_pred = model_3.predict(X_test)
mean_squared_error(y_test, y3_pred)
```


```python
# Plotting Original vs Predicted values (XGBRegressor)

plt.figure(figsize=(10, 6))
plt.plot(np.arange(len(y_test)), y_test, label='Original', color='blue', linestyle='-', linewidth=2)
plt.plot(np.arange(len(y_test)), y2_pred, label='Predicted', color='green', linestyle='--', linewidth=2)
plt.xlabel('Index')
plt.ylabel('Price')
plt.title('Original vs Predicted Prices')
plt.legend()
plt.show()
```


```python
# The XGBRegressor performs best hence using model_2 for final prediction

pred = model_2.predict(test_data)

final = pd.DataFrame()
final['Id'] = test_data.index
final['SalePrice'] = pred

# Write DataFrame to a CSV file without index
final.to_csv('output.csv', index=False)
```
