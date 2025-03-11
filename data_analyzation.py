import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split

from practice import ridge_Model

file_path = "Data/kc_house_data_NaN.csv"
df = pd.read_csv(file_path)

#print(df.head())

#Question1
#   print(df.dtypes)

#Question2
df.drop(["id", "Unnamed: 0"], axis=1, inplace=True)
#   print(df.describe())

#   print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
#   print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())
mean=df['bedrooms'].mean()
df['bedrooms'].replace(np.nan,mean, inplace=True)
mean=df['bathrooms'].mean()
df['bathrooms'].replace(np.nan,mean, inplace=True)

#Question 3
#   print(df.value_counts("floors").to_frame())

#Question 4
#   pd.set_option("display.max_columns", None)
#   print(df.describe())

plt.figure(figsize=(16, 10))
sns.boxplot(
    x=df["waterfront"],
    y=df["price"])
#   plt.show()

#Question 5
sns.regplot(
    x=df["sqft_above"],
    y=df["price"],
    scatter_kws = {"alpha": 0.3, "color": "blue"},
    line_kws = {"color": "red", "linewidth": 2}
)
#   plt.show()

#Question 6
lin_m = LinearRegression()
lin_m.fit(df[["sqft_living"]], df["price"])
#   print(lin_m.score(df[["sqft_living"]], df["price"]))

#Question 7
features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]
lin_m.fit(df[features], df["price"])
#   print(lin_m.score(df[features], df["price"]))

#Question 8
Input=[('scale',StandardScaler()),('polynomial', PolynomialFeatures(include_bias=False)),('model',LinearRegression())]
pipe = Pipeline(Input)
pipe.fit(df[features],df["price"])
#print(pipe.score(df[features],df["price"]))

#Question 9
X_train, X_test, y_train, y_test = train_test_split(df[features], df["price"], test_size=0.15, random_state=1)

ridge_model = Ridge(alpha=0.1)
ridge_model.fit(X_train, y_train)
predicted_y = ridge_model.predict(X_test)
#   print(r2_score(y_test, predicted_y))

#Question 10
pr = PolynomialFeatures(degree=2)
x_train_pr = pr.fit_transform(X_train)
x_test_pr = pr.fit_transform(X_test)
ridge_Model.fit(x_train_pr, y_train)
y_hat = ridge_Model.predict(x_test_pr)
print(r2_score(y_test,y_hat))
