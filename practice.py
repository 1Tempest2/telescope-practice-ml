import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split

file_path = "Data/medical_insurance_dataset.csv"

headers = ["age", "gender", "bmi", "no_of_children", "smoker", "region", "charges"]
df = pd.read_csv(file_path, header = None)
df.columns = headers
#   print(df.head(10))
df.replace('?', np.nan, inplace=True)

most_likely_smoking = df["smoker"].value_counts().idxmax()
df["smoker"].replace(np.nan, most_likely_smoking, inplace=True)
mean_age = df["age"].astype("float").mean()
df["age"].replace(np.nan, mean_age, inplace=True)
df[["age","smoker"]] = df[["age", "smoker"]].astype("int")
df["charges"] = np.round(df[["charges"]],2)
#   print(df.head())

plt.figure(figsize=(8, 5))


sns.regplot(
    x=df["bmi"],
    y=df["charges"],
    scatter_kws={"alpha": 0.5},
    line_kws={"color": "red", "linewidth": 2}
)
#   plt.show()

sns.boxplot(
    x = "smoker",
    y = "charges",
    data = df)
#   plt.show()

#print(df.corr())
#heatmap works for this aswell
sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm",
    linewidths=0.5,
    vmin=-1, vmax=1
)
#   plt.show()

linear_regression_model = LinearRegression()
linear_regression_model.fit(df[["smoker"]], df["charges"])
#   print(linear_regression_model.score(df[["smoker"]], df["charges"]))
other_columns = df.drop("charges", axis = 1).copy()
linear_regression_model.fit(other_columns, df["charges"])
#   print(linear_regression_model.score(other_columns, df["charges"]))
Input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model', LinearRegression())]
pipe=Pipeline(Input)
other_columns = other_columns.astype("float")
pipe.fit(other_columns, df["charges"])
ypipe = pipe.predict(other_columns)
#   print(r2_score(df["charges"], ypipe))

X_train, X_test, y_train, y_test = train_test_split(other_columns, df["charges"], test_size=0.2, random_state=1)

ridge_Model = Ridge(alpha=0.1)
ridge_Model.fit(X_train, y_train)
predicted_y = ridge_Model.predict(X_test)
#   print(r2_score(y_test, predicted_y))

pr = PolynomialFeatures(degree=2)
x_train_pr = pr.fit_transform(X_train)
x_test_pr = pr.fit_transform(X_test)
ridge_Model.fit(x_train_pr, y_train)
y_hat = ridge_Model.predict(x_test_pr)
#print(r2_score(y_test,y_hat))
