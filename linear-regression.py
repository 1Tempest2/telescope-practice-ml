import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
import copy
import seaborn as sns
import tensorflow as tf
from sklearn.linear_model import LinearRegression
from tensorflow.python.keras.utils.vis_utils import plot_model


def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.xlabel('Epoch')
  plt.ylabel('MSE')
  plt.legend()
  plt.grid(True)
  plt.show()

def get_xy(dataframe, y_label, x_labels=None):
  dataframe = copy.deepcopy(dataframe)
  if x_labels is None:
    X = dataframe[[c for c in dataframe.columns if c!=y_label]].values
  else:
    if len(x_labels) == 1:
      X = dataframe[x_labels[0]].values.reshape(-1, 1)
    else:
      X = dataframe[x_labels].values

  y = dataframe[y_label].values.reshape(-1, 1)
  data = np.hstack((X, y))

  return data, X, y

dataset_cols = ["bike_count", "hour", "temp", "humidity", "wind", "visibility", "dew_pt_temp", "radiation", "rain", "snow", "functional"]
df = pd.read_csv("Data/SeoulBikeData.csv").drop(["Date", "Holiday", "Seasons"], axis=1)
df.columns = dataset_cols
df["functional"] = (df["functional"] == "Yes").astype(int)
df = df[df["hour"] == 12]
df = df.drop(["hour"], axis=1)

# for label in df.columns[1:]:
#  plt.scatter(df[label], df["bike_count"])
#  plt.title(label)
#  plt.ylabel("Bike Count at Noon")
#  plt.xlabel(label)
#  plt.show()

df = df.drop(["wind", "visibility", "functional"], axis=1)
train, val, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])

_, X_train_temp, y_train_temp = get_xy(train, "bike_count", x_labels=["temp"])
_, X_val_temp, y_val_temp = get_xy(val, "bike_count", x_labels=["temp"])
_, X_test_temp, y_test_temp = get_xy(test, "bike_count", x_labels=["temp"])

temp_reg = LinearRegression()
temp_reg.fit(X_train_temp, y_train_temp)

# print(temp_reg.coef_, temp_reg.intercept_) gives you the b0 and b coefficients for y = b0 +bx
#print(temp_reg.score(X_test_temp, y_test_temp))


_, X_train_all, y_train_all = get_xy(train, "bike_count", x_labels=df.columns[1:])
_, X_val_all, y_val_all = get_xy(val, "bike_count", x_labels=df.columns[1:])
_, X_test_all, y_test_all = get_xy(test, "bike_count", x_labels=df.columns[1:])

all_reg = LinearRegression()
all_reg.fit(X_train_all, y_train_all)
#print(all_reg.score(X_test_all, y_test_all))

temp_normalizer = tf.keras.layers.Normalization(input_shape=(1,), axis = None)
temp_normalizer.adapt(X_train_temp.reshape(-1, 1))
temp_nn_model = tf.keras.Sequential([
    temp_normalizer,
    tf.keras.layers.Dense(1)
])

temp_nn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), loss='mean_squared_error')
history = temp_nn_model.fit(X_train_temp.reshape(-1), y_train_temp, verbose = 0, epochs = 1000, validation_data = (X_val_temp, y_val_temp))
plot_loss(history)

all_normalizer = tf.keras.layers.Normalization(input_shape=(6,1), axis = None)
all_normalizer.adapt(X_train_all.reshape(-1, 1))
all_nn_model = tf.keras.Sequential([
    all_normalizer,
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])
all_nn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), loss='mean_squared_error')
history2 = all_nn_model.fit(X_train_all.reshape(-1), y_train_all, verbose = 0, epochs = 100, validation_data = (X_val_all, y_val_all))
plot_loss(history2)

