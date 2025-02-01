import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

labels = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]
dataset = pd.read_csv("Data/magic04.data", names = labels)
# print(dataset.head())
# print(dataset["class"].unique())
dataset["class"] = np.where(dataset["class"] == "g", 1, 0)
print(dataset.head())