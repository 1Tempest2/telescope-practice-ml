import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium as fm
from matplotlib.pyplot import subplots

URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/historical_automobile_sales.csv"
df = pd.read_csv(URL)
#print(df.columns)
"""
#Task 1.1
sns.set_style("darkgrid")
sns.lineplot(y=df["Automobile_Sales"],
             x=df["Year"],
             data=df,
             marker="o",
            linestyle="-",
            color="royalblue",
            linewidth=2
)
plt.xlabel("Year", fontsize=12, fontweight='bold')
plt.ylabel("Automobile Sales", fontsize=12, fontweight='bold')
plt.title("Automobile Sales during Recession", fontsize=14, fontweight='bold')
plt.text(1982, 400, '1981-82 Recession', fontsize=8, color="red", fontweight="bold")
plt.text(1991, 400, '1991 Recession',  fontsize=8, color="red", fontweight="bold")
#plt.show()
"""
"""
#Task 1.2
recession_df = df[df["Recession"] == 1][["Vehicle_Type", "Automobile_Sales", "Year"]]

recession_df = recession_df.groupby(["Vehicle_Type", "Year"]).mean().reset_index()

plt.figure(figsize=(10, 5))
sns.set_style("whitegrid")

sns.lineplot(x="Year",
             y="Automobile_Sales",
             hue="Vehicle_Type",
             data=recession_df,
             marker="o",
             linestyle="-",
             linewidth=2)

plt.xlabel("Year", fontsize=12, fontweight='bold')
plt.ylabel("Automobile Sales", fontsize=12, fontweight='bold')
plt.title("Automobile Sales by Vehicle Type During Recession", fontsize=14, fontweight='bold')

plt.legend(title="Vehicle Type")

plt.show()
"""

"""
#Task 1.3

during_recession = df[["Vehicle_Type", "Automobile_Sales", "Recession"]]
during_recession = during_recession.groupby(["Vehicle_Type", "Recession"]).mean().reset_index()

plt.figure(figsize=(10, 5))
sns.set_style("whitegrid")

sns.barplot(x="Recession", y="Automobile_Sales", hue="Vehicle_Type", data=during_recession)

plt.xlabel("Recession (1 = Yes, 0 = No)", fontsize=12, fontweight='bold')
plt.ylabel("Average Automobile Sales", fontsize=12, fontweight='bold')
plt.title("Impact of Recession on Automobile Sales by Vehicle Type", fontsize=14, fontweight='bold')
plt.legend(title="Vehicle Type")

plt.show()
print(df.head())
"""
"""
#Task 1.4

during_recession_gdp = df[df["Recession"] == 1][["Year", "GDP"]]
after_recession_gdp = df[df["Recession"] == 0][["Year", "GDP"]]

fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

sns.lineplot(x="Year", y="GDP", data=during_recession_gdp, ax=axes[0])
axes[0].set_title("GDP During Recession", fontsize=14, fontweight='bold')
axes[0].set_ylabel("GDP", fontsize=12, fontweight='bold')

sns.lineplot(x="Year", y="GDP", data=after_recession_gdp, ax=axes[1])
axes[1].set_title("GDP After Recession", fontsize=14, fontweight='bold')
axes[1].set_xlabel("Year", fontsize=12, fontweight='bold')
axes[1].set_ylabel("GDP", fontsize=12, fontweight='bold')

plt.tight_layout()

plt.show()
"""

"""
#Task 1.5

non_recession = df[df["Recession"] == 0][["Month", "Automobile_Sales"]]
size = df["Seasonality_Weight"]
scatter = sns.scatterplot(x="Automobile_Sales",
                          y="Month",
                          size=size,
                          hue=size,
                          data=non_recession,
                          palette="coolwarm")
plt.xlabel("Month", fontsize=12, fontweight='bold')
plt.ylabel("Automobile Sales", fontsize=12, fontweight='bold')
plt.title("Seasonality impact on Automobile Sales")
plt.show()
"""
"""
#Task 1.6
during_recession = df[df["Recession"] == 1][["Price", "Automobile_Sales", "Consumer_Confidence"]]

fig, axes = plt.subplots(2, 1, figsize=(10, 8))

sns.scatterplot(y="Consumer_Confidence",
                x="Automobile_Sales",
                data=during_recession,
                ax=axes[0])
axes[0].set_title("Consumer Confidence During Recession", fontsize=14, fontweight='bold')
axes[0].set_xlabel("Consumer Confidence", fontsize=12, fontweight='bold')
axes[0].set_ylabel("Automobile Sales", fontsize=12, fontweight='bold')
sns.scatterplot(x="Price",
                y="Automobile_Sales",
                data=during_recession,
                ax=axes[1])
axes[1].set_title("Relationship between Average Vehicle Price and Sales during Recessions", fontsize=14, fontweight='bold')
axes[1].set_xlabel("Price", fontsize=12, fontweight='bold')
axes[1].set_ylabel("Automobile Sales", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()
"""
"""
#Task 1.7
rec_data = df[df["Recession"] == 1]["Advertising_Expenditure"].sum()
nonrec_data = df[df["Recession"] == 0]["Advertising_Expenditure"].sum()


plt.figure(figsize=(8, 6))
labels = ['Recession', 'Non-Recession']
sizes = [rec_data, nonrec_data]

plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
plt.show()
"""
"""
#Task 1.8
rec_data = df[df["Recession"] == 1][["Advertising_Expenditure", "Vehicle_Type"]]
rec_data_grouped = rec_data.groupby("Vehicle_Type")["Advertising_Expenditure"].sum().reset_index()

labels = rec_data_grouped["Vehicle_Type"]
sizes = rec_data_grouped["Advertising_Expenditure"]

plt.figure(figsize=(8, 6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)

plt.title('Advertising Expenditure by Vehicle Type During Recession', fontsize=14, fontweight='bold')
plt.show()
"""
#Task 1.9
rec_data = df[df["Recession"] == 1][["unemployment_rate", "Vehicle_Type","Automobile_Sales"]]
sns.lineplot(x="unemployment_rate",
             y="Automobile_Sales",
             data=rec_data,
             hue="Vehicle_Type",
             palette="coolwarm")
plt.legend(loc='lower left', bbox_to_anchor=(0, 0))
plt.title("Effect of Unemployment Rate on Vehicle Type and Sales")
plt.show()



