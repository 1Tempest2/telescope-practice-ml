import pandas as pd
import numpy as np
import requests
import yfinance as yf
import matplotlib.pyplot as plt
import json
from bs4 import BeautifulSoup
import warnings
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def make_graph(stock_data, revenue_data, stock):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Historical Share Price", "Historical Revenue"), vertical_spacing = .3)
    stock_data_specific = stock_data[stock_data.Date <= '2021-06-14']
    revenue_data_specific = revenue_data[revenue_data.Date <= '2021-04-30']
    fig.add_trace(go.Scatter(x=pd.to_datetime(stock_data_specific.Date, infer_datetime_format=True), y=stock_data_specific.Close.astype("float"), name="Share Price"), row=1, col=1)
    fig.add_trace(go.Scatter(x=pd.to_datetime(revenue_data_specific.Date, infer_datetime_format=True), y=revenue_data_specific.Revenue.astype("float"), name="Revenue"), row=2, col=1)
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price ($US)", row=1, col=1)
    fig.update_yaxes(title_text="Revenue ($US Millions)", row=2, col=1)
    fig.update_layout(showlegend=False,
    height=900,
    title=stock,
    xaxis_rangeslider_visible=True)
    fig.show()

warnings.filterwarnings("ignore", category=FutureWarning)
tesla_ticker = yf.Ticker("TSLA")

tesla_data = tesla_ticker.history(period="max")
tesla_data.reset_index(inplace=True)
#print(tesla_data.head())

#Question 2

url = " https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-PY0220EN-SkillsNetwork/labs/project/revenue.htm"
html_data = requests.get(url).text
soup = BeautifulSoup(html_data, 'html.parser')

tables = soup.find_all("table")
quarterly_revenue_table = tables[1]
rows = quarterly_revenue_table.find_all("tr")
#print(rows)
headers = ["Date", "Revenue"]

data = []
for row in rows[1:]:
    cols = row.find_all("td")
    data.append([col.text.strip() for col in cols])
tesla_revenue = pd.DataFrame(data, columns=headers)
tesla_revenue["Revenue"] = tesla_revenue["Revenue"].str.replace(',|\\$', '', regex=True)

tesla_revenue.dropna(inplace=True)
tesla_revenue = tesla_revenue[tesla_revenue['Revenue'] != ""]
#print(tesla_revenue.tail())

#question 3
gamestop_ticker = yf.Ticker("GME")
gme_data = gamestop_ticker.history(period="max")
gme_data.reset_index(inplace=True)
#print(gme_data.head())

#Question 4
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-PY0220EN-SkillsNetwork/labs/project/stock.html"
html_data_2 = requests.get(url).text
soup2 = BeautifulSoup(html_data_2, "html.parser")

tables = soup2.find_all("table")
quarterly_revenue_table = tables[1]
rows = quarterly_revenue_table.find_all("tr")
headers = ["Date", "Revenue"]
data = []
for row in rows[1:]:
    cols = row.find_all("td")
    data.append([col.text.strip() for col in cols])
gme_revenue = pd.DataFrame(data, columns=headers)
gme_revenue["Revenue"] = gme_revenue["Revenue"].str.replace(',|\\$', '', regex=True)

gme_revenue.dropna(inplace=True)
gme_revenue = gme_revenue[gme_revenue['Revenue'] != ""]
#print(gme_revenue.tail())
#Question 5
make_graph(tesla_data, tesla_revenue, 'Tesla')
#Question 6
make_graph(gme_data, gme_revenue, 'GameStop')



