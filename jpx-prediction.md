```python
import os
import gc
import pandas as pd 
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from decimal import ROUND_HALF_UP, Decimal
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
import re
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 25)
pd.set_option('display.max_colwidth', 3000)
```

**Data Pipeline libraries**


```python
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
```

**MLs Model + Crossvalidation libraries**


```python
from lightgbm import LGBMRegressor
import xgboost
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error,mean_absolute_error
```

**Optimization libraries**


```python
import optuna
```

### Local data path

**Local Datasets**


```python
"""
%%time
financials_df = pd.read_csv("train_files/financials.csv")
options_df = pd.read_csv("train_files/options.csv")
secondary_stock_prices_df = pd.read_csv("train_files/secondary_stock_prices.csv")
stock_prices_df = pd.read_csv("train_files/stock_prices.csv")
trades_df = pd.read_csv("train_files/trades.csv")
stocks_df = pd.read_csv("stock_list.csv")
"""
```

**Local Suplemental files**


```python
"""
%%time
financials_info = pd.read_csv("/data_specifications/stock_fin_spec.csv")
options_info = pd.read_csv("/data_specifications/options_spec.csv")
stock_prices_info = pd.read_csv("/data_specifications/stock_price_spec.csv")
trades_info = pd.read_csv("/data_specifications/trades_spec.csv")
stocks_info = pd.read_csv("/data_specifications/stock_list_spec.csv")
"""
```

### Kaggle data path


```python
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
```

**Datasets**


```python
%%time 
financials_df = pd.read_csv("/kaggle/input/jpx-tokyo-stock-exchange-prediction/train_files/financials.csv")
options_df = pd.read_csv("/kaggle/input/jpx-tokyo-stock-exchange-prediction/train_files/options.csv")
secondary_stock_prices_df = pd.read_csv("/kaggle/input/jpx-tokyo-stock-exchange-prediction/train_files/secondary_stock_prices.csv")
stock_prices_df = pd.read_csv("/kaggle/input/jpx-tokyo-stock-exchange-prediction/train_files/stock_prices.csv")
trades_df = pd.read_csv("/kaggle/input/jpx-tokyo-stock-exchange-prediction/train_files/trades.csv")
stocks_df = pd.read_csv("/kaggle/input/jpx-tokyo-stock-exchange-prediction/stock_list.csv")
```

**Suplemental files**


```python
financials_info = pd.read_csv("/kaggle/input/jpx-tokyo-stock-exchange-prediction/data_specifications/stock_fin_spec.csv")
options_info = pd.read_csv("/kaggle/input/jpx-tokyo-stock-exchange-prediction/data_specifications/options_spec.csv")
stock_prices_info = pd.read_csv("/kaggle/input/jpx-tokyo-stock-exchange-prediction/data_specifications/stock_price_spec.csv")
trades_info = pd.read_csv("/kaggle/input/jpx-tokyo-stock-exchange-prediction/data_specifications/trades_spec.csv")
stocks_info = pd.read_csv("/kaggle/input/jpx-tokyo-stock-exchange-prediction/data_specifications/stock_list_spec.csv")
```

## 0. Utils

**Merging stock prices with stock metadata**


```python
def merge_metadata(stock_prices_df, df_stocks):
    """ This function merges the stock price dataset with stock metadata.
        Args:
        stock_prices_df (pd.DataFrame): prices dataframe
        df_stocks (pd.DataFrame): metadata dataframe

        Returns:
        df_prices (pd.DataFrame): merged dataframe
    """
    col = ["SecuritiesCode","Name","Section/Products","NewMarketSegment","33SectorCode","33SectorName","17SectorCode","17SectorName","NewIndexSeriesSizeCode","NewIndexSeriesSize","IssuedShares","MarketCapitalization"]
    df_prices = pd.merge(stock_prices_df, df_stocks[col], on='SecuritiesCode')
    display(df_prices.head())
    return df_prices
```

**OHLCV Chart plot**


```python
def plot_candle_with_target(df_prices, stock_code):
    """Plot OHLCV plot with target series.
    
      Args:
        df_prices (pd.DataFrame): prices dataframe
        stock_code: int, code of the stock
    """
    df_ = df_prices.copy() 
    df_ = df_[df_['SecuritiesCode'] == stock_code]
    dates = df_['Date'].values
    ohlc = {
        'open': df_['Open'].values, 
        'high': df_['High'].values, 
        'low': df_['Low'].values, 
        'close': df_['Close'].values
    }
    vol = df_['Volume'].values
    target = df_['Target'].values
    
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, x_title='Date')
    fig.add_trace(go.Candlestick(x=dates, name='OHLC', **ohlc),
                  row=1, col=1)
    fig.add_trace(go.Bar(x=dates, y=vol, name='Volume'),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=dates, y=target, name='Target'),
                  row=3, col=1)
    fig.update_layout(
        title=f"OHLCV Chart with Target Series (Stock {stock_code})",
    )
    fig.update(layout_xaxis_rangeslider_visible=False)
    fig.show()
```


```python
def plot_candle_with_target_adjusted(df_prices, stock_code):
    """Plot OHLCV plot with target series.
    
      Args:
        df_prices (pd.DataFrame): prices dataframe
        stock_code: int, code of the stock
    """
    df_ = df_prices.copy()
    df_ = df_[df_['SecuritiesCode'] == stock_code]
    dates = df_['Date'].values
    ohlc = {
        'open': df_['AdjustedOpen'].values, 
        'high': df_['AdjustedHigh'].values, 
        'low': df_['AdjustedLow'].values, 
        'close': df_['AdjustedClose'].values
    }
    vol = df_['Volume'].values
    target = df_['Target'].values
    
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, x_title='Date')
    fig.add_trace(go.Candlestick(x=dates, name='OHLC', **ohlc),
                  row=1, col=1)
    fig.add_trace(go.Bar(x=dates, y=vol, name='Volume'),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=dates, y=target, name='Target'),
                  row=3, col=1)
    fig.update_layout(
        title=f"OHLCV Chart with Target Series (Stock {stock_code})",
    )
    fig.update(layout_xaxis_rangeslider_visible=False)
    fig.show()
```

**Evaluation function (calculates Sharpe Ratio)**


```python
def calc_spread_return_sharpe(df: pd.DataFrame, portfolio_size: int = 200, toprank_weight_ratio: float = 2) -> float:
    """
    Args:
        df (pd.DataFrame): predicted results
        portfolio_size (int): # of equities to buy/sell
        toprank_weight_ratio (float): the relative weight of the most highly ranked stock compared to the least.
    Returns:
        (float): sharpe ratio
    """
    def _calc_spread_return_per_day(df, portfolio_size, toprank_weight_ratio):
        """
        Args:
            df (pd.DataFrame): predicted results
            portfolio_size (int): # of equities to buy/sell
            toprank_weight_ratio (float): the relative weight of the most highly ranked stock compared to the least.
        Returns:
            (float): spread return
        """
        assert df['Rank'].min() == 0
        assert df['Rank'].max() == len(df['Rank']) - 1
        weights = np.linspace(start=toprank_weight_ratio, stop=1, num=portfolio_size)
        purchase = (df.sort_values(by='Rank')['Target'][:portfolio_size] * weights).sum() / weights.mean()
        short = (df.sort_values(by='Rank', ascending=False)['Target'][:portfolio_size] * weights).sum() / weights.mean()
        return purchase - short

    buf = df.groupby('Date').apply(_calc_spread_return_per_day, portfolio_size, toprank_weight_ratio)
    sharpe_ratio = buf.mean() / buf.std()
    return sharpe_ratio
```

More info about Sharpe Ratio:

https://www.investopedia.com/terms/s/sharperatio.asp

**Adjust Split-Reverse/split price**


```python
def adjust_price(price):
    """
    Args:
        price (pd.DataFrame)  : pd.DataFrame include stock_price
    Returns:
        price DataFrame (pd.DataFrame): stock_price with generated AdjustedClose
    """
    # transform Date column into datetime
    price.loc[: ,"Date"] = pd.to_datetime(price.loc[: ,"Date"], format="%Y-%m-%d")

    def generate_adjusted_close(df):
        """
        Args:
            df (pd.DataFrame)  : stock_price for a single SecuritiesCode
        Returns:
            df (pd.DataFrame): stock_price with AdjustedClose for a single SecuritiesCode
        """
        # sort data to generate CumulativeAdjustmentFactor
        df = df.sort_values("Date", ascending=False)
        # generate CumulativeAdjustmentFactor
        df.loc[:, "CumulativeAdjustmentFactor"] = df["AdjustmentFactor"].cumprod()
        
        
        # generate AdjustedClose
        df.loc[:, "AdjustedClose"] = (
            df["CumulativeAdjustmentFactor"] * df["Close"]
        ).map(lambda x: float(
            Decimal(str(x)).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)
        )) 

        # generate AdjustedOpen
        df.loc[:, "AdjustedOpen"] = (
            df["CumulativeAdjustmentFactor"] * df["Open"]
        ).map(lambda x: float(
            Decimal(str(x)).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)
        ))
        
        # generate AdjustedHigh
        df.loc[:, "AdjustedHigh"] = (
            df["CumulativeAdjustmentFactor"] * df["High"]
        ).map(lambda x: float(
            Decimal(str(x)).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)
        ))
        
        # generate Low
        df.loc[:, "AdjustedLow"] = (
            df["CumulativeAdjustmentFactor"] * df["Low"]
        ).map(lambda x: float(
            Decimal(str(x)).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)
        ))
        
        
        # reverse order
        df = df.sort_values("Date")
        # to fill AdjustedClose, replace 0 into np.nan
        df.loc[df["AdjustedClose"] == 0, "AdjustedClose"] = np.nan
        # forward fill AdjustedClose
       
        #df.loc[:, "AdjustedClose"] = df.loc[:, "AdjustedClose"].ffill()
        
        df.loc[:, "AdjustedClose"] = df.loc[:, "AdjustedClose"].interpolate(method='linear',limit_direction='backward')
        df.loc[:, "AdjustedOpen"] = df.loc[:, "AdjustedOpen"].interpolate(method='linear',limit_direction='backward')
        df.loc[:, "AdjustedHigh"] = df.loc[:, "AdjustedHigh"].interpolate(method='linear',limit_direction='backward')
        df.loc[:, "AdjustedLow"] = df.loc[:, "AdjustedLow"].interpolate(method='linear',limit_direction='backward')

        return df

    # generate AdjustedClose
    price = price.sort_values(["SecuritiesCode", "Date"])
    price = price.groupby("SecuritiesCode").apply(generate_adjusted_close).reset_index(drop=True)
    price.drop(columns=['Open', 'High','Low','Close','AdjustmentFactor'],inplace =True)
    #price.set_index("Date", inplace=True)
    return price
```

**Feature extraction**


```python
def generate_feature(price):
    """
    Args:
        price (pd.DataFrame)  : pd.DataFrame include stock_price
    Returns:
        price DataFrame (pd.DataFrame): stock_price with generated features
    """
    # transform Date column into datetime

    def generate_feature_close(df):
        """
        Args:
            df (pd.DataFrame)  : stock_price for a single SecuritiesCode
        Returns:
            df (pd.DataFrame): Features for a single SecuritiesCode
        """
        # sort data to generate CumulativeAdjustmentFactor
        df = df.sort_values("Date", ascending=False)
        
        # generate PriceChanges
        period = [5,10,21,33]
        for i in period:
            df[f"pct{i}"] = df["AdjustedClose"].pct_change(i)
            df.loc[:, f"pct{i}"] = df.loc[:, f"pct{i}"].interpolate(method='linear',limit_direction='backward')
            df.loc[:,f"Volatility_{i}"] = np.log(df["AdjustedClose"]).diff().rolling(i).std()
            df.loc[:, f"Volatility_{i}"] = df.loc[:, f"Volatility_{i}"].interpolate(method='linear',limit_direction='backward')

        period_avg = [10,20,50,60]
        for i in period_avg:
         
            # generate SMA
            df[f"SMA_{i}"] = df['AdjustedClose'].rolling(window=i).mean()
            df.loc[:, f"SMA_{i}"] = df.loc[:, f"SMA_{i}"].interpolate(method='linear',limit_direction='backward')
          
            # generate EMA
            df[f"EMA_{i}"] = df['AdjustedClose'].ewm(span=i,adjust=False).mean()
            df.loc[:, f"EMA_{i}"] = df.loc[:, f"EMA_{i}"].interpolate(method='linear',limit_direction='backward')
        
        # reverse order
        df = df.sort_values("Date")

        return df

    # generate AdjustedClose
    price = price.sort_values(["SecuritiesCode", "Date"])
    price = price.groupby("SecuritiesCode").apply(generate_feature_close).reset_index(drop=True)
    #price.set_index("Date", inplace=True)
    return price
```

**Encoding**


```python
def encoding(df):
    """
    Args:
        df (pd.DataFrame): dataframe w/o encoding
    Returns:
        df DataFrame (pd.DataFrame): one hot encoded dataframe
    """
        
    cat_encoder = OneHotEncoder(handle_unknown='ignore')
    train_set_cat_coded = cat_encoder.fit_transform(df[["33SectorName","17SectorName"]])
    ls=[]
    for i in cat_encoder.categories_:
        for j in i:
            ls.append(j)

    train_set_cat_coded_ready = pd.DataFrame(train_set_cat_coded.todense(),columns = ls,index=df.index)
    for i in train_set_cat_coded_ready.columns.to_list():
        train_set_cat_coded_ready[i] = train_set_cat_coded_ready[i].apply(np.int64)
    
    df = pd.concat([df, train_set_cat_coded_ready], axis=1)
    del train_set_cat_coded
    del train_set_cat_coded_ready
    return df
```

## 1. Exploratory Data Analysis (EDA)

### Stock prices
stock_prices.csv

File Description
The core file of interest, including the daily closing price for each stock and the target column. Following is column information recorded in stock_price_spec.csv:

    RowId: Unique ID of price records, the combination of Date and SecuritiesCode.
    Date: Trade date.
    SecuritiesCode: Local securities code.
    Open: First traded price on a day.
    High: Highest traded price on a day.
    Low: Lowest traded price on a day.
    Close: Last traded price on a day.
    Volume: Number of traded stocks on a day.
    AdjustmentFactor: Used to calculate theoretical price/volume when split/reverse-split happens (NOT including dividend/allotment of shares).
    ExpectedDividend: Expected dividend value for ex-right date. This value is recorded 2 business days before ex-dividend date.
    SupervisionFlag: Flag of securities under supervision and securities to be delisted, for more information, please see here.
    Target: Change ratio of adjusted closing price between t+2 and t+1 where t+0 is trade date.


```python
stock_prices_info
```


```python
display(stock_prices_df.head(5))
```


```python
stock_prices_df.info()
```


```python
stock_prices_df.describe()
```

**Missing values**


```python
display(pd.isna(stock_prices_df).sum()/len(stock_prices_df)*100)
```


```python
missing_high = stock_prices_df[stock_prices_df["High"].isna()]
```


```python
display(missing_high.head(5))
```


```python
display(missing_high.shape[0])
```


```python
missing_high["Date"].nunique()
```


```python
missing_high["Date"].value_counts()
```


```python
plot_missing_high_df = missing_high["Date"].value_counts().to_frame().reset_index()
plot_missing_high_df.rename(columns = {'index':'Date', 'Date':'Count'}, inplace = True)
plot_missing_high_df.head()
```

2020-10-01 is the day with the most amount of missing data


```python
plot_missing_high_df["Date"] = pd.to_datetime(plot_missing_high_df["Date"])
plot_missing_high_df.sort_values(by="Date",inplace=True)
```


```python
fig = go.Figure()
fig.add_trace(go.Scatter(x=plot_missing_high_df["Date"], 
                         y=plot_missing_high_df["Count"], 
                         mode='lines'))

fig.update_layout(
    title=f"Stocks without Prices Count per Date",
    xaxis_title="Date",
    yaxis_title="Count",
)
```


```python
missing_Target = stock_prices_df[stock_prices_df["Target"].isna()]
```


```python
display(missing_Target.head(5))
```


```python
display(missing_Target.shape[0])
```


```python
missing_Target["Date"].nunique()
```


```python
missing_Target["Date"].value_counts()
```


```python
plot_missing_target_df = missing_Target["Date"].value_counts().to_frame().reset_index()
plot_missing_target_df.rename(columns = {'index':'Date', 'Date':'Count'}, inplace = True)
plot_missing_target_df
```


```python
fig = go.Figure()
fig.add_trace(go.Scatter(x=plot_missing_target_df["Date"], 
                         y=plot_missing_target_df["Count"], 
                         mode='lines'))

fig.update_layout(
    title=f"Stocks without Target Count per Date",
    xaxis_title="Date",
    yaxis_title="Count",
)
```

### Stocks

stock_list.csv


SecuritiesCode: Local securities code.
    EffectiveDate: The effective date. (Need clarification.)
    Name: Name of security.
    Section/Products: Section/Product.

    NewMarketSegment: New market segment effective from 2022-04-04 (as of 15:30 JST on Mar 11 2022). For more information, please see Overview of Market Restructuring.
    33SectorCode: 33 sector code.
    33SectorName: 33 sector name.
    17SectorCode: 17 sector code.
    17SectorName: 17 sector name.
　
 For more information about sector code and name, please see TOPIX Sector Indices / TOPIX-17 Series

    NewIndexSeriesSizeCode: TOPIX New Index Series code.
    NewIndexSeriesSize: TOPIX New Index Series name.
For more information about TOPIX New Index Series code and name, please see TOPIX New Index Series / Size-based TOPIX.

    TradeDate: Trade date to calculate MarketCapitalization.
    Close: Close price to calculate MarketCapitalization.
    IssuedShares: Issued shares.
    MarketCapitalization: Market capitalization on December 3, 2021.
    Universe0: A flag of prediction target universe (top 2000 stocks by market capitalization).


```python
display(stocks_info)
```


```python
display(stocks_df.head(5))
```


```python
stocks_df.info()
```


```python
display(pd.isna(stocks_df).sum()/len(stocks_df)*100)
```


```python
stocks_missing_Date = stocks_df[stocks_df["TradeDate"].isna()]
stocks_missing_Date.head()
```


```python
stocks_missing_segment = stocks_df[stocks_df["NewMarketSegment"].isna()]
stocks_missing_segment.head()
```


```python
stocks_df["Universe0"].value_counts()
```


```python
stocks_df["33SectorName"].value_counts()
```


```python
sector = stocks_df["33SectorName"].value_counts().to_frame()
sector.reset_index(inplace=True)
sector["percentage"] = sector["33SectorName"]/sector["33SectorName"].sum()*100
sector.head(5)
```


```python
plt.rcParams["figure.figsize"] = (30,10)
plt.pie(sector["percentage"], labels = sector["index"])
plt.show()
```


```python
stocks_df["Section/Products"].value_counts()
```


```python
section = stocks_df["Section/Products"].value_counts().to_frame()
section.reset_index(inplace=True)
section["percentage"] = section["Section/Products"]/section["Section/Products"].sum()*100
section
```


```python
section["percentage"] = section["Section/Products"]/section["Section/Products"].sum()*100
```


```python
plt.rcParams["figure.figsize"] = (30,10)
plt.pie(section["percentage"], labels = section["index"])
plt.show()
```


```python
del sector
del section
gc.collect()
```

## Selecting the top 2000 Stocks (Universe0 Flag = True)


```python
stocks2000_df = stocks_df[stocks_df["Universe0"]]
display(stocks2000_df.head())
```


```python
stocks2000_df.info()
```


```python
stocks2000_df["33SectorName"].value_counts()
```


```python
sector2000 = stocks2000_df["33SectorName"].value_counts().to_frame()
sector2000.reset_index(inplace=True)
sector2000["percentage"] = sector2000["33SectorName"]/sector2000["33SectorName"].sum()*100
sector2000.head(5)
```


```python
plt.rcParams["figure.figsize"] = (40,10)
plt.pie(sector2000["percentage"], labels = sector2000["index"])
plt.show()
```


```python
section2000 = stocks2000_df["Section/Products"].value_counts().to_frame()
section2000.reset_index(inplace=True)
section2000["percentage"] = section2000["Section/Products"]/section2000["Section/Products"].sum()*100
section2000
```


```python
plt.rcParams["figure.figsize"] = (30,10)
plt.pie(section2000["percentage"], labels = section2000["index"])
plt.show()
```


```python
del stocks_df
del section2000
del sector2000
gc.collect()
```

## Merging Stock metadate with stock price


```python
df_prices =  merge_metadata(stock_prices_df, stocks2000_df)
```

**Free memory**

del stock_prices_df
del stocks2000_df

gc.collect()

**Calculating Target variable statistics**


```python
mean_securities_df = df_prices.groupby(["SecuritiesCode"])["Target"].mean()
total_mean_securities = mean_securities_df.mean()
total_mean_securities
```


```python
fig, ax = plt.subplots(figsize=(10,5))
sns.histplot(data=mean_securities_df.values, bins=100,
             ax=ax)
ax.axvline(x=total_mean_securities, color='red', linestyle='dotted', linewidth=2, 
           label='Mean')
ax.set_title("Target Mean Distibution Securities\n"
             f"Min {round(mean_securities_df.min(), 4)} | "
             f"Max {round(mean_securities_df.max(), 4)} | "
             f"Skewness {round(mean_securities_df.skew(), 2)} | "
             f"Kurtosis {round(mean_securities_df.kurtosis(), 2)}")
ax.set_xlabel("Target Mean")
ax.set_ylabel("Date Count")
ax.legend()
plt.show()
```


```python
mean_date_df = df_prices.groupby(["Date"])["Target"].mean()
total_mean_date = mean_date_df.mean()
total_mean_date
```


```python
fig, ax = plt.subplots(figsize=(10,5))
sns.histplot(data=mean_date_df.values, bins=100,
             ax=ax)
ax.axvline(x=total_mean_date, color='red', linestyle='dotted', linewidth=2, 
           label='Mean')
ax.set_title("Target Mean Distibution Date\n"
             f"Min {round(mean_date_df.min(), 4)} | "
             f"Max {round(mean_date_df.max(), 4)} | "
             f"Skewness {round(mean_date_df.skew(), 2)} | "
             f"Kurtosis {round(mean_date_df.kurtosis(), 2)}")
ax.set_xlabel("Target Mean")
ax.set_ylabel("Date Count")
ax.legend()
plt.show()
```

## Grupying the Target by Sector and Section

### Target vs Sector33


```python
target_sector = df_prices.groupby(["33SectorName"])["Target"].mean()
target_sector.sort_values(inplace=True, ascending=False)

```


```python
target_sector = target_sector.to_frame()
target_sector.reset_index(inplace= True)
```


```python
fig, ax = plt.subplots(figsize = (12,6))    
fig = sns.barplot(x = "33SectorName", y = "Target", data = target_sector, ax=ax)
ax.set_title("Target Mean of Sectors")
ax.tick_params(axis='x', rotation=90)
```

### Target vs Section


```python
target_section = df_prices.groupby(["Section/Products"])["Target"].mean()
target_section.sort_values(inplace=True, ascending=False)

```


```python
target_section = target_section.to_frame()
target_section.reset_index(inplace= True)
```


```python
fig, ax = plt.subplots(figsize = (12,6))    
fig = sns.barplot(x = "Section/Products", y = "Target", data = target_section, ax=ax)
ax.set_title("Target Mean of Sections")
ax.tick_params(axis='x', rotation=90)
```

### Secondary stock prices

Securities with low liquidity (few opportunities to trade)


```python
stock_prices_info
```


```python
display(secondary_stock_prices_df.head(5))
```


```python
secondary_stock_prices_df.info()
```


```python
display(pd.isna(secondary_stock_prices_df).sum()/len(secondary_stock_prices_df)*100)
```

**Missing values**


```python
missing_secondary_high = secondary_stock_prices_df[secondary_stock_prices_df["High"].isna()]
display(missing_secondary_high.head(5))
```


```python
missing_secondary_high["Date"].nunique()
```


```python
plot_missing_secondary_high_df = missing_secondary_high["Date"].value_counts().to_frame().reset_index()
plot_missing_secondary_high_df.rename(columns = {'index':'Date', 'Date':'Count'}, inplace = True)
plot_missing_secondary_high_df.head()
```


```python
del secondary_stock_prices_df
gc.collect()
```

As happened with the primary stock list 2020-10-01 is the day with the most amount of missing data


**NOTE:** Secondary stock prices price movement is correlated with the main stock price movement. Due to the limited amount dedicated to this project, this data will not be included in the prediction model

## Trades

Aggregated summary of trading volumes from the previous business week. Following is column information recorded in trades_spec.csv:

    Date: Data published date, usually Thursday on the following week.
    StartDate: The first trading date in this trading week.
    EndDate: The last trading date in this trading week.
    Section: Market division name.


```python
trades_info
```


```python
display(trades_df.head(5))
```

**NOTE:** Trade volumen info can be used to predict the Target value. Due to the limited amount dedicated to this project, this data will not be included in the prediction model

### Financials 


```python
financials_info
```


```python
financials_df.head()
```


```python
financials_df.info()
```


```python
display(pd.isna(financials_df).sum()/len(financials_df)*100)
```

Notably a high amount of data is missing in financials_df

**NOTE:** Leveraging the financial data in the Fundamental analysis is one popular method in target calculation. Due to the limited amount dedicated to this project, this data will not be included in the prediction model

### Options

**File Description**

Data on the status of a variety of options based on the broader market. Many options include implicit predictions of the future price of the stock market and so may be of interest even though the options are not scored directly. Following is column information recorded in options_spec.csv:

    DateCode: Unique ID for option price records, the combintion of Date and OptionCode.
    Date: Trade date and time.
    OptionsCode: Local securities code. Detailed information is provided in Identification Code Specifications for Futures and Options Transactionssakimono20220208-e.pdf).
    WholeDayOpen: Opening price for whole trading day.
    WholeDayHigh: Highest price for whole trading day.
    WholeDayLow: Lowest price for whole trading day.
    WholeDayClose: Closing price for whole trading day.
    NightSessionOpen: Opening price for night session.
    NightSessionHigh: Highest price for night session.
    NightSessionLow: Lowest price for night session.
    NightSessionClose: Closing price for night session.
    DaySessionOpen: Opening price for day session.
    DaySessionHigh: Highest price for day session.
    DaySessionLow: Lowest price for day session.
    DaySessionClose: Closing price for day session.
    TradingVolume: Trading volume of the product/contract for the whole trading day.
    OpenInterest: Open interest of the product/contract for the whole trading day
    TradingValue: Trading value of the product/contract for the whole trading day
    ContractMonth: Cotract year-month of the product/contract.
    StrikePrice: Exercise price of product/contract.
    DaySessionVolume: Trading volume of the product/contract for day session.
    Putcall: 1 for put and 2 for call.
    LastTradingDay: Last trading day.
    SpecialQuotationDay: The day when the Special Quotation is calculated.
    SettlementPrice: Settlement price.
    TheoreticalPrice: The theoretical price at the end of a day session.
    BaseVolatility: The volatility at the time of calculating the settlement price.
    ImpliedVolatility: Implied volatility.
    InterestRate: Interest rate for calculation.
    DividendRate: Dividend yeild.
    Dividend: Devidend.


```python
options_info
```


```python
options_df.head()
```


```python
options_df.info()
```

**NOTE:** Info provided by Options (for instance the put/call volumen ratio) is interesting to determine the market feeling about a particular derivated product. Due to the limited amount dedicated to this project, this data will not be included in the prediction model

## 2. Data wrangling
- Adjust OHLC prices (Split- Reverse/split)
- Input missing values


#### Generate adjusted OHLC prices
Generate adjusted OHLC prices using AdjustmentFactor attribute. This should reduce historical price gap caused by split/reverse-split.

Furthermore, adjust_price() also inputs the missing data


```python
df_prices_adj = adjust_price(df_prices)
df_prices_adj.head(5)
```


```python
df_prices_adj["CumulativeAdjustmentFactor"].value_counts()
```


```python
df_prices_adj.info()
```

**Less missing values**


```python
display(pd.isna(df_prices_adj).sum()/len(df_prices_adj)*100)
```

**Now the OHTC chart shows continuity**

Before adjustment


```python
plot_candle_with_target(df_prices, 9726) 
```

After adjustment


```python
plot_candle_with_target_adjusted(adjust_price(df_prices), 9726) 
```

Before adjustment


```python
plot_candle_with_target(df_prices, 4582)
```

After adjustment


```python
plot_candle_with_target_adjusted(adjust_price(df_prices), 4582) 
```

Before adjustment


```python
plot_candle_with_target(df_prices, 1805)
```

After adjustment


```python
plot_candle_with_target_adjusted(adjust_price(df_prices), 1805) 
```

## 3. Feature Engineering
- Create basic statics
    1. Moving average
    2. Exponential moving average
    3. volatility
- Do not create Features since you are going to be using RNN
- Merge sentiment analysis data from Options
    1. https://www.boerse-stuttgart.de/de-de/tools/euwax-sentiment/
    2. https://www-mmds.sigmath.es.osaka-u.ac.jp/en/activity/vxj.php#:~:text=The%20Volatility%20Index%20Japan%20(VXJ,based%20on%20Nikkei225%20index%20options.
- Check correlation secondary stock market target vs Primary stock market target


**Correlogram to see if there is any autocorrelation**


```python
df_9726 =df_prices_adj[df_prices_adj["SecuritiesCode"]==9726]
```


```python
plt.rcParams['figure.figsize'] = [10, 5]
pd.plotting.autocorrelation_plot(df_9726["AdjustedClose"])
```


```python
pd.plotting.autocorrelation_plot(df_9726["Target"])
```

It certainly look like we are dealing with a random walk, as there are no indications of any autocorrelation for any lag.

Basically our LSTM found nothing of any real value to model and thus took the average value, along with a slight slope; we would have been just as well off with an extremely simplistic model of the form

C
l
o
s
e
t
∝
C
l
o
s
e
(
t
−
1
)

Price is correlated but price change shows no correlation. Therefore, the target variable itself will not give us much info about future stock movements... We need a good Feature engineering

### Feature Engineering

**From adjusted prices (individual)**
- Price changes
- Moving average
- Exponential moving average
- Volatility
- One hot encoding
- Standarization?

**Price changes**



```python
period = [5,10,21,33]
```


```python
for i in period:
    df_9726[f"pct{i}"] = df_9726["AdjustedClose"].pct_change(i)
display(df_9726.head(5))
```

**Simple Moving Average (SMA)**


```python
period_avg = [10,20,50,60]
```


```python
for i in period_avg:
    df_9726[f"SMA_{i}"] = df_9726['AdjustedClose'].rolling(window=i).mean()

display(df_9726.tail(5))
```

**Exponential Moving Average (EMA)**


```python
for i in period_avg:
    df_9726[f"EMA_{i}"] = df_9726['AdjustedClose'].ewm(span=i,adjust=False).mean()

display(df_9726.tail(5))
```

Plotting the new features


```python
col_avg = ["AdjustedClose","SMA_10","SMA_20","SMA_50","SMA_60","EMA_10","EMA_20","EMA_20","EMA_60"]
```


```python
df_9726Avg = df_9726[col_avg]
df_9726Avg.head()
```


```python
plt.rcParams["figure.figsize"] = (30,10)
df_9726Avg.plot(title = "Avg analysis for Security Code")
```


```python
col_SMA = ["AdjustedClose","SMA_10","SMA_20","SMA_50","SMA_60"]
```


```python
df_9726SMA = df_9726[col_SMA]
df_9726SMA.head()
```


```python
df_9726SMA.plot(title = "Avg analysis for Security Code")
```


```python
col_EMA = ["AdjustedClose","EMA_10","EMA_20","EMA_50","EMA_60"]
```


```python
df_9726EMA = df_9726[col_EMA]
df_9726EMA.head()
```


```python
df_9726EMA.plot(title = "Avg analysis for Security Code")
```


```python
df_prices_feat = generate_feature(df_prices_adj)
df_prices_feat.drop(["RowId","ExpectedDividend","Name","Section/Products","SupervisionFlag","NewMarketSegment","33SectorCode","17SectorCode","NewIndexSeriesSizeCode","NewIndexSeriesSize","IssuedShares","MarketCapitalization"],axis=1,inplace=True)
df_prices_feat.head(5)
```


```python
df_prices_feat.info()
```


```python
del df_prices_adj
del df_9726
del df_9726EMA
del df_9726SMA
del options_df
del financials_df
del trades_df
del financials_info
del options_info
del stock_prices_info
del trades_info
del stocks_info
gc.collect()
```


```python
del target_sector
del target_section
gc.collect()
```

**One Hot encoding Stock Catagorical data**
- SectorCode33
- Section


```python
cat_encoder = OneHotEncoder(handle_unknown='ignore')
train_set_cat_coded = cat_encoder.fit_transform(df_prices_feat[["33SectorName","17SectorName"]])
train_set_cat_coded.shape
```


```python
ls=[]
for i in cat_encoder.categories_:
    for j in i:
        ls.append(j)
```


```python
train_set_cat_coded_ready = pd.DataFrame(train_set_cat_coded.todense(),columns = ls,index=df_prices_feat.index)
train_set_cat_coded_ready
```


```python
for i in train_set_cat_coded_ready.columns.to_list():
    train_set_cat_coded_ready[i] = train_set_cat_coded_ready[i].apply(np.uint8)
```


```python
df_prices_feat = pd.concat([df_prices_feat, train_set_cat_coded_ready], axis=1)
```


```python
del train_set_cat_coded
del train_set_cat_coded_ready
gc.collect()
```

**Selecting features for the model**


```python
col = ["Date","SecuritiesCode","Target","Volume","AdjustedClose","AdjustedHigh","AdjustedOpen","AdjustedLow","pct5","pct10","pct21","pct33","Volatility_5","Volatility_10","Volatility_21","Volatility_33","SMA_10","SMA_20","SMA_50","SMA_60","EMA_10","EMA_20","EMA_50"] +ls
#col
```


```python
X = df_prices_feat[col]
X=X.dropna().sort_values(['Date','SecuritiesCode'])
```

**Correlation Matrix**


```python
corr_mat = np.abs(X.corr())
fig, ax = plt.subplots(figsize=(20,20))
sns.heatmap(corr_mat,square = True, ax = ax)
```

**Removing special characters in the column names**


```python
X = X.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
```

**Drop labels**


```python
y= X["Target"].to_numpy()
X=X.drop(["Target"],axis=1)
```

### Standarize Values


```python
num_pipeline = Pipeline([
        ('std_scaler', StandardScaler()),
    ])
```


```python
num_attribs = ['Volume','AdjustedClose',"AdjustedHigh","AdjustedOpen","AdjustedLow","pct5","pct10","pct21","pct33","Volatility_5","Volatility_10","Volatility_21","Volatility_33",'SMA_10','SMA_20','SMA_50','SMA_60','EMA_10','EMA_20','EMA_50']

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs)
    ])

```

**Removing special characters in the column names**

ts_fold = TimeSeriesSplit(n_splits=3, gap=10000)

for fold, (train_idx, val_idx) in enumerate(ts_fold.split(X, y)):
    
    print(f"\n========================== Fold {fold+1} ==========================")
        
    X_train, y_train = X.iloc[train_idx,:], y[train_idx]
    X_valid, y_val = X.iloc[val_idx,:], y[val_idx]
    print(X_train.shape[0], len(y_train))
    ##### TRAIIIIIIIIIIN
    pipeline_X_df = full_pipeline.fit_transform(X_train)
    pipeline_X_df = pd.DataFrame(pipeline_X_df, columns = num_attribs)
    
    pipeline_X_df.reset_index(drop = True, inplace=True)
    X_train.reset_index(drop = True, inplace=True)
    
    
    X_train.drop(num_attribs,axis=1,inplace=True)
    X_train = pd.concat([X_train,pipeline_X_df], axis=1)
    
    ##### VALIDATION

    pipeline_val_df = full_pipeline.transform(X_valid)
    pipeline_val_df = pd.DataFrame(pipeline_val_df, columns = num_attribs)
   # display(pipeline_val_df)
   # print(num_attribs)
    #display(X_valid)
    pipeline_val_df.reset_index(drop = True, inplace=True)
    X_valid.reset_index(drop = True, inplace=True)
    X_valid.drop(num_attribs,axis=1,inplace=True)
    X_valid = pd.concat([X_valid,pipeline_val_df], axis=1)
    
    
    print(X_train.shape[0], len(y_train))
    print("-------------")
    print(X_valid.shape[0], len(y_val))
    print("***********")

    del X_train, y_train,  X_valid, y_val, pipeline_X_df , pipeline_val_df
    gc.collect()

## Model Selection

Calculate the average Sharpe ratio for the following ML models.

- LGBMRegressor
- XGBoost Regressor
- Random Forest Regressor

The model giving the **highest** Sharpe ratio will be selected for the final model.

**Create list to store all Sharpe ratio for different models**


```python
model_performance = []
```

### Time based Crossvalidation

params = {'n_estimators': 500,
          'num_leaves' : 200,
          'learning_rate': 0.1,
          'colsample_bytree': 0.9,
          'subsample': 0.8,
          'reg_alpha': 0.4,
          'metric': 'mae',
          'random_state': 21}

**LGBMRegressor**


```python
ts_fold = TimeSeriesSplit(n_splits=5,gap=5000)
```


```python
feat_importance=pd.DataFrame()
sharpe_ratio=[]
```


```python
%%time
for fold, (train_idx, val_idx) in enumerate(ts_fold.split(X, y)):
    
    print(f"\n========================== Fold {fold+1} ==========================")
        
    X_train, y_train = X.iloc[train_idx,:], y[train_idx]
    X_valid, y_val = X.iloc[val_idx,:], y[val_idx]
    
    print("Train Date range: {} to {}".format(X_train.Date.min(),X_train.Date.max()))
    print("Valid Date range: {} to {}".format(X_valid.Date.min(),X_valid.Date.max()))
    
    
    ##### TRAIIIIIIIIIIN
    """
    pipeline_X_df = full_pipeline.fit_transform(X_train)
    pipeline_X_df = pd.DataFrame(pipeline_X_df, columns = num_attribs)
    
    pipeline_X_df.reset_index(drop = True, inplace=True)
    X_train.reset_index(drop = True, inplace=True)

    X_train.drop(num_attribs,axis=1,inplace=True)
    X_train = pd.concat([X_train,pipeline_X_df], axis=1)
    #"""
    X_train= X_train.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    
    ##### VALIDATION
    """
    pipeline_val_df = full_pipeline.transform(X_valid)
    pipeline_val_df = pd.DataFrame(pipeline_val_df, columns = num_attribs)
    pipeline_val_df.reset_index(drop = True, inplace=True)
    X_valid.reset_index(drop = True, inplace=True)
    X_valid.drop(num_attribs,axis=1,inplace=True)
    X_valid = pd.concat([X_valid,pipeline_val_df], axis=1)
    
   # """
    X_valid= X_valid.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    
    X_train.drop(['Date','SecuritiesCode'], axis=1, inplace=True)
    X_val=X_valid[X_valid.columns[~X_valid.columns.isin(['Date','SecuritiesCode'])]]
    val_dates=X_valid.Date.unique()[1:-1]
    print("\nTrain Shape: {} {}, Valid Shape: {} {}".format(X_train.shape, y_train.shape, X_val.shape, y_val.shape))
    
    #gbm = LGBMRegressor(**params)
    gbm = LGBMRegressor().fit(X_train, y_train, 
                                      eval_set=[(X_train, y_train), (X_val, y_val)],
                                      verbose=300, 
                                      eval_metric=['mae','mse'])
    y_pred = gbm.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    feat_importance["Importance_Fold"+str(fold)]=gbm.feature_importances_
    feat_importance.set_index(X_train.columns, inplace=True)
        
    rank=[]
    X_val_df=X_valid[X_valid.Date.isin(val_dates)]
    for i in X_val_df.Date.unique():
        temp_df = X_val_df[X_val_df.Date == i].drop(['Date','SecuritiesCode'],axis=1)
        temp_df["pred"] = gbm.predict(temp_df)
        temp_df["Rank"] = (temp_df["pred"].rank(method="first", ascending=False)-1).astype(int)
        rank.append(temp_df["Rank"].values)

    stock_rank=pd.Series([x for y in rank for x in y], name="Rank")
    df=pd.concat([X_val_df.reset_index(drop=True),stock_rank,
                  df_prices_feat[df_prices_feat.Date.isin(val_dates)]['Target'].reset_index(drop=True)], axis=1)
    sharpe=calc_spread_return_sharpe(df)
    sharpe_ratio.append(sharpe)
    print("Valid Sharpe: {}, RMSE: {}, MAE: {}".format(sharpe,rmse,mae))
    
    del X_train, y_train,  X_val, y_val ##, pipeline_val_df, pipeline_X_df
    gc.collect()
    
print("\nAverage cross-validation Sharpe Ratio: {:.4f}, standard deviation = {:.2f}.".format(np.mean(sharpe_ratio),np.std(sharpe_ratio)))
```


```python
feat_importance['avg'] = feat_importance.mean(axis=1)
feat_importance = feat_importance.sort_values(by='avg',ascending=True)
feat_importance = feat_importance.tail(20)
pal=sns.color_palette("plasma_r", 29).as_hex()[2:]

fig=go.Figure()
for i in range(len(feat_importance.index)):
    fig.add_shape(dict(type="line", y0=i, y1=i, x0=0, x1=feat_importance['avg'][i], 
                       line_color=pal[::-1][i],opacity=0.7,line_width=4))
    fig.add_trace(go.Scatter(x=feat_importance['avg'], y=feat_importance.index, mode='markers', 
                             marker_color=pal[::-1], marker_size=8,
                             hovertemplate='%{y} Importance = %{x:.0f}<extra></extra>'))
    fig.update_layout(title='Overall Feature Importance', 
                      xaxis=dict(title='Average Importance',zeroline=False),
                      yaxis_showgrid=False, margin=dict(l=120,t=80),
                      height=700, width=800)
fig.show()
```


```python
model_performance.append({"model":"LGBMRegressor","Mean Sharpe Ratio":np.mean(sharpe_ratio),"SD Sharpe Ratio":np.std(sharpe_ratio)})
```

**XGBoost Regressor**


```python
ts_fold = TimeSeriesSplit(n_splits=5,gap=5000)
feat_importance=pd.DataFrame()
sharpe_ratio=[]
```


```python
%%time
for fold, (train_idx, val_idx) in enumerate(ts_fold.split(X, y)):
    
    print(f"\n========================== Fold {fold+1} ==========================")
        
    X_train, y_train = X.iloc[train_idx,:], y[train_idx]
    X_valid, y_val = X.iloc[val_idx,:], y[val_idx]
    
    print("Train Date range: {} to {}".format(X_train.Date.min(),X_train.Date.max()))
    print("Valid Date range: {} to {}".format(X_valid.Date.min(),X_valid.Date.max()))
    
    
    ##### TRAIIIIIIIIIIN
    """
    pipeline_X_df = full_pipeline.fit_transform(X_train)
    pipeline_X_df = pd.DataFrame(pipeline_X_df, columns = num_attribs)
    
    pipeline_X_df.reset_index(drop = True, inplace=True)
    X_train.reset_index(drop = True, inplace=True)

    X_train.drop(num_attribs,axis=1,inplace=True)
    X_train = pd.concat([X_train,pipeline_X_df], axis=1)
    #"""
    X_train= X_train.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    
    ##### VALIDATION
    """
    pipeline_val_df = full_pipeline.transform(X_valid)
    pipeline_val_df = pd.DataFrame(pipeline_val_df, columns = num_attribs)
    pipeline_val_df.reset_index(drop = True, inplace=True)
    X_valid.reset_index(drop = True, inplace=True)
    X_valid.drop(num_attribs,axis=1,inplace=True)
    X_valid = pd.concat([X_valid,pipeline_val_df], axis=1)
    
   # """
    X_valid= X_valid.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    
    X_train.drop(['Date','SecuritiesCode'], axis=1, inplace=True)
    X_val=X_valid[X_valid.columns[~X_valid.columns.isin(['Date','SecuritiesCode'])]]
    val_dates=X_valid.Date.unique()[1:-1]
    print("\nTrain Shape: {} {}, Valid Shape: {} {}".format(X_train.shape, y_train.shape, X_val.shape, y_val.shape))
    
    xgb = xgboost.XGBRegressor()
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    feat_importance["Importance_Fold"+str(fold)]=xgb.feature_importances_
    feat_importance.set_index(X_train.columns, inplace=True)
        
    rank=[]
    X_val_df=X_valid[X_valid.Date.isin(val_dates)]
    for i in X_val_df.Date.unique():
        temp_df = X_val_df[X_val_df.Date == i].drop(['Date','SecuritiesCode'],axis=1)
        temp_df["pred"] = gbm.predict(temp_df)
        temp_df["Rank"] = (temp_df["pred"].rank(method="first", ascending=False)-1).astype(int)
        rank.append(temp_df["Rank"].values)

    stock_rank=pd.Series([x for y in rank for x in y], name="Rank")
    df=pd.concat([X_val_df.reset_index(drop=True),stock_rank,
                  df_prices_feat[df_prices_feat.Date.isin(val_dates)]['Target'].reset_index(drop=True)], axis=1)
    sharpe=calc_spread_return_sharpe(df)
    sharpe_ratio.append(sharpe)
    print("Valid Sharpe: {}, RMSE: {}, MAE: {}".format(sharpe,rmse,mae))
    
    del X_train, y_train,  X_val, y_val ##, pipeline_val_df, pipeline_X_df
    gc.collect()
    
print("\nAverage cross-validation Sharpe Ratio: {:.4f}, standard deviation = {:.2f}.".format(np.mean(sharpe_ratio),np.std(sharpe_ratio)))
```


```python
feat_importance['avg'] = feat_importance.mean(axis=1)
feat_importance = feat_importance.sort_values(by='avg',ascending=True)
feat_importance = feat_importance.tail(20)
pal=sns.color_palette("plasma_r", 29).as_hex()[2:]

fig=go.Figure()
for i in range(len(feat_importance.index)):
    fig.add_shape(dict(type="line", y0=i, y1=i, x0=0, x1=feat_importance['avg'][i], 
                       line_color=pal[::-1][i],opacity=0.7,line_width=4))
    fig.add_trace(go.Scatter(x=feat_importance['avg'], y=feat_importance.index, mode='markers', 
                             marker_color=pal[::-1], marker_size=8,
                             hovertemplate='%{y} Importance = %{x:.0f}<extra></extra>'))
    fig.update_layout(title='Overall Feature Importance', 
                      xaxis=dict(title='Average Importance',zeroline=False),
                      yaxis_showgrid=False, margin=dict(l=120,t=80),
                      height=700, width=800)
fig.show()
```


```python
model_performance.append({"model":"XGBRegressor","Mean Sharpe Ratio":np.mean(sharpe_ratio),"SD Sharpe Ratio":np.std(sharpe_ratio)})
```

**Random Forest Regressor**

ts_fold = TimeSeriesSplit(n_splits=5,gap=5000)
feat_importance=pd.DataFrame()
sharpe_ratio=[]

%%time
for fold, (train_idx, val_idx) in enumerate(ts_fold.split(X, y)):
    
    print(f"\n========================== Fold {fold+1} ==========================")
        
    X_train, y_train = X.iloc[train_idx,:], y[train_idx]
    X_valid, y_val = X.iloc[val_idx,:], y[val_idx]
    
    print("Train Date range: {} to {}".format(X_train.Date.min(),X_train.Date.max()))
    print("Valid Date range: {} to {}".format(X_valid.Date.min(),X_valid.Date.max()))
    
    
    ##### TRAIIIIIIIIIIN
    """
    pipeline_X_df = full_pipeline.fit_transform(X_train)
    pipeline_X_df = pd.DataFrame(pipeline_X_df, columns = num_attribs)
    
    pipeline_X_df.reset_index(drop = True, inplace=True)
    X_train.reset_index(drop = True, inplace=True)

    X_train.drop(num_attribs,axis=1,inplace=True)
    X_train = pd.concat([X_train,pipeline_X_df], axis=1)
    #"""
    X_train= X_train.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    
    ##### VALIDATION
    """
    pipeline_val_df = full_pipeline.transform(X_valid)
    pipeline_val_df = pd.DataFrame(pipeline_val_df, columns = num_attribs)
    pipeline_val_df.reset_index(drop = True, inplace=True)
    X_valid.reset_index(drop = True, inplace=True)
    X_valid.drop(num_attribs,axis=1,inplace=True)
    X_valid = pd.concat([X_valid,pipeline_val_df], axis=1)
    
   # """
    X_valid= X_valid.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    
    X_train.drop(['Date','SecuritiesCode'], axis=1, inplace=True)
    X_val=X_valid[X_valid.columns[~X_valid.columns.isin(['Date','SecuritiesCode'])]]
    val_dates=X_valid.Date.unique()[1:-1]
    print("\nTrain Shape: {} {}, Valid Shape: {} {}".format(X_train.shape, y_train.shape, X_val.shape, y_val.shape))
    
    regr = RandomForestRegressor()
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    feat_importance["Importance_Fold"+str(fold)]=regr.feature_importances_
    feat_importance.set_index(X_train.columns, inplace=True)
        
    rank=[]
    X_val_df=X_valid[X_valid.Date.isin(val_dates)]
    for i in X_val_df.Date.unique():
        temp_df = X_val_df[X_val_df.Date == i].drop(['Date','SecuritiesCode'],axis=1)
        temp_df["pred"] = gbm.predict(temp_df)
        temp_df["Rank"] = (temp_df["pred"].rank(method="first", ascending=False)-1).astype(int)
        rank.append(temp_df["Rank"].values)

    stock_rank=pd.Series([x for y in rank for x in y], name="Rank")
    df=pd.concat([X_val_df.reset_index(drop=True),stock_rank,
                  df_prices_feat[df_prices_feat.Date.isin(val_dates)]['Target'].reset_index(drop=True)], axis=1)
    sharpe=calc_spread_return_sharpe(df)
    sharpe_ratio.append(sharpe)
    print("Valid Sharpe: {}, RMSE: {}, MAE: {}".format(sharpe,rmse,mae))
    
    del X_train, y_train,  X_val, y_val ##, pipeline_val_df, pipeline_X_df
    gc.collect()
    
print("\nAverage cross-validation Sharpe Ratio: {:.4f}, standard deviation = {:.2f}.".format(np.mean(sharpe_ratio),np.std(sharpe_ratio)))

model_performance.append({"model":"RandomForestRegressor","Mean Sharpe Ratio":np.mean(sharpe_ratio),"SD Sharpe Ratio":std.mean(sharpe_ratio)})

feat_importance['avg'] = feat_importance.mean(axis=1)
feat_importance = feat_importance.sort_values(by='avg',ascending=True)
feat_importance = feat_importance.tail(20)
pal=sns.color_palette("plasma_r", 29).as_hex()[2:]

fig=go.Figure()
for i in range(len(feat_importance.index)):
    fig.add_shape(dict(type="line", y0=i, y1=i, x0=0, x1=feat_importance['avg'][i], 
                       line_color=pal[::-1][i],opacity=0.7,line_width=4))
    fig.add_trace(go.Scatter(x=feat_importance['avg'], y=feat_importance.index, mode='markers', 
                             marker_color=pal[::-1], marker_size=8,
                             hovertemplate='%{y} Importance = %{x:.0f}<extra></extra>'))
    fig.update_layout(title='Overall Feature Importance', 
                      xaxis=dict(title='Average Importance',zeroline=False),
                      yaxis_showgrid=False, margin=dict(l=120,t=80),
                      height=700, width=800)
fig.show()

### Hyperparameter optimization with Optuna

Once the best model is selected, the best hyperparameters are selected using Optuna. Optuna is a state-of-the-art Machine Learning hyperparameter optimization framework. 

More info about Optuna: https://optuna.org/

https://www.kaggle.com/code/swimmy/lgbm-model-fe-portfolio

https://www.kaggle.com/code/marketneutral/purged-time-series-cv-xgboost-optuna/notebook


```python
ts_fold = TimeSeriesSplit(n_splits=5,gap=5000)
```


```python
def objective(trial):

    # Optuna suggest params
    """
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 350, 1000),
        'num_leaves' : trial.suggest_int('n_estimators', 150, 400),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_uniform('learning_rate', 0.01, 0.10),
        'subsample': trial.suggest_uniform('subsample', 0.50, 0.90),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.50, 0.90),
        'device': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 0,
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 10.0)
    }
    """
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 350, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_uniform('learning_rate', 0.01, 0.10),
        'subsample': trial.suggest_uniform('subsample', 0.50, 0.90),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.50, 0.90)
        """,
        'tree_method': 'gpu_hist' """  
    }
    
    for fold, (train_idx, val_idx) in enumerate(ts_fold.split(X, y)):

        print(f"\n========================== Fold {fold+1} ==========================")

        X_train, y_train = X.iloc[train_idx,:], y[train_idx]
        X_valid, y_val = X.iloc[val_idx,:], y[val_idx]

        print("Train Date range: {} to {}".format(X_train.Date.min(),X_train.Date.max()))
        print("Valid Date range: {} to {}".format(X_valid.Date.min(),X_valid.Date.max()))


        ##### TRAIIIIIIIIIIN
        """
        pipeline_X_df = full_pipeline.fit_transform(X_train)
        pipeline_X_df = pd.DataFrame(pipeline_X_df, columns = num_attribs)

        pipeline_X_df.reset_index(drop = True, inplace=True)
        X_train.reset_index(drop = True, inplace=True)

        X_train.drop(num_attribs,axis=1,inplace=True)
        X_train = pd.concat([X_train,pipeline_X_df], axis=1)
      #  """
        X_train= X_train.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

        ##### VALIDATION
        """
        pipeline_val_df = full_pipeline.transform(X_valid)
        pipeline_val_df = pd.DataFrame(pipeline_val_df, columns = num_attribs)
        pipeline_val_df.reset_index(drop = True, inplace=True)
        X_valid.reset_index(drop = True, inplace=True)
        X_valid.drop(num_attribs,axis=1,inplace=True)
        X_valid = pd.concat([X_valid,pipeline_val_df], axis=1)

       # """
        X_valid= X_valid.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

        X_train.drop(['Date','SecuritiesCode'], axis=1, inplace=True)
        X_val=X_valid[X_valid.columns[~X_valid.columns.isin(['Date','SecuritiesCode'])]]
        val_dates=X_valid.Date.unique()[1:-1]
        print("\nTrain Shape: {} {}, Valid Shape: {} {}".format(X_train.shape, y_train.shape, X_val.shape, y_val.shape))
        """
        model = LGBMRegressor(**params)
        model.fit(X_train, y_train, 
                                      eval_set=[(X_train, y_train), (X_val, y_val)],
                                      verbose = False,
                                      eval_metric=['mae','mse'])
        """
        
        y_pred = gbm.predict(X_val)
        
        xgb = xgboost.XGBRegressor()
        xgb.fit(X_train, y_train)
        y_pred = xgb.predict(X_val)
    
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        #feat_importance["Importance_Fold"+str(fold)]=gbm.feature_importances_
        #feat_importance.set_index(X_train.columns, inplace=True)

        rank=[]
        X_val_df=X_valid[X_valid.Date.isin(val_dates)]
        for i in X_val_df.Date.unique():
            temp_df = X_val_df[X_val_df.Date == i].drop(['Date','SecuritiesCode'],axis=1)
            temp_df["pred"] = gbm.predict(temp_df)
            temp_df["Rank"] = (temp_df["pred"].rank(method="first", ascending=False)-1).astype(int)
            rank.append(temp_df["Rank"].values)

        stock_rank=pd.Series([x for y in rank for x in y], name="Rank")
        df=pd.concat([X_val_df.reset_index(drop=True),stock_rank,
                      df_prices_feat[df_prices_feat.Date.isin(val_dates)]['Target'].reset_index(drop=True)], axis=1)
        sharpe=calc_spread_return_sharpe(df)
        sharpe_ratio.append(sharpe)
        print("Valid Sharpe: {}, RMSE: {}, MAE: {}".format(sharpe,rmse,mae))

        del X_train, y_train,  X_val, y_val ##, pipeline_val_df, pipeline_X_df
        gc.collect()
        
    return np.mean(sharpe_ratio) #,np.std(sharpe_ratio)
```

**Select the best model, tune the parameters to find the best ones, retrain the WHOLE DATA with the best parameters**


```python
%%time
opt = optuna.create_study(direction='maximize',sampler=optuna.samplers.RandomSampler(seed=0))
opt.optimize(objective,n_trials=10)

# Save best parameters
trial = opt.best_trial
params_best = dict(trial.params.items())
params_best['random_seed'] = 0
    
# Create the model with best parameters
#model_o = LGBMRegressor(**params_best)#

model_o = xgboost.XGBRegressor(**params_best)

```


```python
params_best
```


```python
#Shows scores for all trials
optuna.visualization.plot_optimization_history(opt)
```


```python
# Vizualize parameter importance
optuna.visualization.plot_param_importances(opt)
```


```python
X1 = X.drop(['Date','SecuritiesCode'], axis=1, inplace=False)
model_o.fit(X1,y)
#X.drop(['Date','SecuritiesCode'], axis=1, inplace=True)
```


```python
del df_prices_feat
gc.collect()
```

### 5. API Submission


```python
import jpx_tokyo_market_prediction
env = jpx_tokyo_market_prediction.make_env()
iter_test = env.iter_test()
```


```python
%%time
cols=['Date','SecuritiesCode','Open','High','Low','Close','Volume','AdjustmentFactor']
stock_prices_df=stock_prices_df[stock_prices_df.Date>='2021-08-01'][cols]

#cols_fin = ["Date","Volume","AdjustedClose","pct5","pct10","pct21","pct33","Volatility_5","Volatility_10","Volatility_21","Volatility_33","SMA_10","SMA_20","SMA_50","SMA_60","EMA_10","EMA_20","EMA_50"]
cols_fin = col
cols_fin.remove("Target")
counter = 0
for (prices, options, financials, trades, secondary_prices, sample_prediction) in iter_test:
    
    ## Loading API data and combine it with the dataset
    current_date = prices["Date"].iloc[0]
    if counter == 0:
        df_price_raw = stock_prices_df.loc[stock_prices_df["Date"] < current_date]
    
    df_price_raw = pd.concat([df_price_raw, prices[cols]]).reset_index(drop=True)
    
    ## Feature engineering
    df_price = adjust_price(df_price_raw)
    features = merge_metadata(df_price, stocks2000_df)  
    features = generate_feature(features)
    features = encoding(features) 
    features = features.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    features = features[X.columns.to_list()]
    
    ## --------- Pipeline
    """
    pipeline_feat_df = full_pipeline.transform(features)
    pipeline_feat_df = pd.DataFrame(pipeline_feat_df, columns = num_attribs)
    pipeline_feat_df.reset_index(drop = True, inplace=True)
    features.reset_index(drop = True, inplace=True)
    features.drop(num_attribs,axis=1,inplace=True)
    features = pd.concat([features,pipeline_feat_df], axis=1)
    """
    ## ---------
    
    feat = features[features.Date == current_date].drop(['SecuritiesCode','Date'],axis=1)
    display(feat)
    
    ## Prediction using the Model
    #feat["pred"] = gbm.predict(feat)
    
    ## Prediction using model with optimized paramat
    feat["pred"] = model_o.predict(feat)

    ## Generate Ranking 0-1999
    feat["Rank"] = (feat["pred"].rank(method="first", ascending=False)-1).astype(int)
    
    ## Input the ranking to the submission file
    sample_prediction["Rank"] = feat["Rank"].values
    display(sample_prediction.head())
    
    ## Input the ranking to the submission file
    assert sample_prediction["Rank"].notna().all()
    assert sample_prediction["Rank"].min() == 0
    assert sample_prediction["Rank"].max() == len(sample_prediction["Rank"]) - 1
    
    ## Submitt prediction file
    env.predict(sample_prediction)
    counter += 1
```

sharpe=calc_spread_return_sharpe(sample_prediction)
sharpe
