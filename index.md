# JPX-Tokyo-Stock-Exchange Prediction 

## Introduction

The competition will involve building portfolios from the stocks eligible for predictions (around 2,000 stocks). Specifically, each participant ranks the stocks from highest to lowest expected returns and is evaluated on the difference in returns between the top and bottom 200 stocks. The participants have access to financial data from the Japanese market, such as stock information and historical stock prices to train and test your model.

The competitor with the highest Sharpe Ratio will result the winner of the challenge. In the competition the Sharpe Ratio is calculated as follows:


![image info](./jpx-prediction_files/Competition_Sharpe.PNG)


The Sharpe ratio was developed by Nobel laureate William F. Sharpe and is used to help investors understand the return of an investment compared to its risk

## Libraries

**Main libraries**


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


<style type='text/css'>
.datatable table.frame { margin-bottom: 0; }
.datatable table.frame thead { border-bottom: none; }
.datatable table.frame tr.coltypes td {  color: #FFFFFF;  line-height: 6px;  padding: 0 0.5em;}
.datatable .bool    { background: #DDDD99; }
.datatable .object  { background: #565656; }
.datatable .int     { background: #5D9E5D; }
.datatable .float   { background: #4040CC; }
.datatable .str     { background: #CC4040; }
.datatable .time    { background: #40CC40; }
.datatable .row_index {  background: var(--jp-border-color3);  border-right: 1px solid var(--jp-border-color0);  color: var(--jp-ui-font-color3);  font-size: 9px;}
.datatable .frame tbody td { text-align: left; }
.datatable .frame tr.coltypes .row_index {  background: var(--jp-border-color0);}
.datatable th:nth-child(2) { padding-left: 12px; }
.datatable .hellipsis {  color: var(--jp-cell-editor-border-color);}
.datatable .vellipsis {  background: var(--jp-layout-color0);  color: var(--jp-cell-editor-border-color);}
.datatable .na {  color: var(--jp-cell-editor-border-color);  font-size: 80%;}
.datatable .sp {  opacity: 0.25;}
.datatable .footer { font-size: 9px; }
.datatable .frame_dimensions {  background: var(--jp-border-color3);  border-top: 1px solid var(--jp-border-color0);  color: var(--jp-ui-font-color3);  display: inline-block;  opacity: 0.6;  padding: 1px 10px 1px 5px;}
</style>



**Optimization libraries**


```python
import optuna
```

## Datasets



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

## 1. Exploratory Data Analysis (EDA)

### Stock prices


File: stock_prices.csv

**File Description**

The core file of interest, including the daily closing price for each stock and the target column. Following is column information recorded in stock_price_spec.csv:


```python
stock_prices_info
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Column</th>
      <th>Sample value</th>
      <th>Type</th>
      <th>Addendum</th>
      <th>Remarks</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>RowId</td>
      <td>20170104_1301</td>
      <td>string</td>
      <td>NaN</td>
      <td>Unique ID of price records</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Date</td>
      <td>2017-01-04 0:00:00</td>
      <td>date</td>
      <td>NaN</td>
      <td>Trade date</td>
    </tr>
    <tr>
      <th>2</th>
      <td>SecuritiesCode</td>
      <td>1301</td>
      <td>Int64</td>
      <td>NaN</td>
      <td>Local securities code</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Open</td>
      <td>2734</td>
      <td>float</td>
      <td>NaN</td>
      <td>first traded price on a day</td>
    </tr>
    <tr>
      <th>4</th>
      <td>High</td>
      <td>2755</td>
      <td>float</td>
      <td>NaN</td>
      <td>highest traded price on a day</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Low</td>
      <td>2730</td>
      <td>float</td>
      <td>NaN</td>
      <td>lowest traded price on a day</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Close</td>
      <td>2742</td>
      <td>float</td>
      <td>NaN</td>
      <td>last traded price on a day</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Volume</td>
      <td>31400</td>
      <td>Int64</td>
      <td>NaN</td>
      <td>number of traded stocks on a day</td>
    </tr>
    <tr>
      <th>8</th>
      <td>AdjustmentFactor</td>
      <td>1</td>
      <td>float</td>
      <td>NaN</td>
      <td>to calculate theoretical price/volume when split/reverse-split happens (NOT including dividend/allotment of shares/)</td>
    </tr>
    <tr>
      <th>9</th>
      <td>SupervisionFlag</td>
      <td>FALSE</td>
      <td>boolean</td>
      <td>NaN</td>
      <td>Flag of Securities Under Supervision &amp; Securities to Be Delisted\nhttps://www.jpx.co.jp/english/listing/market-alerts/supervision/00-archives/index.html )</td>
    </tr>
    <tr>
      <th>10</th>
      <td>ExpectedDividend</td>
      <td>NaN</td>
      <td>float</td>
      <td>NaN</td>
      <td>Expected dividend value for ex-right date. This value is recorded 2 business days before ex-dividend date.</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Target</td>
      <td>0.00073</td>
      <td>float</td>
      <td>NaN</td>
      <td>Change ratio of adjusted closing price between t+2 and t+1 where t+0 is TradeDate</td>
    </tr>
  </tbody>
</table>
</div>




```python
display(stock_prices_df.head(5))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RowId</th>
      <th>Date</th>
      <th>SecuritiesCode</th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
      <th>AdjustmentFactor</th>
      <th>ExpectedDividend</th>
      <th>SupervisionFlag</th>
      <th>Target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>20170104_1301</td>
      <td>2017-01-04</td>
      <td>1301</td>
      <td>2734.0</td>
      <td>2755.0</td>
      <td>2730.0</td>
      <td>2742.0</td>
      <td>31400</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.000730</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20170104_1332</td>
      <td>2017-01-04</td>
      <td>1332</td>
      <td>568.0</td>
      <td>576.0</td>
      <td>563.0</td>
      <td>571.0</td>
      <td>2798500</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.012324</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20170104_1333</td>
      <td>2017-01-04</td>
      <td>1333</td>
      <td>3150.0</td>
      <td>3210.0</td>
      <td>3140.0</td>
      <td>3210.0</td>
      <td>270800</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.006154</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20170104_1376</td>
      <td>2017-01-04</td>
      <td>1376</td>
      <td>1510.0</td>
      <td>1550.0</td>
      <td>1510.0</td>
      <td>1550.0</td>
      <td>11300</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.011053</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20170104_1377</td>
      <td>2017-01-04</td>
      <td>1377</td>
      <td>3270.0</td>
      <td>3350.0</td>
      <td>3270.0</td>
      <td>3330.0</td>
      <td>150800</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.003026</td>
    </tr>
  </tbody>
</table>
</div>



```python
stock_prices_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2332531 entries, 0 to 2332530
    Data columns (total 12 columns):
     #   Column            Dtype  
    ---  ------            -----  
     0   RowId             object 
     1   Date              object 
     2   SecuritiesCode    int64  
     3   Open              float64
     4   High              float64
     5   Low               float64
     6   Close             float64
     7   Volume            int64  
     8   AdjustmentFactor  float64
     9   ExpectedDividend  float64
     10  SupervisionFlag   bool   
     11  Target            float64
    dtypes: bool(1), float64(7), int64(2), object(2)
    memory usage: 198.0+ MB
    

**Missing values**


```python
display(pd.isna(stock_prices_df).sum()/len(stock_prices_df)*100)
```


    RowId                0.000000
    Date                 0.000000
    SecuritiesCode       0.000000
    Open                 0.326169
    High                 0.326169
    Low                  0.326169
    Close                0.326169
    Volume               0.000000
    AdjustmentFactor     0.000000
    ExpectedDividend    99.191222
    SupervisionFlag      0.000000
    Target               0.010204
    dtype: float64




```python
missing_high["Date"].value_counts()
```

    2020-10-01    1988
    2017-03-16      15
    2019-10-09      14
    2019-04-04      14
    2021-10-29      13
                  ... 
    2017-12-25       1
    2017-12-26       1
    2020-03-13       1
    2018-01-04       1
    2018-02-21       1
    Name: Date, Length: 1175, dtype: int64




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2020-10-01</td>
      <td>1988</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2017-03-16</td>
      <td>15</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2019-10-09</td>
      <td>14</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2019-04-04</td>
      <td>14</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2021-10-29</td>
      <td>13</td>
    </tr>
  </tbody>
</table>
</div>


![image info](./jpx-prediction_files/Missing_days.PNG)

2020-10-01 is the day with the most amount of missing data


### Stocks

stock_list.csv


**File Description**

Stock metadata. Following is column information recorded in stock_list_spec.csv:


```python
display(stocks_info)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Column</th>
      <th>Sample value</th>
      <th>Type</th>
      <th>Addendum</th>
      <th>Remarks</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>SecuritiesCode</td>
      <td>1301</td>
      <td>Int64</td>
      <td>NaN</td>
      <td>Local Securities Code</td>
    </tr>
    <tr>
      <th>1</th>
      <td>EffectiveDate</td>
      <td>20211230</td>
      <td>date</td>
      <td>NaN</td>
      <td>the effective date</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Name</td>
      <td>KYOKUYO CO.,LTD.</td>
      <td>string</td>
      <td>NaN</td>
      <td>Name of security</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Section/Products</td>
      <td>First Section (Domestic)</td>
      <td>string</td>
      <td>NaN</td>
      <td>Section/Product</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NewMarketSegment</td>
      <td>Prime Market</td>
      <td>string</td>
      <td>NaN</td>
      <td>New market segment effective from 2022-04-04 (as of 15:30 JST on Mar 11 2022)\nref. https://www.jpx.co.jp/english/equities/market-restructure/market-segments/index.html</td>
    </tr>
    <tr>
      <th>5</th>
      <td>33SectorCode</td>
      <td>50</td>
      <td>Int64</td>
      <td>NaN</td>
      <td>33 Sector Name\n\nref. https://www.jpx.co.jp/english/markets/indices/line-up/files/e_fac_13_sector.pdf</td>
    </tr>
    <tr>
      <th>6</th>
      <td>33SectorName</td>
      <td>Fishery, Agriculture and Forestry</td>
      <td>string</td>
      <td>NaN</td>
      <td>33 Sector Name\n\nref. https://www.jpx.co.jp/english/markets/indices/line-up/files/e_fac_13_sector.pdf</td>
    </tr>
    <tr>
      <th>7</th>
      <td>17SectorCode</td>
      <td>1</td>
      <td>Int64</td>
      <td>NaN</td>
      <td>17 Sector Code\nref. https://www.jpx.co.jp/english/markets/indices/line-up/files/e_fac_13_sector.pdf</td>
    </tr>
    <tr>
      <th>8</th>
      <td>17SectorName</td>
      <td>FOODS</td>
      <td>string</td>
      <td>NaN</td>
      <td>17 Sector Name\nref. https://www.jpx.co.jp/english/markets/indices/line-up/files/e_fac_13_sector.pdf</td>
    </tr>
    <tr>
      <th>9</th>
      <td>NewIndexSeriesSizeCode</td>
      <td>7</td>
      <td>Int64</td>
      <td>NaN</td>
      <td>TOPIX New Index Series code\n\nref. https://www.jpx.co.jp/english/markets/indices/line-up/files/e_fac_12_size.pdf</td>
    </tr>
    <tr>
      <th>10</th>
      <td>NewIndexSeriesSize</td>
      <td>TOPIX Small 2</td>
      <td>string</td>
      <td>NaN</td>
      <td>TOPIX New Index Series Name\n\nref. https://www.jpx.co.jp/english/markets/indices/line-up/files/e_fac_12_size.pdf</td>
    </tr>
    <tr>
      <th>11</th>
      <td>TradeDate</td>
      <td>20211230</td>
      <td>date</td>
      <td>NaN</td>
      <td>Trade date to calculate MarketCapitalization</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Close</td>
      <td>3080</td>
      <td>float</td>
      <td>NaN</td>
      <td>Close price to calculate MarketCapitalization</td>
    </tr>
    <tr>
      <th>13</th>
      <td>IssuedShares</td>
      <td>1.09E+07</td>
      <td>float</td>
      <td>NaN</td>
      <td>Issued shares</td>
    </tr>
    <tr>
      <th>14</th>
      <td>MarketCapitalization</td>
      <td>33659111640</td>
      <td>float</td>
      <td>NaN</td>
      <td>Market capitalization on Dec 3 2021</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Universe0</td>
      <td>TRUE</td>
      <td>boolean</td>
      <td>NaN</td>
      <td>a flag of prediction target universe (top 2000 stocks by market capitalization)</td>
    </tr>
  </tbody>
</table>
</div>



```python
display(stocks_df.head(5))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SecuritiesCode</th>
      <th>EffectiveDate</th>
      <th>Name</th>
      <th>Section/Products</th>
      <th>NewMarketSegment</th>
      <th>33SectorCode</th>
      <th>33SectorName</th>
      <th>17SectorCode</th>
      <th>17SectorName</th>
      <th>NewIndexSeriesSizeCode</th>
      <th>NewIndexSeriesSize</th>
      <th>TradeDate</th>
      <th>Close</th>
      <th>IssuedShares</th>
      <th>MarketCapitalization</th>
      <th>Universe0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1301</td>
      <td>20211230</td>
      <td>KYOKUYO CO.,LTD.</td>
      <td>First Section (Domestic)</td>
      <td>Prime Market</td>
      <td>50</td>
      <td>Fishery, Agriculture and Forestry</td>
      <td>1</td>
      <td>FOODS</td>
      <td>7</td>
      <td>TOPIX Small 2</td>
      <td>20211230.0</td>
      <td>3080.0</td>
      <td>1.092828e+07</td>
      <td>3.365911e+10</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1305</td>
      <td>20211230</td>
      <td>Daiwa ETF-TOPIX</td>
      <td>ETFs/ ETNs</td>
      <td>NaN</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>20211230.0</td>
      <td>2097.0</td>
      <td>3.634636e+09</td>
      <td>7.621831e+12</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1306</td>
      <td>20211230</td>
      <td>NEXT FUNDS TOPIX Exchange Traded Fund</td>
      <td>ETFs/ ETNs</td>
      <td>NaN</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>20211230.0</td>
      <td>2073.5</td>
      <td>7.917718e+09</td>
      <td>1.641739e+13</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1308</td>
      <td>20211230</td>
      <td>Nikko Exchange Traded Index Fund TOPIX</td>
      <td>ETFs/ ETNs</td>
      <td>NaN</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>20211230.0</td>
      <td>2053.0</td>
      <td>3.736943e+09</td>
      <td>7.671945e+12</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1309</td>
      <td>20211230</td>
      <td>NEXT FUNDS ChinaAMC SSE50 Index Exchange Traded Fund</td>
      <td>ETFs/ ETNs</td>
      <td>NaN</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>20211230.0</td>
      <td>44280.0</td>
      <td>7.263200e+04</td>
      <td>3.216145e+09</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



```python
stocks_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 4417 entries, 0 to 4416
    Data columns (total 16 columns):
     #   Column                  Non-Null Count  Dtype  
    ---  ------                  --------------  -----  
     0   SecuritiesCode          4417 non-null   int64  
     1   EffectiveDate           4417 non-null   int64  
     2   Name                    4417 non-null   object 
     3   Section/Products        4417 non-null   object 
     4   NewMarketSegment        3772 non-null   object 
     5   33SectorCode            4417 non-null   object 
     6   33SectorName            4417 non-null   object 
     7   17SectorCode            4417 non-null   object 
     8   17SectorName            4417 non-null   object 
     9   NewIndexSeriesSizeCode  4417 non-null   object 
     10  NewIndexSeriesSize      4417 non-null   object 
     11  TradeDate               4121 non-null   float64
     12  Close                   4121 non-null   float64
     13  IssuedShares            4121 non-null   float64
     14  MarketCapitalization    4121 non-null   float64
     15  Universe0               4417 non-null   bool   
    dtypes: bool(1), float64(4), int64(2), object(9)
    memory usage: 522.1+ KB
    


```python
display(pd.isna(stocks_df).sum()/len(stocks_df)*100)
```


    SecuritiesCode             0.000000
    EffectiveDate              0.000000
    Name                       0.000000
    Section/Products           0.000000
    NewMarketSegment          14.602671
    33SectorCode               0.000000
    33SectorName               0.000000
    17SectorCode               0.000000
    17SectorName               0.000000
    NewIndexSeriesSizeCode     0.000000
    NewIndexSeriesSize         0.000000
    TradeDate                  6.701381
    Close                      6.701381
    IssuedShares               6.701381
    MarketCapitalization       6.701381
    Universe0                  0.000000
    dtype: float64



    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SecuritiesCode</th>
      <th>EffectiveDate</th>
      <th>Name</th>
      <th>Section/Products</th>
      <th>NewMarketSegment</th>
      <th>33SectorCode</th>
      <th>33SectorName</th>
      <th>17SectorCode</th>
      <th>17SectorName</th>
      <th>NewIndexSeriesSizeCode</th>
      <th>NewIndexSeriesSize</th>
      <th>TradeDate</th>
      <th>Close</th>
      <th>IssuedShares</th>
      <th>MarketCapitalization</th>
      <th>Universe0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>62</th>
      <td>1408</td>
      <td>20000101</td>
      <td>ITbook Co.,LTD.</td>
      <td>Mothers (Domestic)</td>
      <td>NaN</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
    </tr>
    <tr>
      <th>164</th>
      <td>1606</td>
      <td>20000101</td>
      <td>Japan Drilling Co.,Ltd.</td>
      <td>First Section (Domestic)</td>
      <td>NaN</td>
      <td>1050</td>
      <td>Mining</td>
      <td>2</td>
      <td>ENERGY RESOURCES</td>
      <td>-</td>
      <td>-</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
    </tr>
    <tr>
      <th>231</th>
      <td>1722</td>
      <td>20000101</td>
      <td>MISAWA HOMES CO.,LTD.</td>
      <td>First Section (Domestic)</td>
      <td>NaN</td>
      <td>2050</td>
      <td>Construction</td>
      <td>3</td>
      <td>CONSTRUCTION &amp; MATERIALS</td>
      <td>-</td>
      <td>-</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
    </tr>
    <tr>
      <th>235</th>
      <td>1728</td>
      <td>20000101</td>
      <td>MISAWA HOMES CHUGOKU CO.,LTD.</td>
      <td>JASDAQ(Standard / Domestic)</td>
      <td>NaN</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
    </tr>
    <tr>
      <th>252</th>
      <td>1782</td>
      <td>20000101</td>
      <td>Joban Kaihatsu Co.,Ltd.</td>
      <td>JASDAQ(Standard / Domestic)</td>
      <td>NaN</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
stocks_df["Universe0"].value_counts()
```




    False    2417
    True     2000
    Name: Universe0, dtype: int64






## Selecting the top 2000 Stocks (Universe0 Flag = True)


```python
stocks2000_df = stocks_df[stocks_df["Universe0"]]
display(stocks2000_df.head())
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SecuritiesCode</th>
      <th>EffectiveDate</th>
      <th>Name</th>
      <th>Section/Products</th>
      <th>NewMarketSegment</th>
      <th>33SectorCode</th>
      <th>33SectorName</th>
      <th>17SectorCode</th>
      <th>17SectorName</th>
      <th>NewIndexSeriesSizeCode</th>
      <th>NewIndexSeriesSize</th>
      <th>TradeDate</th>
      <th>Close</th>
      <th>IssuedShares</th>
      <th>MarketCapitalization</th>
      <th>Universe0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1301</td>
      <td>20211230</td>
      <td>KYOKUYO CO.,LTD.</td>
      <td>First Section (Domestic)</td>
      <td>Prime Market</td>
      <td>50</td>
      <td>Fishery, Agriculture and Forestry</td>
      <td>1</td>
      <td>FOODS</td>
      <td>7</td>
      <td>TOPIX Small 2</td>
      <td>20211230.0</td>
      <td>3080.0</td>
      <td>10928283.0</td>
      <td>3.365911e+10</td>
      <td>True</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1332</td>
      <td>20211230</td>
      <td>Nippon Suisan Kaisha,Ltd.</td>
      <td>First Section (Domestic)</td>
      <td>Prime Market</td>
      <td>50</td>
      <td>Fishery, Agriculture and Forestry</td>
      <td>1</td>
      <td>FOODS</td>
      <td>4</td>
      <td>TOPIX Mid400</td>
      <td>20211230.0</td>
      <td>543.0</td>
      <td>312430277.0</td>
      <td>1.696496e+11</td>
      <td>True</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1333</td>
      <td>20211230</td>
      <td>Maruha Nichiro Corporation</td>
      <td>First Section (Domestic)</td>
      <td>Prime Market</td>
      <td>50</td>
      <td>Fishery, Agriculture and Forestry</td>
      <td>1</td>
      <td>FOODS</td>
      <td>4</td>
      <td>TOPIX Mid400</td>
      <td>20211230.0</td>
      <td>2405.0</td>
      <td>52656910.0</td>
      <td>1.266399e+11</td>
      <td>True</td>
    </tr>
    <tr>
      <th>37</th>
      <td>1375</td>
      <td>20211230</td>
      <td>YUKIGUNI MAITAKE CO.,LTD.</td>
      <td>First Section (Domestic)</td>
      <td>Prime Market</td>
      <td>50</td>
      <td>Fishery, Agriculture and Forestry</td>
      <td>1</td>
      <td>FOODS</td>
      <td>6</td>
      <td>TOPIX Small 1</td>
      <td>20211230.0</td>
      <td>1196.0</td>
      <td>39910700.0</td>
      <td>4.773320e+10</td>
      <td>True</td>
    </tr>
    <tr>
      <th>38</th>
      <td>1376</td>
      <td>20211230</td>
      <td>KANEKO SEEDS CO.,LTD.</td>
      <td>First Section (Domestic)</td>
      <td>Standard Market</td>
      <td>50</td>
      <td>Fishery, Agriculture and Forestry</td>
      <td>1</td>
      <td>FOODS</td>
      <td>7</td>
      <td>TOPIX Small 2</td>
      <td>20211230.0</td>
      <td>1504.0</td>
      <td>11772626.0</td>
      <td>1.770603e+10</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>
   


```python
stocks2000_df["33SectorName"].value_counts()
```




    Information & Communication          229
    Services                             201
    Retail Trade                         179
    Electric Appliances                  156
    Wholesale Trade                      151
                                        ... 
    Oil and Coal Products                  9
    Fishery, Agriculture and Forestry      8
    Marine Transportation                  7
    Mining                                 5
    Air Transportation                     4
    Name: 33SectorName, Length: 33, dtype: int64



![image info](./jpx-prediction_files/sector_pie.png)



```python
section2000 = stocks2000_df["Section/Products"].value_counts().to_frame()
section2000.reset_index(inplace=True)
section2000["percentage"] = section2000["Section/Products"]/section2000["Section/Products"].sum()*100
section2000
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>Section/Products</th>
      <th>percentage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>First Section (Domestic)</td>
      <td>1711</td>
      <td>85.55</td>
    </tr>
    <tr>
      <th>1</th>
      <td>JASDAQ(Standard / Domestic)</td>
      <td>107</td>
      <td>5.35</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Second Section(Domestic)</td>
      <td>90</td>
      <td>4.50</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Mothers (Domestic)</td>
      <td>85</td>
      <td>4.25</td>
    </tr>
    <tr>
      <th>4</th>
      <td>JASDAQ(Growth/Domestic)</td>
      <td>7</td>
      <td>0.35</td>
    </tr>
  </tbody>
</table>
</div>



![image info](./jpx-prediction_files/Section_pie.png)






## Merging Stock metadate with stock price


```python
df_prices =  merge_metadata(stock_prices_df, stocks2000_df)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RowId</th>
      <th>Date</th>
      <th>SecuritiesCode</th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
      <th>AdjustmentFactor</th>
      <th>ExpectedDividend</th>
      <th>SupervisionFlag</th>
      <th>Target</th>
      <th>Name</th>
      <th>Section/Products</th>
      <th>NewMarketSegment</th>
      <th>33SectorCode</th>
      <th>33SectorName</th>
      <th>17SectorCode</th>
      <th>17SectorName</th>
      <th>NewIndexSeriesSizeCode</th>
      <th>NewIndexSeriesSize</th>
      <th>IssuedShares</th>
      <th>MarketCapitalization</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>20170104_1301</td>
      <td>2017-01-04</td>
      <td>1301</td>
      <td>2734.0</td>
      <td>2755.0</td>
      <td>2730.0</td>
      <td>2742.0</td>
      <td>31400</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.000730</td>
      <td>KYOKUYO CO.,LTD.</td>
      <td>First Section (Domestic)</td>
      <td>Prime Market</td>
      <td>50</td>
      <td>Fishery, Agriculture and Forestry</td>
      <td>1</td>
      <td>FOODS</td>
      <td>7</td>
      <td>TOPIX Small 2</td>
      <td>10928283.0</td>
      <td>3.365911e+10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20170105_1301</td>
      <td>2017-01-05</td>
      <td>1301</td>
      <td>2743.0</td>
      <td>2747.0</td>
      <td>2735.0</td>
      <td>2738.0</td>
      <td>17900</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.002920</td>
      <td>KYOKUYO CO.,LTD.</td>
      <td>First Section (Domestic)</td>
      <td>Prime Market</td>
      <td>50</td>
      <td>Fishery, Agriculture and Forestry</td>
      <td>1</td>
      <td>FOODS</td>
      <td>7</td>
      <td>TOPIX Small 2</td>
      <td>10928283.0</td>
      <td>3.365911e+10</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20170106_1301</td>
      <td>2017-01-06</td>
      <td>1301</td>
      <td>2734.0</td>
      <td>2744.0</td>
      <td>2720.0</td>
      <td>2740.0</td>
      <td>19900</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>False</td>
      <td>-0.001092</td>
      <td>KYOKUYO CO.,LTD.</td>
      <td>First Section (Domestic)</td>
      <td>Prime Market</td>
      <td>50</td>
      <td>Fishery, Agriculture and Forestry</td>
      <td>1</td>
      <td>FOODS</td>
      <td>7</td>
      <td>TOPIX Small 2</td>
      <td>10928283.0</td>
      <td>3.365911e+10</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20170110_1301</td>
      <td>2017-01-10</td>
      <td>1301</td>
      <td>2745.0</td>
      <td>2754.0</td>
      <td>2735.0</td>
      <td>2748.0</td>
      <td>24200</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>False</td>
      <td>-0.005100</td>
      <td>KYOKUYO CO.,LTD.</td>
      <td>First Section (Domestic)</td>
      <td>Prime Market</td>
      <td>50</td>
      <td>Fishery, Agriculture and Forestry</td>
      <td>1</td>
      <td>FOODS</td>
      <td>7</td>
      <td>TOPIX Small 2</td>
      <td>10928283.0</td>
      <td>3.365911e+10</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20170111_1301</td>
      <td>2017-01-11</td>
      <td>1301</td>
      <td>2748.0</td>
      <td>2752.0</td>
      <td>2737.0</td>
      <td>2745.0</td>
      <td>9300</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>False</td>
      <td>-0.003295</td>
      <td>KYOKUYO CO.,LTD.</td>
      <td>First Section (Domestic)</td>
      <td>Prime Market</td>
      <td>50</td>
      <td>Fishery, Agriculture and Forestry</td>
      <td>1</td>
      <td>FOODS</td>
      <td>7</td>
      <td>TOPIX Small 2</td>
      <td>10928283.0</td>
      <td>3.365911e+10</td>
    </tr>
  </tbody>
</table>
</div>


**Calculating Target variable statistics**


```python
mean_securities_df = df_prices.groupby(["SecuritiesCode"])["Target"].mean()
total_mean_securities = mean_securities_df.mean()
total_mean_securities
```




    0.0004714963317502812

**Target Securities mean**

![image info](./jpx-prediction_files/Target_securities_histo.png)





    
![png](jpx-prediction_files/jpx-prediction_89_0.png)
    



```python
mean_date_df = df_prices.groupby(["Date"])["Target"].mean()
total_mean_date = mean_date_df.mean()
total_mean_date
```




    0.00044572606297777287

**Target Securities Date**

![image info](./jpx-prediction_files/Target_dates_histo.png)



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


    
![image info](./jpx-prediction_files/target_mean_sector.png)

    


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


    
    
![image info](./jpx-prediction_files/target_mean_section.png)

    


### Secondary stock prices

secondary_stock_prices.csv

**File Description**

Securities with low liquidity (few opportunities to trade). Following is column information recorded in stock_price_spec.csv:


```python
stock_prices_info
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Column</th>
      <th>Sample value</th>
      <th>Type</th>
      <th>Addendum</th>
      <th>Remarks</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>RowId</td>
      <td>20170104_1301</td>
      <td>string</td>
      <td>NaN</td>
      <td>Unique ID of price records</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Date</td>
      <td>2017-01-04 0:00:00</td>
      <td>date</td>
      <td>NaN</td>
      <td>Trade date</td>
    </tr>
    <tr>
      <th>2</th>
      <td>SecuritiesCode</td>
      <td>1301</td>
      <td>Int64</td>
      <td>NaN</td>
      <td>Local securities code</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Open</td>
      <td>2734</td>
      <td>float</td>
      <td>NaN</td>
      <td>first traded price on a day</td>
    </tr>
    <tr>
      <th>4</th>
      <td>High</td>
      <td>2755</td>
      <td>float</td>
      <td>NaN</td>
      <td>highest traded price on a day</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Low</td>
      <td>2730</td>
      <td>float</td>
      <td>NaN</td>
      <td>lowest traded price on a day</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Close</td>
      <td>2742</td>
      <td>float</td>
      <td>NaN</td>
      <td>last traded price on a day</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Volume</td>
      <td>31400</td>
      <td>Int64</td>
      <td>NaN</td>
      <td>number of traded stocks on a day</td>
    </tr>
    <tr>
      <th>8</th>
      <td>AdjustmentFactor</td>
      <td>1</td>
      <td>float</td>
      <td>NaN</td>
      <td>to calculate theoretical price/volume when split/reverse-split happens (NOT including dividend/allotment of shares/)</td>
    </tr>
    <tr>
      <th>9</th>
      <td>SupervisionFlag</td>
      <td>FALSE</td>
      <td>boolean</td>
      <td>NaN</td>
      <td>Flag of Securities Under Supervision &amp; Securities to Be Delisted\nhttps://www.jpx.co.jp/english/listing/market-alerts/supervision/00-archives/index.html )</td>
    </tr>
    <tr>
      <th>10</th>
      <td>ExpectedDividend</td>
      <td>NaN</td>
      <td>float</td>
      <td>NaN</td>
      <td>Expected dividend value for ex-right date. This value is recorded 2 business days before ex-dividend date.</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Target</td>
      <td>0.00073</td>
      <td>float</td>
      <td>NaN</td>
      <td>Change ratio of adjusted closing price between t+2 and t+1 where t+0 is TradeDate</td>
    </tr>
  </tbody>
</table>
</div>




```python
display(secondary_stock_prices_df.head(5))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RowId</th>
      <th>Date</th>
      <th>SecuritiesCode</th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
      <th>AdjustmentFactor</th>
      <th>ExpectedDividend</th>
      <th>SupervisionFlag</th>
      <th>Target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>20170104_1305</td>
      <td>2017-01-04</td>
      <td>1305</td>
      <td>1594.0</td>
      <td>1618.0</td>
      <td>1594.0</td>
      <td>1615.0</td>
      <td>538190</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>False</td>
      <td>-0.001855</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20170104_1306</td>
      <td>2017-01-04</td>
      <td>1306</td>
      <td>1575.0</td>
      <td>1595.0</td>
      <td>1573.0</td>
      <td>1593.0</td>
      <td>2494980</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>False</td>
      <td>-0.000627</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20170104_1308</td>
      <td>2017-01-04</td>
      <td>1308</td>
      <td>1557.0</td>
      <td>1580.0</td>
      <td>1557.0</td>
      <td>1578.0</td>
      <td>526100</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>False</td>
      <td>-0.001900</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20170104_1309</td>
      <td>2017-01-04</td>
      <td>1309</td>
      <td>28810.0</td>
      <td>29000.0</td>
      <td>28520.0</td>
      <td>28780.0</td>
      <td>403</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.005237</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20170104_1311</td>
      <td>2017-01-04</td>
      <td>1311</td>
      <td>717.0</td>
      <td>735.0</td>
      <td>717.0</td>
      <td>734.0</td>
      <td>5470</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.001359</td>
    </tr>
  </tbody>
</table>
</div>



```python
secondary_stock_prices_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2384575 entries, 0 to 2384574
    Data columns (total 12 columns):
     #   Column            Dtype  
    ---  ------            -----  
     0   RowId             object 
     1   Date              object 
     2   SecuritiesCode    int64  
     3   Open              float64
     4   High              float64
     5   Low               float64
     6   Close             float64
     7   Volume            int64  
     8   AdjustmentFactor  float64
     9   ExpectedDividend  float64
     10  SupervisionFlag   bool   
     11  Target            float64
    dtypes: bool(1), float64(7), int64(2), object(2)
    memory usage: 202.4+ MB
    


```python
display(pd.isna(secondary_stock_prices_df).sum()/len(secondary_stock_prices_df)*100)
```


    RowId                0.000000
    Date                 0.000000
    SecuritiesCode       0.000000
    Open                 3.847688
    High                 3.847688
    Low                  3.847688
    Close                3.847688
    Volume               0.000000
    AdjustmentFactor     0.000000
    ExpectedDividend    99.225942
    SupervisionFlag      0.000000
    Target               0.030110
    dtype: float64



As happened with the primary stock list 2020-10-01 is the day with the most amount of missing data


**NOTE:** Secondary stock prices price movement is correlated with the main stock price movement. Due to the limited amount dedicated to this project, this data will not be included in the prediction model

## Trades

trades.csv

**File Description**

Aggregated summary of trading volumes from the previous business week. Following is column information recorded in trades_spec.csv:


```python
trades_info
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>Column</th>
      <th>Sample value</th>
      <th>Type</th>
      <th>Addendum</th>
      <th>Remarks</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>PublishedDate</td>
      <td>2017-01-13</td>
      <td>date</td>
      <td>NaN</td>
      <td>data published data, usually Thursday on the following week.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>StartDate</td>
      <td>2017-01-04</td>
      <td>date</td>
      <td>NaN</td>
      <td>The first trading date in this trading week</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>EndDate</td>
      <td>2017-01-06</td>
      <td>date</td>
      <td>NaN</td>
      <td>The last trading date in this trading week</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>Section</td>
      <td>Prime Market (First Section)</td>
      <td>string</td>
      <td>NaN</td>
      <td>Market division name</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>TotalSales</td>
      <td>8476800009</td>
      <td>Int64</td>
      <td>NaN</td>
      <td>Please check below documents.\n\nPublished file\nhttps://www.jpx.co.jp/english/markets/statistics-equities/investor-type/b5b4pj000004r9zg-att/stock_val_1_220301.pdf\n\nExplanation of the Trading by Type of Investors\nhttps://www.jpx.co.jp/english/markets/statistics-equities/investor-type/07.html</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>51</th>
      <td>50</td>
      <td>TrustBanksBalance</td>
      <td>-47609502</td>
      <td>Int64</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>52</th>
      <td>51</td>
      <td>OtherFinancialInstitutionsSales</td>
      <td>22410692</td>
      <td>Int64</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>53</th>
      <td>52</td>
      <td>OtherFinancialInstitutionsPurchases</td>
      <td>21764485</td>
      <td>Int64</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>54</th>
      <td>53</td>
      <td>OtherFinancialInstitutionsTotal</td>
      <td>44175177</td>
      <td>Int64</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>55</th>
      <td>54</td>
      <td>OtherFinancialInstitutionsBalance</td>
      <td>-646207</td>
      <td>Int64</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>56 rows × 6 columns</p>
</div>




```python
display(trades_df.head(5))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>StartDate</th>
      <th>EndDate</th>
      <th>Section</th>
      <th>TotalSales</th>
      <th>TotalPurchases</th>
      <th>TotalTotal</th>
      <th>TotalBalance</th>
      <th>ProprietarySales</th>
      <th>ProprietaryPurchases</th>
      <th>ProprietaryTotal</th>
      <th>ProprietaryBalance</th>
      <th>BrokerageSales</th>
      <th>BrokeragePurchases</th>
      <th>BrokerageTotal</th>
      <th>BrokerageBalance</th>
      <th>IndividualsSales</th>
      <th>IndividualsPurchases</th>
      <th>IndividualsTotal</th>
      <th>IndividualsBalance</th>
      <th>ForeignersSales</th>
      <th>ForeignersPurchases</th>
      <th>ForeignersTotal</th>
      <th>ForeignersBalance</th>
      <th>SecuritiesCosSales</th>
      <th>SecuritiesCosPurchases</th>
      <th>SecuritiesCosTotal</th>
      <th>SecuritiesCosBalance</th>
      <th>InvestmentTrustsSales</th>
      <th>InvestmentTrustsPurchases</th>
      <th>InvestmentTrustsTotal</th>
      <th>InvestmentTrustsBalance</th>
      <th>BusinessCosSales</th>
      <th>BusinessCosPurchases</th>
      <th>BusinessCosTotal</th>
      <th>BusinessCosBalance</th>
      <th>OtherInstitutionsSales</th>
      <th>OtherInstitutionsPurchases</th>
      <th>OtherInstitutionsTotal</th>
      <th>OtherInstitutionsBalance</th>
      <th>InsuranceCosSales</th>
      <th>InsuranceCosPurchases</th>
      <th>InsuranceCosTotal</th>
      <th>InsuranceCosBalance</th>
      <th>CityBKsRegionalBKsEtcSales</th>
      <th>CityBKsRegionalBKsEtcPurchase</th>
      <th>CityBKsRegionalBKsEtcTotal</th>
      <th>CityBKsRegionalBKsEtcBalance</th>
      <th>TrustBanksSales</th>
      <th>TrustBanksPurchases</th>
      <th>TrustBanksTotal</th>
      <th>TrustBanksBalance</th>
      <th>OtherFinancialInstitutionsSales</th>
      <th>OtherFinancialInstitutionsPurchases</th>
      <th>OtherFinancialInstitutionsTotal</th>
      <th>OtherFinancialInstitutionsBalance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2017-01-04</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2017-01-05</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2017-01-06</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2017-01-10</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2017-01-11</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>


**NOTE:** Trade volumen info can be used to predict the Target value. Due to the limited amount dedicated to this project, this data will not be included in the prediction model

### Financials 

financials.csv

**File Description**

Financial information about stock products. Following is column information recorded in stock_fin_spec.csv:


```python
financials_info
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Column</th>
      <th>Sample value</th>
      <th>Type</th>
      <th>Addendum</th>
      <th>Remarks</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>DisclosureNumber</td>
      <td>20161025419878</td>
      <td>Int64</td>
      <td>NaN</td>
      <td>Unique ID for disclosure documents.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>DateCode</td>
      <td>20170106_7888</td>
      <td>string</td>
      <td>NaN</td>
      <td>combination of TradeDate and LocalCode (this is not unique for stock_fin as a company can disclose multiple documents on a day)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Date</td>
      <td>2017-01-06 0:00:00</td>
      <td>date</td>
      <td>NaN</td>
      <td>Trade date. This column is used to align with stock_price's TradeDate</td>
    </tr>
    <tr>
      <th>3</th>
      <td>SecuritiesCode</td>
      <td>7888</td>
      <td>Int64</td>
      <td>NaN</td>
      <td>Local Securities Code</td>
    </tr>
    <tr>
      <th>4</th>
      <td>DisclosedDate</td>
      <td>2017-01-06 0:00:00</td>
      <td>date</td>
      <td>NaN</td>
      <td>Date on which the document disclosed.</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>40</th>
      <td>ChangesInAccountingEstimates</td>
      <td>FALSE</td>
      <td>boolean</td>
      <td>TRUE, FALSE, or blank</td>
      <td>Changes in accounting estimates. (changes in accounting estimates that had been made for the preparation of consolidated financial statements for the previous consolidated fiscal year or any prior consolidated fiscal year, or quarterly consolidated financial statements for the immediately preceding or prior quarterly consolidated accounting period or cumulative quarterly consolidated accounting period based on new information that has become available)</td>
    </tr>
    <tr>
      <th>41</th>
      <td>RetrospectiveRestatement</td>
      <td>FALSE</td>
      <td>boolean</td>
      <td>TRUE, FALSE, or blank</td>
      <td>The reflection, in consolidated financial statements or quarterly consolidated financial statements, of the correction of an error in consolidated financial statements for the previous consolidated fiscal year or any prior consolidated fiscal year or quarterly consolidated financial statements for the immediately preceding or prior quarterly consolidated accounting period or cumulative quarterly consolidated accounting period.\n\n[Note] Blank if is difficult to distinguish changes in accounting policies from changes in accounting estimates.</td>
    </tr>
    <tr>
      <th>42</th>
      <td>NumberOfIssuedAndOutstandingSharesAtTheEndOfFiscalYearIncludingTreasuryStock</td>
      <td>25688569</td>
      <td>Int64</td>
      <td>Actual value for the current accounting period.</td>
      <td>Number of issued shares at the end of the period (including treasury shares) as of the current accounting period.</td>
    </tr>
    <tr>
      <th>43</th>
      <td>NumberOfTreasuryStockAtTheEndOfFiscalYear</td>
      <td>203627</td>
      <td>Int64</td>
      <td>Actual value for the current accounting period.</td>
      <td>Number of treasury shares at the end of the period as of the current accounting period.</td>
    </tr>
    <tr>
      <th>44</th>
      <td>AverageNumberOfShares</td>
      <td>25485430</td>
      <td>Int64</td>
      <td>Actual value for the current accounting period.</td>
      <td>Average number of shares between the start date of the current fiscal year and the end date of in the current period.</td>
    </tr>
  </tbody>
</table>
<p>45 rows × 5 columns</p>
</div>




```python
financials_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DisclosureNumber</th>
      <th>DateCode</th>
      <th>Date</th>
      <th>SecuritiesCode</th>
      <th>DisclosedDate</th>
      <th>DisclosedTime</th>
      <th>DisclosedUnixTime</th>
      <th>TypeOfDocument</th>
      <th>CurrentPeriodEndDate</th>
      <th>TypeOfCurrentPeriod</th>
      <th>CurrentFiscalYearStartDate</th>
      <th>CurrentFiscalYearEndDate</th>
      <th>NetSales</th>
      <th>OperatingProfit</th>
      <th>OrdinaryProfit</th>
      <th>Profit</th>
      <th>EarningsPerShare</th>
      <th>TotalAssets</th>
      <th>Equity</th>
      <th>EquityToAssetRatio</th>
      <th>BookValuePerShare</th>
      <th>ResultDividendPerShare1stQuarter</th>
      <th>ResultDividendPerShare2ndQuarter</th>
      <th>ResultDividendPerShare3rdQuarter</th>
      <th>ResultDividendPerShareFiscalYearEnd</th>
      <th>ResultDividendPerShareAnnual</th>
      <th>ForecastDividendPerShare1stQuarter</th>
      <th>ForecastDividendPerShare2ndQuarter</th>
      <th>ForecastDividendPerShare3rdQuarter</th>
      <th>ForecastDividendPerShareFiscalYearEnd</th>
      <th>ForecastDividendPerShareAnnual</th>
      <th>ForecastNetSales</th>
      <th>ForecastOperatingProfit</th>
      <th>ForecastOrdinaryProfit</th>
      <th>ForecastProfit</th>
      <th>ForecastEarningsPerShare</th>
      <th>ApplyingOfSpecificAccountingOfTheQuarterlyFinancialStatements</th>
      <th>MaterialChangesInSubsidiaries</th>
      <th>ChangesBasedOnRevisionsOfAccountingStandard</th>
      <th>ChangesOtherThanOnesBasedOnRevisionsOfAccountingStandard</th>
      <th>ChangesInAccountingEstimates</th>
      <th>RetrospectiveRestatement</th>
      <th>NumberOfIssuedAndOutstandingSharesAtTheEndOfFiscalYearIncludingTreasuryStock</th>
      <th>NumberOfTreasuryStockAtTheEndOfFiscalYear</th>
      <th>AverageNumberOfShares</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.016121e+13</td>
      <td>20170104_2753</td>
      <td>2017-01-04</td>
      <td>2753.0</td>
      <td>2017-01-04</td>
      <td>07:30:00</td>
      <td>1.483483e+09</td>
      <td>3QFinancialStatements_Consolidated_JP</td>
      <td>2016-12-31</td>
      <td>3Q</td>
      <td>2016-04-01</td>
      <td>2017-03-31</td>
      <td>22761000000</td>
      <td>2147000000</td>
      <td>2234000000</td>
      <td>1494000000</td>
      <td>218.23</td>
      <td>22386000000.0</td>
      <td>18295000000.0</td>
      <td>0.817</td>
      <td>2671.42</td>
      <td>－</td>
      <td>50.0</td>
      <td>－</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>50.0</td>
      <td>100.0</td>
      <td>31800000000</td>
      <td>3255000000</td>
      <td>3300000000</td>
      <td>2190000000</td>
      <td>319.76</td>
      <td>NaN</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>6848800.0</td>
      <td>－</td>
      <td>6848800.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.017010e+13</td>
      <td>20170104_3353</td>
      <td>2017-01-04</td>
      <td>3353.0</td>
      <td>2017-01-04</td>
      <td>15:00:00</td>
      <td>1.483510e+09</td>
      <td>3QFinancialStatements_Consolidated_JP</td>
      <td>2016-11-30</td>
      <td>3Q</td>
      <td>2016-03-01</td>
      <td>2017-02-28</td>
      <td>22128000000</td>
      <td>820000000</td>
      <td>778000000</td>
      <td>629000000</td>
      <td>328.57</td>
      <td>25100000000.0</td>
      <td>7566000000.0</td>
      <td>0.301</td>
      <td>NaN</td>
      <td>－</td>
      <td>36.0</td>
      <td>－</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>36.0</td>
      <td>72.0</td>
      <td>30200000000</td>
      <td>1350000000</td>
      <td>1300000000</td>
      <td>930000000</td>
      <td>485.36</td>
      <td>NaN</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>2035000.0</td>
      <td>118917</td>
      <td>1916083.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.016123e+13</td>
      <td>20170104_4575</td>
      <td>2017-01-04</td>
      <td>4575.0</td>
      <td>2017-01-04</td>
      <td>12:00:00</td>
      <td>1.483499e+09</td>
      <td>ForecastRevision</td>
      <td>2016-12-31</td>
      <td>2Q</td>
      <td>2016-07-01</td>
      <td>2017-06-30</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>110000000</td>
      <td>-465000000</td>
      <td>-466000000</td>
      <td>-467000000</td>
      <td>-93.11</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.017010e+13</td>
      <td>20170105_2659</td>
      <td>2017-01-05</td>
      <td>2659.0</td>
      <td>2017-01-05</td>
      <td>15:00:00</td>
      <td>1.483596e+09</td>
      <td>3QFinancialStatements_Consolidated_JP</td>
      <td>2016-11-30</td>
      <td>3Q</td>
      <td>2016-03-01</td>
      <td>2017-02-28</td>
      <td>134781000000</td>
      <td>11248000000</td>
      <td>11558000000</td>
      <td>7171000000</td>
      <td>224.35</td>
      <td>128464000000.0</td>
      <td>100905000000.0</td>
      <td>0.765</td>
      <td>3073.12</td>
      <td>－</td>
      <td>0.0</td>
      <td>－</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>42.0</td>
      <td>42.0</td>
      <td>177683000000</td>
      <td>14168000000</td>
      <td>14473000000</td>
      <td>9111000000</td>
      <td>285.05</td>
      <td>NaN</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>31981654.0</td>
      <td>18257</td>
      <td>31963405.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.017011e+13</td>
      <td>20170105_3050</td>
      <td>2017-01-05</td>
      <td>3050.0</td>
      <td>2017-01-05</td>
      <td>15:30:00</td>
      <td>1.483598e+09</td>
      <td>ForecastRevision</td>
      <td>2017-02-28</td>
      <td>FY</td>
      <td>2016-02-29</td>
      <td>2017-02-28</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>－</td>
      <td>－</td>
      <td>13.0</td>
      <td>24.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
financials_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 92956 entries, 0 to 92955
    Data columns (total 45 columns):
     #   Column                                                                        Non-Null Count  Dtype  
    ---  ------                                                                        --------------  -----  
     0   DisclosureNumber                                                              92954 non-null  float64
     1   DateCode                                                                      92954 non-null  object 
     2   Date                                                                          92956 non-null  object 
     3   SecuritiesCode                                                                92954 non-null  float64
     4   DisclosedDate                                                                 92954 non-null  object 
     5   DisclosedTime                                                                 92954 non-null  object 
     6   DisclosedUnixTime                                                             92954 non-null  float64
     7   TypeOfDocument                                                                92954 non-null  object 
     8   CurrentPeriodEndDate                                                          92954 non-null  object 
     9   TypeOfCurrentPeriod                                                           92954 non-null  object 
     10  CurrentFiscalYearStartDate                                                    92954 non-null  object 
     11  CurrentFiscalYearEndDate                                                      92954 non-null  object 
     12  NetSales                                                                      75448 non-null  object 
     13  OperatingProfit                                                               73446 non-null  object 
     14  OrdinaryProfit                                                                75328 non-null  object 
     15  Profit                                                                        75380 non-null  object 
     16  EarningsPerShare                                                              74958 non-null  object 
     17  TotalAssets                                                                   75433 non-null  object 
     18  Equity                                                                        75431 non-null  object 
     19  EquityToAssetRatio                                                            74739 non-null  object 
     20  BookValuePerShare                                                             35773 non-null  object 
     21  ResultDividendPerShare1stQuarter                                              74380 non-null  object 
     22  ResultDividendPerShare2ndQuarter                                              55940 non-null  object 
     23  ResultDividendPerShare3rdQuarter                                              37677 non-null  object 
     24  ResultDividendPerShareFiscalYearEnd                                           19416 non-null  object 
     25  ResultDividendPerShareAnnual                                                  19415 non-null  object 
     26  ForecastDividendPerShare1stQuarter                                            19241 non-null  object 
     27  ForecastDividendPerShare2ndQuarter                                            42619 non-null  object 
     28  ForecastDividendPerShare3rdQuarter                                            60807 non-null  object 
     29  ForecastDividendPerShareFiscalYearEnd                                         79021 non-null  object 
     30  ForecastDividendPerShareAnnual                                                79022 non-null  object 
     31  ForecastNetSales                                                              82842 non-null  object 
     32  ForecastOperatingProfit                                                       81083 non-null  object 
     33  ForecastOrdinaryProfit                                                        82718 non-null  object 
     34  ForecastProfit                                                                83856 non-null  object 
     35  ForecastEarningsPerShare                                                      82842 non-null  object 
     36  ApplyingOfSpecificAccountingOfTheQuarterlyFinancialStatements                 7249 non-null   object 
     37  MaterialChangesInSubsidiaries                                                 64504 non-null  object 
     38  ChangesBasedOnRevisionsOfAccountingStandard                                   74895 non-null  object 
     39  ChangesOtherThanOnesBasedOnRevisionsOfAccountingStandard                      74895 non-null  object 
     40  ChangesInAccountingEstimates                                                  74126 non-null  object 
     41  RetrospectiveRestatement                                                      70396 non-null  object 
     42  NumberOfIssuedAndOutstandingSharesAtTheEndOfFiscalYearIncludingTreasuryStock  74918 non-null  object 
     43  NumberOfTreasuryStockAtTheEndOfFiscalYear                                     74918 non-null  object 
     44  AverageNumberOfShares                                                         74349 non-null  object 
    dtypes: float64(3), object(42)
    memory usage: 31.9+ MB
    


```python
display(pd.isna(financials_df).sum()/len(financials_df)*100)
```


    DisclosureNumber                                                                 0.002152
    DateCode                                                                         0.002152
    Date                                                                             0.000000
    SecuritiesCode                                                                   0.002152
    DisclosedDate                                                                    0.002152
                                                                                      ...    
    ChangesInAccountingEstimates                                                    20.256896
    RetrospectiveRestatement                                                        24.269547
    NumberOfIssuedAndOutstandingSharesAtTheEndOfFiscalYearIncludingTreasuryStock    19.404880
    NumberOfTreasuryStockAtTheEndOfFiscalYear                                       19.404880
    AverageNumberOfShares                                                           20.016997
    Length: 45, dtype: float64


Notably a high amount of data is missing in financials_df

**NOTE:** Leveraging the financial data in the Fundamental analysis is one popular method in target calculation. Due to the limited amount dedicated to this project, this data will not be included in the prediction model

### Options

options.csv

**File Description**

Data on the status of a variety of options based on the broader market. Many options include implicit predictions of the future price of the stock market and so may be of interest even though the options are not scored directly. Following is column information recorded in options_spec.csv:



```python
options_info
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Column</th>
      <th>Sample value</th>
      <th>Type</th>
      <th>Addendum</th>
      <th>Remarks</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>DateCode</td>
      <td>20170104_144122718</td>
      <td>string</td>
      <td>NaN</td>
      <td>Unique ID for option price records</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Date</td>
      <td>2017-01-04 0:00:00</td>
      <td>date</td>
      <td>NaN</td>
      <td>Trade date and time</td>
    </tr>
    <tr>
      <th>2</th>
      <td>OptionsCode</td>
      <td>144122718</td>
      <td>string</td>
      <td>NaN</td>
      <td>Local Securities Code (link to https://www.jpx.co.jp/english/sicc/regulations/b5b4pj0000023mqo-att/(HP)sakimono20220208-e.pdf )</td>
    </tr>
    <tr>
      <th>3</th>
      <td>WholeDayOpen</td>
      <td>0</td>
      <td>float</td>
      <td>NaN</td>
      <td>Opening Price for Whole Trading Day</td>
    </tr>
    <tr>
      <th>4</th>
      <td>WholeDayHigh</td>
      <td>0</td>
      <td>float</td>
      <td>NaN</td>
      <td>High Price for Whole Trading Day</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>26</th>
      <td>BaseVolatility</td>
      <td>17.4736</td>
      <td>float</td>
      <td>NaN</td>
      <td>The volatility at the time of calculating the settlement price</td>
    </tr>
    <tr>
      <th>27</th>
      <td>ImpliedVolatility</td>
      <td>26.0651</td>
      <td>float</td>
      <td>NaN</td>
      <td>Implied Volatility</td>
    </tr>
    <tr>
      <th>28</th>
      <td>InterestRate</td>
      <td>0.1282</td>
      <td>float</td>
      <td>NaN</td>
      <td>Interest rate for calculation</td>
    </tr>
    <tr>
      <th>29</th>
      <td>DividendRate</td>
      <td>1.6817</td>
      <td>float</td>
      <td>NaN</td>
      <td>Dividendv yeild</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Dividend</td>
      <td>0.00E+00</td>
      <td>float</td>
      <td>NaN</td>
      <td>Devidend</td>
    </tr>
  </tbody>
</table>
<p>31 rows × 5 columns</p>
</div>




```python
options_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DateCode</th>
      <th>Date</th>
      <th>OptionsCode</th>
      <th>WholeDayOpen</th>
      <th>WholeDayHigh</th>
      <th>WholeDayLow</th>
      <th>WholeDayClose</th>
      <th>NightSessionOpen</th>
      <th>NightSessionHigh</th>
      <th>NightSessionLow</th>
      <th>NightSessionClose</th>
      <th>DaySessionOpen</th>
      <th>DaySessionHigh</th>
      <th>DaySessionLow</th>
      <th>DaySessionClose</th>
      <th>TradingVolume</th>
      <th>OpenInterest</th>
      <th>TradingValue</th>
      <th>ContractMonth</th>
      <th>StrikePrice</th>
      <th>WholeDayVolume</th>
      <th>Putcall</th>
      <th>LastTradingDay</th>
      <th>SpecialQuotationDay</th>
      <th>SettlementPrice</th>
      <th>TheoreticalPrice</th>
      <th>BaseVolatility</th>
      <th>ImpliedVolatility</th>
      <th>InterestRate</th>
      <th>DividendRate</th>
      <th>Dividend</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>20170104_132010018</td>
      <td>2017-01-04</td>
      <td>132010018</td>
      <td>650.0</td>
      <td>650.0</td>
      <td>480.0</td>
      <td>480.0</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>650.0</td>
      <td>650.0</td>
      <td>480.0</td>
      <td>480.0</td>
      <td>6</td>
      <td>19</td>
      <td>3455000</td>
      <td>201701</td>
      <td>20000.0</td>
      <td>6</td>
      <td>1</td>
      <td>20170112</td>
      <td>20170113</td>
      <td>480.0</td>
      <td>478.4587</td>
      <td>17.4736</td>
      <td>17.5865</td>
      <td>0.0091</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20170104_132010118</td>
      <td>2017-01-04</td>
      <td>132010118</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>201701</td>
      <td>20125.0</td>
      <td>0</td>
      <td>1</td>
      <td>20170112</td>
      <td>20170113</td>
      <td>575.0</td>
      <td>571.1385</td>
      <td>17.4736</td>
      <td>16.5000</td>
      <td>0.0091</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20170104_132010218</td>
      <td>2017-01-04</td>
      <td>132010218</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>201701</td>
      <td>20250.0</td>
      <td>0</td>
      <td>1</td>
      <td>20170112</td>
      <td>20170113</td>
      <td>680.0</td>
      <td>677.3710</td>
      <td>17.4736</td>
      <td>15.8644</td>
      <td>0.0091</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20170104_132010318</td>
      <td>2017-01-04</td>
      <td>132010318</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>201701</td>
      <td>20375.0</td>
      <td>0</td>
      <td>1</td>
      <td>20170112</td>
      <td>20170113</td>
      <td>795.0</td>
      <td>791.0383</td>
      <td>17.4736</td>
      <td>15.2288</td>
      <td>0.0091</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20170104_132010518</td>
      <td>2017-01-04</td>
      <td>132010518</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>201701</td>
      <td>20500.0</td>
      <td>0</td>
      <td>1</td>
      <td>20170112</td>
      <td>20170113</td>
      <td>910.0</td>
      <td>909.9947</td>
      <td>17.4736</td>
      <td>14.5932</td>
      <td>0.0091</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
options_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3567694 entries, 0 to 3567693
    Data columns (total 31 columns):
     #   Column               Dtype  
    ---  ------               -----  
     0   DateCode             object 
     1   Date                 object 
     2   OptionsCode          int64  
     3   WholeDayOpen         float64
     4   WholeDayHigh         float64
     5   WholeDayLow          float64
     6   WholeDayClose        float64
     7   NightSessionOpen     object 
     8   NightSessionHigh     object 
     9   NightSessionLow      object 
     10  NightSessionClose    object 
     11  DaySessionOpen       float64
     12  DaySessionHigh       float64
     13  DaySessionLow        float64
     14  DaySessionClose      float64
     15  TradingVolume        int64  
     16  OpenInterest         int64  
     17  TradingValue         int64  
     18  ContractMonth        int64  
     19  StrikePrice          float64
     20  WholeDayVolume       int64  
     21  Putcall              int64  
     22  LastTradingDay       int64  
     23  SpecialQuotationDay  int64  
     24  SettlementPrice      float64
     25  TheoreticalPrice     float64
     26  BaseVolatility       float64
     27  ImpliedVolatility    float64
     28  InterestRate         float64
     29  DividendRate         float64
     30  Dividend             float64
    dtypes: float64(16), int64(9), object(6)
    memory usage: 843.8+ MB
    

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




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RowId</th>
      <th>Date</th>
      <th>SecuritiesCode</th>
      <th>Volume</th>
      <th>ExpectedDividend</th>
      <th>SupervisionFlag</th>
      <th>Target</th>
      <th>Name</th>
      <th>Section/Products</th>
      <th>NewMarketSegment</th>
      <th>33SectorCode</th>
      <th>33SectorName</th>
      <th>17SectorCode</th>
      <th>17SectorName</th>
      <th>NewIndexSeriesSizeCode</th>
      <th>NewIndexSeriesSize</th>
      <th>IssuedShares</th>
      <th>MarketCapitalization</th>
      <th>CumulativeAdjustmentFactor</th>
      <th>AdjustedClose</th>
      <th>AdjustedOpen</th>
      <th>AdjustedHigh</th>
      <th>AdjustedLow</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>20170104_1301</td>
      <td>2017-01-04</td>
      <td>1301</td>
      <td>31400</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.000730</td>
      <td>KYOKUYO CO.,LTD.</td>
      <td>First Section (Domestic)</td>
      <td>Prime Market</td>
      <td>50</td>
      <td>Fishery, Agriculture and Forestry</td>
      <td>1</td>
      <td>FOODS</td>
      <td>7</td>
      <td>TOPIX Small 2</td>
      <td>10928283.0</td>
      <td>3.365911e+10</td>
      <td>1.0</td>
      <td>2742.0</td>
      <td>2734.0</td>
      <td>2755.0</td>
      <td>2730.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20170105_1301</td>
      <td>2017-01-05</td>
      <td>1301</td>
      <td>17900</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.002920</td>
      <td>KYOKUYO CO.,LTD.</td>
      <td>First Section (Domestic)</td>
      <td>Prime Market</td>
      <td>50</td>
      <td>Fishery, Agriculture and Forestry</td>
      <td>1</td>
      <td>FOODS</td>
      <td>7</td>
      <td>TOPIX Small 2</td>
      <td>10928283.0</td>
      <td>3.365911e+10</td>
      <td>1.0</td>
      <td>2738.0</td>
      <td>2743.0</td>
      <td>2747.0</td>
      <td>2735.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20170106_1301</td>
      <td>2017-01-06</td>
      <td>1301</td>
      <td>19900</td>
      <td>NaN</td>
      <td>False</td>
      <td>-0.001092</td>
      <td>KYOKUYO CO.,LTD.</td>
      <td>First Section (Domestic)</td>
      <td>Prime Market</td>
      <td>50</td>
      <td>Fishery, Agriculture and Forestry</td>
      <td>1</td>
      <td>FOODS</td>
      <td>7</td>
      <td>TOPIX Small 2</td>
      <td>10928283.0</td>
      <td>3.365911e+10</td>
      <td>1.0</td>
      <td>2740.0</td>
      <td>2734.0</td>
      <td>2744.0</td>
      <td>2720.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20170110_1301</td>
      <td>2017-01-10</td>
      <td>1301</td>
      <td>24200</td>
      <td>NaN</td>
      <td>False</td>
      <td>-0.005100</td>
      <td>KYOKUYO CO.,LTD.</td>
      <td>First Section (Domestic)</td>
      <td>Prime Market</td>
      <td>50</td>
      <td>Fishery, Agriculture and Forestry</td>
      <td>1</td>
      <td>FOODS</td>
      <td>7</td>
      <td>TOPIX Small 2</td>
      <td>10928283.0</td>
      <td>3.365911e+10</td>
      <td>1.0</td>
      <td>2748.0</td>
      <td>2745.0</td>
      <td>2754.0</td>
      <td>2735.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20170111_1301</td>
      <td>2017-01-11</td>
      <td>1301</td>
      <td>9300</td>
      <td>NaN</td>
      <td>False</td>
      <td>-0.003295</td>
      <td>KYOKUYO CO.,LTD.</td>
      <td>First Section (Domestic)</td>
      <td>Prime Market</td>
      <td>50</td>
      <td>Fishery, Agriculture and Forestry</td>
      <td>1</td>
      <td>FOODS</td>
      <td>7</td>
      <td>TOPIX Small 2</td>
      <td>10928283.0</td>
      <td>3.365911e+10</td>
      <td>1.0</td>
      <td>2745.0</td>
      <td>2748.0</td>
      <td>2752.0</td>
      <td>2737.0</td>
    </tr>
  </tbody>
</table>
</div>



    

**Less missing values**


```python
display(pd.isna(df_prices_adj).sum()/len(df_prices_adj)*100)
```


    RowId                          0.000000
    Date                           0.000000
    SecuritiesCode                 0.000000
    Volume                         0.000000
    ExpectedDividend              99.191222
    SupervisionFlag                0.000000
    Target                         0.010204
    Name                           0.000000
    Section/Products               0.000000
    NewMarketSegment               0.000000
    33SectorCode                   0.000000
    33SectorName                   0.000000
    17SectorCode                   0.000000
    17SectorName                   0.000000
    NewIndexSeriesSizeCode         0.000000
    NewIndexSeriesSize             0.000000
    IssuedShares                   0.000000
    MarketCapitalization           0.000000
    CumulativeAdjustmentFactor     0.000000
    AdjustedClose                  0.000429
    AdjustedOpen                   0.000429
    AdjustedHigh                   0.000429
    AdjustedLow                    0.000429
    dtype: float64


**Now the OHTC chart shows continuity**

Before adjustment

![image info](./jpx-prediction_files/Before_adjustment.PNG)

After adjustment

![image info](./jpx-prediction_files/After_adjustment.PNG)

## 3. Feature Engineering


**Correlogram to see if there is any autocorrelation**


Correlogram adjusted Close price
![image info](./jpx-prediction_files/Autocorrelation_price.png)

    


Correlogram Target
![image info](./jpx-prediction_files/Autocorrelation_target.png)

    

    


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

Close Price shows autocorrelation but stock return not. Therefore, the target variable itself will not give us much info about future stock movements... At this point the original idea of using a LSTM model was discarded at it was proceed to continue with the feature engineering.

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


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RowId</th>
      <th>Date</th>
      <th>SecuritiesCode</th>
      <th>Volume</th>
      <th>ExpectedDividend</th>
      <th>SupervisionFlag</th>
      <th>Target</th>
      <th>Name</th>
      <th>Section/Products</th>
      <th>NewMarketSegment</th>
      <th>33SectorCode</th>
      <th>33SectorName</th>
      <th>17SectorCode</th>
      <th>17SectorName</th>
      <th>NewIndexSeriesSizeCode</th>
      <th>NewIndexSeriesSize</th>
      <th>IssuedShares</th>
      <th>MarketCapitalization</th>
      <th>CumulativeAdjustmentFactor</th>
      <th>AdjustedClose</th>
      <th>AdjustedOpen</th>
      <th>AdjustedHigh</th>
      <th>AdjustedLow</th>
      <th>pct5</th>
      <th>pct10</th>
      <th>pct21</th>
      <th>pct33</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2245987</th>
      <td>20170104_9726</td>
      <td>2017-01-04</td>
      <td>9726</td>
      <td>472000</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.013605</td>
      <td>KNT-CT Holdings Co.,Ltd.</td>
      <td>First Section (Domestic)</td>
      <td>Standard Market</td>
      <td>9050</td>
      <td>Services</td>
      <td>10</td>
      <td>IT &amp; SERVICES, OTHERS</td>
      <td>7</td>
      <td>TOPIX Small 2</td>
      <td>27331013.0</td>
      <td>3.799011e+10</td>
      <td>10.0</td>
      <td>1470.0</td>
      <td>1460.0</td>
      <td>1480.0</td>
      <td>1450.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2245988</th>
      <td>20170105_9726</td>
      <td>2017-01-05</td>
      <td>9726</td>
      <td>420000</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.020134</td>
      <td>KNT-CT Holdings Co.,Ltd.</td>
      <td>First Section (Domestic)</td>
      <td>Standard Market</td>
      <td>9050</td>
      <td>Services</td>
      <td>10</td>
      <td>IT &amp; SERVICES, OTHERS</td>
      <td>7</td>
      <td>TOPIX Small 2</td>
      <td>27331013.0</td>
      <td>3.799011e+10</td>
      <td>10.0</td>
      <td>1470.0</td>
      <td>1480.0</td>
      <td>1490.0</td>
      <td>1470.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2245989</th>
      <td>20170106_9726</td>
      <td>2017-01-06</td>
      <td>9726</td>
      <td>462000</td>
      <td>NaN</td>
      <td>False</td>
      <td>-0.006579</td>
      <td>KNT-CT Holdings Co.,Ltd.</td>
      <td>First Section (Domestic)</td>
      <td>Standard Market</td>
      <td>9050</td>
      <td>Services</td>
      <td>10</td>
      <td>IT &amp; SERVICES, OTHERS</td>
      <td>7</td>
      <td>TOPIX Small 2</td>
      <td>27331013.0</td>
      <td>3.799011e+10</td>
      <td>10.0</td>
      <td>1490.0</td>
      <td>1470.0</td>
      <td>1490.0</td>
      <td>1470.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2245990</th>
      <td>20170110_9726</td>
      <td>2017-01-10</td>
      <td>9726</td>
      <td>1170000</td>
      <td>NaN</td>
      <td>False</td>
      <td>-0.006623</td>
      <td>KNT-CT Holdings Co.,Ltd.</td>
      <td>First Section (Domestic)</td>
      <td>Standard Market</td>
      <td>9050</td>
      <td>Services</td>
      <td>10</td>
      <td>IT &amp; SERVICES, OTHERS</td>
      <td>7</td>
      <td>TOPIX Small 2</td>
      <td>27331013.0</td>
      <td>3.799011e+10</td>
      <td>10.0</td>
      <td>1520.0</td>
      <td>1500.0</td>
      <td>1530.0</td>
      <td>1490.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2245991</th>
      <td>20170111_9726</td>
      <td>2017-01-11</td>
      <td>9726</td>
      <td>797000</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.006667</td>
      <td>KNT-CT Holdings Co.,Ltd.</td>
      <td>First Section (Domestic)</td>
      <td>Standard Market</td>
      <td>9050</td>
      <td>Services</td>
      <td>10</td>
      <td>IT &amp; SERVICES, OTHERS</td>
      <td>7</td>
      <td>TOPIX Small 2</td>
      <td>27331013.0</td>
      <td>3.799011e+10</td>
      <td>10.0</td>
      <td>1510.0</td>
      <td>1530.0</td>
      <td>1530.0</td>
      <td>1490.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>


**Simple Moving Average (SMA)**


```python
period_avg = [10,20,50,60]
```


```python
for i in period_avg:
    df_9726[f"SMA_{i}"] = df_9726['AdjustedClose'].rolling(window=i).mean()

display(df_9726.tail(5))
```
![image info](./jpx-prediction_files/SMA.png)



**Exponential Moving Average (EMA)**


```python
for i in period_avg:
    df_9726[f"EMA_{i}"] = df_9726['AdjustedClose'].ewm(span=i,adjust=False).mean()

display(df_9726.tail(5))
```

![image info](./jpx-prediction_files/EMA.png)




**One Hot encoding Stock Catagorical data**
- SectorCode33
- Section


```python
cat_encoder = OneHotEncoder(handle_unknown='ignore')
train_set_cat_coded = cat_encoder.fit_transform(df_prices_feat[["33SectorName","17SectorName"]])
train_set_cat_coded.shape
```




    (2332531, 50)




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




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Air Transportation</th>
      <th>Banks</th>
      <th>Chemicals</th>
      <th>Construction</th>
      <th>Electric Appliances</th>
      <th>Electric Power and Gas</th>
      <th>Fishery, Agriculture and Forestry</th>
      <th>Foods</th>
      <th>Glass and Ceramics Products</th>
      <th>Information &amp; Communication</th>
      <th>Insurance</th>
      <th>Iron and Steel</th>
      <th>Land Transportation</th>
      <th>Machinery</th>
      <th>Marine Transportation</th>
      <th>Metal Products</th>
      <th>Mining</th>
      <th>Nonferrous Metals</th>
      <th>Oil and Coal Products</th>
      <th>Other Financing Business</th>
      <th>Other Products</th>
      <th>Pharmaceutical</th>
      <th>Precision Instruments</th>
      <th>Pulp and Paper</th>
      <th>Real Estate</th>
      <th>Retail Trade</th>
      <th>Rubber Products</th>
      <th>Securities and Commodities Futures</th>
      <th>Services</th>
      <th>Textiles and Apparels</th>
      <th>Transportation Equipment</th>
      <th>Warehousing and Harbor Transportation Service</th>
      <th>Wholesale Trade</th>
      <th>AUTOMOBILES &amp; TRANSPORTATION EQUIPMENT</th>
      <th>BANKS</th>
      <th>COMMERCIAL &amp; WHOLESALE TRADE</th>
      <th>CONSTRUCTION &amp; MATERIALS</th>
      <th>ELECTRIC APPLIANCES &amp; PRECISION INSTRUMENTS</th>
      <th>ELECTRIC POWER &amp; GAS</th>
      <th>ENERGY RESOURCES</th>
      <th>FINANCIALS （EX BANKS）</th>
      <th>FOODS</th>
      <th>IT &amp; SERVICES, OTHERS</th>
      <th>MACHINERY</th>
      <th>PHARMACEUTICAL</th>
      <th>RAW MATERIALS &amp; CHEMICALS</th>
      <th>REAL ESTATE</th>
      <th>RETAIL TRADE</th>
      <th>STEEL &amp; NONFERROUS METALS</th>
      <th>TRANSPORTATION &amp; LOGISTICS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2332526</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2332527</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2332528</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2332529</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2332530</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>2332531 rows × 50 columns</p>
</div>





**Selecting features for the model**


```python
col = ["Date","SecuritiesCode","Target","Volume","AdjustedClose","AdjustedHigh","AdjustedOpen","AdjustedLow","pct5","pct10","pct21","pct33","Volatility_5","Volatility_10","Volatility_21","Volatility_33","SMA_10","SMA_20","SMA_50","SMA_60","EMA_10","EMA_20","EMA_50"] +ls
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


![image info](./jpx-prediction_files/correlation_matrix.png)


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

A data pipeline was originally created to standarize the input attribute for the model. After several tries, it was concluded that the model works better without standarized values.


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

![image info](./jpx-prediction_files/timeseries.png)



**LGBMRegressor**


```python
ts_fold = TimeSeriesSplit(n_splits=10,gap=10000)
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
    
    
    ##### TRAIN standarization pipeline. See comment above
    """
    pipeline_X_df = full_pipeline.fit_transform(X_train)
    pipeline_X_df = pd.DataFrame(pipeline_X_df, columns = num_attribs)
    
    pipeline_X_df.reset_index(drop = True, inplace=True)
    X_train.reset_index(drop = True, inplace=True)

    X_train.drop(num_attribs,axis=1,inplace=True)
    X_train = pd.concat([X_train,pipeline_X_df], axis=1)
    #"""
    X_train= X_train.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    
    ##### VALIDATION. See comment above
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

 ![image info](./jpx-prediction_files/LGBMlight_features.PNG)


```python
model_performance.append({"model":"LGBMRegressor","Mean Sharpe Ratio":np.mean(sharpe_ratio),"SD Sharpe Ratio":np.std(sharpe_ratio)})
```

**XGBoost Regressor**


```python
ts_fold = TimeSeriesSplit(n_splits=10,gap=10000)
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
    
    
    ##### TRAIN
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
        temp_df["pred"] = xgb.predict(temp_df)
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

 ![image info](./jpx-prediction_files/XGBoost_features.PNG)

```python
model_performance.append({"model":"XGBRegressor","Mean Sharpe Ratio":np.mean(sharpe_ratio),"SD Sharpe Ratio":np.std(sharpe_ratio)})
```

**Random Forest Regressor**

**Disclaimer**:

Random Forest Regressor was tried several time. The trainning was extremely slow plus the model would crash several times at the 3-Fold. Somehow the model got stuck. Therefore, the code is maintained on the notebook just for educational purposed. The code is commented though.


```python
# %%time
# ts_fold = TimeSeriesSplit(n_splits=5,gap=5000)
# feat_importance=pd.DataFrame()
# sharpe_ratio=[]

# for fold, (train_idx, val_idx) in enumerate(ts_fold.split(X, y)):
    
#     print(f"\n========================== Fold {fold+1} ==========================")
        
#     X_train, y_train = X.iloc[train_idx,:], y[train_idx]
#     X_valid, y_val = X.iloc[val_idx,:], y[val_idx]
    
#     print("Train Date range: {} to {}".format(X_train.Date.min(),X_train.Date.max()))
#     print("Valid Date range: {} to {}".format(X_valid.Date.min(),X_valid.Date.max()))
    
    
#     ##### TRAIIIIIIIIIIN
#     """
#     pipeline_X_df = full_pipeline.fit_transform(X_train)
#     pipeline_X_df = pd.DataFrame(pipeline_X_df, columns = num_attribs)
    
#     pipeline_X_df.reset_index(drop = True, inplace=True)
#     X_train.reset_index(drop = True, inplace=True)

#     X_train.drop(num_attribs,axis=1,inplace=True)
#     X_train = pd.concat([X_train,pipeline_X_df], axis=1)
#     #"""
    
    
#     ##### VALIDATION
#     """
#     pipeline_val_df = full_pipeline.transform(X_valid)
#     pipeline_val_df = pd.DataFrame(pipeline_val_df, columns = num_attribs)
#     pipeline_val_df.reset_index(drop = True, inplace=True)
#     X_valid.reset_index(drop = True, inplace=True)
#     X_valid.drop(num_attribs,axis=1,inplace=True)
#     X_valid = pd.concat([X_valid,pipeline_val_df], axis=1)
    
#     """
    
#     X_train= X_train.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
#     X_valid= X_valid.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    
#     X_train.drop(['Date','SecuritiesCode'], axis=1, inplace=True)
#     X_val=X_valid[X_valid.columns[~X_valid.columns.isin(['Date','SecuritiesCode'])]]
#     val_dates=X_valid.Date.unique()[1:-1]
#     print("\nTrain Shape: {} {}, Valid Shape: {} {}".format(X_train.shape, y_train.shape, X_val.shape, y_val.shape))
    
#     regr = RandomForestRegressor()
#     regr.fit(X_train, y_train)
#     y_pred = regr.predict(X_val)
#     rmse = np.sqrt(mean_squared_error(y_val, y_pred))
#     mae = mean_absolute_error(y_val, y_pred)
#     feat_importance["Importance_Fold"+str(fold)]=regr.feature_importances_
#     feat_importance.set_index(X_train.columns, inplace=True)
        
#     rank=[]
#     X_val_df=X_valid[X_valid.Date.isin(val_dates)]
#     for i in X_val_df.Date.unique():
#         temp_df = X_val_df[X_val_df.Date == i].drop(['Date','SecuritiesCode'],axis=1)
#         temp_df["pred"] = regr.predict(temp_df)
#         temp_df["Rank"] = (temp_df["pred"].rank(method="first", ascending=False)-1).astype(int)
#         rank.append(temp_df["Rank"].values)

#     stock_rank=pd.Series([x for y in rank for x in y], name="Rank")
#     df=pd.concat([X_val_df.reset_index(drop=True),stock_rank,
#                   df_prices_feat[df_prices_feat.Date.isin(val_dates)]['Target'].reset_index(drop=True)], axis=1)
#     sharpe=calc_spread_return_sharpe(df)
#     sharpe_ratio.append(sharpe)
#     print("Valid Sharpe: {}, RMSE: {}, MAE: {}".format(sharpe,rmse,mae))
    
#     del X_train, y_train,  X_val, y_val ##, pipeline_val_df, pipeline_X_df
#     gc.collect()
    
# print("\nAverage cross-validation Sharpe Ratio: {:.4f}, standard deviation = {:.2f}.".format(np.mean(sharpe_ratio),np.std(sharpe_ratio)))
# model_performance.append({"model":"RandomForestRegressor","Mean Sharpe Ratio":np.mean(sharpe_ratio),"SD Sharpe Ratio":std.mean(sharpe_ratio)})

```


```python

# feat_importance['avg'] = feat_importance.mean(axis=1)
# feat_importance = feat_importance.sort_values(by='avg',ascending=True)
# feat_importance = feat_importance.tail(20)
# pal=sns.color_palette("plasma_r", 29).as_hex()[2:]

# fig=go.Figure()
# for i in range(len(feat_importance.index)):
#     fig.add_shape(dict(type="line", y0=i, y1=i, x0=0, x1=feat_importance['avg'][i], 
#                        line_color=pal[::-1][i],opacity=0.7,line_width=4))
#     fig.add_trace(go.Scatter(x=feat_importance['avg'], y=feat_importance.index, mode='markers', 
#                              marker_color=pal[::-1], marker_size=8,
#                              hovertemplate='%{y} Importance = %{x:.0f}<extra></extra>'))
#     fig.update_layout(title='Overall Feature Importance', 
#                       xaxis=dict(title='Average Importance',zeroline=False),
#                       yaxis_showgrid=False, margin=dict(l=120,t=80),
#                       height=700, width=800)
# fig.show()

```

**Compare Performance from different models**


```python
for result in model_performance:
    print(result)
```

    {'model': 'LGBMRegressor', 'Mean Sharpe Ratio': 0.026261532021756462, 'SD Sharpe Ratio': 0.07761409966205923}
    {'model': 'XGBRegressor', 'Mean Sharpe Ratio': 0.033681211272249253, 'SD Sharpe Ratio': 0.07079662216785994}
    

**Model selection argumentation**

XGBoost presented a better performance. However, the algorithm need considerably more time to be trained. Since this project was carried out using solely Kaggle due to the lack of a proper workstation, it was decided to go for the LGBMlight model. This way, the hyperparameter tuning time was reduced significantly.

### Hyperparameter optimization with Optuna

Once the best model is selected, the best hyperparameters are selected using Optuna. Optuna is a state-of-the-art Machine Learning hyperparameter optimization framework. 

More info about Optuna: https://optuna.org/


```python
ts_fold = TimeSeriesSplit(n_splits=10,gap=10000)
```


```python
sharpe_ratio=[]
def objective(trial):

    # Optuna suggest params
    
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 350, 1000),
        'num_leaves' : trial.suggest_int('n_estimators', 150, 400),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_uniform('learning_rate', 0.01, 0.10),
        'subsample': trial.suggest_uniform('subsample', 0.50, 0.90),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.50, 0.90),
       # 'metric': 'mae',
       # 'device': 'gpu',
       # 'gpu_platform_id': 0,
       # 'gpu_device_id': 0,
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

    }
    """
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
        
        model = LGBMRegressor(**params)
        model.fit(X_train, y_train, 
                                      eval_set=[(X_train, y_train), (X_val, y_val)],
                                      verbose = False,
                                      eval_metric=['mae','mse'])
        y_pred = model.predict(X_val)
        """
        xgb = xgboost.XGBRegressor()
        xgb.fit(X_train, y_train)
        y_pred = xgb.predict(X_val)
        """
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        #feat_importance["Importance_Fold"+str(fold)]=gbm.feature_importances_
        #feat_importance.set_index(X_train.columns, inplace=True)

        rank=[]
        X_val_df=X_valid[X_valid.Date.isin(val_dates)]
        for i in X_val_df.Date.unique():
            temp_df = X_val_df[X_val_df.Date == i].drop(['Date','SecuritiesCode'],axis=1)
            temp_df["pred"] = model.predict(temp_df)
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
model_o = LGBMRegressor(**params_best)
#model_o = xgboost.XGBRegressor(**params_best)

```

    [32m[I 2022-06-22 20:50:02,727][0m A new study created in memory with name: no-name-639cb349-bec5-40c4-999d-904efadbd859[0m
    

    
    ========================== Fold 1 ==========================
    Train Date range: 2017-01-04 00:00:00 to 2017-06-12 00:00:00
    Valid Date range: 2017-06-19 00:00:00 to 2017-12-01 00:00:00
    
    Train Shape: (202033, 70) (202033,), Valid Shape: (212025, 70) (212025,)
    Valid Sharpe: 0.029690761231897945, RMSE: 0.017238235208123513, MAE: 0.011213619574465043
    
    ========================== Fold 2 ==========================
    Train Date range: 2017-01-04 00:00:00 to 2017-11-22 00:00:00
    Valid Date range: 2017-12-01 00:00:00 to 2018-05-18 00:00:00
    
    

    [32m[I 2022-06-22 22:42:01,338][0m Trial 9 finished with value: 0.020638830150889232 and parameters: {'n_estimators': 439, 'max_depth': 4, 'learning_rate': 0.043185265359486774, 'subsample': 0.828397291939174, 'colsample_bytree': 0.5388405103172245, 'reg_alpha': 2.2479136784899816, 'reg_lambda': 0.002423224390060893}. Best is trial 7 with value: 0.02211597946527774.[0m
    

    Valid Sharpe: 0.04370587680535442, RMSE: 0.01973687281664798, MAE: 0.013765479199336473
    CPU times: user 7h 7min 33s, sys: 1min 20s, total: 7h 8min 54s
    Wall time: 1h 51min 58s
    


```python
params_best
```




    {'n_estimators': 586,
     'max_depth': 7,
     'learning_rate': 0.049474136211608837,
     'subsample': 0.8953495352236904,
     'colsample_bytree': 0.5408179242992113,
     'reg_alpha': 0.006847105576684045,
     'reg_lambda': 0.004418125737902547,
     'random_seed': 0}




```python
#Shows scores for all trials
optuna.visualization.plot_optimization_history(opt)
```


<div>                            <div id="ace4a8a4-735e-47f3-84c6-c0391d60bc14" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("ace4a8a4-735e-47f3-84c6-c0391d60bc14")) {                    Plotly.newPlot(                        "ace4a8a4-735e-47f3-84c6-c0391d60bc14",                        [{"mode":"markers","name":"Objective Value","x":[0,1,2,3,4,5,6,7,8,9],"y":[0.011371539123610446,0.021493277479801027,0.02122699810372552,0.01872503160357821,0.017023429220655865,0.019161407905114735,0.020657448117291248,0.02211597946527774,0.02098921760869187,0.020638830150889232],"type":"scatter"},{"name":"Best Value","x":[0,1,2,3,4,5,6,7,8,9],"y":[0.011371539123610446,0.021493277479801027,0.021493277479801027,0.021493277479801027,0.021493277479801027,0.021493277479801027,0.021493277479801027,0.02211597946527774,0.02211597946527774,0.02211597946527774],"type":"scatter"}],                        {"title":{"text":"Optimization History Plot"},"xaxis":{"title":{"text":"#Trials"}},"yaxis":{"title":{"text":"Objective Value"}},"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('ace4a8a4-735e-47f3-84c6-c0391d60bc14');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>



```python
# Vizualize parameter importance
optuna.visualization.plot_param_importances(opt)
```


<div>                            <div id="9998f2f7-d62d-4882-960f-7fb7adca3550" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("9998f2f7-d62d-4882-960f-7fb7adca3550")) {                    Plotly.newPlot(                        "9998f2f7-d62d-4882-960f-7fb7adca3550",                        [{"cliponaxis":false,"hovertemplate":["reg_alpha (LogUniformDistribution): 0.036008177728607316<extra></extra>","reg_lambda (LogUniformDistribution): 0.04030047289633241<extra></extra>","n_estimators (IntUniformDistribution): 0.07149298868879846<extra></extra>","colsample_bytree (UniformDistribution): 0.11559788989085713<extra></extra>","subsample (UniformDistribution): 0.20772500672342112<extra></extra>","max_depth (IntUniformDistribution): 0.2524804648864894<extra></extra>","learning_rate (UniformDistribution): 0.27639499918549404<extra></extra>"],"marker":{"color":"rgb(66,146,198)"},"orientation":"h","text":["0.036008177728607316","0.04030047289633241","0.07149298868879846","0.11559788989085713","0.20772500672342112","0.2524804648864894","0.27639499918549404"],"textposition":"outside","texttemplate":"%{text:.2f}","x":[0.036008177728607316,0.04030047289633241,0.07149298868879846,0.11559788989085713,0.20772500672342112,0.2524804648864894,0.27639499918549404],"y":["reg_alpha","reg_lambda","n_estimators","colsample_bytree","subsample","max_depth","learning_rate"],"type":"bar"}],                        {"showlegend":false,"title":{"text":"Hyperparameter Importances"},"xaxis":{"title":{"text":"Importance for Objective Value"}},"yaxis":{"title":{"text":"Hyperparameter"}},"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('9998f2f7-d62d-4882-960f-7fb7adca3550');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>



```python
X1 = X.drop(['Date','SecuritiesCode'], axis=1, inplace=False)
model_o.fit(X1,y)
#X.drop(['Date','SecuritiesCode'], axis=1, inplace=True)
```




    LGBMRegressor(colsample_bytree=0.5408179242992113,
                  learning_rate=0.049474136211608837, max_depth=7, n_estimators=586,
                  random_seed=0, reg_alpha=0.006847105576684045,
                  reg_lambda=0.004418125737902547, subsample=0.8953495352236904)




```python
del df_prices_feat
gc.collect()
```




    173



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

    This version of the API is not optimized and should not be used to estimate the runtime of your code on the hidden test set.
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>SecuritiesCode</th>
      <th>Volume</th>
      <th>CumulativeAdjustmentFactor</th>
      <th>AdjustedClose</th>
      <th>AdjustedOpen</th>
      <th>AdjustedHigh</th>
      <th>AdjustedLow</th>
      <th>Name</th>
      <th>Section/Products</th>
      <th>NewMarketSegment</th>
      <th>33SectorCode</th>
      <th>33SectorName</th>
      <th>17SectorCode</th>
      <th>17SectorName</th>
      <th>NewIndexSeriesSizeCode</th>
      <th>NewIndexSeriesSize</th>
      <th>IssuedShares</th>
      <th>MarketCapitalization</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2021-08-02</td>
      <td>1301</td>
      <td>20600</td>
      <td>1.0</td>
      <td>3010.0</td>
      <td>2985.0</td>
      <td>3010.0</td>
      <td>2985.0</td>
      <td>KYOKUYO CO.,LTD.</td>
      <td>First Section (Domestic)</td>
      <td>Prime Market</td>
      <td>50</td>
      <td>Fishery, Agriculture and Forestry</td>
      <td>1</td>
      <td>FOODS</td>
      <td>7</td>
      <td>TOPIX Small 2</td>
      <td>10928283.0</td>
      <td>3.365911e+10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2021-08-03</td>
      <td>1301</td>
      <td>9100</td>
      <td>1.0</td>
      <td>3000.0</td>
      <td>3015.0</td>
      <td>3020.0</td>
      <td>3000.0</td>
      <td>KYOKUYO CO.,LTD.</td>
      <td>First Section (Domestic)</td>
      <td>Prime Market</td>
      <td>50</td>
      <td>Fishery, Agriculture and Forestry</td>
      <td>1</td>
      <td>FOODS</td>
      <td>7</td>
      <td>TOPIX Small 2</td>
      <td>10928283.0</td>
      <td>3.365911e+10</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2021-08-04</td>
      <td>1301</td>
      <td>12700</td>
      <td>1.0</td>
      <td>2989.0</td>
      <td>2985.0</td>
      <td>2989.0</td>
      <td>2972.0</td>
      <td>KYOKUYO CO.,LTD.</td>
      <td>First Section (Domestic)</td>
      <td>Prime Market</td>
      <td>50</td>
      <td>Fishery, Agriculture and Forestry</td>
      <td>1</td>
      <td>FOODS</td>
      <td>7</td>
      <td>TOPIX Small 2</td>
      <td>10928283.0</td>
      <td>3.365911e+10</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2021-08-05</td>
      <td>1301</td>
      <td>26800</td>
      <td>1.0</td>
      <td>3030.0</td>
      <td>2989.0</td>
      <td>3035.0</td>
      <td>2981.0</td>
      <td>KYOKUYO CO.,LTD.</td>
      <td>First Section (Domestic)</td>
      <td>Prime Market</td>
      <td>50</td>
      <td>Fishery, Agriculture and Forestry</td>
      <td>1</td>
      <td>FOODS</td>
      <td>7</td>
      <td>TOPIX Small 2</td>
      <td>10928283.0</td>
      <td>3.365911e+10</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2021-08-06</td>
      <td>1301</td>
      <td>10900</td>
      <td>1.0</td>
      <td>2996.0</td>
      <td>3035.0</td>
      <td>3040.0</td>
      <td>2996.0</td>
      <td>KYOKUYO CO.,LTD.</td>
      <td>First Section (Domestic)</td>
      <td>Prime Market</td>
      <td>50</td>
      <td>Fishery, Agriculture and Forestry</td>
      <td>1</td>
      <td>FOODS</td>
      <td>7</td>
      <td>TOPIX Small 2</td>
      <td>10928283.0</td>
      <td>3.365911e+10</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Volume</th>
      <th>AdjustedClose</th>
      <th>AdjustedHigh</th>
      <th>AdjustedOpen</th>
      <th>AdjustedLow</th>
      <th>pct5</th>
      <th>pct10</th>
      <th>pct21</th>
      <th>pct33</th>
      <th>Volatility_5</th>
      <th>Volatility_10</th>
      <th>Volatility_21</th>
      <th>Volatility_33</th>
      <th>SMA_10</th>
      <th>SMA_20</th>
      <th>SMA_50</th>
      <th>SMA_60</th>
      <th>EMA_10</th>
      <th>EMA_20</th>
      <th>EMA_50</th>
      <th>AirTransportation</th>
      <th>Banks</th>
      <th>Chemicals</th>
      <th>Construction</th>
      <th>ElectricAppliances</th>
      <th>ElectricPowerandGas</th>
      <th>FisheryAgricultureandForestry</th>
      <th>Foods</th>
      <th>GlassandCeramicsProducts</th>
      <th>InformationCommunication</th>
      <th>Insurance</th>
      <th>IronandSteel</th>
      <th>LandTransportation</th>
      <th>Machinery</th>
      <th>MarineTransportation</th>
      <th>MetalProducts</th>
      <th>Mining</th>
      <th>NonferrousMetals</th>
      <th>OilandCoalProducts</th>
      <th>OtherFinancingBusiness</th>
      <th>OtherProducts</th>
      <th>Pharmaceutical</th>
      <th>PrecisionInstruments</th>
      <th>PulpandPaper</th>
      <th>RealEstate</th>
      <th>RetailTrade</th>
      <th>RubberProducts</th>
      <th>SecuritiesandCommoditiesFutures</th>
      <th>Services</th>
      <th>TextilesandApparels</th>
      <th>TransportationEquipment</th>
      <th>WarehousingandHarborTransportationService</th>
      <th>WholesaleTrade</th>
      <th>AUTOMOBILESTRANSPORTATIONEQUIPMENT</th>
      <th>BANKS</th>
      <th>COMMERCIALWHOLESALETRADE</th>
      <th>CONSTRUCTIONMATERIALS</th>
      <th>ELECTRICAPPLIANCESPRECISIONINSTRUMENTS</th>
      <th>ELECTRICPOWERGAS</th>
      <th>ENERGYRESOURCES</th>
      <th>FINANCIALSEXBANKS</th>
      <th>FOODS</th>
      <th>ITSERVICESOTHERS</th>
      <th>MACHINERY</th>
      <th>PHARMACEUTICAL</th>
      <th>RAWMATERIALSCHEMICALS</th>
      <th>REALESTATE</th>
      <th>RETAILTRADE</th>
      <th>STEELNONFERROUSMETALS</th>
      <th>TRANSPORTATIONLOGISTICS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>85</th>
      <td>8900</td>
      <td>2971.0</td>
      <td>2982.0</td>
      <td>2982.0</td>
      <td>2965.0</td>
      <td>-0.006732</td>
      <td>0.023225</td>
      <td>0.093908</td>
      <td>0.021542</td>
      <td>0.012787</td>
      <td>0.010636</td>
      <td>0.014061</td>
      <td>0.015689</td>
      <td>2969.8</td>
      <td>3017.90</td>
      <td>3027.10</td>
      <td>3037.666667</td>
      <td>2971.0</td>
      <td>2971.0</td>
      <td>2971.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>171</th>
      <td>1360800</td>
      <td>589.0</td>
      <td>599.0</td>
      <td>592.0</td>
      <td>588.0</td>
      <td>-0.037351</td>
      <td>0.003396</td>
      <td>0.078098</td>
      <td>0.112054</td>
      <td>0.008705</td>
      <td>0.015544</td>
      <td>0.019127</td>
      <td>0.016932</td>
      <td>583.4</td>
      <td>598.50</td>
      <td>626.76</td>
      <td>628.383333</td>
      <td>589.0</td>
      <td>589.0</td>
      <td>589.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>257</th>
      <td>125900</td>
      <td>2377.0</td>
      <td>2388.0</td>
      <td>2368.0</td>
      <td>2360.0</td>
      <td>-0.045435</td>
      <td>-0.005469</td>
      <td>0.097181</td>
      <td>0.124106</td>
      <td>0.008615</td>
      <td>0.013454</td>
      <td>0.015080</td>
      <td>0.013963</td>
      <td>2331.7</td>
      <td>2383.65</td>
      <td>2529.92</td>
      <td>2551.966667</td>
      <td>2377.0</td>
      <td>2377.0</td>
      <td>2377.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>343</th>
      <td>81100</td>
      <td>1224.0</td>
      <td>1239.0</td>
      <td>1230.0</td>
      <td>1224.0</td>
      <td>0.019608</td>
      <td>0.090686</td>
      <td>0.176471</td>
      <td>0.191993</td>
      <td>0.018463</td>
      <td>0.014261</td>
      <td>0.011966</td>
      <td>0.011624</td>
      <td>1254.9</td>
      <td>1315.00</td>
      <td>1415.92</td>
      <td>1451.816667</td>
      <td>1224.0</td>
      <td>1224.0</td>
      <td>1224.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>429</th>
      <td>6200</td>
      <td>1351.0</td>
      <td>1372.0</td>
      <td>1339.0</td>
      <td>1339.0</td>
      <td>0.022206</td>
      <td>0.051073</td>
      <td>0.122132</td>
      <td>0.081421</td>
      <td>0.016534</td>
      <td>0.012212</td>
      <td>0.012299</td>
      <td>0.012915</td>
      <td>1374.1</td>
      <td>1407.75</td>
      <td>1457.22</td>
      <td>1473.483333</td>
      <td>1351.0</td>
      <td>1351.0</td>
      <td>1351.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>171655</th>
      <td>65300</td>
      <td>517.0</td>
      <td>531.0</td>
      <td>528.0</td>
      <td>516.0</td>
      <td>0.021277</td>
      <td>0.090909</td>
      <td>0.197292</td>
      <td>0.098646</td>
      <td>0.027837</td>
      <td>0.021519</td>
      <td>0.022232</td>
      <td>0.024255</td>
      <td>533.1</td>
      <td>558.60</td>
      <td>563.02</td>
      <td>563.966667</td>
      <td>517.0</td>
      <td>517.0</td>
      <td>517.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>171741</th>
      <td>29100</td>
      <td>785.0</td>
      <td>800.0</td>
      <td>796.0</td>
      <td>785.0</td>
      <td>-0.015287</td>
      <td>0.008917</td>
      <td>0.047134</td>
      <td>0.121019</td>
      <td>0.012773</td>
      <td>0.016062</td>
      <td>0.015273</td>
      <td>0.016153</td>
      <td>789.5</td>
      <td>800.90</td>
      <td>846.24</td>
      <td>857.733333</td>
      <td>785.0</td>
      <td>785.0</td>
      <td>785.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>171827</th>
      <td>6200</td>
      <td>1627.0</td>
      <td>1653.0</td>
      <td>1645.0</td>
      <td>1627.0</td>
      <td>0.032575</td>
      <td>0.060234</td>
      <td>0.056546</td>
      <td>0.044868</td>
      <td>0.005328</td>
      <td>0.007613</td>
      <td>0.006306</td>
      <td>0.006767</td>
      <td>1681.8</td>
      <td>1700.90</td>
      <td>1712.56</td>
      <td>1720.183333</td>
      <td>1627.0</td>
      <td>1627.0</td>
      <td>1627.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>171913</th>
      <td>7800</td>
      <td>2418.0</td>
      <td>2433.0</td>
      <td>2394.0</td>
      <td>2393.0</td>
      <td>-0.024814</td>
      <td>-0.012821</td>
      <td>0.014888</td>
      <td>-0.002481</td>
      <td>0.010325</td>
      <td>0.009556</td>
      <td>0.007744</td>
      <td>0.007076</td>
      <td>2379.1</td>
      <td>2395.90</td>
      <td>2417.56</td>
      <td>2439.616667</td>
      <td>2418.0</td>
      <td>2418.0</td>
      <td>2418.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>171999</th>
      <td>284000</td>
      <td>706.0</td>
      <td>717.0</td>
      <td>702.0</td>
      <td>695.0</td>
      <td>-0.053824</td>
      <td>0.014164</td>
      <td>0.075071</td>
      <td>0.181303</td>
      <td>0.012321</td>
      <td>0.018099</td>
      <td>0.015071</td>
      <td>0.014817</td>
      <td>691.5</td>
      <td>718.10</td>
      <td>778.60</td>
      <td>793.733333</td>
      <td>706.0</td>
      <td>706.0</td>
      <td>706.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>2000 rows × 70 columns</p>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>SecuritiesCode</th>
      <th>Rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2021-12-06</td>
      <td>1301</td>
      <td>1138</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2021-12-06</td>
      <td>1332</td>
      <td>358</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2021-12-06</td>
      <td>1333</td>
      <td>255</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2021-12-06</td>
      <td>1375</td>
      <td>1565</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2021-12-06</td>
      <td>1376</td>
      <td>1607</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>SecuritiesCode</th>
      <th>Volume</th>
      <th>CumulativeAdjustmentFactor</th>
      <th>AdjustedClose</th>
      <th>AdjustedOpen</th>
      <th>AdjustedHigh</th>
      <th>AdjustedLow</th>
      <th>Name</th>
      <th>Section/Products</th>
      <th>NewMarketSegment</th>
      <th>33SectorCode</th>
      <th>33SectorName</th>
      <th>17SectorCode</th>
      <th>17SectorName</th>
      <th>NewIndexSeriesSizeCode</th>
      <th>NewIndexSeriesSize</th>
      <th>IssuedShares</th>
      <th>MarketCapitalization</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2021-08-02</td>
      <td>1301</td>
      <td>20600</td>
      <td>1.0</td>
      <td>3010.0</td>
      <td>2985.0</td>
      <td>3010.0</td>
      <td>2985.0</td>
      <td>KYOKUYO CO.,LTD.</td>
      <td>First Section (Domestic)</td>
      <td>Prime Market</td>
      <td>50</td>
      <td>Fishery, Agriculture and Forestry</td>
      <td>1</td>
      <td>FOODS</td>
      <td>7</td>
      <td>TOPIX Small 2</td>
      <td>10928283.0</td>
      <td>3.365911e+10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2021-08-03</td>
      <td>1301</td>
      <td>9100</td>
      <td>1.0</td>
      <td>3000.0</td>
      <td>3015.0</td>
      <td>3020.0</td>
      <td>3000.0</td>
      <td>KYOKUYO CO.,LTD.</td>
      <td>First Section (Domestic)</td>
      <td>Prime Market</td>
      <td>50</td>
      <td>Fishery, Agriculture and Forestry</td>
      <td>1</td>
      <td>FOODS</td>
      <td>7</td>
      <td>TOPIX Small 2</td>
      <td>10928283.0</td>
      <td>3.365911e+10</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2021-08-04</td>
      <td>1301</td>
      <td>12700</td>
      <td>1.0</td>
      <td>2989.0</td>
      <td>2985.0</td>
      <td>2989.0</td>
      <td>2972.0</td>
      <td>KYOKUYO CO.,LTD.</td>
      <td>First Section (Domestic)</td>
      <td>Prime Market</td>
      <td>50</td>
      <td>Fishery, Agriculture and Forestry</td>
      <td>1</td>
      <td>FOODS</td>
      <td>7</td>
      <td>TOPIX Small 2</td>
      <td>10928283.0</td>
      <td>3.365911e+10</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2021-08-05</td>
      <td>1301</td>
      <td>26800</td>
      <td>1.0</td>
      <td>3030.0</td>
      <td>2989.0</td>
      <td>3035.0</td>
      <td>2981.0</td>
      <td>KYOKUYO CO.,LTD.</td>
      <td>First Section (Domestic)</td>
      <td>Prime Market</td>
      <td>50</td>
      <td>Fishery, Agriculture and Forestry</td>
      <td>1</td>
      <td>FOODS</td>
      <td>7</td>
      <td>TOPIX Small 2</td>
      <td>10928283.0</td>
      <td>3.365911e+10</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2021-08-06</td>
      <td>1301</td>
      <td>10900</td>
      <td>1.0</td>
      <td>2996.0</td>
      <td>3035.0</td>
      <td>3040.0</td>
      <td>2996.0</td>
      <td>KYOKUYO CO.,LTD.</td>
      <td>First Section (Domestic)</td>
      <td>Prime Market</td>
      <td>50</td>
      <td>Fishery, Agriculture and Forestry</td>
      <td>1</td>
      <td>FOODS</td>
      <td>7</td>
      <td>TOPIX Small 2</td>
      <td>10928283.0</td>
      <td>3.365911e+10</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Volume</th>
      <th>AdjustedClose</th>
      <th>AdjustedHigh</th>
      <th>AdjustedOpen</th>
      <th>AdjustedLow</th>
      <th>pct5</th>
      <th>pct10</th>
      <th>pct21</th>
      <th>pct33</th>
      <th>Volatility_5</th>
      <th>Volatility_10</th>
      <th>Volatility_21</th>
      <th>Volatility_33</th>
      <th>SMA_10</th>
      <th>SMA_20</th>
      <th>SMA_50</th>
      <th>SMA_60</th>
      <th>EMA_10</th>
      <th>EMA_20</th>
      <th>EMA_50</th>
      <th>AirTransportation</th>
      <th>Banks</th>
      <th>Chemicals</th>
      <th>Construction</th>
      <th>ElectricAppliances</th>
      <th>ElectricPowerandGas</th>
      <th>FisheryAgricultureandForestry</th>
      <th>Foods</th>
      <th>GlassandCeramicsProducts</th>
      <th>InformationCommunication</th>
      <th>Insurance</th>
      <th>IronandSteel</th>
      <th>LandTransportation</th>
      <th>Machinery</th>
      <th>MarineTransportation</th>
      <th>MetalProducts</th>
      <th>Mining</th>
      <th>NonferrousMetals</th>
      <th>OilandCoalProducts</th>
      <th>OtherFinancingBusiness</th>
      <th>OtherProducts</th>
      <th>Pharmaceutical</th>
      <th>PrecisionInstruments</th>
      <th>PulpandPaper</th>
      <th>RealEstate</th>
      <th>RetailTrade</th>
      <th>RubberProducts</th>
      <th>SecuritiesandCommoditiesFutures</th>
      <th>Services</th>
      <th>TextilesandApparels</th>
      <th>TransportationEquipment</th>
      <th>WarehousingandHarborTransportationService</th>
      <th>WholesaleTrade</th>
      <th>AUTOMOBILESTRANSPORTATIONEQUIPMENT</th>
      <th>BANKS</th>
      <th>COMMERCIALWHOLESALETRADE</th>
      <th>CONSTRUCTIONMATERIALS</th>
      <th>ELECTRICAPPLIANCESPRECISIONINSTRUMENTS</th>
      <th>ELECTRICPOWERGAS</th>
      <th>ENERGYRESOURCES</th>
      <th>FINANCIALSEXBANKS</th>
      <th>FOODS</th>
      <th>ITSERVICESOTHERS</th>
      <th>MACHINERY</th>
      <th>PHARMACEUTICAL</th>
      <th>RAWMATERIALSCHEMICALS</th>
      <th>REALESTATE</th>
      <th>RETAILTRADE</th>
      <th>STEELNONFERROUSMETALS</th>
      <th>TRANSPORTATIONLOGISTICS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>86</th>
      <td>19100</td>
      <td>3065.0</td>
      <td>3065.0</td>
      <td>2998.0</td>
      <td>2990.0</td>
      <td>-0.053834</td>
      <td>-0.006525</td>
      <td>0.014682</td>
      <td>-0.004894</td>
      <td>0.013389</td>
      <td>0.015034</td>
      <td>0.012960</td>
      <td>0.016617</td>
      <td>2971.8</td>
      <td>3017.90</td>
      <td>3027.10</td>
      <td>3037.333333</td>
      <td>3065.0</td>
      <td>3065.0</td>
      <td>3065.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>173</th>
      <td>6449200</td>
      <td>556.0</td>
      <td>569.0</td>
      <td>569.0</td>
      <td>535.0</td>
      <td>0.030576</td>
      <td>0.080935</td>
      <td>0.188849</td>
      <td>0.178058</td>
      <td>0.030104</td>
      <td>0.022640</td>
      <td>0.019843</td>
      <td>0.019386</td>
      <td>578.9</td>
      <td>594.85</td>
      <td>624.70</td>
      <td>627.250000</td>
      <td>556.0</td>
      <td>556.0</td>
      <td>556.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>260</th>
      <td>127300</td>
      <td>2409.0</td>
      <td>2417.0</td>
      <td>2382.0</td>
      <td>2371.0</td>
      <td>-0.054795</td>
      <td>-0.013699</td>
      <td>0.080531</td>
      <td>0.101287</td>
      <td>0.008071</td>
      <td>0.014009</td>
      <td>0.015568</td>
      <td>0.014251</td>
      <td>2335.0</td>
      <td>2374.70</td>
      <td>2524.42</td>
      <td>2548.933333</td>
      <td>2409.0</td>
      <td>2409.0</td>
      <td>2409.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>347</th>
      <td>128600</td>
      <td>1264.0</td>
      <td>1266.0</td>
      <td>1227.0</td>
      <td>1227.0</td>
      <td>-0.037184</td>
      <td>0.043513</td>
      <td>0.121044</td>
      <td>0.170095</td>
      <td>0.019705</td>
      <td>0.019123</td>
      <td>0.014619</td>
      <td>0.012941</td>
      <td>1249.4</td>
      <td>1307.55</td>
      <td>1409.16</td>
      <td>1445.466667</td>
      <td>1264.0</td>
      <td>1264.0</td>
      <td>1264.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>434</th>
      <td>5800</td>
      <td>1395.0</td>
      <td>1395.0</td>
      <td>1374.0</td>
      <td>1366.0</td>
      <td>-0.033692</td>
      <td>0.013620</td>
      <td>0.075269</td>
      <td>0.056631</td>
      <td>0.018690</td>
      <td>0.016936</td>
      <td>0.014699</td>
      <td>0.014120</td>
      <td>1372.2</td>
      <td>1402.80</td>
      <td>1453.52</td>
      <td>1471.550000</td>
      <td>1395.0</td>
      <td>1395.0</td>
      <td>1395.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>173651</th>
      <td>57800</td>
      <td>535.0</td>
      <td>535.0</td>
      <td>526.0</td>
      <td>524.0</td>
      <td>-0.028037</td>
      <td>0.035514</td>
      <td>0.185047</td>
      <td>0.056075</td>
      <td>0.031483</td>
      <td>0.025065</td>
      <td>0.023081</td>
      <td>0.025092</td>
      <td>531.2</td>
      <td>555.35</td>
      <td>562.06</td>
      <td>563.333333</td>
      <td>535.0</td>
      <td>535.0</td>
      <td>535.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>173738</th>
      <td>48500</td>
      <td>805.0</td>
      <td>806.0</td>
      <td>795.0</td>
      <td>792.0</td>
      <td>-0.045963</td>
      <td>0.002484</td>
      <td>0.045963</td>
      <td>0.096894</td>
      <td>0.014561</td>
      <td>0.017030</td>
      <td>0.015371</td>
      <td>0.016870</td>
      <td>789.3</td>
      <td>799.35</td>
      <td>844.10</td>
      <td>855.783333</td>
      <td>805.0</td>
      <td>805.0</td>
      <td>805.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>173825</th>
      <td>6600</td>
      <td>1620.0</td>
      <td>1640.0</td>
      <td>1640.0</td>
      <td>1620.0</td>
      <td>0.037037</td>
      <td>0.077778</td>
      <td>0.060494</td>
      <td>0.057407</td>
      <td>0.004276</td>
      <td>0.004407</td>
      <td>0.006298</td>
      <td>0.006590</td>
      <td>1669.2</td>
      <td>1695.90</td>
      <td>1709.86</td>
      <td>1718.050000</td>
      <td>1620.0</td>
      <td>1620.0</td>
      <td>1620.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>173912</th>
      <td>5200</td>
      <td>2440.0</td>
      <td>2440.0</td>
      <td>2437.0</td>
      <td>2423.0</td>
      <td>-0.045902</td>
      <td>-0.011475</td>
      <td>0.002459</td>
      <td>-0.007787</td>
      <td>0.002703</td>
      <td>0.009420</td>
      <td>0.008003</td>
      <td>0.007221</td>
      <td>2381.9</td>
      <td>2395.90</td>
      <td>2415.32</td>
      <td>2438.183333</td>
      <td>2440.0</td>
      <td>2440.0</td>
      <td>2440.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>173999</th>
      <td>204500</td>
      <td>719.0</td>
      <td>719.0</td>
      <td>711.0</td>
      <td>706.0</td>
      <td>-0.072323</td>
      <td>-0.009736</td>
      <td>0.070932</td>
      <td>0.152990</td>
      <td>0.010283</td>
      <td>0.019032</td>
      <td>0.015320</td>
      <td>0.015360</td>
      <td>692.2</td>
      <td>715.70</td>
      <td>775.90</td>
      <td>791.233333</td>
      <td>719.0</td>
      <td>719.0</td>
      <td>719.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>2000 rows × 70 columns</p>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>SecuritiesCode</th>
      <th>Rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2021-12-07</td>
      <td>1301</td>
      <td>696</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2021-12-07</td>
      <td>1332</td>
      <td>1913</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2021-12-07</td>
      <td>1333</td>
      <td>748</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2021-12-07</td>
      <td>1375</td>
      <td>1272</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2021-12-07</td>
      <td>1376</td>
      <td>1278</td>
    </tr>
  </tbody>
</table>
</div>


    CPU times: user 2min 51s, sys: 517 ms, total: 2min 51s
    Wall time: 2min 51s
    
