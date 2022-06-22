## JPX Tokyo Stock Exchange Prediction Challenge

Based on [jpx-tokyo-stock-exchange challenge](https://www.kaggle.com/competitions/jpx-tokyo-stock-exchange-prediction)

## Description
Capstone project by Gari Zabaleta for *Introducción a la Data Science y al Machine Learning 2021-2022* Postgraduate.

## Getting Started

### Enviroment

Anaconda for Python 3.8. To install [Anaconda](https://www.anaconda.com/products/distribution)

### 1. Run locally

**1.1 Download the Dataset using the kaggle API**

Install the Kaggle API

```
pip install kaggle
```

More about the Kaggle API in https://github.com/Kaggle/kaggle-api.

**1.2 Download API Token**

Go to Kaggle account settings and download the *kaggle.json* file to the following folder (Windows)
```
C:\Users\<USER>\.kaggle
```


**1.3 Download the dataset to your local machine**

```
mkdir CapstoneProject
cd CapstoneProject

kaggle competitions download -c jpx-tokyo-stock-exchange-prediction
cd jpx-tokyo-stock-exchange-prediction
```

**1.4 Clone this repository**

```
git clone https://github.com/GariZabaleta/jpx-tokyo-stock-exchange-prediction.git
```

**1.5 Install Requirements**

Open Anaconda terminal and install required packages

```
conda install -c plotly plotly
conda install -c conda-forge xgboost
conda install -c conda-forge lightgbm
conda install -c conda-forge optuna
```

**1.6 Run the jupyter notebook**

Open the Anaconda Prompt

```
cd CapstoneProject/jpx-tokyo-stock-exchange-prediction
jupyter notebook
```

**1.7 Select local file path**

Uncomment local data path in the following cells

*Local Datasets*
```
%%time
financials_df = pd.read_csv("train_files/financials.csv")
options_df = pd.read_csv("train_files/options.csv")
secondary_stock_prices_df = pd.read_csv("train_files/secondary_stock_prices.csv")
stock_prices_df = pd.read_csv("train_files/stock_prices.csv")
trades_df = pd.read_csv("train_files/trades.csv")
stocks_df = pd.read_csv("stock_list.csv")
```
and 

*Local Suplemental files*


```
%%time
financials_info = pd.read_csv("/data_specifications/stock_fin_spec.csv")
options_info = pd.read_csv("/data_specifications/options_spec.csv")
stock_prices_info = pd.read_csv("/data_specifications/stock_price_spec.csv")
trades_info = pd.read_csv("/data_specifications/trades_spec.csv")
stocks_info = pd.read_csv("/data_specifications/stock_list_spec.csv")
```

Comment Kaggle data path in the following cells

*Datasets*

```
"""
%%time 
financials_df = pd.read_csv("/kaggle/input/jpx-tokyo-stock-exchange-prediction/train_files/financials.csv")
options_df = pd.read_csv("/kaggle/input/jpx-tokyo-stock-exchange-prediction/train_files/options.csv")
secondary_stock_prices_df = pd.read_csv("/kaggle/input/jpx-tokyo-stock-exchange-prediction/train_files/secondary_stock_prices.csv")
stock_prices_df = pd.read_csv("/kaggle/input/jpx-tokyo-stock-exchange-prediction/train_files/stock_prices.csv")
trades_df = pd.read_csv("/kaggle/input/jpx-tokyo-stock-exchange-prediction/train_files/trades.csv")
stocks_df = pd.read_csv("/kaggle/input/jpx-tokyo-stock-exchange-prediction/stock_list.csv")
"""
```

*Suplemental files*

```
"""
financials_info = pd.read_csv("/kaggle/input/jpx-tokyo-stock-exchange-prediction/data_specifications/stock_fin_spec.csv")
options_info = pd.read_csv("/kaggle/input/jpx-tokyo-stock-exchange-prediction/data_specifications/options_spec.csv")
stock_prices_info = pd.read_csv("/kaggle/input/jpx-tokyo-stock-exchange-prediction/data_specifications/stock_price_spec.csv")
trades_info = pd.read_csv("/kaggle/input/jpx-tokyo-stock-exchange-prediction/data_specifications/trades_spec.csv")
stocks_info = pd.read_csv("/kaggle/input/jpx-tokyo-stock-exchange-prediction/data_specifications/stock_list_spec.csv")
"""
```

### 2. Run in Kaggle


**2.1 Clone this repository**

```
git clone https://github.com/GariZabaleta/jpx-tokyo-stock-exchange-prediction.git
```

**2.2 Go to Kaggle website**

* [Kaggle](https://www.kaggle.com/)
* Create a new Notebook
* Click Add data
* Go to Competition data and select JPX Tokyo Stock Exchange Prediction and click on *Add*
* In the Notebook go to *File* and click *Import Notebook*. 
* Upload j*px-prediction.ipynb* file from your local machine


## HELP

* To speed up the optimization process with Optuna activate when available the GPU option by uncommenting the following  parameters in the *objective* function

```
def objective(trial):
    params = {
       # 'device': 'gpu',
       # 'gpu_platform_id': 0,
       # 'gpu_device_id': 0,
        }
```

## Authors

Gari Zabaleta

Email: [garikoitz.zabaleta@gmail.com](garikoitz.zabaleta@gmail.com)

Linkedin: [https://www.linkedin.com/in/garizabaleta/](https://www.linkedin.com/in/garizabaleta/)


## Agurrak

Eta azkenik, eta gehienbat nire Gurasoeri. Zuri ere Izaro, eduki duzun pazientziarengatik. Mil esker!
