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

```
pip install -r requirements.txt
```

**1.6 Run the jupyter notebook**

Open the Anaconda Prompt

```
cd CapstoneProject/jpx-tokyo-stock-exchange-prediction
jupyter notebook
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
* Select j*px-prediction.ipynb* file from your local machine


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
