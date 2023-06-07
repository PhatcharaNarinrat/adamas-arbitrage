import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import pickle
from datetime import datetime
import pytz
import datetime as dt
import os
import inspect
import matplotlib.pyplot as plt
import requests
import time
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder


colName = pd.Series(
    ["symbol", "lastPrice", "openPrice", "highPrice", "lowPrice", "bidPrice", "volume"]
)
Top10Coin = ["BTC", "ETH", "LTC", "DOGE", "BNB", "XRP", "ADA", "XLM", "SOL", "DOT"]


def get_binance_dataframe():
    # BInance
    urlBinace = "https://data.binance.com/api/v3/ticker/24hr"
    clientBinance = requests.get(urlBinace)
    dataBinance = clientBinance.json()
    selectedBinance = pd.DataFrame(dataBinance)
    selectedBinance = selectedBinance[
        selectedBinance["symbol"].str.endswith("USDT", na=False)
    ]
    colName = pd.Series(
        [
            "symbol",
            "lastPrice",
            "openPrice",
            "highPrice",
            "lowPrice",
            "bidPrice",
            "volume",
        ]
    )
    df = selectedBinance[(colName)].copy()
    timeStamp = datetime.now(pytz.timezone("Asia/Bangkok"))
    unix_time = int(time.mktime(timeStamp.timetuple()))
    df.loc[:, "time"] = unix_time
    df = df.astype(
        {
            "lastPrice": "float",
            "openPrice": "float",
            "highPrice": "float",
            "lowPrice": "float",
            "bidPrice": "float",
            "volume": "float64",
        }
    )
    df = df[df["symbol"].str.endswith("USDT", na=False)].loc[
        df["lastPrice"] > 0
    ]  # เลือก currency USDT/USD เป็นมาตรฐาน
    df["symbol"] = df["symbol"].str.replace(
        "USDT", ""
    )  # ทำการตัดคำเพื่อให้เห็นเฉพาะชื่อเหรียญเมื่อแปลงเป็น USDT แล้ว
    Top10Coin = ["BTC", "ETH", "LTC", "DOGE", "BNB", "XRP", "ADA", "XLM", "SOL", "DOT"]
    df = df[np.isin(df["symbol"], Top10Coin)]  # เลือก 10 เหรียญจาก list ด้านบน
    df = df.round(decimals=2)
    pd.options.display.float_format = "{:.4f}".format
    df.rename(
        columns={
            "openPrice": "open",
            "highPrice": "high",
            "lowPrice": "low",
            "lastPrice": "close",
        },
        inplace=True,
    )
    df = df[["time", "volume", "high", "low", "open", "symbol", "close"]]
    le = LabelEncoder()
    le.fit(df["symbol"])
    df.loc[:, "symbol"] = le.transform(df["symbol"])
    pd.options.display.float_format = "{:.4f}".format
    # print("binance : ", df)
    return df


def get_bitget_dataframe():
    urlBitget = "https://api.bitget.com/api/mix/v1/market/tickers?productType=umcbl"
    clientBitget = requests.get(urlBitget)
    data = clientBitget.json()
    selectedBitget = pd.DataFrame(data["data"])
    colNameBitget = pd.Series(
        ["symbol", "last", "openUtc", "high24h", "low24h", "bestBid", "usdtVolume"]
    )
    newDataBitget = selectedBitget[(colNameBitget)].copy()
    timeStamp = datetime.now(pytz.timezone("Asia/Bangkok"))
    unix_time = int(time.mktime(timeStamp.timetuple()))
    newDataBitget.loc[:, "time"] = unix_time
    newDataBitget = newDataBitget.astype(
        {
            "last": "float",
            "openUtc": "float",
            "high24h": "float",
            "low24h": "float",
            "bestBid": "float",
            "usdtVolume": "float64",
        }
    )
    newDataBitget["symbol"] = newDataBitget["symbol"].str.replace("USDT_UMCBL", "")
    newDataBitget = newDataBitget[np.isin(newDataBitget["symbol"], Top10Coin)]
    newDataBitget = newDataBitget.round(decimals=2)
    newDataBitget.rename(
        columns={
            "openUtc": "open",
            "high24h": "high",
            "low24h": "low",
            "usdtVolume": "volume",
            "last": "close",
        },
        inplace=True,
    )
    df = newDataBitget[["time", "volume", "high", "low", "open", "symbol", "close"]]
    le = LabelEncoder()
    le.fit(df["symbol"])
    df.loc[:, "symbol"] = le.transform(df["symbol"])
    pd.options.display.float_format = "{:.4f}".format
    # print("bitget : ", df)
    return df


def get_bingx_dataframe():
    urlBingX = "https://api-swap-rest.bingbon.pro/api/v1/market/getTicker"
    clientBingX = requests.get(urlBingX)
    dataBingX = clientBingX.json()
    dataBingX = dataBingX["data"]
    newDataBingX = pd.DataFrame(dataBingX["tickers"])
    colNameBingX = pd.Series(
        [
            "symbol",
            "lastPrice",
            "openPrice",
            "highPrice",
            "lowPrice",
            "bidPrice",
            "volume",
        ]
    )
    newDataBingX = newDataBingX[(colNameBingX)].copy()
    newDataBingX.columns = colName
    timeStamp = datetime.now(pytz.timezone("Asia/Bangkok"))
    unix_time = int(time.mktime(timeStamp.timetuple()))
    newDataBingX.loc[:, "time"] = unix_time
    newDataBingX = newDataBingX.astype(
        {
            "lastPrice": "float",
            "openPrice": "float",
            "highPrice": "float",
            "lowPrice": "float",
            "bidPrice": "float",
            "volume": "float64",
        }
    )
    newDataBingX["symbol"] = newDataBingX["symbol"].str.replace("-USDT", "")
    newDataBingX = newDataBingX[np.isin(newDataBingX["symbol"], Top10Coin)]
    newDataBingX = newDataBingX.round(decimals=2)
    newDataBingX.rename(
        columns={
            "openPrice": "open",
            "highPrice": "high",
            "lowPrice": "low",
            "volume": "volume",
            "lastPrice": "close",
        },
        inplace=True,
    )
    df = newDataBingX[["time", "volume", "high", "low", "open", "symbol", "close"]]
    le = LabelEncoder()
    le.fit(df["symbol"])
    df.loc[:, "symbol"] = le.transform(df["symbol"])
    pd.options.display.float_format = "{:.4f}".format
    # print("bingx : ", df)
    return df


def get_tapbit_dataframe():
    apiKey = "QIA4JAMGVULFKIUNR5XH46OPVS6DWA6A"
    urlTapbit = "https://openapi.tapbit.com/spot/api/spot/instruments/ticker_list"
    clientTapbit = requests.get(urlTapbit)
    dataTapbit = clientTapbit.json()
    dataTapbit = dataTapbit["data"]
    newDataTapbit = pd.DataFrame(dataTapbit)
    colNameTapbit = pd.Series(
        [
            "trade_pair_name",
            "last_price",
            "lowest_ask",
            "highest_price_24h",
            "lowest_price_24h",
            "highest_bid",
            "volume24h",
        ]
    )
    newDataTapbit = newDataTapbit[(colNameTapbit)].copy()
    newDataTapbit.columns = colName
    timeStamp = datetime.now(pytz.timezone("Asia/Bangkok"))
    unix_time = int(time.mktime(timeStamp.timetuple()))
    # newDataTapbit.loc[:, "time"] = unix_time
    newDataTapbit["time"] = unix_time
    newDataTapbit = newDataTapbit.astype(
        {
            "lastPrice": "float",
            "openPrice": "float",
            "highPrice": "float",
            "lowPrice": "float",
            "bidPrice": "float",
            "volume": "float64",
        }
    )
    newDataTapbit["symbol"] = newDataTapbit["symbol"].str.replace("/USDT", "")
    newDataTapbit = newDataTapbit[np.isin(newDataTapbit["symbol"], Top10Coin)]
    newDataTapbit = newDataTapbit.round(decimals=2)
    newDataTapbit.rename(
        columns={
            "openPrice": "open",
            "highPrice": "high",
            "lowPrice": "low",
            "volume": "volume",
            "lastPrice": "close",
        },
        inplace=True,
    )
    df = newDataTapbit[["time", "volume", "high", "low", "open", "symbol", "close"]]
    le = LabelEncoder()
    le.fit(df["symbol"])
    df.loc[:, "symbol"] = le.transform(df["symbol"])
    pd.options.display.float_format = "{:.4f}".format
    # print("tapbit : ", df)
    return df


# get_binance_dataframe()
# get_bitget_dataframe()
# get_bingx_dataframe()
# get_tapbit_dataframe()
# get_binance_dataframe()
