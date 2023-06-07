import requests
import pandas as pd
import joblib


apiList = [
    "https://min-api.cryptocompare.com/data/symbol/histohour?fsym={cryp}&tsym=USD&limit={limit}",
    "https://min-api.cryptocompare.com/data/exchange/histohour?tsym={cryp}&limit={limit}",
    "https://min-api.cryptocompare.com/data/v2/histohour?fsym={cryp}&tsym=USD&limit={limit}",
]

api_key = "66560c6362a39314d9e75ab469ba080547bcb992f7d962f4a8e8b358872fba37"
ExchangeList = ["Binance", "Bitkub", "coinbase", "HuobiGlobal", "UPbit", "KuCoin"]
cryptoList = ["BTC", "ETH", "LTC", "DOGE", "BNB", "XRP", "ADA", "XLM", "SOL", "DOT"]
limit = 2000  # 24*90
vlDf = pd.DataFrame()
symDf = pd.DataFrame()
exHisDf = pd.DataFrame()
symOHLCVDf = pd.DataFrame()
index = 1
for a in apiList:
    match index:
        case 1:
            for cryp in cryptoList:
                listofAPI = a.format(cryp=cryp, limit=limit)
                # Add the API key to the request headers
                headers = {"Authorization": f"Apikey {api_key}"}
                # Send the GET request
                response = requests.get(listofAPI, headers=headers)

                df# Check the response status
                if response.status_code == 200:
                    data = response.json()
                    data = data["Data"]
                    # Process the data as needed
                    dfData = pd.DataFrame(data)
                    dfData["symbol"] = cryp
                    # print(dfData)
                    symDf = symDf._append(dfData, ignore_index=True)

                else:
                    print(f"Request failed with status code: {response.status_code}")
        case 2:
            for cryp in cryptoList:
                listofAPI = a.format(cryp=cryp, limit=limit)
                headers = {"Authorization": f"Apikey {api_key}"}
                # Send the GET request
                response = requests.get(listofAPI, headers=headers)

                # Check the response status
                if response.status_code == 200:
                    data = response.json()
                    data = data["Data"]
                    # Process the data as needed
                    dfData = pd.DataFrame(data)
                    dfData["symbol"] = cryp
                    # print(dfData)
                    exHisDf = exHisDf._append(dfData, ignore_index=True)

                else:
                    print(f"Request failed with status code: {response.status_code}")
        case 3:
            for cryp in cryptoList:
                listofAPI = a.format(cryp=cryp, limit=limit)
                headers = {"Authorization": f"Apikey {api_key}"}
                # Send the GET request
                response = requests.get(listofAPI, headers=headers)

                # Check the response status
                if response.status_code == 200:
                    data = response.json()
                    data_array = data["Data"]["Data"]
                    # Process the data as needed
                    dfData = pd.DataFrame(data_array)
                    dfData["symbol"] = cryp
                    # print(dfData)
                    symOHLCVDf = symOHLCVDf._append(dfData, ignore_index=True)

                else:
                    print(f"Request failed with status code: {response.status_code}")
    index = index + 1

pd.options.display.float_format = "{:.4f}".format
# Merge Data From 3 API
df = symDf.merge(exHisDf, on=["time", "symbol"], how="inner")
df = df.merge(symOHLCVDf, on=["time", "symbol"], how="inner")

# Check for missing values in the entire DataFrame
missing_count = df.isna().sum()
any_missing = df.isna().values.any()
total_missing = df.isna().sum().sum()
print(missing_count)
if any_missing:
    print("There are missing values in the DataFrame.")
    print("Total number of missing values:", total_missing)
else:
    print("There are no missing values in the DataFrame.")


# Label Data
le = preprocessing.LabelEncoder()
le.fit(df.symbol)
df["symbol"] = le.transform(df.symbol)

########## GradientBoostingRegressor Model ###########
# Step 1: Data Preparation
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
)
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from math import sqrt

X = df[["time", "volume", "high", "low", "open", "symbol"]]
y = df["close"]

# Step 2: Create and Train the Model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=50
)

# Step 3: Call Model
modelGB = GradientBoostingRegressor(n_estimators=100, max_depth=10, random_state=50)
modelGB.fit(X_train, y_train)

# Step 4: Make Predictions
y_pred = modelGB.predict(X_test)

# Step 5: Evaluate the Model
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
mse = mean_squared_error(y_test, y_pred)
rmse = sqrt(mse)
nrmse = rmse / (np.max(y) - np.min(y))
wape = np.sum(np.abs(y_test - y_pred)) / np.sum(np.abs(y_test)) * 100

print("R-Squared:", r2)
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Mean Absolute Percentage Error:", mape)
print("Root Mean Squared Error:", rmse)
print("Normalized Root Mean Squared Error:", nrmse)
print("Weighted Absolute Percentage Error:", wape)
joblib.dump(modelGB, "gradient_boosting_model.pkl")
