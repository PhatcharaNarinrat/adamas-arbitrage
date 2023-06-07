from flask import Flask, request, jsonify
from sklearn.ensemble import GradientBoostingRegressor
import pickle
import matplotlib
import joblib
import pandas as pd

from load_data import (
    get_binance_dataframe,
    get_bingx_dataframe,
    get_bitget_dataframe,
    get_tapbit_dataframe,
)

app = Flask(__name__)
modelGB = joblib.load("gradient_boosting_model.pkl")


@app.route("/", methods=["GET"])
def predict():
    exchanges = ["binance", "bitget", "bingx", "tapbit"]
    predictions = {}

    for exchange in exchanges:
        print(exchange)
        exchange_df = {}
        if exchange == "binance":
            exchange_df = get_binance_dataframe()
        elif exchange == "bitget":
            exchange_df = get_bitget_dataframe()
        elif exchange == "bingx":
            exchange_df = get_bingx_dataframe()
        elif exchange == "tapbit":
            exchange_df = get_tapbit_dataframe()

        X = exchange_df[["time", "volume", "high", "low", "open", "symbol"]]
        y_pred = modelGB.predict(X)
        pd.options.display.float_format = "{:.4f}".format
        labels = exchange_df["symbol"].values.tolist()

        predictions[exchange] = y_pred.tolist()
        organized_predictions = {label: price for label, price in zip(labels, y_pred)}
        predictions[exchange] = organized_predictions
    print(pd.DataFrame(predictions))
    return jsonify(predictions)


if __name__ == "__main__":
    app.run()
