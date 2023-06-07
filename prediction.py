from flask import Flask, request, jsonify
from sklearn.ensemble import GradientBoostingRegressor
import pickle
import matplotlib
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from load_data import (
    get_binance_dataframe,
    get_bingx_dataframe,
    get_bitget_dataframe,
    get_tapbit_dataframe,
)

modelGB = joblib.load("gradient_boosting_model.pkl")


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
        actual = exchange_df[["symbol", "close"]]

        print("actual data:", actual)
        y_pred = modelGB.predict(X)
        pd.options.display.float_format = "{:.4f}".format
        labels = exchange_df["symbol"].values.tolist()

        organized_predictions = {label: price for label, price in zip(labels, y_pred)}
        predictions[exchange] = organized_predictions

        merged_predictions = pd.DataFrame.from_dict(
            predictions[exchange], orient="index", columns=[f"predicted_{exchange}"]
        )

        # Merge actual data with predictions based on symbol
        merged_predictions = merged_predictions.merge(
            actual, left_index=True, right_on="symbol", how="inner"
        )
        merged_predictions.rename(columns={"close": f"actual_{exchange}"}, inplace=True)
        merged_predictions.set_index("symbol", inplace=True)

        # Calculate difference between predicted and actual values
        merged_predictions[f"diff_{exchange}"] = (
            merged_predictions[f"predicted_{exchange}"]
            - merged_predictions[f"actual_{exchange}"]
        )

        predictions[exchange] = merged_predictions

    result_df = pd.concat(predictions.values(), axis=1)
    print(result_df)

    return jsonify(result_df.to_dict())


predict()
