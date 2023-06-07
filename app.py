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
    symbol = [
        ["BNB", 1],
        ["BTC", 2],
        ["DOGE", 3],
        ["DOT", 4],
        ["ETH", 5],
        ["LTC", 6],
        ["SOL", 7],
        ["XLM", 8],
        ["XRP", 9],
        ["ADA", 0],
    ]
    dfSymbol = pd.DataFrame(symbol, columns=["symbol_name", "symbol"])

    for exchange in exchanges:
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
        pd.options.display.float_format = "{:.4f}".format
        actual = exchange_df[["symbol", "close"]]
        print("actual : ", actual)
        # Round the 'close' column to 4 decimal places
        y_pred = modelGB.predict(X)
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

        merged_predictions = merged_predictions.merge(
            dfSymbol, left_on="symbol", right_on="symbol", how="inner"
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
        result_df = result_df.loc[:, ~result_df.columns.duplicated()]
        print("result_df : ", result_df)
    result_df[
        ["actual_binance", "actual_bingx", "actual_bitget", "actual_tapbit"]
    ] = result_df[
        ["actual_binance", "actual_bingx", "actual_bitget", "actual_tapbit"]
    ].apply(
        lambda x: round(x.astype(float), 4)
    )
    formatted_data = result_df.to_dict(orient="list")
    formatted_data = {
        column: [
            f"{value:.4f}" if isinstance(value, float) else value for value in values
        ]
        for column, values in formatted_data.items()
    }

    return jsonify(formatted_data)


if __name__ == "__main__":
    app.run()
