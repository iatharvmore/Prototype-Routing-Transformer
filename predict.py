
import os
import torch
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

from src.config import *
from src.prototype_routing_transformer import PrototypeRoutingTransformer
from src.data_loader import add_indicators  # we already have this
import warnings
warnings.filterwarnings("ignore")


def load_model():
    """Load the trained PRT model."""
    model = PrototypeRoutingTransformer(
        input_dim=INPUT_DIM,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_ENCODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        pred_len=PRED_LEN,
        n_prototypes=16
    )
    model.load_state_dict(torch.load(PRT_MODEL, map_location="cpu"))
    model.eval()
    return model


def prepare_input(ticker):
    """Fetch recent stock data, add indicators, and prepare scaled sequence."""
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=120)  # 4 months context

    df = yf.download(ticker, start=start_date, end=end_date)
    if df.empty:
        raise ValueError(f"No data found for {ticker}")

    df = add_indicators(df)
    df.dropna(inplace=True)

    # Scale using MinMaxScaler
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    # Use last SEQ_LEN days as input
    last_seq = scaled[-SEQ_LEN:]
    last_seq = np.expand_dims(last_seq, axis=0)  # shape (1, seq_len, features)

    return torch.tensor(last_seq, dtype=torch.float32), scaler, df


def predict_next_days(ticker="RELIANCE.NS"):
    """Predict next PRED_LEN days for a given ticker."""
    model = load_model()
    X, scaler, df = prepare_input(ticker)

    with torch.no_grad():
        preds = model(X).squeeze().numpy()  # shape (pred_len,)

    # Inverse scale (only 'Close' column index)
    close_index = 3  # Assuming 'Close' is 4th column
    dummy = np.zeros((len(preds), df.shape[1]))
    dummy[:, close_index] = preds
    inv = scaler.inverse_transform(dummy)[:, close_index]

    # Build prediction dataframe
    future_dates = [df.index[-1] + timedelta(days=i+1) for i in range(PRED_LEN)]
    pred_df = pd.DataFrame({
        "Ticker": ticker,
        "Date": future_dates,
        "Predicted_Close": inv
    })

    os.makedirs(RESULTS_DIR, exist_ok=True)
    pred_path = os.path.join(RESULTS_DIR, "predictions.csv")

    if os.path.exists(pred_path):
        old = pd.read_csv(pred_path)
        combined = pd.concat([old, pred_df], ignore_index=True)
        combined.to_csv(pred_path, index=False)
    else:
        pred_df.to_csv(pred_path, index=False)

    print(f"✅ Saved predictions for {ticker} to {pred_path}")
    print(pred_df)


def batch_predict():
    """Run prediction for all tickers in config."""
    for ticker in TICKERS:
        print(f"\n🔹 Predicting for {ticker}...")
        try:
            predict_next_days(ticker)
        except Exception as e:
            print(f"[ERROR] {ticker}: {e}")


if __name__ == "__main__":
    batch_predict()
