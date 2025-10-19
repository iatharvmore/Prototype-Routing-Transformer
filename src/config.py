# src/config.py

import os

# -----------------------------
# Data Paths
# -----------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

PROCESSED_X = os.path.join(PROCESSED_DATA_DIR, "X.npy")
PROCESSED_Y = os.path.join(PROCESSED_DATA_DIR, "y.npy")

# -----------------------------
# Results / Outputs
# -----------------------------
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(RESULTS_DIR, "models")
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

BASE_MODEL = os.path.join(MODELS_DIR, "base_transformer.pth")
PRT_MODEL = os.path.join(MODELS_DIR, "prt_model.pth")

BASE_MODEL_LOG = os.path.join(LOGS_DIR, "base_transformer.log")
PRT_MODEL_LOG = os.path.join(LOGS_DIR, "prt_model.log")
METRICS_FILE = "results/metrics.json"
# -----------------------------
# Stock / Finance Config
# -----------------------------
TICKERS = ["RELIANCE.NS", "NIFTY50.NS", "INFY.NS", "HDFCBANK.NS"]
START_DATE = "2020-01-01"
END_DATE = "2025-06-30"
SEQ_LEN = 30
PRED_LEN = 5  # number of steps to predict

# -----------------------------
# Model Hyperparameters
# -----------------------------
INPUT_DIM = 9  # OHLC + Volume + RSI + EMA20 + EMA50 + MACD
D_MODEL = 64
NHEAD = 4
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
DIM_FEEDFORWARD = 256
DROPOUT = 0.1

# -----------------------------
# Training Hyperparameters
# -----------------------------
BATCH_SIZE = 16
EPOCHS = 50
LR = 1e-3
DEVICE = "cuda"  # "cpu" if GPU not available

# -----------------------------
# Other settings
# -----------------------------
RANDOM_SEED = 42
