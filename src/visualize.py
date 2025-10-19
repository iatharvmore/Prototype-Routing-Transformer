# src/visualize.py

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
import json
import os
import pandas as pd

# -----------------------------
# Helper for device selection
# -----------------------------
def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# Plot Predicted vs Actual Prices for a ticker
# -----------------------------
def plot_predictions_for_ticker(model, X, y, ticker, save_dir="results/plots"):
    device = get_device()
    model.eval().to(device)
    
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        preds = model(X_tensor).cpu().numpy()
    
    plt.figure(figsize=(10,5))
    plt.plot(range(len(y)), y, label='Actual', color='blue')

    preds_to_plot = preds.squeeze()
    if len(preds_to_plot) > len(y):
        preds_to_plot = preds_to_plot[:len(y)]
    plt.plot(range(len(preds_to_plot)), preds_to_plot, label='Predicted', color='red')

    plt.title(f"{ticker} - Actual vs Predicted Prices")
    plt.xlabel("Time Step")
    plt.ylabel("Normalized Price")
    plt.legend()
    
    save_path = os.path.join(save_dir, f"predictions_{ticker.lower()}.png")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] Saved prediction plot: {save_path}")


# -----------------------------
# Plot all predictions from CSV
# -----------------------------
def plot_predictions_from_csv(csv_path, save_dir="results/plots"):
    if not os.path.exists(csv_path):
        print(f"[WARN] CSV file {csv_path} does not exist.")
        return
    
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    tickers = df['Ticker'].unique()
    
    for ticker in tickers:
        ticker_df = df[df['Ticker'] == ticker].sort_values("Date")
        y = ticker_df['Predicted_Close'].values
        # X can be None if only plotting predictions
        X_dummy = np.zeros((len(y), INPUT_DIM))  # dummy X
        plot_predictions_for_ticker(model=None, X=X_dummy, y=y, ticker=ticker, save_dir=save_dir)


# -----------------------------
# Plot Training Loss Curve
# -----------------------------
def plot_loss(log_path="results/logs/prt_model.log", save_dir="results/plots"):
    if not os.path.exists(log_path):
        print(f"[WARN] Log file {log_path} does not exist.")
        return
    
    with open(log_path, 'r') as f:
        history = json.load(f)
    
    if 'loss' not in history or len(history['loss']) == 0:
        print("[WARN] No loss data to plot.")
        return
    
    plt.figure(figsize=(8,5))
    plt.plot(history['loss'], marker='o', color='green')
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    
    save_path = os.path.join(save_dir, "loss_curve.png")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] Saved loss curve: {save_path}")


# -----------------------------
# Visualize Prototypes (t-SNE)
# -----------------------------
def plot_prototypes(model, save_dir="results/plots"):
    device = get_device()
    model.eval().to(device)
    
    if not hasattr(model, 'prototypes') or model.prototypes is None:
        print("[WARN] Model has no 'prototypes' attribute for visualization.")
        return
    
    prototypes = model.prototypes.detach().cpu().numpy()
    if len(prototypes) == 0:
        print("[WARN] No prototypes to visualize.")
        return
    
    tsne = TSNE(n_components=2, random_state=42)
    proj = tsne.fit_transform(prototypes)
    
    plt.figure(figsize=(8,6))
    plt.scatter(proj[:,0], proj[:,1], c=np.arange(len(prototypes)), cmap='viridis', s=50)
    plt.title("PRT Prototypes t-SNE Projection")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    
    save_path = os.path.join(save_dir, "prototypes_tsne.png")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] Saved prototype visualization: {save_path}")
