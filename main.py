# # main.py
# main.py

import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from src.base_transformer import FinanceSeq2SeqTransformer
from src.prototype_routing_transformer import PrototypeRoutingTransformer
from src.train_utils import train_model, create_target_sequences
from src.config import *
from src.visualize import plot_loss, plot_prototypes, plot_predictions_for_ticker  # Import visualization
import os

def evaluate_model(model, X, y, device='cuda'):
    model.eval()
    pred_len = model.pred_len

    # Create sequences
    y_seq = create_target_sequences(y, pred_len)
    X_trimmed = X[:len(y_seq)]

    X_tensor = torch.tensor(X_trimmed, dtype=torch.float32).to(device)
    with torch.no_grad():
        out = model(X_tensor)  # (num_samples, pred_len, 1)
    
    # Flatten to compare
    mse = mean_squared_error(y_seq.flatten(), out.cpu().numpy().flatten())
    return mse


def main():
    # --- Load processed data ---
    print("[INFO] Loading processed data...")
    X = np.load(PROCESSED_X)
    y = np.load(PROCESSED_Y)
    print(f"[INFO] X shape: {X.shape}, y shape: {y.shape}")

    # --- Train Base Transformer ---
    print("\n[INFO] Initializing Base Transformer...")
    base_model = FinanceSeq2SeqTransformer(input_dim=X.shape[2], pred_len=PRED_LEN)
    print("[INFO] Training Base Transformer...")
    base_history = train_model(
        base_model, X, y,
        device=DEVICE,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        lr=LR,
        save_path=BASE_MODEL,
        log_path=BASE_MODEL_LOG
    )

    # --- Train PRT Transformer ---
    print("\n[INFO] Initializing Prototype Routing Transformer...")
    prt_model = PrototypeRoutingTransformer(input_dim=X.shape[2], pred_len=PRED_LEN)
    print("[INFO] Training PRT Transformer...")
    prt_history = train_model(
        prt_model, X, y,
        device=DEVICE,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        lr=LR,
        save_path=PRT_MODEL,
        log_path=PRT_MODEL_LOG
    )

    # --- Evaluate & Compare ---
    print("\n[INFO] Evaluating models...")
    base_mse = evaluate_model(base_model, X, y)
    prt_mse = evaluate_model(prt_model, X, y)
    print(f"\n[RESULTS] Base Transformer MSE: {base_mse:.6f}")
    print(f"[RESULTS] PRT Transformer MSE:  {prt_mse:.6f}")

    # Save metrics
    import json
    metrics = {
        "base_transformer": {"mse": float(base_mse)},
        "prt_transformer": {"mse": float(prt_mse)}
    }
    os.makedirs(os.path.dirname(METRICS_FILE), exist_ok=True)
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"[INFO] Metrics saved to {METRICS_FILE}")

    # -----------------------------
    # Visualizations
    # -----------------------------
    print("\n[INFO] Generating visualizations...")

    # 1️⃣ Plot loss curves
    plot_loss(log_path=BASE_MODEL_LOG)
    plot_loss(log_path=PRT_MODEL_LOG)

    # 2️⃣ Plot prototype t-SNE (only for PRT model)
    plot_prototypes(prt_model)

    # 3️⃣ Plot predicted vs actual for a few sequences
    # Use first 5 sequences as an example
    for i in range(min(5, X.shape[0])):
        plot_predictions_for_ticker(
            model=prt_model,
            X=X[i:i+1],
            y=y[i:i+prt_model.pred_len].flatten(),
            ticker=f"Sample_{i+1}"
        )


if __name__ == "__main__":
    main()


# import numpy as np
# import torch
# from sklearn.metrics import mean_squared_error
# from src.base_transformer import FinanceSeq2SeqTransformer
# from src.prototype_routing_transformer import PrototypeRoutingTransformer
# from src.train_utils import train_model, create_target_sequences
# from src.config import *

# def evaluate_model(model, X, y, device='cuda'):
#     model.eval()
#     pred_len = model.pred_len

#     # Create sequences
#     y_seq = create_target_sequences(y, pred_len)
#     X_trimmed = X[:len(y_seq)]

#     X_tensor = torch.tensor(X_trimmed, dtype=torch.float32).to(device)
#     with torch.no_grad():
#         out = model(X_tensor)  # (num_samples, pred_len, 1)
    
#     # Flatten to compare
#     mse = mean_squared_error(y_seq.flatten(), out.cpu().numpy().flatten())
#     return mse


# def main():
#     # --- Load processed data ---
#     print("[INFO] Loading processed data...")
#     X = np.load(PROCESSED_X)
#     y = np.load(PROCESSED_Y)
#     print(f"[INFO] X shape: {X.shape}, y shape: {y.shape}")

#     # --- Train Base Transformer ---
#     print("\n[INFO] Initializing Base Transformer...")
#     base_model = FinanceSeq2SeqTransformer(input_dim=X.shape[2], pred_len=PRED_LEN)
#     print("[INFO] Training Base Transformer...")
#     base_history = train_model(
#         base_model, X, y,
#         device=DEVICE,
#         batch_size=BATCH_SIZE,
#         epochs=EPOCHS,
#         lr=LR,
#         save_path=BASE_MODEL,
#         log_path=BASE_MODEL_LOG
#     )

#     # --- Train PRT Transformer ---
#     print("\n[INFO] Initializing Prototype Routing Transformer...")
#     prt_model = PrototypeRoutingTransformer(input_dim=X.shape[2], pred_len=PRED_LEN)
#     print("[INFO] Training PRT Transformer...")
#     prt_history = train_model(
#         prt_model, X, y,
#         device=DEVICE,
#         batch_size=BATCH_SIZE,
#         epochs=EPOCHS,
#         lr=LR,
#         save_path=PRT_MODEL,
#         log_path=PRT_MODEL_LOG
#     )

#     # --- Evaluate & Compare ---
#     print("\n[INFO] Evaluating models...")
#     base_mse = evaluate_model(base_model, X, y)
#     prt_mse = evaluate_model(prt_model, X, y)
#     print(f"\n[RESULTS] Base Transformer MSE: {base_mse:.6f}")
#     print(f"[RESULTS] PRT Transformer MSE:  {prt_mse:.6f}")

#     # Save metrics
#     import json
#     metrics = {
#         "base_transformer": {"mse": float(base_mse)},
#         "prt_transformer": {"mse": float(prt_mse)}
#     }
#     with open(METRICS_FILE, "w") as f:
#         json.dump(metrics, f, indent=4)
#     print(f"[INFO] Metrics saved to {METRICS_FILE}")

# if __name__ == "__main__":
#     main()
