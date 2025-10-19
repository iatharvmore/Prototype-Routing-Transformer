# # src/train_utils.py
# src/train_utils.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os
import json
from tqdm import tqdm
import numpy as np
from src.config import *
from src.prototype_routing_transformer import PrototypeRoutingTransformer

# --- Helper: create multi-step target sequences ---
def create_target_sequences(y, pred_len):
    """
    Converts 1D array y into overlapping multi-step sequences
    y: numpy array of shape (num_samples,)
    pred_len: number of steps to predict
    Returns: y_seq of shape (num_samples - pred_len, pred_len)
    """
    seqs = []
    for i in range(len(y) - pred_len):
        seqs.append(y[i:i+pred_len])
    return np.array(seqs)

# --- Main training function ---
def train_model(model, X, y, device=DEVICE, 
                batch_size=BATCH_SIZE, epochs=EPOCHS, lr=LR,
                save_path=PRT_MODEL, log_path=PRT_MODEL_LOG):

    # --- Prepare multi-step targets ---
    y_seq = create_target_sequences(y, model.pred_len)
    X_trimmed = X[:len(y_seq)]
    y_seq = y_seq[..., None]  # Convert to 3D: (num_samples, pred_len, 1)

    # --- Dataset & Dataloader ---
    dataset = TensorDataset(torch.tensor(X_trimmed, dtype=torch.float32),
                            torch.tensor(y_seq, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # --- Model setup ---
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Ensure directories exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    history = {'loss': []}

    # --- Training loop ---
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for xb, yb in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            out = model(xb)  # (batch, pred_len, 1)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * xb.size(0)

        epoch_loss /= len(dataset)
        history['loss'].append(epoch_loss)

        # Logging
        print(f"[Epoch {epoch+1}/{epochs}] Loss: {epoch_loss:.6f}")
        with open(log_path, 'w') as f:
            json.dump(history, f, indent=4)

    # Save final model
    torch.save(model.state_dict(), save_path)
    print(f"[INFO] PRT model saved to {save_path}")

    return history

# --- Script entry (optional) ---
# if __name__ == "__main__":
#     # Load processed data
#     X = np.load(PROCESSED_X)
#     y = np.load(PROCESSED_Y)

#     # Initialize PRT model
#     input_dim = X.shape[2]
#     model = PrototypeRoutingTransformer(input_dim=input_dim, pred_len=PRED_LEN)

#     # Train
#     history = train_model(model, X, y)


# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, TensorDataset
# import os
# import json
# from tqdm import tqdm
# import numpy as np
# # from base_transformer import FinanceSeq2SeqTransformer
# from prototype_routing_transformer import PrototypeRoutingTransformer

# # --- Helper: create target sequences for multi-step prediction ---
# def create_target_sequences(y, pred_len):
#     """
#     y: 1D numpy array of shape (num_samples,)
#     pred_len: number of steps to predict
#     Returns: y_seq of shape (num_samples - pred_len, pred_len)
#     """
#     seqs = []
#     for i in range(len(y) - pred_len):
#         seqs.append(y[i:i+pred_len])
#     return np.array(seqs)

# # --- Main training function ---
# def train_model(model, X, y, device='cuda', 
#                 batch_size=16, epochs=50, lr=1e-3,
#                 save_path="results/models/model.pth",
#                 log_path="results/logs/model.log"):

#     # --- Create target sequences for multi-step prediction ---
#     y_seq = create_target_sequences(y, model.pred_len)
#     # Trim X to match y_seq length
#     X_trimmed = X[:len(y_seq)]

#     # Ensure y_seq is 3D
#     y_seq = y_seq[..., None]  # (num_samples, pred_len, 1)

#     # Prepare dataset & dataloader
#     dataset = TensorDataset(torch.tensor(X_trimmed, dtype=torch.float32),
#                             torch.tensor(y_seq, dtype=torch.float32))
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#     # Move model to device
#     model.to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     criterion = nn.MSELoss()

#     # Ensure directories exist
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     os.makedirs(os.path.dirname(log_path), exist_ok=True)

#     history = {'loss': []}

#     # --- Training loop ---
#     for epoch in range(epochs):
#         model.train()
#         epoch_loss = 0.0

#         for xb, yb in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
#             xb, yb = xb.to(device), yb.to(device)

#             optimizer.zero_grad()
#             out = model(xb)  # (batch, pred_len, 1)
#             loss = criterion(out, yb)  # Shapes match now
#             loss.backward()
#             optimizer.step()

#             epoch_loss += loss.item() * xb.size(0)

#         epoch_loss /= len(dataset)
#         history['loss'].append(epoch_loss)

#         # Logging
#         print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.6f}")
#         with open(log_path, 'w') as f:
#             json.dump(history, f, indent=4)

#     # Save final model
#     torch.save(model.state_dict(), save_path)
#     print(f"Model saved to {save_path}")
#     return history

# # --- Script entry ---
# # if __name__ == "__main__":
# #     # Load processed data
# #     X = np.load("data/processed/X.npy")
# #     y = np.load("data/processed/y.npy")  # 1D array of closing prices

# #     input_dim = X.shape[2]
# #     pred_len = 5  # multi-step prediction
# #     model = FinanceSeq2SeqTransformer(input_dim=input_dim, pred_len=pred_len)

# #     # Train
# #     history = train_model(
# #         model, X, y, device='cuda', batch_size=16, epochs=20,
# #         save_path="results/models/base_transformer.pth",
# #         log_path="results/logs/base_transformer.log"
# #     )
