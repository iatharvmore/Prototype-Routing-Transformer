import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class FinanceSeq2SeqTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_encoder_layers=3,
                 num_decoder_layers=3, dim_feedforward=256, dropout=0.1, pred_len=5):
        super().__init__()
        self.d_model = d_model
        self.pred_len = pred_len

        # 1. Project numeric features to d_model
        self.input_proj = nn.Linear(input_dim, d_model)

        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # 3. Transformer Encoder & Decoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout,
                                                   activation='relu',
                                                   batch_first=True)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout,
                                                   activation='relu',
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # 4. Decoder input projection (to d_model)
        self.decoder_input_proj = nn.Linear(1, d_model)  # Each step has 1 feature (closing price)
        
        # 5. Final regression output
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, src, tgt=None):
        """
        src: (batch, seq_len, input_dim) -> historical features
        tgt: (batch, pred_len, 1) -> previous target values (for teacher forcing)
             If None, we use zeros to start prediction
        Returns:
            out: (batch, pred_len, 1)
        """
        batch_size = src.size(0)

        # --- Encode ---
        src = self.input_proj(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        memory = self.encoder(src)  # (batch, seq_len, d_model)

        # --- Prepare decoder input ---
        if tgt is None:
            # Use zeros as initial input
            tgt = torch.zeros(batch_size, self.pred_len, 1, device=src.device)
        tgt = self.decoder_input_proj(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)

        # --- Decode ---
        out = self.decoder(tgt, memory)  # (batch, pred_len, d_model)

        # --- Final linear layer ---
        out = self.fc_out(out)  # (batch, pred_len, 1)
        return out

# import torch

# seq_len = 30
# pred_len = 5
# input_dim = 9  # e.g., OHLC + Volume + RSI + EMA20 + EMA50 + MACD

# # Create dummy batch
# batch_size = 16
# src = torch.randn(batch_size, seq_len, input_dim)
# tgt = torch.randn(batch_size, pred_len, 1)  # previous target values (teacher forcing)

# # Initialize model
# from base_transformer import FinanceSeq2SeqTransformer
# model = FinanceSeq2SeqTransformer(input_dim=input_dim, pred_len=pred_len)

# # Forward pass
# out = model(src, tgt)
# print("Output shape:", out.shape)  # Should be (16, 5, 1)
