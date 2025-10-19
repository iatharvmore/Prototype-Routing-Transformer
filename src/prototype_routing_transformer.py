# src/prototype_routing_transformer.py

import torch
import torch.nn as nn
import math
from src.base_transformer import PositionalEncoding

class PrototypeRoutingLayer(nn.Module):
    """
    One layer of Prototype Routing: input attends to learned prototype vectors.
    """
    def __init__(self, d_model, n_prototypes=16):
        super().__init__()
        self.n_prototypes = n_prototypes
        # Learnable prototype vectors
        self.prototypes = nn.Parameter(torch.randn(n_prototypes, d_model))
        # Linear projection for queries
        self.query_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        returns: (batch, seq_len, d_model)
        """
        q = self.query_proj(x)  # (batch, seq_len, d_model)
        # Attention: queries vs prototypes
        attn_scores = torch.matmul(q, self.prototypes.T) / math.sqrt(x.size(-1))  # (batch, seq_len, n_prototypes)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        routed = torch.matmul(attn_weights, self.prototypes)  # (batch, seq_len, d_model)
        return routed

class PrototypeRoutingTransformer(nn.Module):
    """
    Multi-step forecasting PRT model
    """
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=3,
                 dim_feedforward=256, dropout=0.1, pred_len=5, n_prototypes=16):
        super().__init__()
        self.d_model = d_model
        self.pred_len = pred_len

        # Input projection for numeric features
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Prototype routing layers
        self.pr_layers = nn.ModuleList([PrototypeRoutingLayer(d_model, n_prototypes) for _ in range(num_layers)])

        # Decoder input projection
        self.decoder_input_proj = nn.Linear(1, d_model)
        self.decoder_pos_encoder = PositionalEncoding(d_model, dropout)

        # Transformer decoder for multi-step prediction
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout,
                                                   activation='relu',
                                                   batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Final output layer
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, src, tgt=None):
        """
        src: (batch, seq_len, input_dim)
        tgt: (batch, pred_len, 1)
        returns: (batch, pred_len, 1)
        """
        batch_size = src.size(0)

        # --- Encoder ---
        x = self.input_proj(src) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        for pr in self.pr_layers:
            x = x + pr(x)  # residual connection
        memory = x  # (batch, seq_len, d_model)

        # --- Decoder ---
        if tgt is None:
            tgt = torch.zeros(batch_size, self.pred_len, 1, device=src.device)
        tgt = self.decoder_input_proj(tgt) * math.sqrt(self.d_model)
        tgt = self.decoder_pos_encoder(tgt)

        out = self.decoder(tgt, memory)
        out = self.fc_out(out)
        return out

# --- Quick test ---
# if __name__ == "__main__":
#     seq_len = 30
#     pred_len = 5
#     input_dim = 9
#     batch_size = 16

#     src = torch.randn(batch_size, seq_len, input_dim)
#     tgt = torch.randn(batch_size, pred_len, 1)

#     model = PrototypeRoutingTransformer(input_dim=input_dim, pred_len=pred_len)
#     out = model(src, tgt)
#     print("PRT Output shape:", out.shape)  # (16, 5, 1)
