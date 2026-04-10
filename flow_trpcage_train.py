# ===============================================================
# Unconditional Flow-Matching Model for TRP-Cage psi phi Data
# Model 1: Without residual connection
# Model 2:  With residual connections
# Model 3: With Transformer
# Data: trpcage_psi_phi.npy  -> shape (50000, 38)
# Batch size: 4096
# Train: 50000 epochs
# Save model every 1000 epochs
# Log loss to file
# Sampling via ODE
# ===============================================================

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
import matplotlib.pyplot as plt

# ===============================================================
# Device
# ===============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device, flush=True)

# ===============================================================
# Hyperparameters
# ===============================================================
DATA_PATH = "trpcage_psi_phi.npy"
BATCH_SIZE = 4096
DATA_DIM = 38
HIDDEN_DIM = 256
EPOCHS = 50000
LR = 5e-5
DROPOUT    = 0.2

SAVE_DIR = "with_trans/models_fm_rs"
SAMPLE_DIR = "with_trans/samples_fm_rs"
LOSS_FILE = "with_trans/loss_log_rs.txt"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)

# ===============================================================
# Time Embedding
# ===============================================================

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU()
        )

    def forward(self, t):
        t = t.unsqueeze(-1).float()
        return self.net(t)





# ===============================================================
# Unconditional MLP for Flow Matching
# ===============================================================



class MLPFM(nn.Module):
    def __init__(self, data_dim=38, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.time_embed = TimeEmbedding(hidden_dim)

        in_dim = data_dim + hidden_dim

        self.block1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.block3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.block4 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.out = nn.Linear(hidden_dim, data_dim)

    def forward(self, x, t):
        t_emb = self.time_embed(t)                 # (B, hidden_dim)
        h = torch.cat([x, t_emb], dim=-1)          # (B, 38+256)

        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)

        return self.out(h)


# ===============================================================
# Residual Block
# ===============================================================
class ResBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return x + self.block(x)   # Residual connection


# ===============================================================
# MLP with Residuals for Flow Matching
# ===============================================================
class MLPFM_RS(nn.Module):
    def __init__(self, data_dim=38, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.time_embed = TimeEmbedding(hidden_dim)

        # First projection: x + t → hidden
        self.input_layer = nn.Sequential(
            nn.Linear(data_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Residual blocks
        self.res1 = ResBlock(hidden_dim, dropout)
        self.res2 = ResBlock(hidden_dim, dropout)
        self.res3 = ResBlock(hidden_dim, dropout)
        self.res4 = ResBlock(hidden_dim, dropout)

        # Output projection
        self.out = nn.Linear(hidden_dim, data_dim)

    def forward(self, x, t):
        t_emb = self.time_embed(t)                # (B, hidden_dim)
        h = torch.cat([x, t_emb], dim=-1)         # (B, 38 + 256)

        h = self.input_layer(h)
        h = self.res1(h)
        h = self.res2(h)
        h = self.res3(h)
        h = self.res4(h)

        return self.out(h)





# ===============================================================
# Transformer based Flow-Matching Model
# ===============================================================

class TransformerFM(nn.Module):
    def __init__(
        self,
        data_dim=38,
        hidden_dim=256,
        num_layers=6,
        num_heads=8,
        dropout=0.2
    ):
        super().__init__()
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim

        # Time embedding
        self.time_embed = TimeEmbedding(hidden_dim)

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(data_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, data_dim)
        )

    def forward(self, x, t):
        """
        x : (B, 38)
        t : (B,)
        """

        # Project x → hidden
        h = self.input_proj(x)          # (B, H)

        # Time embedding
        t_emb = self.time_embed(t)      # (B, H)

        # Add time as conditioning (FiLM-style)
        h = h + t_emb

        # Transformer (B, T, H)
        h = h.unsqueeze(1)              # (B, 1, H)

        # Encode
        h = self.encoder(h)             # (B, 1, H)

        # Remove sequence dimension
        h = h.squeeze(1)                # (B, H)

        # Output velocity
        return self.output_proj(h)








# ===============================================================
# Flow Matching Loss
# ===============================================================
def fm_loss(model, x1):
    """
    x1 : (B, 38)
    """
    bsz = x1.size(0)

    # Sample t ∈ [0,1]
    t = torch.rand(bsz, device=device)

    # Base distribution z ~ N(0, I)
    z = torch.randn_like(x1)

    # Interpolation
    xt = t[:, None] * x1 + (1 - t)[:, None] * z

    # Target velocity
    v_target = x1 - z

    # Predicted velocity
    pred = model(xt, t)

    return F.mse_loss(pred, v_target)

# ===============================================================
# ODE Sampler
# ===============================================================
class FlowODE(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x):
        t_batch = torch.full((x.size(0),), float(t), device=device)
        return self.model(x, t_batch)

@torch.no_grad()
def sample_flow(model, n_samples=1024, t_eval=200):
    model.eval()
    x0 = torch.randn(n_samples, DATA_DIM, device=device)
    t = torch.linspace(0, 1, t_eval, device=device)

    ode = FlowODE(model)
    x_t = odeint(ode, x0, t)

    # Final samples
    return x_t[-1].cpu().numpy()

# ===============================================================
# Load Data
# ===============================================================
data = np.load(DATA_PATH).astype(np.float32)   # (50000, 38)
N = data.shape[0]
print("Loaded data:", data.shape, flush=True)

# ===============================================================
# Model, Optimizer
# ===============================================================
# model = MLPFM(data_dim=DATA_DIM, hidden_dim=HIDDEN_DIM).to(device)
# model = MLPFM_RS(data_dim=DATA_DIM, hidden_dim=HIDDEN_DIM).to(device)

model = TransformerFM(
    data_dim=DATA_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=6,
    num_heads=8,
    dropout=DROPOUT
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

# ===============================================================
# Training
# ===============================================================
print("\n=== TRAINING STARTED ===", flush=True)
start_time = time.time()

loss_log = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    num_batches = 0

    # Shuffle data each epoch
    perm = torch.randperm(N)

    for i in range(0, N, BATCH_SIZE):
        idx = perm[i:i + BATCH_SIZE]
        x1 = torch.tensor(data[idx], device=device)

        loss = fm_loss(model, x1)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

    avg_loss = epoch_loss / num_batches
    loss_log.append(avg_loss)

    print(f"Epoch {epoch:04d}/{EPOCHS} | Loss: {avg_loss:.6f}", flush=True)

    # Save loss log every epoch
    np.savetxt(LOSS_FILE, np.array(loss_log))

    # Save model and generate samples every 1000 epochs
    if epoch % 5000 == 0:
        model_path = os.path.join(SAVE_DIR, f"unet_fm_epoch_{epoch:04d}.pt")
        torch.save(model.state_dict(), model_path)
        print(f"Saved model: {model_path}", flush=True)

        # Sampling
        samples = sample_flow(model, n_samples=1024)

        # Save samples
        sample_path = os.path.join(SAMPLE_DIR, f"samples_epoch_{epoch:04d}.npy")
        np.save(sample_path, samples)

        # Simple diagnostic plot for first two dimensions
        plt.figure(figsize=(6,6))
        plt.scatter(samples[:,0], samples[:,1], s=5, alpha=0.6)
        plt.xlabel("Dim 1")
        plt.ylabel("Dim 2")
        plt.title(f"Flow Matching Samples (Epoch {epoch})")
        plt.grid(True)
        plt.gca().set_aspect("equal", "box")
        plot_path = os.path.join(SAMPLE_DIR, f"samples_epoch_{epoch:04d}.png")
        plt.savefig(plot_path, dpi=200)
        plt.close()

        print(f"Saved samples and plot for epoch {epoch}", flush=True)

total_time = time.time() - start_time
print("\n=== TRAINING COMPLETE ===", flush=True)
print(f"Total time: {total_time/60:.2f} minutes ({total_time:.1f} seconds)", flush=True)

# ===============================================================
#Sample After Training
# ===============================================================
"""

model = MLPFM(data_dim=38).to(device)
model.load_state_dict(torch.load("models_fm/unet_fm_epoch_5000.pt", map_location=device))
model.eval()

samples = sample_flow(model, n_samples=5000)
np.save("final_samples.npy", samples)
"""
