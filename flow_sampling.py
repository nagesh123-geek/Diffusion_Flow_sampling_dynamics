# ===============================================================
# Flow-Matching Sampling 
# Supports:
#   1) MLP
#   2) MLP_RS (Residual)
#   3) TransformerFM
# Batched sampling + timing + FRAME SAVING
# ===============================================================

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint

# ===============================================================
# Device
# ===============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device, flush=True)

# ===============================================================
# Hyperparameters
# ===============================================================
DATA_DIM    = 38
HIDDEN_DIM  = 256
DROPOUT     = 0.2
BATCH_SIZE  = 4096
TOTAL_SAMPLES = 500000
T_EVAL      = 1000    # ODE steps

#MODEL_PATH = "models_fm/unet_fm_epoch_50000.pt"
#MODEL_PATH = "with_rs/models_fm_rs/unet_fm_epoch_50000.pt"
MODEL_PATH = "with_trans/models_fm_trans/unet_fm_epoch_50000.pt"

SAVE_FILE  = "final_samples_trans.npy"

FRAME_DIR = "Flow_frames_trans"
os.makedirs(FRAME_DIR, exist_ok=True)

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
        return self.net(t.unsqueeze(-1).float())

# ===============================================================
# UNetFM
# ===============================================================
class UNetFM(nn.Module):
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
        t_emb = self.time_embed(t)
        h = torch.cat([x, t_emb], dim=-1)
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
            nn.LayerNorm(dim)
        )

    def forward(self, x):
        return x + self.block(x)

# ===============================================================
# UNetFM_RS
# ===============================================================
class UNetFM_RS(nn.Module):
    def __init__(self, data_dim=38, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.time_embed = TimeEmbedding(hidden_dim)

        self.input_layer = nn.Sequential(
            nn.Linear(data_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.res1 = ResBlock(hidden_dim, dropout)
        self.res2 = ResBlock(hidden_dim, dropout)
        self.res3 = ResBlock(hidden_dim, dropout)
        self.res4 = ResBlock(hidden_dim, dropout)

        self.out = nn.Linear(hidden_dim, data_dim)

    def forward(self, x, t):
        t_emb = self.time_embed(t)
        h = torch.cat([x, t_emb], dim=-1)
        h = self.input_layer(h)
        h = self.res1(h)
        h = self.res2(h)
        h = self.res3(h)
        h = self.res4(h)
        return self.out(h)

# ===============================================================
# TransformerFM
# ===============================================================
class TransformerFM(nn.Module):
    def __init__(self, data_dim=38, hidden_dim=256, num_layers=6, num_heads=8, dropout=0.2):
        super().__init__()
        self.time_embed = TimeEmbedding(hidden_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(data_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, data_dim)
        )

    def forward(self, x, t):
        h = self.input_proj(x)
        h = h + self.time_embed(t)
        h = self.encoder(h.unsqueeze(1)).squeeze(1)
        return self.output_proj(h)

# ===============================================================
# ODE Wrapper
# ===============================================================
class FlowODE(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x):
        t_batch = torch.full((x.size(0),), float(t), device=x.device)
        return self.model(x, t_batch)

# ===============================================================
# Batched Flow Sampling WITH FRAME SAVING
# ===============================================================
@torch.no_grad()
def sample_flow_batched(model, total_samples, batch_size=4096, t_eval=1000):
    model.eval()
    frame_buffers = {i: [] for i in range(t_eval)}
    all_samples = []

    start_time = time.time()

    for i in range(0, total_samples, batch_size):
        bs = min(batch_size, total_samples - i)
        print(f"Sampling batch {i} → {i+bs}", flush=True)

        x0 = torch.randn(bs, DATA_DIM, device=device)
        t = torch.linspace(0, 1, t_eval, device=device)

        ode = FlowODE(model)
        x_t = odeint(ode, x0, t)  # (T, B, D)

        for ti in range(t_eval):
            frame_buffers[ti].append(x_t[ti].cpu().numpy())

        all_samples.append(x_t[-1].cpu().numpy())

    total_time = time.time() - start_time
    all_samples = np.concatenate(all_samples, axis=0)

    print("\n==============================",flush=True)
    print("Sampling completed",flush=True)
    print("Total samples:", all_samples.shape[0],flush=True)
    print(f"Total time: {total_time:.2f} s",flush=True)
    print(f"Samples / second: {total_samples / total_time:.2f}",flush=True)
    print("==============================\n",flush=True)

    return all_samples, frame_buffers

# ===============================================================
# Choose model
# ===============================================================
#model = UNetFM(DATA_DIM, HIDDEN_DIM, DROPOUT).to(device)
#model = UNetFM_RS(DATA_DIM, HIDDEN_DIM, DROPOUT).to(device)
model = TransformerFM(DATA_DIM, HIDDEN_DIM).to(device)

# ===============================================================
# Load weights
# ===============================================================
print("Loading model from:", MODEL_PATH,flush=True)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ===============================================================
# Run sampling
# ===============================================================
samples, frame_buffers = sample_flow_batched(
    model,
    total_samples=TOTAL_SAMPLES,
    batch_size=BATCH_SIZE,
    t_eval=T_EVAL
)

# ===============================================================
# Save final samples
# ===============================================================
np.save(SAVE_FILE, samples)
print("Saved samples to:", SAVE_FILE,flush=True)

# ===============================================================
# Save ODE frames
# ===============================================================
print("Saving flow frames...",flush=True)

for t in range(T_EVAL):
    frame_t = np.concatenate(frame_buffers[t], axis=0)
    np.save(f"{FRAME_DIR}/frame_{t:04d}.npy", frame_t)

print(f"Saved {T_EVAL} frames to {FRAME_DIR}/",flush=True)
