import os, time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ===============================================================
# Device
# ===============================================================
device = "cuda:6" if torch.cuda.is_available() else "cpu"
print("Using device:", device,flush=True)

# ===============================================================
# Hyperparameters (must match training)
# ===============================================================
DATA_DIM   = 38
HIDDEN_DIM = 256
DROPOUT    = 0.2
T          = 1000

N_SAMPLES = 500000
BATCH     = 4096

MODEL_TYPE = "transformer"   # "mlp", "mlp_rs", "transformer"

MODEL_PATH = {
    "mlp":         "unet/models/model_epoch_50000.pt",
    "mlp_rs":      "unet_rs/models/model_epoch_50000.pt",
    "transformer": "transformer/models/model_epoch_50000.pt",
}[MODEL_TYPE]

SAVE_FILE = f"final_samples_{MODEL_TYPE}.npy"
FRAME_DIR = "diff_frames/transformer"
os.makedirs(FRAME_DIR, exist_ok=True)

# ===============================================================
# Diffusion Schedule
# ===============================================================
betas = torch.linspace(1e-4, 0.02, T).to(device)
alphas = 1.0 - betas
alpha_bar = torch.cumprod(alphas, dim=0)

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
# Model 1 : Diffusion MLP
# ===============================================================
class DiffusionMLP(nn.Module):
    def __init__(self, data_dim, hidden_dim, dropout):
        super().__init__()
        self.time_embed = TimeEmbedding(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(data_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, data_dim)
        )

    def forward(self, x, t):
        t_emb = self.time_embed(t)
        return self.net(torch.cat([x, t_emb], dim=-1))

# ===============================================================
# Model 2 : Diffusion MLP + Residuals
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

class DiffusionMLP_RS(nn.Module):
    def __init__(self, data_dim, hidden_dim, dropout):
        super().__init__()
        self.time_embed = TimeEmbedding(hidden_dim)

        self.input = nn.Sequential(
            nn.Linear(data_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        self.res1 = ResBlock(hidden_dim, dropout)
        self.res2 = ResBlock(hidden_dim, dropout)
        self.res3 = ResBlock(hidden_dim, dropout)
        self.res4 = ResBlock(hidden_dim, dropout)

        self.out = nn.Linear(hidden_dim, data_dim)

    def forward(self, x, t):
        t_emb = self.time_embed(t)
        h = self.input(torch.cat([x, t_emb], dim=-1))
        h = self.res1(h)
        h = self.res2(h)
        h = self.res3(h)
        h = self.res4(h)
        return self.out(h)

# ===============================================================
# Model 3 : Diffusion Transformer
# ===============================================================
class DiffusionTransformer(nn.Module):
    def __init__(self, data_dim, hidden_dim, num_layers=6, num_heads=8, dropout=0.2):
        super().__init__()
        self.time_embed = TimeEmbedding(hidden_dim)
        self.input_proj = nn.Linear(data_dim, hidden_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, data_dim)
        )

    def forward(self, x, t):
        h = self.input_proj(x)
        h = h + self.time_embed(t)
        h = self.encoder(h.unsqueeze(1)).squeeze(1)
        return self.out(h)

# ===============================================================
# Reverse Diffusion Sampler WITH FRAME SAVING
# ===============================================================
@torch.no_grad()
def sample_diffusion_with_frames(model, n_samples, frame_buffers):
    model.eval()
    x = torch.randn(n_samples, DATA_DIM, device=device)

    for t in reversed(range(T)):
        t_batch = torch.full((n_samples,), t / T, device=device)

        beta  = betas[t]
        alpha = alphas[t]
        a_bar = alpha_bar[t]

        eps = model(x, t_batch)
        mean = (1 / torch.sqrt(alpha)) * (x - beta / torch.sqrt(1 - a_bar) * eps)

        if t > 0:
            x = mean + torch.sqrt(beta) * torch.randn_like(x)
        else:
            x = mean

        # Save this timestep
        frame_buffers[t].append(x.detach().cpu().numpy())

    return x.cpu().numpy()

# ===============================================================
# Build model
# ===============================================================
if MODEL_TYPE == "mlp":
    model = DiffusionMLP(DATA_DIM, HIDDEN_DIM, DROPOUT).to(device)
elif MODEL_TYPE == "mlp_rs":
    model = DiffusionMLP_RS(DATA_DIM, HIDDEN_DIM, DROPOUT).to(device)
elif MODEL_TYPE == "transformer":
    model = DiffusionTransformer(DATA_DIM, HIDDEN_DIM).to(device)
else:
    raise ValueError("Unknown MODEL_TYPE")

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print("Loaded model:", MODEL_TYPE,flush=True)

# ===============================================================
# Batched Sampling
# ===============================================================
frame_buffers = {t: [] for t in range(T)}
all_samples = []

start = time.time()

with torch.no_grad():
    for i in range(0, N_SAMPLES, BATCH):
        bs = min(BATCH, N_SAMPLES - i)
        print(f"Sampling {i} → {i + bs}", flush=True)

        samples = sample_diffusion_with_frames(
            model,
            bs,
            frame_buffers
        )
        all_samples.append(samples)

all_samples = np.concatenate(all_samples, axis=0)
elapsed = time.time() - start

print("Final shape:", all_samples.shape,flush=True)
print(f"Total time: {elapsed/60:.2f} min",flush=True)
print(f"Samples/sec: {N_SAMPLES/elapsed:.1f}",flush=True)

# ===============================================================
# Save final samples
# ===============================================================
np.save(SAVE_FILE, all_samples)
print("Saved:", SAVE_FILE,flush=True)

# ===============================================================
# Save diffusion frames (1000 files)
# ===============================================================
print("Saving diffusion frames...",flush=True)

for t in range(T):
    frame_t = np.concatenate(frame_buffers[t], axis=0)
    np.save(f"{FRAME_DIR}/frame_{t:04d}.npy", frame_t)

print(f"Saved {T} frames to {FRAME_DIR}/",flush=True)

# ===============================================================
# Quick visualization
# ===============================================================
plt.figure(figsize=(6, 6))
plt.scatter(all_samples[:5000, 0], all_samples[:5000, 1], s=2, alpha=0.5)
plt.xlabel("ψ")
plt.ylabel("φ")
plt.title(f"Generated samples ({MODEL_TYPE})")
plt.tight_layout()
plt.savefig(f"scatter_{MODEL_TYPE}.png")
plt.close()
