# ===============================================================
# Diffusion Models for TRP-Cage psi - phi data(38D)
# Model 1 : MLP (no residuals)
# Model 2 : MLP + Residuals
# Model 3 : Transformer
# ===============================================================

import os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ===============================================================
# Device
# ===============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ===============================================================
# Hyperparameters
# ===============================================================
DATA_PATH = "trpcage_psi_phi.npy"
BATCH_SIZE = 4096
DATA_DIM = 38
HIDDEN_DIM = 256
EPOCHS = 50000
LR = 5e-5
DROPOUT = 0.2

SAVE_DIR   = "Diffusion/unet_rs/models"
SAMPLE_DIR = "Diffusion/unet_rs/samples"
LOSS_FILE  = "Diffusion/unet_rs/loss.txt"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)

# ===============================================================
# Diffusion Schedule
# ===============================================================
T = 1000
betas = torch.linspace(1e-4, 0.02, T).to(device)
alphas = 1.0 - betas
alpha_bar = torch.cumprod(alphas, dim=0)

def q_sample(x0, t, noise):
    a_bar = alpha_bar[t].unsqueeze(1)
    return torch.sqrt(a_bar) * x0 + torch.sqrt(1 - a_bar) * noise

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
# Model 1 : Diffusion MLP (No residuals)
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
        h = torch.cat([x, t_emb], dim=-1)
        return self.net(h)

# ===============================================================
# Model 2 : Diffusion MLP with Residuals
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
        h = torch.cat([x, t_emb], dim=-1)

        h = self.input(h)
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
        t_emb = self.time_embed(t)
        h = h + t_emb

        h = h.unsqueeze(1)
        h = self.encoder(h)
        h = h.squeeze(1)

        return self.out(h)

# ===============================================================
# Diffusion Loss
# ===============================================================
def diffusion_loss(model, x0):
    bsz = x0.size(0)
    t = torch.randint(0, T, (bsz,), device=device)
    noise = torch.randn_like(x0)

    xt = q_sample(x0, t, noise)
    pred_noise = model(xt, t.float() / T)

    return F.mse_loss(pred_noise, noise)

# ===============================================================
# Reverse Diffusion Sampler
# ===============================================================
@torch.no_grad()
def sample_diffusion(model, n_samples=1024):
    model.eval()
    x = torch.randn(n_samples, DATA_DIM, device=device)

    for t in reversed(range(T)):
        t_batch = torch.full((n_samples,), t/T, device=device)
        beta = betas[t]
        alpha = alphas[t]
        a_bar = alpha_bar[t]

        eps = model(x, t_batch)

        mean = (1/torch.sqrt(alpha)) * (x - beta/torch.sqrt(1 - a_bar) * eps)

        if t > 0:
            x = mean + torch.sqrt(beta) * torch.randn_like(x)
        else:
            x = mean

    return x.cpu().numpy()

# ===============================================================
# Load Data
# ===============================================================
data = np.load(DATA_PATH).astype(np.float32)
N = data.shape[0]
print("Loaded:", data.shape)

# ===============================================================
# Choose Model Here
# ===============================================================
#model = DiffusionMLP(DATA_DIM, HIDDEN_DIM, DROPOUT).to(device)
#model = DiffusionMLP_RS(DATA_DIM, HIDDEN_DIM, DROPOUT).to(device)
model = DiffusionTransformer(DATA_DIM, HIDDEN_DIM, num_layers=6, num_heads=8, dropout=DROPOUT).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

# ===============================================================
# Training
# ===============================================================
print("=== TRAINING STARTED ===")
loss_log = []
start = time.time()

for epoch in range(1, EPOCHS + 1):
    perm = torch.randperm(N)
    epoch_loss = 0
    nb = 0

    for i in range(0, N, BATCH_SIZE):
        idx = perm[i:i+BATCH_SIZE]
        x1 = torch.tensor(data[idx], device=device)

        loss = diffusion_loss(model, x1)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        epoch_loss += loss.item()
        nb += 1

    avg_loss = epoch_loss / nb
    loss_log.append(avg_loss)
    print(f"Epoch {epoch:04d} | Loss: {avg_loss:.6f}",flush=True)

    np.savetxt(LOSS_FILE, np.array(loss_log))

    if epoch % 5000 == 0:
        path = os.path.join(SAVE_DIR, f"model_epoch_{epoch}.pt")
        torch.save(model.state_dict(), path)

        samples = sample_diffusion(model, 1024)
        np.save(os.path.join(SAMPLE_DIR, f"samples_{epoch}.npy"), samples)

        plt.figure(figsize=(6,6))
        plt.scatter(samples[:,0], samples[:,1], s=5, alpha=0.6)
        plt.title(f"Epoch {epoch}")
        plt.savefig(os.path.join(SAMPLE_DIR, f"samples_{epoch}.png"))
        plt.close()

print("Training finished in %.2f minutes" % ((time.time()-start)/60))

