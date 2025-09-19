so what changes are required:

# meanflow_mnist_optimized.py
"""
Improved MeanFlow for MNIST with:
 - Fourier time embedding
 - UNet with higher capacity + residual blocks
 - Heun's method sampler
 - Average-velocity loss with higher MC samples
 - EMA weights for sampling (optional)
 - Cosine LR scheduler + checkpointing (saves EMA state)
"""

import os, math, time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

# ---------------- Config ----------------
BATCH_SIZE     = 128
LR             = 1e-4
MAX_STEPS      = 100000
LOG_EVERY      = 200
SAMPLE_EVERY   = 1000
SAMPLE_STEPS   = 100
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# safe: clear cache if cuda is available
if torch.cuda.is_available():
    torch.cuda.empty_cache()

SAMPLE_FOLDER  = "samples"
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_FILE= os.path.join(CHECKPOINT_DIR, "meanflow_mnist.pth")

LAMBDA_AVG = 2.0   # weight on avg-velocity
EMA_DECAY  = 0.999 # EMA decay for weights

# ---------------- Data ----------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 2.0 - 1.0)
])
def unnormalize_img(x):
    return (x + 1.0) * 0.5

train_ds = datasets.MNIST(root="data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available())

# ---------------- Fourier Time embedding ----------------
class FourierTimeEmbed(nn.Module):
    def __init__(self, dim=128, scale=50.0):
        super().__init__()
        self.dim = dim
        freqs = torch.exp(torch.linspace(0., math.log(scale), dim//2))
        self.register_buffer("freqs", freqs)
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
    def forward(self, t):
        # t: (B,)
        t = t.view(-1,1).to(self.freqs.device)
        angles = t * self.freqs[None]
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return self.proj(emb)

# ---------------- Residual block ----------------
class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
        )
    def forward(self, x):
        return x + 0.1 * self.net(x)

# ---------------- Stronger UNet with ResBlocks ----------------
class UNetSmall(nn.Module):
    def __init__(self, time_dim=128, base_ch=128):
        super().__init__()
        self.time_proj = nn.Linear(time_dim, base_ch)

        self.enc1 = nn.Sequential(nn.Conv2d(1, base_ch, 3, padding=1), nn.ReLU(), ResBlock(base_ch))
        self.enc2 = nn.Sequential(nn.Conv2d(base_ch, base_ch, 3, padding=1), nn.ReLU(), ResBlock(base_ch))
        self.down = nn.AvgPool2d(2)  # 28 -> 14

        self.mid = nn.Sequential(nn.Conv2d(base_ch, base_ch, 3, padding=1), nn.ReLU(), ResBlock(base_ch))

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = nn.Sequential(nn.Conv2d(base_ch*2, base_ch, 3, padding=1), nn.ReLU(), ResBlock(base_ch))
        self.out_conv = nn.Conv2d(base_ch, 1, 3, padding=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x, t_emb):
        B, _, H, W = x.shape
        tproj = self.time_proj(t_emb).view(B, -1, 1, 1)

        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        d = self.down(e2)

        m = self.mid(d) + tproj.expand(-1, -1, d.shape[2], d.shape[3])

        u = self.up(m)
        # ensure u and e1 have compatible channels: e1 is base_ch, u is base_ch -> cat works
        cat = torch.cat([u, e1], dim=1)
        h = self.dec1(cat)
        return self.out_conv(h)

# ---------------- Heun Sampler ----------------
def sample_and_save(model, tembed, step, n_samples=64, steps=SAMPLE_STEPS, folder=SAMPLE_FOLDER, filename=None, device=None):
    model.eval()
    os.makedirs(folder, exist_ok=True)
    if device is None:
        # infer device from model params
        device = next(model.parameters()).device
    with torch.no_grad():
        z = torch.randn(n_samples, 1, 28, 28, device=device)
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((n_samples,), i * dt, device=device)
            v1 = model(z, tembed(t.to(device)))
            z_euler = z + dt * v1
            t_next = torch.full((n_samples,), (i+1)*dt, device=device)
            v2 = model(z_euler, tembed(t_next.to(device)))
            z = z + dt * 0.5 * (v1 + v2)
        imgs = unnormalize_img(z).clamp(0.0, 1.0)
        fname = filename if filename else f"sample_{step:06d}.png"
        utils.save_image(imgs, os.path.join(folder, fname), nrow=8)
    model.train()
    return imgs

# ---------------- Training step ----------------
def training_step(batch, model, tembed, optimizer, mse_loss, avg_samples=4):
    # default avg_samples lowered to 4 (memory-friendly)
    x, _ = batch
    x = x.to(next(model.parameters()).device)
    B = x.shape[0]

    z0 = torch.randn_like(x, device=x.device)

    # Flow matching loss: local target x - zt (more stable)
    t_fm = torch.rand(B, device=x.device)
    zt = (1.0 - t_fm.view(B,1,1,1)) * z0 + t_fm.view(B,1,1,1) * x
    v_target = x - zt
    v_pred = model(zt, tembed(t_fm))
    loss_fm = mse_loss(v_pred, v_target)

    # Average velocity loss (Monte Carlo approximation)
    t0 = torch.zeros(B, device=x.device)
    v0 = model(z0, tembed(t0))
    v_sum = torch.zeros_like(v0)
    for _ in range(avg_samples):
        ti = torch.rand(B, device=x.device)
        zti = (1.0 - ti.view(B,1,1,1)) * z0 + ti.view(B,1,1,1) * x
        vti = model(zti, tembed(ti))
        v_sum = v_sum + vti
    v_bar = v_sum / float(avg_samples)
    loss_avg = mse_loss(v0, v_bar)

    loss = loss_fm + LAMBDA_AVG * loss_avg
    optimizer.zero_grad()
    loss.backward()
    # ---- tiny fix: clip grads for both model and tembed (not just model) ----
    torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(tembed.parameters()), 1.0)
    optimizer.step()

    return loss.item(), loss_fm.item(), loss_avg.item()

# ---------------- Checkpoint utils (robust: ema_model optional) ----------------
def save_checkpoint(model, tembed, optimizer, step, ema_model=None, fname=CHECKPOINT_FILE):
    """
    Save model, tembed, optimizer and optional ema_model state.
    ema_model: optional. If provided, its state is saved under 'ema_state'.
    """
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ck = {
        "model_state": model.state_dict(),
        "tembed_state": tembed.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "ema_state": ema_model.state_dict() if ema_model is not None else None,
        "step": step
    }
    torch.save(ck, fname)
    print(f"✅ Saved checkpoint: {fname}")

def load_checkpoint_if_exists(model, tembed, optimizer=None, ema_model=None, fname=CHECKPOINT_FILE):
    """
    Loads checkpoint if exists. If ema_model is provided and an ema_state exists in the checkpoint,
    it will be loaded into ema_model. Returns the saved step (0 if none).
    """
    if os.path.exists(fname):
        ck = torch.load(fname, map_location=DEVICE)
        if "model_state" in ck and ck["model_state"] is not None:
            model.load_state_dict(ck["model_state"])
        if "tembed_state" in ck and ck["tembed_state"] is not None:
            tembed.load_state_dict(ck["tembed_state"])
        if ema_model is not None and ck.get("ema_state") is not None:
            try:
                ema_model.load_state_dict(ck["ema_state"])
            except Exception as e:
                print("Warning: failed to load ema_state:", e)
        if optimizer is not None and ck.get("optimizer_state") is not None:
            try:
                optimizer.load_state_dict(ck["optimizer_state"])
            except Exception as e:
                print("Warning: failed to load optimizer state (incompatible):", e)
        step = ck.get("step", 0)
        print(f"✅ Loaded checkpoint from {fname} (step {step})")
        return step
    return 0

# ---------------- Training loop ----------------
def train():
    # reduce channels back to 128 to fit memory
    model = UNetSmall(time_dim=128, base_ch=128).to(DEVICE)
    tembed = FourierTimeEmbed(dim=128, scale=50.0).to(DEVICE)

    # ema_model omitted to save memory (optional to enable)
    ema_model = None

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(tembed.parameters()), 
        lr=LR, weight_decay=1e-6
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_STEPS)
    mse_loss = nn.MSELoss()

    step = load_checkpoint_if_exists(model, tembed, optimizer, ema_model)
    print("🚀 Training started on", DEVICE)

    t0 = time.time()
    while step < MAX_STEPS:
        for batch in train_loader:
            # reduce avg_samples to 4 (memory-friendly)
            loss, loss_fm, loss_avg = training_step(
                batch, model, tembed, optimizer, mse_loss, avg_samples=4
            )

            # scheduler step per-iteration (---- tiny fix: actually step the scheduler ----)
            scheduler.step()

            if step % LOG_EVERY == 0:
                elapsed = (time.time() - t0) / 60
                print(f"[step {step:06d}] loss={loss:.5f} fm={loss_fm:.5f} avg={loss_avg:.5f} time={elapsed:.1f}m lr={scheduler.get_last_lr()[0]:.2e}")
            if step % SAMPLE_EVERY == 0:
                sample_and_save(model, tembed, step, steps=SAMPLE_STEPS)
                save_checkpoint(model, tembed, optimizer, step, ema_model=None)
            step += 1
            if step >= MAX_STEPS:
                break

    save_checkpoint(model, tembed, optimizer, step, ema_model=None)
    sample_and_save(model, tembed, step, steps=SAMPLE_STEPS, folder="outputs", filename="mnist_digits_final.png")
    return model, tembed


if __name__ == "__main__":
    train()