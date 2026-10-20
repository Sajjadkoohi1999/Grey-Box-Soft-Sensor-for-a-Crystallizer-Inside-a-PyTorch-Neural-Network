"""
=============================================================================
 GREY-BOX SOFT SENSOR FOR A CRYSTALLIZER
 First-principles MSMPR model EMBEDDED INSIDE a neural network.

 Architecture:
     L50 = L50_physics(X_raw) * (1 + ANN_correction(X_scaled))
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import torch
import torch.nn as nn
import torch.optim as optim


# =============================================================================
# 1. TRUE PROCESS SIMULATOR
# =============================================================================
def simulate_true_process(n_samples=3000, seed=42):
    rng = np.random.default_rng(seed)

    T_jacket = rng.uniform(300, 320, n_samples)
    F_feed   = rng.uniform(0.5, 2.0, n_samples)
    C_feed   = rng.uniform(0.25, 0.45, n_samples)
    RPM      = rng.uniform(200, 600, n_samples)
    T_feed   = rng.uniform(320, 340, n_samples)
    tau      = rng.uniform(20, 80, n_samples)

    k_g   = 2.0e-8
    C_sat = 0.20
    S = np.clip(C_feed - C_sat * (1.0 + 0.02 * (T_jacket - 310)), 1e-4, None)

    G_true = (k_g
              * S**0.85
              * (1.0 + 0.03*(T_jacket - 310) + 0.0015*(T_jacket - 310)**2)
              * (1.0 + 0.0002*(RPM - 400)))

    L50 = 3.0 * G_true * (tau * 60.0) * 1e6 + rng.normal(0, 1.0, n_samples)

    conductivity = 0.8*C_feed + 0.02*RPM/100 + rng.normal(0, 0.005, n_samples)
    dT           = T_feed - T_jacket + rng.normal(0, 1.0, n_samples)
    density      = 1000 + 250*C_feed + 0.05*RPM + rng.normal(0, 3, n_samples)

    X = pd.DataFrame({
        "T_jacket": T_jacket, "F_feed": F_feed, "C_feed": C_feed,
        "RPM": RPM, "T_feed": T_feed, "tau": tau,
        "conductivity": conductivity, "dT": dT, "density": density,
    })
    y = pd.Series(L50, name="L50_um")
    return X, y


# =============================================================================
# 2. FIRST-PRINCIPLES MODEL (RAW inputs)
# =============================================================================
def first_principles_L50(X_raw_tensor):
    """MSMPR physics. RAW units: T_jacket (K), C_feed (kg/kg), tau (min)."""
    T_jacket = X_raw_tensor[:, 0]
    C_feed   = X_raw_tensor[:, 2]
    tau      = X_raw_tensor[:, 5]

    k_g   = 2.0e-8
    C_sat = 0.20

    S = torch.clamp(C_feed - C_sat * (1.0 + 0.02 * (T_jacket - 310.0)), min=1e-4)
    G = k_g * S**0.8 * (1.0 + 0.03 * (T_jacket - 310.0))
    L50 = 3.0 * G * (tau * 60.0) * 1e6
    return L50


# =============================================================================
# 3. GREY-BOX NETWORK
# =============================================================================
class GreyBoxModel(nn.Module):
    """L50 = L50_physics(X_raw) * (1 + ANN_correction(X_scaled))"""
    def __init__(self, input_dim, hidden=(32, 16), use_multiplicative=True):
        super().__init__()
        self.use_multiplicative = use_multiplicative

        layers = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.ann = nn.Sequential(*layers)

        for m in self.ann.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)
                nn.init.normal_(m.weight, std=0.01)

    def forward(self, x_raw, x_scaled):
        L50_phys = first_principles_L50(x_raw)
        delta    = self.ann(x_scaled)

        if self.use_multiplicative:
            L50 = L50_phys.unsqueeze(-1) * (1.0 + delta)
        else:
            L50 = L50_phys.unsqueeze(-1) + delta

        return L50.squeeze(-1), L50_phys, delta.squeeze(-1)


# =============================================================================
# 4. TRAINING
# =============================================================================
def train_greybox(X_train_raw, X_train_scaled, y_train,
                  X_val_raw,   X_val_scaled,   y_val,
                  input_dim, hidden=(32, 16), epochs=2000, lr=1e-3,
                  use_multiplicative=True, lambda_phys=0.0, verbose=True):

    Xr_tr = torch.tensor(X_train_raw,    dtype=torch.float32)
    Xs_tr = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_tr  = torch.tensor(y_train,        dtype=torch.float32)
    Xr_vl = torch.tensor(X_val_raw,      dtype=torch.float32)
    Xs_vl = torch.tensor(X_val_scaled,   dtype=torch.float32)
    y_vl  = torch.tensor(y_val,          dtype=torch.float32)

    model = GreyBoxModel(input_dim, hidden=hidden,
                         use_multiplicative=use_multiplicative)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    train_losses, val_losses = [], []
    best_val, best_state = np.inf, None

    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        y_pred, _, delta = model(Xr_tr, Xs_tr)
        mse = loss_fn(y_pred, y_tr)
        phys_reg = lambda_phys * (delta ** 2).mean()
        (mse + phys_reg).backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            y_val_pred, _, _ = model(Xr_vl, Xs_vl)
            v_loss = loss_fn(y_val_pred, y_vl).item()

        train_losses.append(mse.item())
        val_losses.append(v_loss)

        if v_loss < best_val:
            best_val = v_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if verbose and (ep + 1) % 200 == 0:
            print(f"  epoch {ep+1:4d} | train MSE = {mse.item():8.3f} "
                  f"| val MSE = {v_loss:8.3f}")

    model.load_state_dict(best_state)
    model.eval()
    return model, train_losses, val_losses


# =============================================================================
# 5. METRICS
# =============================================================================
def report(name, y_true, y_pred):
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    print(f"{name:<30s} | R2={r2:7.4f} | RMSE={rmse:7.3f} um | MAE={mae:7.3f} um")
    return dict(r2=r2, rmse=rmse, mae=mae)


# =============================================================================
# 6. MAIN
# =============================================================================
def main():
    print("=" * 78)
    print("  GREY-BOX SOFT SENSOR - Physics inside the neural network")
    print("=" * 78)

    X_raw, y = simulate_true_process(n_samples=3000, seed=42)
    print(f"\n[data] shape={X_raw.shape}, "
          f"target mean={y.mean():.2f}, std={y.std():.2f}")

    X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
        X_raw, y.values, test_size=0.25, random_state=0)
    X_tr_raw, X_vl_raw, y_tr, y_vl = train_test_split(
        X_tr_raw, y_tr, test_size=0.15, random_state=1)

    scaler = StandardScaler().fit(X_tr_raw)

    X_tr_raw_np = X_tr_raw.values.astype(np.float32)
    X_vl_raw_np = X_vl_raw.values.astype(np.float32)
    X_te_raw_np = X_te_raw.values.astype(np.float32)

    X_tr_sc = scaler.transform(X_tr_raw).astype(np.float32)
    X_vl_sc = scaler.transform(X_vl_raw).astype(np.float32)
    X_te_sc = scaler.transform(X_te_raw).astype(np.float32)

    print(f"[split] train={X_tr_raw_np.shape}, val={X_vl_raw_np.shape}, "
          f"test={X_te_raw_np.shape}")

    # Physics baseline
    print("\n--- Baseline: physics only (RAW inputs) ---")
    L50_phys_te = first_principles_L50(
        torch.tensor(X_te_raw_np, dtype=torch.float32)
    ).detach().numpy()
    report("Physics only", y_te, L50_phys_te)
    print(f"  Physics prediction stats: "
          f"mean={L50_phys_te.mean():.2f}, std={L50_phys_te.std():.2f}")

    # Multiplicative grey-box
    print("\n--- GREY-BOX: L50 = physics * (1 + ANN(X)) ---")
    model, tr_loss, vl_loss = train_greybox(
        X_tr_raw_np, X_tr_sc, y_tr,
        X_vl_raw_np, X_vl_sc, y_vl,
        input_dim=X_tr_raw_np.shape[1],
        hidden=(32, 16), epochs=2000, lr=1e-3,
        use_multiplicative=True, lambda_phys=0.0, verbose=True)

    with torch.no_grad():
        y_gb_pred, _, delta_te = model(
            torch.tensor(X_te_raw_np, dtype=torch.float32),
            torch.tensor(X_te_sc,     dtype=torch.float32),
        )
        y_gb_pred = y_gb_pred.numpy()
        delta_te  = delta_te.numpy()

    report("GREY-BOX (multiplicative)", y_te, y_gb_pred)
    print(f"  ANN correction stats: mean={delta_te.mean():+.4f}, "
          f"std={delta_te.std():.4f}, "
          f"range=[{delta_te.min():+.4f}, {delta_te.max():+.4f}]")

    # Additive grey-box
    print("\n--- GREY-BOX: L50 = physics + ANN(X) ---")
    model_add, _, _ = train_greybox(
        X_tr_raw_np, X_tr_sc, y_tr,
        X_vl_raw_np, X_vl_sc, y_vl,
        input_dim=X_tr_raw_np.shape[1],
        hidden=(32, 16), epochs=2000, lr=1e-3,
        use_multiplicative=False, lambda_phys=0.0, verbose=False)

    with torch.no_grad():
        y_add_pred, _, _ = model_add(
            torch.tensor(X_te_raw_np, dtype=torch.float32),
            torch.tensor(X_te_sc,     dtype=torch.float32),
        )
        y_add_pred = y_add_pred.numpy()

    report("GREY-BOX (additive)", y_te, y_add_pred)

    # Summary
    print("\n" + "=" * 78)
    print("  SUMMARY")
    print("=" * 78)
    print(f"{'Model':<30s} | {'R2':>8s} | {'RMSE (um)':>10s} | {'MAE (um)':>9s}")
    print("-" * 78)
    for name, pred in [
        ("Physics only",          L50_phys_te),
        ("Grey-box (multiplic.)", y_gb_pred),
        ("Grey-box (additive)",   y_add_pred),
    ]:
        r2   = r2_score(y_te, pred)
        rmse = np.sqrt(mean_squared_error(y_te, pred))
        mae  = mean_absolute_error(y_te, pred)
        print(f"{name:<30s} | {r2:8.4f} | {rmse:10.3f} | {mae:9.3f}")
    print("=" * 78)

    # Plots
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))

    ax[0,0].semilogy(tr_loss, label="Train")
    ax[0,0].semilogy(vl_loss, label="Val")
    ax[0,0].set(xlabel="Epoch", ylabel="MSE loss (log)",
                title="Grey-box training curve")
    ax[0,0].legend(); ax[0,0].grid(alpha=0.3)

    lims = [y_te.min(), y_te.max()]
    ax[0,1].scatter(y_te, L50_phys_te, s=8, alpha=0.4,
                    label="Physics", color="tab:blue")
    ax[0,1].scatter(y_te, y_gb_pred,   s=8, alpha=0.6,
                    label="Grey-box", color="tab:green")
    ax[0,1].plot(lims, lims, 'r--', lw=1.5)
    ax[0,1].set(xlabel="Measured L50 (um)", ylabel="Predicted",
                title="Parity: physics vs grey-box")
    ax[0,1].legend(); ax[0,1].grid(alpha=0.3)

    ax[1,0].hist(y_te - L50_phys_te, bins=40, alpha=0.5,
                 label="Physics", color="tab:blue")
    ax[1,0].hist(y_te - y_gb_pred,   bins=40, alpha=0.6,
                 label="Grey-box", color="tab:green")
    ax[1,0].set(xlabel="Residual (um)", ylabel="Count", title="Residuals")
    ax[1,0].legend(); ax[1,0].grid(alpha=0.3)

    true_resid = y_te - L50_phys_te
    predicted_correction = delta_te * L50_phys_te
    ax[1,1].scatter(true_resid, predicted_correction, s=8, alpha=0.5,
                    color="tab:green")
    lims2 = [min(true_resid.min(), predicted_correction.min()),
             max(true_resid.max(), predicted_correction.max())]
    ax[1,1].plot(lims2, lims2, 'r--', lw=1.5)
    ax[1,1].set(xlabel="True physics error (um)",
                ylabel="ANN-predicted correction (um)",
                title="Does the ANN correct the missing physics?")
    ax[1,1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("greybox_results.png", dpi=150)
    print("\n[plot] saved to greybox_results.png")


if __name__ == "__main__":
    main()