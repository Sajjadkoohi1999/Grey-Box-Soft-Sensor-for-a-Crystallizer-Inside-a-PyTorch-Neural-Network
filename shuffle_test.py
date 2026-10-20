"""
shuffle_test.py — negative control.
Shuffling y should destroy any real signal → R² drops to ~0.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from main import simulate_crystallizer, preprocess_pipeline, train_pls

# --- 1. Real data ---
X, y = simulate_crystallizer(n_samples=2000, seed=42)
d = preprocess_pipeline(X, y)
pls = train_pls(d["X_train"], d["y_train"])
r2_real = r2_score(d["y_test"], pls.predict(d["X_test"]).ravel())
print(f"\nReal data:        PLS R2 = {r2_real:.4f}  (should be ~0.91)")

# --- 2. Shuffled data (negative control) ---
y_shuffled = pd.Series(
    np.random.default_rng(0).permutation(y.values),
    name="L50_um"
)
d_shuf = preprocess_pipeline(X, y_shuffled)
pls_shuf = train_pls(d_shuf["X_train"], d_shuf["y_train"])
r2_shuf = r2_score(d_shuf["y_test"], pls_shuf.predict(d_shuf["X_test"]).ravel())
print(f"Shuffled data:    PLS R2 = {r2_shuf:.4f}  (should be ~0.00)")

# --- 3. Verdict ---
print()
if r2_real > 0.85 and r2_shuf < 0.2:
    print("✅ PASS — model learned real physics; no leakage.")
elif r2_shuf > 0.5:
    print("❌ FAIL — leakage detected! Investigate pipeline.")
else:
    print("⚠️ UNCLEAR — check the numbers.")