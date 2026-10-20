"""
test.py — quick end-to-end verification of the soft sensor project.
"""
import numpy as np
from main import (simulate_crystallizer, preprocess_pipeline,
                  train_pls, train_gpr, pls_vip, report)

print("=" * 60)
print("SOFT SENSOR - QUICK TEST")
print("=" * 60)

# 1. Simulator
X, y = simulate_crystallizer(n_samples=800, seed=0)
assert X.shape == (830, 9)
assert y.mean() > 10 and y.std() > 5
print(f"[TEST 1] Simulator OK — X{X.shape}, y_mean={y.mean():.1f}, y_std={y.std():.1f}")

# 2. Preprocess
d = preprocess_pipeline(X, y)
assert d["X_train"].shape[0] > 400
assert d["X_test"].shape[0] > 100
print(f"[TEST 2] Preprocessing OK — train={d['X_train'].shape}, test={d['X_test'].shape}")

# 3. PLS
pls = train_pls(d["X_train"], d["y_train"])
y_pls = pls.predict(d["X_test"]).ravel()
r2_pls = report("PLS", d["y_test"], y_pls)
assert r2_pls["r2"] > 0.5
print(f"[TEST 3] PLS OK — R2={r2_pls['r2']:.3f}")

# 4. GPR
gpr = train_gpr(d["X_train"], d["y_train"], max_samples=200)
y_gpr, s_gpr = gpr.predict(d["X_test"], return_std=True)
r2_gpr = report("GPR", d["y_test"], y_gpr, y_std=s_gpr)
assert r2_gpr["r2"] > 0.5
print(f"[TEST 4] GPR OK — R2={r2_gpr['r2']:.3f}")

# 5. VIP
vip = pls_vip(pls, d["feature_names"])
top = list(vip.keys())[0]
print(f"[TEST 5] VIP OK — top feature = {top}")

print("\n" + "=" * 60)
print("ALL TESTS PASSED ✅")
print("=" * 60)