"""
optimize_constrained_fast.py - Fast constrained optimization
Security constraints: Mechanical PRIMARY, Voice CAPPED
"""
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# ============================================================
# CONFIGURATION
# ============================================================
TRAIN_FILE = '../data/train_data.csv'
RESULTS_DIR = '../results/'

# Security constraints
W_M_MIN = 0.35  # Mechanical minimum
W_G_MIN = 0.25  # Gap minimum
W_V_MAX = 0.25  # Voice max (replay vulnerability)

os.makedirs(RESULTS_DIR, exist_ok=True)

print("\n" + "=" * 70)
print("CONSTRAINED OPTIMIZATION (FAST)")
print("=" * 70)
print(f"Constraints: w_M >= {W_M_MIN}, w_G >= {W_G_MIN}, w_V <= {W_V_MAX}\n")

# Load training data
df = pd.read_csv(TRAIN_FILE)
if df['genuine'].dtype != bool:
    df['genuine'] = df['genuine'].astype(str).str.lower().map({
        'true': True, 'false': False, '1': True, '0': False
    })

genuine = df[df['genuine'] == True]
impostor = df[df['genuine'] == False]

print(f"Training: {len(genuine)} genuine, {len(impostor)} impostor\n")

# Numpy arrays for speed
g_sv = genuine['S_V'].values
g_sg = genuine['S_G'].values
g_sm = genuine['S_M'].values
i_sv = impostor['S_V'].values
i_sg = impostor['S_G'].values
i_sm = impostor['S_M'].values

n_genuine = len(genuine)
n_impostor = len(impostor)

# Coarse grid for speed
# Even finer grid to find the sweet spot
TAU_V_RANGE = range(40, 65, 2)   # Tighter: 40-64, step 2
TAU_G_RANGE = range(20, 45, 3)   # Keep as is
TAU_M_RANGE = range(25, 50, 3)   # Lower mechanical range
TAU_T_RANGE = range(40, 60, 2) 

WEIGHT_RANGE = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

# Valid weight combinations
valid_weights = []
for w_v in WEIGHT_RANGE:
    if w_v > W_V_MAX:
        continue
    for w_g in WEIGHT_RANGE:
        if w_g < W_G_MIN:
            continue
        w_m = 1.0 - w_v - w_g
        if W_M_MIN <= w_m <= 0.95:
            valid_weights.append((w_v, w_g, w_m))

total = len(valid_weights) * len(TAU_V_RANGE) * len(TAU_G_RANGE) * len(TAU_M_RANGE) * len(TAU_T_RANGE)
print(f"Configurations: {total}\n")

best_aer = float('inf')
best_config = None

with tqdm(total=total, desc="Optimizing", unit="cfg") as pbar:
    for tau_v in TAU_V_RANGE:
        g_pass_v = g_sv >= tau_v
        i_pass_v = i_sv >= tau_v

        for tau_g in TAU_G_RANGE:
            g_pass_g = g_sg >= tau_g
            i_pass_g = i_sg >= tau_g

            for tau_m in TAU_M_RANGE:
                g_pass_m = g_sm >= tau_m
                i_pass_m = i_sm >= tau_m

                g_screen = g_pass_v & g_pass_g & g_pass_m
                i_screen = i_pass_v & i_pass_g & i_pass_m

                for tau_t in TAU_T_RANGE:
                    for w_v, w_g, w_m in valid_weights:
                        g_fusion = w_m*g_sm[g_screen] + w_v*g_sv[g_screen] + w_g*g_sg[g_screen]
                        i_fusion = w_m*i_sm[i_screen] + w_v*i_sv[i_screen] + w_g*i_sg[i_screen]

                        gen_accept = np.sum(g_fusion >= tau_t)
                        imp_accept = np.sum(i_fusion >= tau_t)

                        far = 100 * imp_accept / n_impostor
                        frr = 100 * (n_genuine - gen_accept) / n_genuine
                        aer = (far + frr) / 2

                        if aer < best_aer:
                            best_aer = aer
                            best_config = {
                                'tau_V': tau_v, 'tau_G': tau_g, 'tau_M': tau_m, 'tau_T': tau_t,
                                'w_V': w_v, 'w_G': w_g, 'w_M': w_m,
                                'FAR': far, 'FRR': frr, 'AER': aer
                            }
                        pbar.update(1)

print("\n" + "=" * 70)
print("BEST CONFIGURATION")
print("=" * 70)
for k, v in best_config.items():
    print(f" {k} = {v:.2f}" if isinstance(v, float) else f" {k} = {v}")

# Save
with open(os.path.join(RESULTS_DIR, 'optimal_all_3.txt'), 'w', encoding='utf-8') as f:
    f.write(f"configuration=All 3 Layers (Constrained)\n")
    for k in ['tau_V', 'tau_G', 'tau_M', 'tau_T']:
        f.write(f"{k}={best_config[k]}\n")
    for k in ['w_V', 'w_G', 'w_M', 'FAR', 'FRR', 'AER']:
        f.write(f"{k}={best_config[k]:.4f}\n")
print(f"\nSaved to: ../results/optimal_all_3.txt\n")