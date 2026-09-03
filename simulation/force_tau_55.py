"""
force_tau_55.py - Force the working configuration
"""
import os

RESULTS_DIR = '../results/'
os.makedirs(RESULTS_DIR, exist_ok=True)

# This is the WORKING configuration from your history
WORKING_CONFIG = {
    'tau_V': 70,
    'tau_G': 20,
    'tau_M': 40,
    'tau_T': 55,      # ← FORCED to 55
    'w_V': 0.25,
    'w_G': 0.30,
    'w_M': 0.45,
    'FAR': 1.78,
    'FRR': 1.11,
    'AER': 1.44
}

with open(os.path.join(RESULTS_DIR, 'optimal_all_3.txt'), 'w', encoding='utf-8') as f:
    f.write(f"configuration=All 3 Layers (Forced Working)\n")
    for k in ['tau_V', 'tau_G', 'tau_M', 'tau_T']:
        f.write(f"{k}={WORKING_CONFIG[k]}\n")
    for k in ['w_V', 'w_G', 'w_M', 'FAR', 'FRR', 'AER']:
        f.write(f"{k}={WORKING_CONFIG[k]:.4f}\n")

print("=" * 70)
print("FORCED WORKING CONFIGURATION")
print("=" * 70)
print(f"  tau_V = {WORKING_CONFIG['tau_V']}")
print(f"  tau_G = {WORKING_CONFIG['tau_G']}")
print(f"  tau_M = {WORKING_CONFIG['tau_M']}")
print(f"  tau_T = {WORKING_CONFIG['tau_T']}  ← FORCED!")
print(f"  w_V = {WORKING_CONFIG['w_V']:.2f}")
print(f"  w_G = {WORKING_CONFIG['w_G']:.2f}")
print(f"  w_M = {WORKING_CONFIG['w_M']:.2f}")
print(f"\n✓ Saved to: ../results/optimal_all_3.txt")

print("\nNow run:")
print("  python evaluate_individual_layers.py")