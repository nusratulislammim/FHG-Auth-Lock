"""
evaluate_test_complete.py - Final evaluation on held-out test data
"""
import os
import sys
import pandas as pd
import numpy as np

TEST_FILE = "../data/test_data.csv"
OPTIMAL_FILE = "../results/optimal_all_3.txt"
RESULTS_DIR = "../results"
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 75)
print("FINAL EVALUATION ON HELD-OUT TEST DATA")
print("=" * 75)

# Load test data
df_test = pd.read_csv(TEST_FILE)
if df_test['genuine'].dtype != bool:
    df_test['genuine'] = df_test['genuine'].astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False})

genuine = df_test[df_test['genuine'] == True]
impostor = df_test[df_test['genuine'] == False]
print(f"\nTest data: {len(df_test)} attempts")
print(f" Genuine: {len(genuine)}, Impostor: {len(impostor)}")

# Load optimal config
with open(OPTIMAL_FILE, 'r') as f:
    lines = f.readlines()
    tau_v = float(lines[1].split('=')[1].strip())
    tau_g = float(lines[2].split('=')[1].strip())
    tau_m = float(lines[3].split('=')[1].strip())
    tau_t = float(lines[4].split('=')[1].strip())
    w_v = float(lines[5].split('=')[1].strip())
    w_g = float(lines[6].split('=')[1].strip())
    w_m = float(lines[7].split('=')[1].strip())

print(f"\nOptimal config: τV={tau_v:.0f}, τG={tau_g:.0f}, τM={tau_m:.0f}, τT={tau_t:.0f}")
print(f"wV={w_v:.2f}, wG={w_g:.2f}, wM={w_m:.2f}")


def compute_metrics(thresh_v, thresh_g, thresh_m, thresh_t, w_v, w_g, w_m):
    genuine_screen = genuine.apply(lambda r: (r['S_V'] >= thresh_v) & (r['S_G'] >= thresh_g) & (r['S_M'] >= thresh_m), axis=1)
    impostor_screen = impostor.apply(lambda r: (r['S_V'] >= thresh_v) & (r['S_G'] >= thresh_g) & (r['S_M'] >= thresh_m), axis=1)
    
    genuine_accept = genuine[genuine_screen].apply(lambda r: (w_m*r['S_M'] + w_v*r['S_V'] + w_g*r['S_G']) >= thresh_t, axis=1).sum()
    impostor_accept = impostor[impostor_screen].apply(lambda r: (w_m*r['S_M'] + w_v*r['S_V'] + w_g*r['S_G']) >= thresh_t, axis=1).sum()
    
    FAR = 100 * impostor_accept / len(impostor)
    FRR = 100 * (len(genuine) - genuine_accept) / len(genuine)
    return FAR, FRR, int(genuine_accept), int(impostor_accept)

# Final evaluation
far, frr, tp, fp = compute_metrics(tau_v, tau_g, tau_m, tau_t, w_v, w_g, w_m)
tn = len(impostor) - fp
fn = len(genuine) - tp
acc = 100 * (tp + tn) / (len(genuine) + len(impostor))

print("\n=== FINAL EVALUATION ===")
print(f"{'Metric':<15} {'Value':<15}")
print("-" * 30)
print(f"{'FAR':<15} {far:.2f}%")
print(f"{'FRR':<15} {frr:.2f}%")
print(f"{'Accuracy':<15} {acc:.2f}%")
print(f"\nConfusion Matrix:")
print(f"TP: {tp}, FN: {fn}")
print(f"FP: {fp}, TN: {tn}")
print(f"Total correct: {tp + tn} / {len(df_test)}")

# True EER
eer_results = []
for tau_t_sweep in range(35, 85, 1):
    far_s, frr_s, _, _ = compute_metrics(tau_v, tau_g, tau_m, tau_t_sweep, w_v, w_g, w_m)
    eer_results.append({'threshold': tau_t_sweep, 'FAR': far_s, 'FRR': frr_s, 'diff': abs(far_s - frr_s)})

eer_df = pd.DataFrame(eer_results)
eer_point = eer_df.loc[eer_df['diff'].idxmin()]
eer = (eer_point['FAR'] + eer_point['FRR']) / 2

print(f"\n=== TRUE EER ===")
print(f"EER = {eer:.2f}% (at τ={eer_point['threshold']:.0f})")

# Save results
with open(os.path.join(RESULTS_DIR, "final_test_results.txt"), 'w') as f:
    f.write(f"FAR={far:.2f}\nFRR={frr:.2f}\nEER={eer:.2f}\nAccuracy={acc:.2f}\n")
    f.write(f"TP={tp}\nFN={fn}\nFP={fp}\nTN={tn}\n")
    f.write(f"tau_V={tau_v:.0f}\ntau_G={tau_g:.0f}\ntau_M={tau_m:.0f}\ntau_T={tau_t:.0f}\n")
    f.write(f"w_V={w_v:.2f}\nw_G={w_g:.2f}\nw_M={w_m:.2f}\n")

print(f"\n✓ Results saved to: {os.path.join(RESULTS_DIR, 'final_test_results.txt')}")