"""
optimize_complete.py - Training-only optimization with ablations
Optimizes: Voice+Gap, Voice+Mech, Gap+Mech, All 3
"""
import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm

TRAIN_FILE = "../data/train_data.csv"
RESULTS_DIR = "../results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Search ranges
TAU_V_RANGE = range(20, 60, 2)
TAU_G_RANGE = range(20, 60, 2)
TAU_M_RANGE = range(40, 80, 2)
TAU_T_RANGE = range(40, 76, 2)
WEIGHT_RANGE = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


def load_training_data():
    df_train = pd.read_csv(TRAIN_FILE)
    if df_train["genuine"].dtype != bool:
        df_train["genuine"] = df_train["genuine"].astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False})
    
    genuine = df_train[df_train["genuine"] == True]
    impostor = df_train[df_train["genuine"] == False]
    
    print("=" * 75)
    print("TRAINING-ONLY HYBRID AUTHENTICATION OPTIMIZATION")
    print("=" * 75)
    print(f"\nTraining attempts: {len(df_train)}")
    print(f"Genuine: {len(genuine)}, Impostor: {len(impostor)}")
    
    return genuine, impostor


def decision_vg(s_v, s_g, tau_v, tau_g, tau_t, w_v, w_g):
    return (s_v >= tau_v) & (s_g >= tau_g) & (w_v * s_v + w_g * s_g >= tau_t)


def decision_vm(s_v, s_m, tau_v, tau_m, tau_t, w_v, w_m):
    return (s_v >= tau_v) & (s_m >= tau_m) & (w_v * s_v + w_m * s_m >= tau_t)


def decision_gm(s_g, s_m, tau_g, tau_m, tau_t, w_g, w_m):
    return (s_g >= tau_g) & (s_m >= tau_m) & (w_g * s_g + w_m * s_m >= tau_t)


def decision_all(s_v, s_g, s_m, tau_v, tau_g, tau_m, tau_t, w_v, w_g, w_m):
    return (s_v >= tau_v) & (s_g >= tau_g) & (s_m >= tau_m) & (w_v * s_v + w_g * s_g + w_m * s_m >= tau_t)


def calculate_metrics(genuine_accept, impostor_accept, n_genuine, n_impostor):
    far = 100.0 * impostor_accept / n_impostor if n_impostor > 0 else 0.0
    frr = 100.0 * (n_genuine - genuine_accept) / n_genuine if n_genuine > 0 else 0.0
    aer = (far + frr) / 2.0
    return far, frr, aer


def optimize_vg(g, i, n_g, n_i):
    results = []
    for tau_v in TAU_V_RANGE:
        for tau_g in TAU_G_RANGE:
            for tau_t in TAU_T_RANGE:
                for w_v in WEIGHT_RANGE:
                    w_g = 1.0 - w_v
                    if w_g < 0.05:
                        continue
                    g_pass = decision_vg(g['S_V'].values, g['S_G'].values, tau_v, tau_g, tau_t, w_v, w_g)
                    i_pass = decision_vg(i['S_V'].values, i['S_G'].values, tau_v, tau_g, tau_t, w_v, w_g)
                    far, frr, aer = calculate_metrics(np.sum(g_pass), np.sum(i_pass), n_g, n_i)
                    results.append({"tau_V": tau_v, "tau_G": tau_g, "tau_T": tau_t, "w_V": round(w_v, 2), "w_G": round(w_g, 2), "FAR": far, "FRR": frr, "AER": aer})
    return pd.DataFrame(results)


def optimize_vm(g, i, n_g, n_i):
    results = []
    for tau_v in TAU_V_RANGE:
        for tau_m in TAU_M_RANGE:
            for tau_t in TAU_T_RANGE:
                for w_v in WEIGHT_RANGE:
                    w_m = 1.0 - w_v
                    if w_m < 0.05:
                        continue
                    g_pass = decision_vm(g['S_V'].values, g['S_M'].values, tau_v, tau_m, tau_t, w_v, w_m)
                    i_pass = decision_vm(i['S_V'].values, i['S_M'].values, tau_v, tau_m, tau_t, w_v, w_m)
                    far, frr, aer = calculate_metrics(np.sum(g_pass), np.sum(i_pass), n_g, n_i)
                    results.append({"tau_V": tau_v, "tau_M": tau_m, "tau_T": tau_t, "w_V": round(w_v, 2), "w_M": round(w_m, 2), "FAR": far, "FRR": frr, "AER": aer})
    return pd.DataFrame(results)


def optimize_gm(g, i, n_g, n_i):
    results = []
    for tau_g in TAU_G_RANGE:
        for tau_m in TAU_M_RANGE:
            for tau_t in TAU_T_RANGE:
                for w_g in WEIGHT_RANGE:
                    w_m = 1.0 - w_g
                    if w_m < 0.05:
                        continue
                    g_pass = decision_gm(g['S_G'].values, g['S_M'].values, tau_g, tau_m, tau_t, w_g, w_m)
                    i_pass = decision_gm(i['S_G'].values, i['S_M'].values, tau_g, tau_m, tau_t, w_g, w_m)
                    far, frr, aer = calculate_metrics(np.sum(g_pass), np.sum(i_pass), n_g, n_i)
                    results.append({"tau_G": tau_g, "tau_M": tau_m, "tau_T": tau_t, "w_G": round(w_g, 2), "w_M": round(w_m, 2), "FAR": far, "FRR": frr, "AER": aer})
    return pd.DataFrame(results)


def optimize_all(g, i, n_g, n_i):
    results = []
    for tau_v in TAU_V_RANGE:
        for tau_g in TAU_G_RANGE:
            for tau_m in TAU_M_RANGE:
                for tau_t in TAU_T_RANGE:
                    for w_v in WEIGHT_RANGE:
                        for w_g in WEIGHT_RANGE:
                            w_m = 1.0 - w_v - w_g
                            if w_m < 0.05 or w_m > 0.95:
                                continue
                            g_pass = decision_all(g['S_V'].values, g['S_G'].values, g['S_M'].values, tau_v, tau_g, tau_m, tau_t, w_v, w_g, w_m)
                            i_pass = decision_all(i['S_V'].values, i['S_G'].values, i['S_M'].values, tau_v, tau_g, tau_m, tau_t, w_v, w_g, w_m)
                            far, frr, aer = calculate_metrics(np.sum(g_pass), np.sum(i_pass), n_g, n_i)
                            results.append({"tau_V": tau_v, "tau_G": tau_g, "tau_M": tau_m, "tau_T": tau_t, "w_V": round(w_v, 2), "w_G": round(w_g, 2), "w_M": round(w_m, 2), "FAR": far, "FRR": frr, "AER": aer})
    return pd.DataFrame(results)


def save_best(name, df_results):
    df_sorted = df_results.sort_values(by=["AER", "FAR", "FRR"], ascending=[True, True, True])
    best = df_sorted.iloc[0]
    print("\n" + "-" * 75)
    print(f"BEST TRAINING CONFIGURATION: {name}")
    print("-" * 75)
    if not pd.isna(best['tau_V']): print(f"  τV = {int(best['tau_V'])}")
    if not pd.isna(best['tau_G']): print(f"  τG = {int(best['tau_G'])}")
    if not pd.isna(best['tau_M']): print(f"  τM = {int(best['tau_M'])}")
    print(f"  τT = {int(best['tau_T'])}")
    if not pd.isna(best['w_V']): print(f"  wV = {best['w_V']:.2f}")
    if not pd.isna(best['w_G']): print(f"  wG = {best['w_G']:.2f}")
    if not pd.isna(best['w_M']): print(f"  wM = {best['w_M']:.2f}")
    print(f"\n  FAR = {best['FAR']:.2f}%")
    print(f"  FRR = {best['FRR']:.2f}%")
    print(f"  AER = {best['AER']:.2f}%")
    return best.to_dict()


def main():
    genuine, impostor = load_training_data()
    n_g = len(genuine)
    n_i = len(impostor)
    
    configs = [
        ("Voice + Gap", optimize_vg),
        ("Voice + Mechanical", optimize_vm),
        ("Gap + Mechanical", optimize_gm),
        ("All 3", optimize_all)
    ]
    
    results_summary = []
    for name, func in configs:
        print(f"\n{'='*75}\nOPTIMIZING: {name}\n{'='*75}")
        df_results = func(genuine, impostor, n_g, n_i)
        print(f"  ✓ Evaluated {len(df_results)} configurations")
        best = save_best(name, df_results)
        results_summary.append({"Configuration": name, **best})
    
    summary_df = pd.DataFrame(results_summary)
    print("\n" + "=" * 75)
    print("TRAINING OPTIMIZATION SUMMARY")
    print("=" * 75)
    print(summary_df[['Configuration', 'FAR', 'FRR', 'AER']].to_string(index=False))
    
    # Save best All 3 config
    best_all = summary_df[summary_df['Configuration'] == 'All 3'].iloc[0]
    with open(os.path.join(RESULTS_DIR, "optimal_all_3.txt"), 'w') as f:
        f.write(f"configuration=All 3\n")
        f.write(f"tau_V={int(best_all['tau_V'])}\n")
        f.write(f"tau_G={int(best_all['tau_G'])}\n")
        f.write(f"tau_M={int(best_all['tau_M'])}\n")
        f.write(f"tau_T={int(best_all['tau_T'])}\n")
        f.write(f"w_V={best_all['w_V']:.2f}\n")
        f.write(f"w_G={best_all['w_G']:.2f}\n")
        f.write(f"w_M={best_all['w_M']:.2f}\n")
        f.write(f"FAR={best_all['FAR']:.4f}\n")
        f.write(f"FRR={best_all['FRR']:.4f}\n")
        f.write(f"AER={best_all['AER']:.4f}\n")
    
    print(f"\n✓ Optimal config saved to: {os.path.join(RESULTS_DIR, 'optimal_all_3.txt')}")
    print("\nIMPORTANT: AER = (FAR+FRR)/2 is NOT EER. True EER from test set.")


if __name__ == "__main__":
    main()