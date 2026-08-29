"""
analyze_results.py - Statistical Analysis & Visualization
Generates LaTeX tables and figures for paper

Authors: Nusratul Islam Mim, Firuze Tasnim Sneha, Al Musabbir, Dr. Md. Sujan Ali
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================
# FILE PATHS
# ============================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(SCRIPT_DIR, "..", "data", "experimental_data_N8.csv")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

OUTPUT_TABLE_III = os.path.join(RESULTS_DIR, "table_III_error_rates.tex")

def load_data():
    """Load experimental dataset"""
    if not os.path.exists(INPUT_CSV):
        print(f"ERROR: File not found at {INPUT_CSV}")
        return None
    
    df = pd.read_csv(INPUT_CSV)
    print(f"✅ Loaded {len(df)} authentication attempts")
    print(f"  Genuine: {df['genuine'].sum()}")
    print(f"  Impostor: {(~df['genuine']).sum()}\n")
    return df

def compute_error_rates_from_csv(df, config_name, score_cols, thresholds):
    """Compute FAR, FRR directly from CSV data"""
    genuine = df[df['genuine'] == True]
    impostor = df[df['genuine'] == False]
    
    if len(score_cols) == 1:
        col = score_cols[0]
        genuine_accept = (genuine[col] > thresholds[col]).sum()
        impostor_accept = (impostor[col] > thresholds[col]).sum()
    else:
        genuine_accept = genuine.apply(
            lambda row: all(row[col] > thresholds[col] for col in score_cols),
            axis=1
        ).sum()
        impostor_accept = impostor.apply(
            lambda row: all(row[col] > thresholds[col] for col in score_cols),
            axis=1
        ).sum()
    
    num_genuine = len(genuine)
    num_impostor = len(impostor)
    
    FAR = 100 * impostor_accept / num_impostor if num_impostor > 0 else 0
    FRR = 100 * (num_genuine - genuine_accept) / num_genuine if num_genuine > 0 else 0
    Accuracy = 100 * (genuine_accept + (num_impostor - impostor_accept)) / (num_genuine + num_impostor)
    EER = (FAR + FRR) / 2
    
    return {
        'config': config_name,
        'FAR': FAR,
        'FRR': FRR,
        'EER': EER,
        'Accuracy': Accuracy,
        'TP': int(genuine_accept),
        'FN': int(num_genuine - genuine_accept),
        'TN': int(num_impostor - impostor_accept),
        'FP': int(impostor_accept)
    }

def generate_table_III(df):
    """Generate Table III: Error Rates"""
    print("=== Computing Error Rates ===\n")
    
    
    thresholds = {'S_M': 70, 'S_A': 50, 'S_U': 70}
    
    configs = [
        ('Mechanical only ($S_M$)', ['S_M']),
        ('Acoustic only ($S_A$)', ['S_A']),
        ('Ultrasonic only ($S_U$)', ['S_U']),
        ('Mech. + Acous. (AND)', ['S_M', 'S_A']),
        ('Mech. + Ultra. (AND)', ['S_M', 'S_U']),
        ('Acous. + Ultra. (AND)', ['S_A', 'S_U']),
        ('\\textbf{All three (AND)}', ['S_M', 'S_A', 'S_U']),
    ]
    
    results = []
    for config_name, score_cols in configs:
        res = compute_error_rates_from_csv(df, config_name, score_cols, thresholds)
        results.append(res)
        print(f"{config_name:30s}: FAR={res['FAR']:6.3f}% FRR={res['FRR']:5.1f}% Acc={res['Accuracy']:5.1f}%")
    
    print()
    
    latex = r"""\begin{table}[h]
\centering
\caption{Measured Authentication Performance (Simulation, N=8)}
\label{tab:results}
\scriptsize
\begin{tabular}{lrrrr}
\toprule
\textbf{Configuration} & \textbf{FAR (\%)} & \textbf{FRR (\%)} & \textbf{EER (\%)} & \textbf{Accuracy (\%)} \\
\midrule
"""
    
    for res in results:
        latex += f"{res['config']:30s} & {res['FAR']:5.1f} & {res['FRR']:5.1f} & {res['EER']:5.1f} & {res['Accuracy']:5.1f} \\\\\n"
        if res['config'] == 'Ultrasonic only ($S_U$)' or res['config'] == 'Acous. + Ultra. (AND)':
            latex += r"\midrule" + "\n"
    
    num_impostor = len(df[df['genuine'] == False])
    num_genuine = len(df[df['genuine'] == True])
    final_result = results[-1]
    
    latex += r"""\bottomrule
\end{tabular}
"""
    latex += f"\\\\[2pt]\n\\scriptsize FAR: {final_result['FP']} false accepts / {num_impostor} impostor attempts.\\\\\n"
    latex += f"\\scriptsize FRR: {final_result['FN']} false rejects / {num_genuine} genuine attempts.\\\\\n"
    latex += r"""\scriptsize Simulation using empirically-derived sensor noise models.
\end{table}
"""
    
    with open(OUTPUT_TABLE_III, 'w') as f:
        f.write(latex)
    
    print(f"✅ Table III saved to {OUTPUT_TABLE_III}\n")
    return results

def main():
    print("\n" + "=" * 60)
    print("FHG AUTHENTICATION SYSTEM - SIMULATION ANALYSIS")
    print("=" * 60 + "\n")
    
    df = load_data()
    if df is None:
        print("ERROR: Could not load data. Please run generate_dataset.py first.")
        return
    
    results = generate_table_III(df)
    print("✅ Analysis complete!")
    print(f"✅ Results saved to {RESULTS_DIR}")

if __name__ == "__main__":
    main()