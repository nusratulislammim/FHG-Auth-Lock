"""
analyze_results.py - Statistical Analysis & Visualization
Generates LaTeX tables and figures for paper

UPDATED: Now supports 4-layer authentication:
- S_M: Mechanical (FHG)
- S_V: Voice (3 digits)
- S_G: Spatial Gap
- S_U: Spatial Position

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
    print(f"  Columns: {df.columns.tolist()}")
    print()
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
    """Generate Table III: Error Rates for 4-layer system"""
    print("=== Computing Error Rates ===\n")
    
    # Updated thresholds for hybrid fusion
    thresholds = {
        'S_M': 55,   # Mechanical (FHG)
        'S_V': 45,   # Voice
        'S_G': 50,   # Spatial Gap
        'S_U': 55    # Spatial Position
    }
    
    # Single-layer and combined configurations
    configs = [
        ('Voice only ($S_V$)', ['S_V']),
        ('Spatial Gap only ($S_G$)', ['S_G']),
        ('Mechanical only ($S_M$)', ['S_M']),
        ('Spatial Position only ($S_U$)', ['S_U']),
        ('Voice + Gap (AND)', ['S_V', 'S_G']),
        ('Voice + Mech. (AND)', ['S_V', 'S_M']),
        ('Voice + Spatial (AND)', ['S_V', 'S_U']),
        ('Gap + Mech. (AND)', ['S_G', 'S_M']),
        ('Gap + Spatial (AND)', ['S_G', 'S_U']),
        ('Mech. + Spatial (AND)', ['S_M', 'S_U']),
        ('Voice + Gap + Mech. (AND)', ['S_V', 'S_G', 'S_M']),
        ('Voice + Gap + Spatial (AND)', ['S_V', 'S_G', 'S_U']),
        ('Voice + Mech. + Spatial (AND)', ['S_V', 'S_M', 'S_U']),
        ('Gap + Mech. + Spatial (AND)', ['S_G', 'S_M', 'S_U']),
        ('\\textbf{All 4 layers (Hybrid)}', ['S_V', 'S_G', 'S_M', 'S_U']),
    ]
    
    results = []
    for config_name, score_cols in configs:
        res = compute_error_rates_from_csv(df, config_name, score_cols, thresholds)
        results.append(res)
        print(f"{config_name:35s}: FAR={res['FAR']:6.3f}% FRR={res['FRR']:5.1f}% Acc={res['Accuracy']:5.1f}%")
    
    print()
    
    # Generate LaTeX table
    latex = r"""\begin{table}[h]
\centering
\caption{Simulated Authentication Performance (4-Layer Hybrid System, N=8)}
\label{tab:results}
\scriptsize
\begin{tabular}{lrrrr}
\toprule
\textbf{Configuration} & \textbf{FAR (\%)} & \textbf{FRR (\%)} & \textbf{EER (\%)} & \textbf{Accuracy (\%)} \\
\midrule
"""
    
    for res in results:
        latex += f"{res['config']:35s} & {res['FAR']:5.1f} & {res['FRR']:5.1f} & {res['EER']:5.1f} & {res['Accuracy']:5.1f} \\\\\n"
        if res['config'] == 'Spatial Position only ($S_U$)' or res['config'] == 'Gap + Spatial (AND)':
            latex += r"\midrule" + "\n"
    
    num_impostor = len(df[df['genuine'] == False])
    num_genuine = len(df[df['genuine'] == True])
    final_result = results[-1]
    
    latex += r"""\bottomrule
\end{tabular}
"""
    latex += f"\\\\[2pt]\n\\scriptsize FAR: {final_result['FP']} false accepts / {num_impostor} impostor attempts.\\\\\n"
    latex += f"\\scriptsize FRR: {final_result['FN']} false rejects / {num_genuine} genuine attempts.\\\\\n"
    latex += r"""\scriptsize Hybrid fusion: Sequential + Weighted AND-Check. \\
\scriptsize Thresholds: $\tau_V=45, \tau_G=50, \tau_M=55, \tau_U=55$, weighted threshold $=58$.
\end{table}
"""
    
    with open(OUTPUT_TABLE_III, 'w') as f:
        f.write(latex)
    
    print(f"✅ Table III saved to {OUTPUT_TABLE_III}\n")
    return results

def generate_correlation_table(df):
    """Generate correlation matrix for all 4 layers"""
    print("=== Computing Score Correlations ===\n")
    
    scores = df[['S_V', 'S_G', 'S_M', 'S_U']]
    corr_matrix = scores.corr(method='pearson')
    
    print("Correlation Matrix:")
    print(corr_matrix.round(3))
    print()
    
    latex = r"""\begin{table}[h]
\centering
\caption{Per-Layer Score Correlation (Pearson $r$, N=280)}
\label{tab:correlation}
\scriptsize
\begin{tabular}{lrrrr}
\toprule
& \textbf{Voice ($S_V$)} & \textbf{Gap ($S_G$)} & \textbf{Mech. ($S_M$)} & \textbf{Spatial ($S_U$)} \\
\midrule
"""
    
    labels = ['Voice ($S_V$)', 'Gap ($S_G$)', 'Mech. ($S_M$)', 'Spatial ($S_U$)']
    for i, label in enumerate(labels):
        row = [label]
        for j in range(4):
            val = corr_matrix.iloc[i, j]
            row.append(f"{val:.2f}")
        latex += " & ".join(row) + " \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\\[2pt]
\scriptsize Low correlations ($|r| < 0.4$) indicate independent layers.
\end{table}
"""
    
    corr_file = os.path.join(RESULTS_DIR, "table_correlation.tex")
    with open(corr_file, 'w') as f:
        f.write(latex)
    
    print(f"✅ Correlation table saved to {corr_file}\n")
    return corr_matrix

def main():
    print("\n" + "=" * 60)
    print("HYBRID AUTHENTICATION SYSTEM - SIMULATION ANALYSIS")
    print("(4 Layers: Voice + Spatial Gap + Mechanical + Spatial)")
    print("=" * 60 + "\n")
    
    df = load_data()
    if df is None:
        print("ERROR: Could not load data. Please run generate_dataset.py first.")
        return
    
    results = generate_table_III(df)
    corr_matrix = generate_correlation_table(df)
    
    print("=" * 60)
    print("✅ Analysis complete!")
    print(f"✅ Results saved to {RESULTS_DIR}")
    print("=" * 60)
    
    # Print final summary
    final = results[-1]
    print(f"\n=== FINAL PERFORMANCE SUMMARY ===")
    print(f"Hybrid Fusion (Voice + Spatial Gap + Mechanical + Spatial Position)")
    print(f"  FAR:  {final['FAR']:.3f}%")
    print(f"  FRR:  {final['FRR']:.1f}%")
    print(f"  EER:  {final['EER']:.1f}%")
    print(f"  Acc:  {final['Accuracy']:.1f}%")

if __name__ == "__main__":
    main()