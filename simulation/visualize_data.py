"""
visualize_data.py - Generate publication-quality visualizations
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

TEST_DATA_FILE = '../data/test_data.csv'
RESULTS_DIR = '../results/'
os.makedirs(RESULTS_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-darkgrid')
COLORS = {'genuine': '#2ecc71', 'impostor': '#e74c3c', 'threshold': '#3498db'}

def load_data():
    df = pd.read_csv(TEST_DATA_FILE)
    if df['genuine'].dtype != bool:
        df['genuine'] = df['genuine'].astype(str).str.lower().map({
            'true': True, 'false': False, '1': True, '0': False
        })
    return df

def get_fusion_score(row, w_m=0.45, w_v=0.25, w_g=0.30):
    return w_m * row['S_M'] + w_v * row['S_V'] + w_g * row['S_G']

def load_config():
    config = {}
    with open('../results/optimal_all_3.txt', 'r', encoding='utf-8') as f:
        for line in f:
            if '=' in line:
                key, value = line.strip().split('=', 1)
                if key.startswith('tau_'):
                    config[key] = int(float(value))
                elif key.startswith('w_'):
                    config[key] = float(value)
    return config

def plot_score_distributions():
    df = load_data()
    genuine = df[df['genuine'] == True]
    impostor = df[df['genuine'] == False]
    config = load_config()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    layers = [('S_V', 'Voice', config.get('tau_V', 34)), 
              ('S_G', 'Spatial Gap', config.get('tau_G', 38)), 
              ('S_M', 'Mechanical', config.get('tau_M', 62))]
    
    for idx, (col, name, thresh) in enumerate(layers):
        ax = axes[idx]
        ax.hist(genuine[col], bins=25, alpha=0.6, label='Genuine', color=COLORS['genuine'], edgecolor='black')
        ax.hist(impostor[col], bins=25, alpha=0.6, label='Impostor', color=COLORS['impostor'], edgecolor='black')
        ax.axvline(thresh, color=COLORS['threshold'], linestyle='--', linewidth=2, label=f'Threshold ({thresh})')
        ax.set_xlabel('Score', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title(name, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Score Distributions (Test Set)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'score_distributions.png'), dpi=300, bbox_inches='tight')
    print("✓ Saved: score_distributions.png")
    plt.close()

def plot_confusion_matrix():
    df = load_data()
    config = load_config()
    w_m = config.get('w_M', 0.45)
    w_v = config.get('w_V', 0.25)
    w_g = config.get('w_G', 0.30)
    tau_v = config.get('tau_V', 34)
    tau_g = config.get('tau_G', 38)
    tau_m = config.get('tau_M', 62)
    tau_t = config.get('tau_T', 60)
    
    genuine = df[df['genuine'] == True]
    impostor = df[df['genuine'] == False]
    
    genuine_pass = genuine.apply(
        lambda row: row['S_V'] > tau_v and row['S_G'] > tau_g and row['S_M'] > tau_m,
        axis=1
    )
    impostor_pass = impostor.apply(
        lambda row: row['S_V'] > tau_v and row['S_G'] > tau_g and row['S_M'] > tau_m,
        axis=1
    )
    
    gen_accept = genuine[genuine_pass].apply(
        lambda row: get_fusion_score(row, w_m, w_v, w_g) > tau_t, axis=1
    ).sum()
    imp_accept = impostor[impostor_pass].apply(
        lambda row: get_fusion_score(row, w_m, w_v, w_g) > tau_t, axis=1
    ).sum()
    
    tp = int(gen_accept)
    fn = len(genuine) - tp
    fp = int(imp_accept)
    tn = len(impostor) - fp
    
    cm = np.array([[tn, fp], [fn, tp]])
    
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Reject', 'Accept'], 
                yticklabels=['Impostor', 'Genuine'],
                cbar_kws={'label': 'Count'}, ax=ax)
    
    ax.set_xlabel('Predicted', fontweight='bold')
    ax.set_ylabel('Actual', fontweight='bold')
    ax.set_title('Confusion Matrix', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    print("✓ Saved: confusion_matrix.png")
    plt.close()

def plot_roc_curve():
    df = load_data()
    config = load_config()
    w_m = config.get('w_M', 0.45)
    w_v = config.get('w_V', 0.25)
    w_g = config.get('w_G', 0.30)
    
    df['fusion_score'] = df.apply(lambda row: get_fusion_score(row, w_m, w_v, w_g), axis=1)
    
    y_true = df['genuine'].astype(int)
    y_scores = df['fusion_score']
    
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(fpr, tpr, color='darkorange', lw=2.5, label=f'ROC (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax.set_xlabel('False Positive Rate', fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontweight='bold')
    ax.set_title('ROC Curve', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'roc_curve.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: roc_curve.png (AUC={roc_auc:.3f})")
    plt.close()

def plot_threshold_tradeoff():
    df = load_data()
    config = load_config()
    w_m = config.get('w_M', 0.45)
    w_v = config.get('w_V', 0.25)
    w_g = config.get('w_G', 0.30)
    tau_v = config.get('tau_V', 34)
    tau_g = config.get('tau_G', 38)
    tau_m = config.get('tau_M', 62)
    
    genuine = df[df['genuine'] == True]
    impostor = df[df['genuine'] == False]
    
    results = []
    for thresh in range(30, 80, 1):
        gen_pass = genuine.apply(
            lambda row: row['S_V'] > tau_v and row['S_G'] > tau_g and row['S_M'] > tau_m,
            axis=1
        )
        imp_pass = impostor.apply(
            lambda row: row['S_V'] > tau_v and row['S_G'] > tau_g and row['S_M'] > tau_m,
            axis=1
        )
        gen_accept = genuine[gen_pass].apply(lambda row: get_fusion_score(row, w_m, w_v, w_g) > thresh, axis=1).sum()
        imp_accept = impostor[imp_pass].apply(lambda row: get_fusion_score(row, w_m, w_v, w_g) > thresh, axis=1).sum()
        results.append({'threshold': thresh, 
                       'FAR': 100*imp_accept/len(impostor) if len(impostor) > 0 else 0, 
                       'FRR': 100*(len(genuine)-gen_accept)/len(genuine) if len(genuine) > 0 else 0})
    
    df_results = pd.DataFrame(results)
    if len(df_results) > 0:
        eer_idx = (df_results['FAR'] - df_results['FRR']).abs().idxmin()
        eer_point = df_results.loc[eer_idx]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df_results['threshold'], df_results['FAR'], 'b-', linewidth=2, label='FAR')
        ax.plot(df_results['threshold'], df_results['FRR'], 'r-', linewidth=2, label='FRR')
        ax.plot(eer_point['threshold'], eer_point['FAR'], 'go', markersize=12, label=f'EER = {eer_point["FAR"]:.1f}%')
        ax.set_xlabel('Threshold', fontweight='bold')
        ax.set_ylabel('Rate (%)', fontweight='bold')
        ax.set_title('FAR vs FRR Trade-off', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'threshold_tradeoff.png'), dpi=300, bbox_inches='tight')
        print(f"✓ Saved: threshold_tradeoff.png (EER={eer_point['FAR']:.1f}%)")
    else:
        print("⚠ Skipped threshold_tradeoff.png - no data")
    plt.close()

def main():
    print("\nGenerating visualizations...")
    plot_score_distributions()
    plot_confusion_matrix()
    plot_roc_curve()
    plot_threshold_tradeoff()
    print(f"\n✓ All plots saved to {RESULTS_DIR}\n")

if __name__ == "__main__":
    main()