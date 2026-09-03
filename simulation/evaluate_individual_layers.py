"""
evaluate_individual_layers.py - Evaluate each layer individually on test data
"""
import os
import sys
import pandas as pd
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================
TEST_DATA_FILE = '../data/test_data.csv'
OPTIMAL_CONFIG_FILE = '../results/optimal_all_3.txt'
RESULTS_DIR = '../results/'
OUTPUT_FILE = os.path.join(RESULTS_DIR, 'individual_layer_performance.txt')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# DATA LOADING
# ============================================================
def load_test_data():
    """Load test data and split into genuine/impostor."""
    df = pd.read_csv(TEST_DATA_FILE)
    if df['genuine'].dtype != bool:
        df['genuine'] = df['genuine'].astype(str).str.lower().map({
            'true': True, 'false': False, '1': True, '0': False
        })
    genuine = df[df['genuine'] == True]
    impostor = df[df['genuine'] == False]
    return genuine, impostor

def load_config():
    """Load optimal configuration from training."""
    config = {}
    with open(OPTIMAL_CONFIG_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if '=' in line:
                key, value = line.strip().split('=', 1)
                if key.startswith('tau_'):
                    config[key] = int(float(value))
                elif key.startswith('w_'):
                    config[key] = float(value)
    return config

# ============================================================
# EVALUATION FUNCTIONS
# ============================================================
def find_eer(genuine_scores, impostor_scores):
    """Find Equal Error Rate point."""
    results = []
    for thresh in range(0, 101, 1):
        gen_accept = (genuine_scores >= thresh).sum()
        imp_accept = (impostor_scores >= thresh).sum()
        
        far = 100 * imp_accept / len(impostor_scores)
        frr = 100 * (len(genuine_scores) - gen_accept) / len(genuine_scores)
        acc = 100 * (gen_accept + (len(impostor_scores) - imp_accept)) / (len(genuine_scores) + len(impostor_scores))
        
        results.append({'threshold': thresh, 'far': far, 'frr': frr, 'diff': abs(far - frr), 'accuracy': acc})

    df_results = pd.DataFrame(results)
    eer_idx = df_results['diff'].idxmin()
    eer_point = df_results.loc[eer_idx]
    eer = (eer_point['far'] + eer_point['frr']) / 2
    return eer_point['threshold'], eer, eer_point['far'], eer_point['frr'], eer_point['accuracy']

def evaluate_fusion(genuine, impostor, config):
    """Evaluate complete 3-layer fusion system."""
    # Stage 1: Sequential screening
    g_screen = genuine.apply(
        lambda row: (row['S_V'] >= config['tau_V']) and (row['S_G'] >= config['tau_G']) and (row['S_M'] >= config['tau_M']),
        axis=1
    )
    i_screen = impostor.apply(
        lambda row: (row['S_V'] >= config['tau_V']) and (row['S_G'] >= config['tau_G']) and (row['S_M'] >= config['tau_M']),
        axis=1
    )
    
    # Stage 2: Weighted fusion
    gen_accept = genuine[g_screen].apply(
        lambda row: (config['w_M']*row['S_M'] + config['w_V']*row['S_V'] + config['w_G']*row['S_G']) >= config['tau_T'],
        axis=1
    ).sum()
    imp_accept = impostor[i_screen].apply(
        lambda row: (config['w_M']*row['S_M'] + config['w_V']*row['S_V'] + config['w_G']*row['S_G']) >= config['tau_T'],
        axis=1
    ).sum()
    
    far = 100 * imp_accept / len(impostor)
    frr = 100 * (len(genuine) - gen_accept) / len(genuine)
    acc = 100 * (gen_accept + (len(impostor) - imp_accept)) / (len(genuine) + len(impostor))
    
    return {
        'far': far, 'frr': frr, 'aer': (far + frr) / 2, 'accuracy': acc,
        'tp': int(gen_accept), 'fn': len(genuine) - int(gen_accept),
        'fp': int(imp_accept), 'tn': len(impostor) - int(imp_accept)
    }

# ============================================================
# MAIN FUNCTION
# ============================================================
def main():
    """Main evaluation workflow."""
    print("\n" + "=" * 80)
    print("INDIVIDUAL LAYER PERFORMANCE EVALUATION")
    print("=" * 80)
    
    genuine, impostor = load_test_data()
    config = load_config()
    
    print(f"\nTest Set: {len(genuine)} genuine, {len(impostor)} impostor")
    print(f"Configuration: tau_V={config['tau_V']}, tau_G={config['tau_G']}, tau_M={config['tau_M']}, tau_T={config['tau_T']}")
    print(f"Weights: w_V={config['w_V']:.2f}, w_G={config['w_G']:.2f}, w_M={config['w_M']:.2f}\n")
    
    # Individual layers
    layers = {'Voice': 'S_V', 'Spatial Gap': 'S_G', 'Mechanical': 'S_M'}
    layer_results = {}
    
    for name, col in layers.items():
        thresh, eer, far, frr, acc = find_eer(genuine[col], impostor[col])
        layer_results[name] = {
            'optimal_threshold': thresh, 'eer': eer,
            'far': far, 'frr': frr, 'accuracy': acc
        }
    
    # Fusion
    fusion_results = evaluate_fusion(genuine, impostor, config)
    
    # Report
    report = []
    report.append("=" * 80)
    report.append("INDIVIDUAL LAYER PERFORMANCE (Test Set)")
    report.append("=" * 80)
    report.append(f"\n{'Layer':<20} {'Threshold':<12} {'FAR (%)':<10} {'FRR (%)':<10} {'EER (%)':<10} {'Acc (%)':<10}")
    report.append("-" * 80)
    
    for name in ['Voice', 'Spatial Gap', 'Mechanical']:
        res = layer_results[name]
        report.append(f"{name:<20} {res['optimal_threshold']:<12.0f} {res['far']:<10.2f} {res['frr']:<10.2f} "
                      f"{res['eer']:<10.2f} {res['accuracy']:<10.2f}")
    
    report.append("\n3-LAYER FUSION SYSTEM")
    report.append("-" * 80)
    report.append(f"FAR = {fusion_results['far']:.2f}%")
    report.append(f"FRR = {fusion_results['frr']:.2f}%")
    report.append(f"Accuracy = {fusion_results['accuracy']:.2f}%")
    report.append(f"Confusion: TP={fusion_results['tp']}, FN={fusion_results['fn']}, "
                  f"FP={fusion_results['fp']}, TN={fusion_results['tn']}")
    report.append("=" * 80)
    
    report_text = "\n".join(report)
    
    # Save with UTF-8 encoding to avoid Windows charmap errors
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\n✓ Saved: {OUTPUT_FILE}\n")
    return 0

if __name__ == "__main__":
    sys.exit(main())