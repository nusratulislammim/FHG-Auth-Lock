"""
evaluate_individual_layers.py - Per-layer + fusion evaluation on test set
"""
import os
import sys
import pandas as pd
import numpy as np

TEST_DATA_FILE      = '../data/test_data.csv'
OPTIMAL_CONFIG_FILE = '../results/optimal_all_3.txt'
RESULTS_DIR         = '../results/'
OUTPUT_FILE         = os.path.join(RESULTS_DIR, 'individual_layer_performance.txt')

os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------

def load_test_data():
    df = pd.read_csv(TEST_DATA_FILE)
    if df['genuine'].dtype != bool:
        df['genuine'] = df['genuine'].astype(str).str.lower().map({
            'true': True, 'false': False, '1': True, '0': False
        })
    return df[df['genuine'] == True], df[df['genuine'] == False]

def load_config():
    config = {}
    with open(OPTIMAL_CONFIG_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if '=' in line:
                k, v = line.strip().split('=', 1)
                if k.startswith('tau_'):
                    config[k] = int(float(v))
                elif k.startswith('w_'):
                    config[k] = float(v)
    return config

def find_eer(g_scores, i_scores):
    best = {'diff': float('inf')}
    for thresh in range(0, 101):
        ga  = (g_scores >= thresh).sum()
        ia  = (i_scores >= thresh).sum()
        far = 100 * ia / len(i_scores)
        frr = 100 * (len(g_scores) - ga) / len(g_scores)
        acc = 100 * (ga + (len(i_scores) - ia)) / (len(g_scores) + len(i_scores))
        diff = abs(far - frr)
        if diff < best['diff']:
            best = {'threshold': thresh, 'far': far, 'frr': frr,
                    'diff': diff, 'accuracy': acc}
    eer = (best['far'] + best['frr']) / 2
    return best['threshold'], eer, best['far'], best['frr'], best['accuracy']

def evaluate_fusion(genuine, impostor, cfg):
    g_screen = genuine.apply(
        lambda r: r['S_V'] >= cfg['tau_V'] and
                  r['S_G'] >= cfg['tau_G'] and
                  r['S_M'] >= cfg['tau_M'], axis=1)
    i_screen = impostor.apply(
        lambda r: r['S_V'] >= cfg['tau_V'] and
                  r['S_G'] >= cfg['tau_G'] and
                  r['S_M'] >= cfg['tau_M'], axis=1)

    ga = genuine[g_screen].apply(
        lambda r: cfg['w_M']*r['S_M'] + cfg['w_V']*r['S_V'] + cfg['w_G']*r['S_G'] >= cfg['tau_T'],
        axis=1).sum()
    ia = impostor[i_screen].apply(
        lambda r: cfg['w_M']*r['S_M'] + cfg['w_V']*r['S_V'] + cfg['w_G']*r['S_G'] >= cfg['tau_T'],
        axis=1).sum()

    ng, ni = len(genuine), len(impostor)
    far = 100 * ia / ni
    frr = 100 * (ng - ga) / ng
    acc = 100 * (ga + (ni - ia)) / (ng + ni)
    return {
        'far': far, 'frr': frr, 'aer': (far+frr)/2, 'accuracy': acc,
        'tp': int(ga), 'fn': ng - int(ga),
        'fp': int(ia), 'tn': ni - int(ia)
    }

# ---------------------------------------------------------------

def main():
    print("\n" + "=" * 80)
    print("INDIVIDUAL LAYER PERFORMANCE EVALUATION")
    print("=" * 80)

    genuine, impostor = load_test_data()
    cfg               = load_config()

    print(f"\nTest Set: {len(genuine)} genuine, {len(impostor)} impostor")
    print(f"Config: tau_V={cfg['tau_V']}, tau_G={cfg['tau_G']}, "
          f"tau_M={cfg['tau_M']}, tau_T={cfg['tau_T']}")
    print(f"Weights: w_V={cfg['w_V']:.2f}, w_G={cfg['w_G']:.2f}, w_M={cfg['w_M']:.2f}\n")

    layers = {'Voice': 'S_V', 'Spatial Gap': 'S_G', 'Mechanical': 'S_M'}
    layer_results = {}

    for name, col in layers.items():
        t, eer, far, frr, acc = find_eer(genuine[col], impostor[col])
        layer_results[name] = {
            'optimal_threshold': t, 'eer': eer,
            'far': far, 'frr': frr, 'accuracy': acc
        }

    fusion = evaluate_fusion(genuine, impostor, cfg)

    lines = []
    lines.append("=" * 80)
    lines.append("INDIVIDUAL LAYER PERFORMANCE (Test Set)")
    lines.append("=" * 80)
    lines.append(f"\n{'Layer':<20} {'Threshold':<12} {'FAR (%)':<10} "
                 f"{'FRR (%)':<10} {'EER (%)':<10} {'Acc (%)':<10}")
    lines.append("-" * 80)

    for name in ['Voice', 'Spatial Gap', 'Mechanical']:
        r = layer_results[name]
        lines.append(f"{name:<20} {r['optimal_threshold']:<12.0f} "
                     f"{r['far']:<10.2f} {r['frr']:<10.2f} "
                     f"{r['eer']:<10.2f} {r['accuracy']:<10.2f}")

    lines.append("\n3-LAYER FUSION SYSTEM")
    lines.append("-" * 80)
    lines.append(f"FAR      = {fusion['far']:.2f}%")
    lines.append(f"FRR      = {fusion['frr']:.2f}%")
    lines.append(f"AER      = {fusion['aer']:.2f}%")
    lines.append(f"Accuracy = {fusion['accuracy']:.2f}%")
    lines.append(f"Confusion: TP={fusion['tp']}, FN={fusion['fn']}, "
                 f"FP={fusion['fp']}, TN={fusion['tn']}")
    lines.append("=" * 80)

    # Layer ranking
    ranked = sorted(layer_results.items(), key=lambda x: x[1]['eer'])
    lines.append("\nLAYER RANKING (best to worst EER):")
    for rank, (name, r) in enumerate(ranked, 1):
        lines.append(f"  {rank}. {name:<15} EER = {r['eer']:.2f}%")

    lines.append(f"\nFusion AER = {fusion['aer']:.2f}%  "
                 f"(vs best single-layer EER = {ranked[0][1]['eer']:.2f}%)")

    report = "\n".join(lines)
    print(report)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\nSaved: {OUTPUT_FILE}\n")
    return 0

if __name__ == "__main__":
    sys.exit(main())