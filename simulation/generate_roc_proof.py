"""
generate_roc_proof.py - Generate ROC curve with optimal operating points
UPDATED: Now supports 4-layer hybrid authentication system
Layers: Voice (S_V), Spatial Gap (S_G), Mechanical (S_M), Spatial Position (S_U)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Load your data
df = pd.read_csv('../data/experimental_data_N8.csv')

print(f"Loaded {len(df)} attempts")
print(f"Genuine: {df['genuine'].sum()}, Impostor: {(~df['genuine']).sum()}")

genuine = df[df['genuine'] == True]
impostor = df[df['genuine'] == False]

# ========================================
# FUNCTION TO COMPUTE METRICS (4-LAYER)
# ========================================
def compute_metrics(thresh_v, thresh_g, thresh_m, thresh_u):
    """
    Compute FAR and FRR for 4-layer system:
    Voice (S_V), Spatial Gap (S_G), Mechanical (S_M), Spatial Position (S_U)
    """
    genuine_accept = genuine.apply(
        lambda row: (row['S_V'] > thresh_v) and 
                    (row['S_G'] > thresh_g) and 
                    (row['S_M'] > thresh_m) and 
                    (row['S_U'] > thresh_u),
        axis=1
    ).sum()
    impostor_accept = impostor.apply(
        lambda row: (row['S_V'] > thresh_v) and 
                    (row['S_G'] > thresh_g) and 
                    (row['S_M'] > thresh_m) and 
                    (row['S_U'] > thresh_u),
        axis=1
    ).sum()
    
    FAR = 100 * impostor_accept / len(impostor) if len(impostor) > 0 else 0
    FRR = 100 * (len(genuine) - genuine_accept) / len(genuine) if len(genuine) > 0 else 0
    return FAR, FRR

# ========================================
# TEST CURRENT OPERATING POINT
# ========================================
print("\n=== CURRENT HYBRID OPERATING POINT ===")
current_far, current_frr = compute_metrics(40, 45, 50, 50)
print(f"tau_V=40, tau_G=45, tau_M=50, tau_U=50")
print(f"FAR={current_far:.2f}%, FRR={current_frr:.2f}%")
print(f"Expected from analyze_results.py: FAR=5.0%, FRR=20.6%")

# Also test paper's old point (for comparison)
print("\n=== OLD PAPER OPERATING POINT (for comparison) ===")
old_far, old_frr = compute_metrics(70, 70, 70, 70)
print(f"Old thresholds: FAR={old_far:.2f}%, FRR={old_frr:.2f}%")

# ========================================
# THRESHOLD SWEEP (4-LAYER)
# ========================================
thresholds = []
for tv in range(20, 65, 5):
    for tg in range(20, 65, 5):
        for tm in range(20, 65, 5):
            for tu in range(20, 65, 5):
                far, frr = compute_metrics(tv, tg, tm, tu)
                thresholds.append({
                    'tau_V': tv,
                    'tau_G': tg,
                    'tau_M': tm,
                    'tau_U': tu,
                    'FAR': round(far, 2),
                    'FRR': round(frr, 2)
                })

# Convert to DataFrame
results = pd.DataFrame(thresholds)
results = results.sort_values(['FAR', 'FRR'])

print(f"\n=== OPERATING POINTS SUMMARY ===")
print(f"Total combinations tested: {len(results)}")

# ========================================
# FIND BEST OPERATING POINTS
# ========================================

# Points with FRR < 30%
print("\n=== POINTS WITH FRR < 30% (Best Usability) ===")
low_frr = results[results['FRR'] < 30]
if len(low_frr) > 0:
    print(low_frr.sort_values('FRR').head(10))
else:
    print(f"No points with FRR < 30% found. Minimum FRR: {results['FRR'].min():.2f}%")

# Points with FAR < 10%
print("\n=== POINTS WITH FAR < 10% (Best Security) ===")
low_far = results[results['FAR'] < 10]
if len(low_far) > 0:
    print(low_far.sort_values('FRR').head(10))
else:
    print(f"No points with FAR < 10% found. Minimum FAR: {results['FAR'].min():.2f}%")

# Balanced points (FAR ≈ FRR)
print("\n=== BALANCED POINTS (FAR ≈ FRR) ===")
results['diff'] = abs(results['FAR'] - results['FRR'])
balanced = results.sort_values('diff').head(10)
print(balanced[['tau_V', 'tau_G', 'tau_M', 'tau_U', 'FAR', 'FRR']])

# Optimal point (best FRR with FAR < 10%)
print("\n=== BEST OPERATING POINT (FRR < 25% with FAR < 10%) ===")
optimal = results[(results['FRR'] < 25) & (results['FAR'] < 10)]
if len(optimal) > 0:
    best = optimal.sort_values('FRR').iloc[0]
    print(f"tau_V={best['tau_V']}, tau_G={best['tau_G']}, tau_M={best['tau_M']}, tau_U={best['tau_U']}")
    print(f"FAR={best['FAR']:.2f}%, FRR={best['FRR']:.2f}%")
else:
    print("No point found with FRR < 25% and FAR < 10%")
    closest = results.iloc[(results['FRR'] - 25).abs().argsort()[:5]]
    print("\nClosest to target:")
    print(closest[['tau_V', 'tau_G', 'tau_M', 'tau_U', 'FAR', 'FRR']])

# ========================================
# PLOT: FAR vs FRR TRADE-OFF
# ========================================
fig, ax = plt.subplots(figsize=(10, 7))

# Scatter plot of all points
scatter = ax.scatter(results['FAR'], results['FRR'], 
                    c=results['FRR'], cmap='RdYlGn_r', alpha=0.5, s=20)

# Mark current operating point
ax.plot(current_far, current_frr, 'ro', markersize=14, 
        label=f'Current (FAR={current_far:.1f}%, FRR={current_frr:.1f}%)',
        markerfacecolor='red', markeredgecolor='black', markeredgewidth=2)

# Mark optimal point if found
if len(optimal) > 0:
    best = optimal.sort_values('FRR').iloc[0]
    ax.plot(best['FAR'], best['FRR'], 'go', markersize=12,
            label=f'Optimal (FRR={best["FRR"]:.1f}%)',
            markerfacecolor='green', markeredgecolor='black', markeredgewidth=2)

# Mark balanced point
if len(balanced) > 0:
    bal = balanced.iloc[0]
    ax.plot(bal['FAR'], bal['FRR'], 'bo', markersize=12,
            label=f'Balanced (EER≈{bal["FRR"]:.1f}%)',
            markerfacecolor='blue', markeredgecolor='black', markeredgewidth=2)

ax.set_xlabel('False Accept Rate (FAR %)', fontsize=12)
ax.set_ylabel('False Reject Rate (FRR %)', fontsize=12)
ax.set_title('4-Layer Hybrid System: FAR vs FRR Trade-off', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right', fontsize=10)
ax.set_xlim(0, 85)
ax.set_ylim(0, 85)

# Add colorbar
cbar = plt.colorbar(scatter)
cbar.set_label('FRR (%)', fontsize=10)

plt.tight_layout()
plt.savefig('../results/threshold_tradeoff_4layer.png', dpi=300, bbox_inches='tight')
print("\n✅ Threshold trade-off plot saved to ../results/threshold_tradeoff_4layer.png")

# ========================================
# PRINT SUMMARY STATISTICS
# ========================================
print("\n" + "=" * 60)
print("=== SUMMARY STATISTICS ===")
print("=" * 60)
print(f"Current Operating Point (Hybrid):")
print(f"  tau_V=40, tau_G=45, tau_M=50, tau_U=50")
print(f"  FAR = {current_far:.2f}%")
print(f"  FRR = {current_frr:.2f}%")
print()

if len(optimal) > 0:
    best = optimal.sort_values('FRR').iloc[0]
    print(f"Optimal Operating Point:")
    print(f"  tau_V={best['tau_V']}, tau_G={best['tau_G']}, tau_M={best['tau_M']}, tau_U={best['tau_U']}")
    print(f"  FAR = {best['FAR']:.2f}%")
    print(f"  FRR = {best['FRR']:.2f}%")
    print(f"  EER = {(best['FAR'] + best['FRR']) / 2:.2f}%")
    print(f"  Accuracy = {100 - (best['FAR'] + best['FRR']) / 2:.2f}%")