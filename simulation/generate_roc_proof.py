"""
generate_roc_proof.py - Generate ROC curve with optimal operating points
Uses correct threshold scale (0-100, matching the scores)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load your data
df = pd.read_csv('../data/experimental_data_N8.csv')

print(f"Loaded {len(df)} attempts")
print(f"Genuine: {df['genuine'].sum()}, Impostor: {(~df['genuine']).sum()}")

genuine = df[df['genuine'] == True]
impostor = df[df['genuine'] == False]

# Function to compute FAR and FRR for a given threshold
def compute_metrics(thresh_m, thresh_a, thresh_u):
    genuine_accept = genuine.apply(
        lambda row: (row['S_M'] > thresh_m) and (row['S_A'] > thresh_a) and (row['S_U'] > thresh_u),
        axis=1
    ).sum()
    impostor_accept = impostor.apply(
        lambda row: (row['S_M'] > thresh_m) and (row['S_A'] > thresh_a) and (row['S_U'] > thresh_u),
        axis=1
    ).sum()
    
    FAR = 100 * impostor_accept / len(impostor) if len(impostor) > 0 else 0
    FRR = 100 * (len(genuine) - genuine_accept) / len(genuine) if len(genuine) > 0 else 0
    return FAR, FRR

# Test the paper's operating point
print("\n=== PAPER'S OPERATING POINT ===")
paper_far, paper_frr = compute_metrics(70, 66, 70)
print(f"tau_M=70, tau_A=66, tau_U=70: FAR={paper_far:.2f}%, FRR={paper_frr:.2f}%")
print(f"Expected from analyze_results.py: FAR=0.83%, FRR=75.6%")

# Try different threshold combinations (using 0-100 scale)
thresholds = []
for tm in range(20, 80, 5):
    for ta in range(20, 80, 5):
        for tu in range(20, 80, 5):
            far, frr = compute_metrics(tm, ta, tu)
            thresholds.append({
                'tau_M': tm,
                'tau_A': ta,
                'tau_U': tu,
                'FAR': round(far, 2),
                'FRR': round(frr, 2)
            })

# Convert to DataFrame
results = pd.DataFrame(thresholds)
results = results.sort_values(['FAR', 'FRR'])

print(f"\n=== OPERATING POINTS SUMMARY ===")
print(f"Total combinations tested: {len(results)}")

# Find points with good FRR
print("\n=== POINTS WITH FRR < 30% ===")
low_frr = results[results['FRR'] < 30]
if len(low_frr) > 0:
    print(low_frr.sort_values('FRR').head(10))
else:
    print(f"No points with FRR < 30% found. Minimum FRR: {results['FRR'].min():.2f}%")

# Find points with good FAR
print("\n=== POINTS WITH FAR < 5% ===")
low_far = results[results['FAR'] < 5]
if len(low_far) > 0:
    print(low_far.sort_values('FRR').head(10))
else:
    print(f"No points with FAR < 5% found. Minimum FAR: {results['FAR'].min():.2f}%")

# Find balanced points
print("\n=== BALANCED POINTS (FAR ≈ FRR) ===")
results['diff'] = abs(results['FAR'] - results['FRR'])
balanced = results.sort_values('diff').head(10)
print(balanced[['tau_M', 'tau_A', 'tau_U', 'FAR', 'FRR']])

# Find optimal point
print("\n=== BEST OPERATING POINT (Lowest FRR with FAR < 10%) ===")
optimal = results[(results['FRR'] < 40) & (results['FAR'] < 10)]
if len(optimal) > 0:
    best = optimal.sort_values('FRR').iloc[0]
    print(f"tau_M={best['tau_M']}, tau_A={best['tau_A']}, tau_U={best['tau_U']}")
    print(f"FAR={best['FAR']:.2f}%, FRR={best['FRR']:.2f}%")
else:
    print("No point found with FRR < 40% and FAR < 10%")
    # Show closest to target
    closest = results.iloc[(results['FRR'] - 40).abs().argsort()[:5]]
    print("\nClosest to target:")
    print(closest[['tau_M', 'tau_A', 'tau_U', 'FAR', 'FRR']])

# Plot
fig, ax = plt.subplots(figsize=(8, 6))

# Scatter plot
scatter = ax.scatter(results['FAR'], results['FRR'], 
                    c=results['FRR'], cmap='RdYlGn_r', alpha=0.5, s=20)

# Mark paper's operating point
ax.plot(paper_far, paper_frr, 'ro', markersize=12, label='Paper Operating Point',
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
ax.set_title('FAR vs FRR Trade-off with Threshold Tuning', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right', fontsize=10)
ax.set_xlim(0, 105)
ax.set_ylim(0, 105)

# Add colorbar
cbar = plt.colorbar(scatter)
cbar.set_label('FRR (%)', fontsize=10)

plt.tight_layout()
plt.savefig('../results/threshold_tradeoff.png', dpi=300, bbox_inches='tight')
print("\n✅ Threshold trade-off plot saved to ../results/threshold_tradeoff.png")