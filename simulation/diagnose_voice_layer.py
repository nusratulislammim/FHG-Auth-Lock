"""
diagnose_voice_layer.py - Diagnose voice authentication issues
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================

TEST_DATA_FILE = '../data/test_data.csv'
RESULTS_DIR = '../results/'
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# MAIN DIAGNOSIS
# ============================================================

def main():
    # Load test data
    if not os.path.exists(TEST_DATA_FILE):
        print(f"❌ ERROR: Test data not found at {TEST_DATA_FILE}")
        print("   Run generate_dataset.py first!")
        return 1
    
    df_test = pd.read_csv(TEST_DATA_FILE)
    
    # Ensure boolean type
    if df_test['genuine'].dtype != bool:
        df_test['genuine'] = df_test['genuine'].astype(str).str.lower().map({
            'true': True, 'false': False, '1': True, '0': False
        })
    
    genuine = df_test[df_test['genuine'] == True]
    impostor = df_test[df_test['genuine'] == False]
    
    print("\n" + "=" * 80)
    print("VOICE LAYER DIAGNOSIS")
    print("=" * 80)
    print(f"\nTest Set: {len(genuine)} genuine, {len(impostor)} impostor attempts")
    
    # Score statistics
    print("\n" + "-" * 80)
    print("VOICE SCORE (S_V) STATISTICS")
    print("-" * 80)
    print(f"{'Category':<12} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print("-" * 80)
    print(f"{'Genuine':<12} {genuine['S_V'].mean():<10.2f} {genuine['S_V'].std():<10.2f} "
          f"{genuine['S_V'].min():<10.2f} {genuine['S_V'].max():<10.2f}")
    print(f"{'Impostor':<12} {impostor['S_V'].mean():<10.2f} {impostor['S_V'].std():<10.2f} "
          f"{impostor['S_V'].min():<10.2f} {impostor['S_V'].max():<10.2f}")
    
    separation = genuine['S_V'].mean() - impostor['S_V'].mean()
    print(f"\nSeparation (Genuine - Impostor): {separation:.2f}")
    
    if separation < 20:
        print("❌ CRITICAL: Separation < 20 points - poor discriminability!")
    elif separation < 30:
        print("⚠ WARNING: Separation < 30 points - marginal discriminability")
    else:
        print("✓ Good separation between genuine and impostor")
    
    # DTW cost statistics
    print("\n" + "-" * 80)
    print("DTW COST STATISTICS (per digit)")
    print("-" * 80)
    print(f"{'Digit':<10} {'Gen Mean':<12} {'Gen Std':<12} {'Imp Mean':<12} {'Imp Std':<12} {'Separation':<12}")
    print("-" * 80)
    
    for i in [1, 2, 3]:
        col = f'voice_dtw_{i}'
        g_mean = genuine[col].mean()
        g_std = genuine[col].std()
        i_mean = impostor[col].mean()
        i_std = impostor[col].std()
        sep = i_mean - g_mean
        print(f"{'Digit ' + str(i):<10} {g_mean:<12.4f} {g_std:<12.4f} "
              f"{i_mean:<12.4f} {i_std:<12.4f} {sep:<12.4f}")
    
    # Distribution overlap
    print("\n" + "-" * 80)
    print("DISTRIBUTION ANALYSIS")
    print("-" * 80)
    
    overlap = ((impostor['S_V'].min() <= genuine['S_V'].max()) and 
               (genuine['S_V'].min() <= impostor['S_V'].max()))
    print(f"Distributions overlap: {overlap}")
    print(f"Genuine range:  [{genuine['S_V'].min():.2f}, {genuine['S_V'].max():.2f}]")
    print(f"Impostor range: [{impostor['S_V'].min():.2f}, {impostor['S_V'].max():.2f}]")
    
    # Threshold analysis
    print("\n" + "-" * 80)
    print("PERFORMANCE AT DIFFERENT THRESHOLDS")
    print("-" * 80)
    print(f"{'Threshold':<12} {'FAR (%)':<12} {'FRR (%)':<12} {'Accuracy (%)':<12} {'EER Diff':<12}")
    print("-" * 80)
    
    best_eer = float('inf')
    best_thresh = 0
    
    for thresh in range(10, 100, 5):
        gen_accept = (genuine['S_V'] >= thresh).sum()
        imp_accept = (impostor['S_V'] >= thresh).sum()
        
        far = 100 * imp_accept / len(impostor)
        frr = 100 * (len(genuine) - gen_accept) / len(genuine)
        acc = 100 * (gen_accept + (len(impostor) - imp_accept)) / (len(genuine) + len(impostor))
        eer_diff = abs(far - frr)
        
        print(f"{thresh:<12} {far:<12.2f} {frr:<12.2f} {acc:<12.2f} {eer_diff:<12.2f}")
        
        if eer_diff < best_eer:
            best_eer = eer_diff
            best_thresh = thresh
            best_far = far
            best_frr = frr
    
    print("-" * 80)
    print(f"EER Point: Threshold={best_thresh}, FAR={best_far:.2f}%, FRR={best_frr:.2f}%, EER={(best_far+best_frr)/2:.2f}%")
    
    # Critical checks
    print("\n" + "=" * 80)
    print("CRITICAL CHECKS")
    print("=" * 80)
    
    issues = []
    
    # Check 1: 100% FAR at low threshold
    if (impostor['S_V'] >= 20).all():
        issues.append("❌ CRITICAL: 100% of impostors pass at threshold 20 - ZERO security!")
    
    # Check 2: Mean impostor score too high
    if impostor['S_V'].mean() > 70:
        issues.append("❌ CRITICAL: Mean impostor score > 70 - too lenient!")
    
    # Check 3: Poor separation
    if separation < 15:
        issues.append("❌ CRITICAL: Score separation < 15 - cannot discriminate!")
    
    # Check 4: EER too high
    eer = (best_far + best_frr) / 2
    if eer > 40:
        issues.append(f"❌ CRITICAL: EER={eer:.1f}% - voice layer ineffective!")
    
    if issues:
        for issue in issues:
            print(issue)
        print("\n🔧 FIXES REQUIRED:")
        print("   1. Reduce DTW score divisor (15 → 8) in sensor_models.py")
        print("   2. Increase speaker variability (pitch_base std: 25 → 40)")
        print("   3. Increase impostor digit error rate (0.30 → 0.50)")
    else:
        print("✓ All checks passed - voice layer functioning properly")
    
    # Visualization
    print("\n" + "-" * 80)
    print("GENERATING VISUALIZATIONS")
    print("-" * 80)
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Score distribution
        axes[0, 0].hist(genuine['S_V'], bins=30, alpha=0.6, label='Genuine', 
                        color='green', edgecolor='black', density=True)
        axes[0, 0].hist(impostor['S_V'], bins=30, alpha=0.6, label='Impostor', 
                        color='red', edgecolor='black', density=True)
        axes[0, 0].axvline(best_thresh, color='blue', linestyle='--', linewidth=2, 
                           label=f'EER Threshold ({best_thresh})')
        axes[0, 0].set_xlabel('Voice Score (S_V)', fontweight='bold')
        axes[0, 0].set_ylabel('Density', fontweight='bold')
        axes[0, 0].set_title('Voice Score Distribution', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. DTW costs
        dtw_genuine = pd.concat([genuine[f'voice_dtw_{i}'] for i in [1, 2, 3]])
        dtw_impostor = pd.concat([impostor[f'voice_dtw_{i}'] for i in [1, 2, 3]])
        axes[0, 1].hist(dtw_genuine, bins=30, alpha=0.6, label='Genuine', 
                        color='green', edgecolor='black', density=True)
        axes[0, 1].hist(dtw_impostor, bins=30, alpha=0.6, label='Impostor', 
                        color='red', edgecolor='black', density=True)
        axes[0, 1].set_xlabel('DTW Cost', fontweight='bold')
        axes[0, 1].set_ylabel('Density', fontweight='bold')
        axes[0, 1].set_title('DTW Cost Distribution', fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Box plots (FIXED: tick_labels instead of labels)
        data_to_plot = [genuine['S_V'], impostor['S_V']]
        bp = axes[1, 0].boxplot(data_to_plot, tick_labels=['Genuine', 'Impostor'],
                                patch_artist=True, widths=0.6)
        bp['boxes'][0].set_facecolor('green')
        bp['boxes'][0].set_alpha(0.6)
        bp['boxes'][1].set_facecolor('red')
        bp['boxes'][1].set_alpha(0.6)
        axes[1, 0].set_ylabel('Voice Score (S_V)', fontweight='bold')
        axes[1, 0].set_title('Score Distribution (Box Plot)', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 4. FAR/FRR curve
        thresholds = range(0, 101, 2)
        fars = []
        frrs = []
        
        for thresh in thresholds:
            gen_accept = (genuine['S_V'] >= thresh).sum()
            imp_accept = (impostor['S_V'] >= thresh).sum()
            
            far = 100 * imp_accept / len(impostor)
            frr = 100 * (len(genuine) - gen_accept) / len(genuine)
            
            fars.append(far)
            frrs.append(frr)
        
        axes[1, 1].plot(thresholds, fars, 'b-', linewidth=2, label='FAR')
        axes[1, 1].plot(thresholds, frrs, 'r-', linewidth=2, label='FRR')
        axes[1, 1].axvline(best_thresh, color='green', linestyle='--', 
                           linewidth=2, label=f'EER Point ({best_thresh})')
        axes[1, 1].set_xlabel('Threshold', fontweight='bold')
        axes[1, 1].set_ylabel('Rate (%)', fontweight='bold')
        axes[1, 1].set_title('FAR vs FRR Trade-off', fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_xlim(0, 100)
        axes[1, 1].set_ylim(0, 100)
        
        plt.tight_layout()
        output_path = os.path.join(RESULTS_DIR, 'voice_diagnosis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved diagnosis plot: {output_path}")
        plt.close()
        
    except Exception as e:
        print(f"⚠ Could not generate plots: {e}")
        print("  (Continuing without visualization)")
    
    # Summary
    print("\n" + "=" * 80)
    print("DIAGNOSIS SUMMARY")
    print("=" * 80)
    print(f"Voice Layer Status: {'❌ BROKEN' if issues else '✓ FUNCTIONAL'}")
    print(f"EER at optimal threshold: {eer:.2f}%")
    print(f"Score separation: {separation:.2f} points")
    print(f"Issues found: {len(issues)}")
    
    if issues:
        print("\n⚠ ACTION REQUIRED: Apply fixes in sensor_models.py and regenerate data")
        return 1
    else:
        print("\n✓ Voice layer is working correctly")
        return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())