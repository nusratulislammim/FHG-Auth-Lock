"""
visualize_data.py - Additional Data Visualizations
Generates supplementary figures for presentation/supplementary materials

Authors: Nusratul Islam Mim, Firuze Tasnim Sneha, Al Musabbir, Dr. Md. Sujan Ali
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Input file from data folder
INPUT_CSV = "../data/experimental_data_N8.csv"

# Output folder
OUTPUT_FOLDER = "../results/"

# Create results folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def plot_score_distributions():
    """Plot score distributions (genuine vs impostor)"""
    df = pd.read_csv(INPUT_CSV)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, (score, name) in enumerate([('S_M', 'Mechanical'), 
                                          ('S_A', 'Acoustic'), 
                                          ('S_U', 'Spatial')]):
        ax = axes[idx]
        
        genuine = df[df['genuine'] == True][score]
        impostor = df[df['genuine'] == False][score]
        
        ax.hist(genuine, bins=20, alpha=0.6, label='Genuine', color='green', edgecolor='black')
        ax.hist(impostor, bins=20, alpha=0.6, label='Impostor', color='red', edgecolor='black')
        
        ax.axvline(70 if score != 'S_A' else 66, color='blue', linestyle='--', 
                   linewidth=2, label='Threshold')
        
        ax.set_xlabel('Score', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'{name} Layer ($S_{score[2]}$)', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, 'score_distributions.png'), dpi=300, bbox_inches='tight')
    print("✓ Saved score_distributions.png to results/")
    plt.close()

def plot_user_consistency():
    """Plot per-user score consistency (intra-user variance)"""
    df = pd.read_csv(INPUT_CSV)
    genuine = df[df['genuine'] == True]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    user_ids = sorted(genuine['participant_id'].unique())
    scores_by_user = {score: [] for score in ['S_M', 'S_A', 'S_U']}
    
    for user_id in user_ids:
        user_data = genuine[genuine['participant_id'] == user_id]
        for score in ['S_M', 'S_A', 'S_U']:
            scores_by_user[score].append(user_data[score].tolist())
    
    positions = np.arange(len(user_ids))
    width = 0.25
    
    colors = {'S_M': 'blue', 'S_A': 'green', 'S_U': 'orange'}
    
    for idx, (score, label) in enumerate([('S_M', 'Mechanical'), 
                                           ('S_A', 'Acoustic'), 
                                           ('S_U', 'Spatial')]):
        means = [np.mean(scores_by_user[score][i]) for i in range(len(user_ids))]
        stds = [np.std(scores_by_user[score][i]) for i in range(len(user_ids))]
        
        ax.bar(positions + idx*width, means, width, yerr=stds, 
               label=label, color=colors[score], alpha=0.7, capsize=5)
    
    ax.set_xlabel('User', fontsize=12)
    ax.set_ylabel('Score (mean ± std)', fontsize=12)
    ax.set_title('Per-User Score Consistency (Genuine Attempts)', fontsize=14, fontweight='bold')
    ax.set_xticks(positions + width)
    ax.set_xticklabels(user_ids)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, 'user_consistency.png'), dpi=300, bbox_inches='tight')
    print("✓ Saved user_consistency.png to results/")
    plt.close()

def plot_temporal_stability():
    """Plot score stability across sessions (temporal consistency)"""
    df = pd.read_csv(INPUT_CSV)
    genuine = df[df['genuine'] == True]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, (score, name) in enumerate([('S_M', 'Mechanical'), 
                                          ('S_A', 'Acoustic'), 
                                          ('S_U', 'Spatial')]):
        ax = axes[idx]
        
        sessions = [1, 2, 3, 4]
        session_means = []
        session_stds = []
        
        for session in sessions:
            session_data = genuine[genuine['session'] == session][score]
            session_means.append(session_data.mean())
            session_stds.append(session_data.std())
        
        ax.errorbar(sessions, session_means, yerr=session_stds, 
                    marker='o', markersize=8, linewidth=2, capsize=5, color='purple')
        
        ax.axhline(70 if score != 'S_A' else 66, color='red', linestyle='--', 
                   linewidth=2, label='Threshold', alpha=0.7)
        
        ax.set_xlabel('Session (Day 1, 3, 7, 14)', fontsize=11)
        ax.set_ylabel('Score (mean ± std)', fontsize=11)
        ax.set_title(f'{name} Layer', fontsize=12, fontweight='bold')
        ax.set_xticks(sessions)
        ax.set_xticklabels(['1\n(Day 1)', '2\n(Day 3)', '3\n(Day 7)', '4\n(Day 14)'])
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.suptitle('Temporal Stability: Score Consistency Across Sessions', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, 'temporal_stability.png'), dpi=300, bbox_inches='tight')
    print("✓ Saved temporal_stability.png to results/")
    plt.close()

def plot_confusion_matrix():
    """Plot confusion matrix for AND-rule system"""
    df = pd.read_csv(INPUT_CSV)
    
    # Compute decisions
    df['predicted'] = ((df['S_M'] > 70) & (df['S_A'] > 66) & (df['S_U'] > 70))
    
    # Confusion matrix
    tp = ((df['genuine'] == True) & (df['predicted'] == True)).sum()
    fn = ((df['genuine'] == True) & (df['predicted'] == False)).sum()
    tn = ((df['genuine'] == False) & (df['predicted'] == False)).sum()
    fp = ((df['genuine'] == False) & (df['predicted'] == True)).sum()
    
    cm = np.array([[tn, fp], [fn, tp]])
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Reject', 'Accept'], 
                yticklabels=['Impostor', 'Genuine'],
                cbar_kws={'label': 'Count'}, ax=ax, annot_kws={'size': 16})
    
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title('Confusion Matrix (Three-Layer AND Fusion)', fontsize=13, fontweight='bold')
    
    # Add percentages
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            pct = 100 * cm[i, j] / total
            ax.text(j+0.5, i+0.7, f'({pct:.1f}%)', 
                   ha='center', va='center', fontsize=10, color='gray')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    print("✓ Saved confusion_matrix.png to results/")
    plt.close()

def main():
    print("\n=== Generating Supplementary Visualizations ===\n")
    
    # Verify input file exists
    if not os.path.exists(INPUT_CSV):
        print(f"ERROR: Input file not found at {INPUT_CSV}")
        print("Please run generate_dataset.py first.")
        return
    
    plot_score_distributions()
    plot_user_consistency()
    plot_temporal_stability()
    plot_confusion_matrix()
    
    print("\n✓ All visualizations generated!")
    print(f"  → Files saved to {OUTPUT_FOLDER}")
    print("  → Use these for presentation slides or supplementary materials\n")

if __name__ == "__main__":
    main()