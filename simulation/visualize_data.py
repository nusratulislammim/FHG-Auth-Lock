"""
visualize_data.py - Additional Data Visualizations
Generates supplementary figures for presentation/supplementary materials

UPDATED: Now supports 4-layer hybrid authentication system
Layers: Voice (S_V), Spatial Gap (S_G), Mechanical (S_M), Spatial Position (S_U)

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
    """Plot score distributions (genuine vs impostor) for all 4 layers"""
    df = pd.read_csv(INPUT_CSV)
    
    # 4 layers: Voice (S_V), Spatial Gap (S_G), Mechanical (S_M), Spatial Position (S_U)
    layers = [
        ('S_V', 'Voice', 40),   # Voice threshold: 40
        ('S_G', 'Spatial Gap', 45),  # Spatial Gap threshold: 45
        ('S_M', 'Mechanical', 50),   # Mechanical threshold: 50
        ('S_U', 'Spatial Position', 50)  # Spatial Position threshold: 50
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, (score, name, threshold) in enumerate(layers):
        ax = axes[idx]
        
        genuine = df[df['genuine'] == True][score]
        impostor = df[df['genuine'] == False][score]
        
        ax.hist(genuine, bins=20, alpha=0.6, label='Genuine', color='green', edgecolor='black')
        ax.hist(impostor, bins=20, alpha=0.6, label='Impostor', color='red', edgecolor='black')
        
        ax.axvline(threshold, color='blue', linestyle='--', 
                   linewidth=2, label=f'Threshold ({threshold})')
        
        ax.set_xlabel('Score', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'{name} Layer', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, 'score_distributions.png'), dpi=300, bbox_inches='tight')
    print("✓ Saved score_distributions.png to results/")
    plt.close()

def plot_user_consistency():
    """Plot per-user score consistency (intra-user variance) for all 4 layers"""
    df = pd.read_csv(INPUT_CSV)
    genuine = df[df['genuine'] == True]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    user_ids = sorted(genuine['participant_id'].unique())
    score_cols = ['S_V', 'S_G', 'S_M', 'S_U']
    labels = ['Voice', 'Spatial Gap', 'Mechanical', 'Spatial']
    colors = {'S_V': 'purple', 'S_G': 'orange', 'S_M': 'blue', 'S_U': 'green'}
    
    scores_by_user = {score: [] for score in score_cols}
    
    for user_id in user_ids:
        user_data = genuine[genuine['participant_id'] == user_id]
        for score in score_cols:
            scores_by_user[score].append(user_data[score].tolist())
    
    positions = np.arange(len(user_ids))
    width = 0.2
    
    for idx, (score, label) in enumerate(zip(score_cols, labels)):
        means = [np.mean(scores_by_user[score][i]) for i in range(len(user_ids))]
        stds = [np.std(scores_by_user[score][i]) for i in range(len(user_ids))]
        
        ax.bar(positions + idx*width, means, width, yerr=stds, 
               label=label, color=colors[score], alpha=0.7, capsize=3)
    
    ax.set_xlabel('User', fontsize=12)
    ax.set_ylabel('Score (mean ± std)', fontsize=12)
    ax.set_title('Per-User Score Consistency (Genuine Attempts - 4 Layers)', fontsize=14, fontweight='bold')
    ax.set_xticks(positions + width * 1.5)
    ax.set_xticklabels(user_ids)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, 'user_consistency.png'), dpi=300, bbox_inches='tight')
    print("✓ Saved user_consistency.png to results/")
    plt.close()

def plot_temporal_stability():
    """Plot score stability across sessions (temporal consistency) for all 4 layers"""
    df = pd.read_csv(INPUT_CSV)
    genuine = df[df['genuine'] == True]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    layers = [
        ('S_V', 'Voice', 40),
        ('S_G', 'Spatial Gap', 45),
        ('S_M', 'Mechanical', 50),
        ('S_U', 'Spatial Position', 50)
    ]
    
    for idx, (score, name, threshold) in enumerate(layers):
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
        
        ax.axhline(threshold, color='red', linestyle='--', 
                   linewidth=2, label=f'Threshold ({threshold})', alpha=0.7)
        
        ax.set_xlabel('Session (Day 1, 3, 7, 14)', fontsize=11)
        ax.set_ylabel('Score (mean ± std)', fontsize=11)
        ax.set_title(f'{name} Layer', fontsize=12, fontweight='bold')
        ax.set_xticks(sessions)
        ax.set_xticklabels(['1\n(Day 1)', '2\n(Day 3)', '3\n(Day 7)', '4\n(Day 14)'])
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.suptitle('Temporal Stability: Score Consistency Across Sessions (4 Layers)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, 'temporal_stability.png'), dpi=300, bbox_inches='tight')
    print("✓ Saved temporal_stability.png to results/")
    plt.close()

def plot_confusion_matrix():
    """Plot confusion matrix for 4-layer hybrid system"""
    df = pd.read_csv(INPUT_CSV)
    
    # Use the actual decisions from the dataset (already computed with hybrid fusion)
    # The 'decision' column contains ACCEPT/REJECT from the simulation
    
    # Confusion matrix
    tp = ((df['genuine'] == True) & (df['decision'] == 'ACCEPT')).sum()
    fn = ((df['genuine'] == True) & (df['decision'] == 'REJECT')).sum()
    tn = ((df['genuine'] == False) & (df['decision'] == 'REJECT')).sum()
    fp = ((df['genuine'] == False) & (df['decision'] == 'ACCEPT')).sum()
    
    cm = np.array([[tn, fp], [fn, tp]])
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Reject', 'Accept'], 
                yticklabels=['Impostor', 'Genuine'],
                cbar_kws={'label': 'Count'}, ax=ax, annot_kws={'size': 16})
    
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title('Confusion Matrix (4-Layer Hybrid Fusion)', fontsize=13, fontweight='bold')
    
    # Add percentages
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            pct = 100 * cm[i, j] / total
            ax.text(j+0.5, i+0.7, f'({pct:.1f}%)', 
                   ha='center', va='center', fontsize=10, color='black')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    print("✓ Saved confusion_matrix.png to results/")
    plt.close()

def plot_layer_radar():
    """Plot radar chart showing per-layer performance"""
    df = pd.read_csv(INPUT_CSV)
    
    genuine = df[df['genuine'] == True]
    impostor = df[df['genuine'] == False]
    
    layers = ['S_V', 'S_G', 'S_M', 'S_U']
    labels = ['Voice', 'Spatial Gap', 'Mechanical', 'Spatial']
    
    genuine_means = [genuine[col].mean() for col in layers]
    impostor_means = [impostor[col].mean() for col in layers]
    
    angles = np.linspace(0, 2 * np.pi, len(layers), endpoint=False).tolist()
    genuine_means += genuine_means[:1]
    impostor_means += impostor_means[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    ax.plot(angles, genuine_means, 'o-', linewidth=2, label='Genuine', color='green')
    ax.fill(angles, genuine_means, alpha=0.25, color='green')
    
    ax.plot(angles, impostor_means, 'o-', linewidth=2, label='Impostor', color='red')
    ax.fill(angles, impostor_means, alpha=0.25, color='red')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.set_title('Per-Layer Performance Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, 'layer_radar.png'), dpi=300, bbox_inches='tight')
    print("✓ Saved layer_radar.png to results/")
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
    plot_layer_radar()
    
    print("\n✓ All visualizations generated!")
    print(f"  → Files saved to {OUTPUT_FOLDER}")
    print("  → Use these for presentation slides or supplementary materials\n")

if __name__ == "__main__":
    main()