"""
fix_voice_scores.py - Recalculate S_V from existing DTW costs
"""
import pandas as pd
import numpy as np

# CHANGE THIS to adjust voice strength
DIVISOR = 0.3

files = [
    '../data/train_data.csv',
    '../data/test_data.csv',
    '../data/experimental_data_N50.csv'
]

print(f"\nRecalculating voice scores (divisor = {DIVISOR})")

for filepath in files:
    df = pd.read_csv(filepath)
    print(f"\n{filepath}:")
    print(f"  Old S_V mean: {df['S_V'].mean():.2f}")
    
    for i in range(len(df)):
        dtws = [df.loc[i, 'voice_dtw_1'], 
                df.loc[i, 'voice_dtw_2'], 
                df.loc[i, 'voice_dtw_3']]
        scores = [100 * np.exp(-d / DIVISOR) for d in dtws]
        df.loc[i, 'S_V'] = int(np.mean(scores))
    
    print(f"  New S_V mean: {df['S_V'].mean():.2f}")
    df.to_csv(filepath, index=False)
    print(f"  ✓ Saved")

print("\n✓ Done! Now run: python diagnose_voice_layer.py")