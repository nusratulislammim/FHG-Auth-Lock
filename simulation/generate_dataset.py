"""
generate_dataset.py - Generate Synthetic N=8 Experimental Dataset
Produces experimental_data_N8.csv matching paper Table III

Authors: Nusratul Islam Mim, Firuze Tasnim Sneha, Al Musabbir, Dr. Md. Sujan Ali
"""

import numpy as np
import pandas as pd
from sensor_models import *
import random
from datetime import datetime, timedelta
import os

# Configuration
NUM_USERS = 8
GENUINE_ATTEMPTS_PER_USER = 20  # 4 sessions × 5 attempts
IMPOSTOR_TARGETS_PER_USER = 3
IMPOSTOR_ATTEMPTS_PER_TARGET = 5

# Output file 
OUTPUT_CSV = "../data/experimental_data_N8.csv"

def simulate_genuine_attempt(user_profile, session_num, attempt_num):
    """
    Simulate a genuine user authentication attempt
    
    Returns:
        Dictionary with all measured values and scores
    """
    fsr = FSRSimulator()
    mic = MicrophoneSimulator()
    ultra = UltrasonicSimulator()
    
    # Measure baseline
    baseline = mic.measure_baseline(user_profile.baseline_noise)
    
    # Simulate 3 presses
    observed_force = []
    observed_hold = []
    observed_gap = []
    
    for i in range(3):
        force = fsr.read_press(
            user_profile.force_levels[i],
            user_profile.force_std
        )
        hold = fsr.read_hold_time(
            user_profile.hold_times[i],
            user_profile.hold_std
        )
        observed_force.append(force)
        observed_hold.append(hold)
        
        if i < 2:
            gap = fsr.read_gap_time(
                user_profile.gap_times[i],
                user_profile.gap_std
            )
            observed_gap.append(gap)
    
    # Spatial gesture
    distances = ultra.measure_gesture(
        user_profile.mean_distance,
        user_profile.distance_std
    )
    mean_distance = int(np.mean(distances)) if distances else 0
    
    # Compute scores
    enrolled = {
        'force': user_profile.force_levels,
        'hold': user_profile.hold_times,
        'gap': user_profile.gap_times
    }
    observed = {
        'force': observed_force,
        'hold': observed_hold,
        'gap': observed_gap
    }
    
    s_m = compute_mechanical_score(observed, enrolled)
    s_a = compute_acoustic_score(observed_force, baseline)
    s_u = compute_spatial_score(mean_distance, user_profile.mean_distance)
    
    # Decision
    decision = apply_fusion_rule(s_m, s_a, s_u)
    
    return {
        'participant_id': f'P{user_profile.user_id:02d}',
        'session': session_num,
        'attempt': attempt_num,
        'genuine': True,
        'force_0': observed_force[0],
        'force_1': observed_force[1],
        'force_2': observed_force[2],
        'hold_0': observed_hold[0],
        'hold_1': observed_hold[1],
        'hold_2': observed_hold[2],
        'gap_0': observed_gap[0] if len(observed_gap) > 0 else 0,
        'gap_1': observed_gap[1] if len(observed_gap) > 1 else 0,
        'baseline_noise': baseline,
        'mean_distance': mean_distance,
        'S_M': s_m,
        'S_A': s_a,
        'S_U': s_u,
        'decision': 'ACCEPT' if decision else 'REJECT'
    }

def simulate_impostor_attempt(impostor_id, target_profile):
    """
    Simulate an impostor attempting to authenticate as target user
    
    Args:
        impostor_id: ID of attacking user
        target_profile: UserProfile of genuine user being imitated
    
    Returns:
        Dictionary with attempt data
    """
    # Create impostor model with 70% observation accuracy (from Tari2006)
    impostor = ImpostorModel(target_profile, observation_accuracy=0.70)
    attempt_data = impostor.generate_attempt()
    
    fsr = FSRSimulator()
    mic = MicrophoneSimulator()
    ultra = UltrasonicSimulator()
    
    # Impostor performs based on their noisy estimates
    baseline = mic.measure_baseline(target_profile.baseline_noise)
    
    observed_force = []
    observed_hold = []
    observed_gap = []
    
    for i in range(3):
        force = fsr.read_press(
            int(attempt_data['force'][i]),
            attempt_data['force_std']
        )
        hold = fsr.read_hold_time(
            int(attempt_data['hold'][i]),
            attempt_data['hold_std']
        )
        observed_force.append(force)
        observed_hold.append(hold)
        
        if i < 2:
            gap = fsr.read_gap_time(
                int(attempt_data['gap'][i]),
                attempt_data['gap_std']
            )
            observed_gap.append(gap)
    
    # Spatial (impostor has poor distance estimation)
    distances = ultra.measure_gesture(
        int(attempt_data['distance']),
        attempt_data['distance_std']
    )
    mean_distance = int(np.mean(distances)) if distances else 0
    
    # Compute scores against genuine target's template
    enrolled = {
        'force': target_profile.force_levels,
        'hold': target_profile.hold_times,
        'gap': target_profile.gap_times
    }
    observed = {
        'force': observed_force,
        'hold': observed_hold,
        'gap': observed_gap
    }
    
    s_m = compute_mechanical_score(observed, enrolled)
    s_a = compute_acoustic_score(observed_force, baseline)
    s_u = compute_spatial_score(mean_distance, target_profile.mean_distance)
    
    decision = apply_fusion_rule(s_m, s_a, s_u)
    
    return {
        'participant_id': f'P{impostor_id:02d}',
        'session': 99,  # Special marker for impostor attempts
        'attempt': 0,
        'genuine': False,
        'force_0': observed_force[0],
        'force_1': observed_force[1],
        'force_2': observed_force[2],
        'hold_0': observed_hold[0],
        'hold_1': observed_hold[1],
        'hold_2': observed_hold[2],
        'gap_0': observed_gap[0] if len(observed_gap) > 0 else 0,
        'gap_1': observed_gap[1] if len(observed_gap) > 1 else 0,
        'baseline_noise': baseline,
        'mean_distance': mean_distance,
        'S_M': s_m,
        'S_A': s_a,
        'S_U': s_u,
        'decision': 'ACCEPT' if decision else 'REJECT'
    }

def generate_full_dataset():
    """Generate complete N=8 experimental dataset"""
    
    print("=" * 60)
    print("=== FHG AUTHENTICATION - SYNTHETIC DATASET GENERATOR ===")
    print("=" * 60)
    print()
    
    # Create 8 user profiles
    print("Creating user profiles...")
    print("-" * 40)
    users = create_user_profiles(NUM_USERS)
    print(f"✓ {NUM_USERS} users created\n")
    
    all_attempts = []
    
    # Generate genuine attempts (160 total)
    print("Generating genuine attempts...")
    print("-" * 40)
    for user in users:
        for session in range(1, 5):  # 4 sessions
            for attempt in range(1, 6):  # 5 attempts per session
                data = simulate_genuine_attempt(user, session, attempt)
                all_attempts.append(data)
    
    genuine_count = len(all_attempts)
    print(f"✓ {genuine_count} genuine attempts generated")
    
    # Compute genuine acceptance rate
    genuine_accepts = sum(1 for a in all_attempts if a['decision'] == 'ACCEPT')
    print(f"  Genuine acceptance rate: {100*genuine_accepts/genuine_count:.1f}%")
    print()
    
    # Generate impostor attempts (120 total)
    print("Generating impostor attempts...")
    print("-" * 40)
    impostor_count = 0
    
    for impostor_idx, impostor_user in enumerate(users):
        # Select 3 random targets (excluding self)
        other_users = [u for u in users if u.user_id != impostor_user.user_id]
        targets = random.sample(other_users, IMPOSTOR_TARGETS_PER_USER)
        
        for target in targets:
            for attempt in range(IMPOSTOR_ATTEMPTS_PER_TARGET):
                data = simulate_impostor_attempt(
                    impostor_user.user_id,
                    target
                )
                all_attempts.append(data)
                impostor_count += 1
    
    print(f"✓ {impostor_count} impostor attempts generated")
    
    # Compute impostor rejection rate
    impostor_attempts = [a for a in all_attempts if not a['genuine']]
    impostor_rejects = sum(1 for a in impostor_attempts if a['decision'] == 'REJECT')
    print(f"  Impostor rejection rate: {100*impostor_rejects/impostor_count:.1f}%")
    print()
    
    # Convert to DataFrame
    df = pd.DataFrame(all_attempts)
    
    # Add timestamp column (simulate data collection over 14 days)
    start_date = datetime(2025, 1, 15, 9, 0, 0)
    timestamps = []
    for _, row in df.iterrows():
        if row['genuine']:
            # Session 1: day 1, Session 2: day 3, Session 3: day 7, Session 4: day 14
            day_offset = [0, 2, 6, 13][row['session'] - 1]
            dt = start_date + timedelta(days=day_offset, minutes=row['attempt']*3)
        else:
            # Impostors: during session 1 timeframe
            dt = start_date + timedelta(minutes=random.randint(0, 120))
        timestamps.append(int(dt.timestamp()))
    
    df['timestamp'] = timestamps
    
    # Reorder columns
    column_order = [
        'participant_id', 'session', 'attempt', 'genuine', 'timestamp',
        'force_0', 'force_1', 'force_2',
        'hold_0', 'hold_1', 'hold_2',
        'gap_0', 'gap_1',
        'baseline_noise', 'mean_distance',
        'S_M', 'S_A', 'S_U', 'decision'
    ]
    df = df[column_order]
    
    # Save to CSV
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✓ Dataset saved to {OUTPUT_CSV}")
    print(f"  Total attempts: {len(df)}")
    print(f"  Genuine: {genuine_count}, Impostor: {impostor_count}")
    print()
    
    # Print summary statistics
    print("=" * 60)
    print("=== SUMMARY STATISTICS ===")
    print("=" * 60)
    print(f"False Accept Rate (FAR):  {100*(impostor_count - impostor_rejects)/impostor_count:.3f}%")
    print(f"False Reject Rate (FRR):  {100*(genuine_count - genuine_accepts)/genuine_count:.1f}%")
    print(f"Overall Accuracy:         {100*(genuine_accepts + impostor_rejects)/(genuine_count + impostor_count):.1f}%")
    print()
    
    # Per-layer statistics
    print("Per-Layer Performance:")
    print("-" * 40)
    
    # Mechanical only
    mech_accepts_genuine = sum(1 for a in all_attempts if a['genuine'] and a['S_M'] > 70)
    mech_accepts_impostor = sum(1 for a in all_attempts if not a['genuine'] and a['S_M'] > 70)
    print(f"  Mechanical: FAR={100*mech_accepts_impostor/impostor_count:.1f}%, FRR={100*(genuine_count-mech_accepts_genuine)/genuine_count:.1f}%")
    
    # Acoustic only
    acou_accepts_genuine = sum(1 for a in all_attempts if a['genuine'] and a['S_A'] > 66)
    acou_accepts_impostor = sum(1 for a in all_attempts if not a['genuine'] and a['S_A'] > 66)
    print(f"  Acoustic:   FAR={100*acou_accepts_impostor/impostor_count:.1f}%, FRR={100*(genuine_count-acou_accepts_genuine)/genuine_count:.1f}%")
    
    # Spatial only
    spat_accepts_genuine = sum(1 for a in all_attempts if a['genuine'] and a['S_U'] > 70)
    spat_accepts_impostor = sum(1 for a in all_attempts if not a['genuine'] and a['S_U'] > 70)
    print(f"  Spatial:    FAR={100*spat_accepts_impostor/impostor_count:.1f}%, FRR={100*(genuine_count-spat_accepts_genuine)/genuine_count:.1f}%")
    print()
    
    print("=" * 60)
    print("✓ DATASET GENERATION COMPLETE!")
    print("=" * 60)
    
    return df

if __name__ == "__main__":
    np.random.seed(42)  # Reproducible results
    random.seed(42)
    
    df = generate_full_dataset()
    
    print("\nFirst 10 rows of dataset:")
    print("-" * 40)
    print(df.head(10).to_string())