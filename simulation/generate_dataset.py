"""
generate_dataset.py - Generate Synthetic N=8 Experimental Dataset
HYBRID FUSION: Sequential + Weighted AND-Check
"""

import numpy as np
import pandas as pd
from sensor_models import *
import random
from datetime import datetime, timedelta

NUM_USERS = 8
GENUINE_ATTEMPTS_PER_USER = 20
IMPOSTOR_TARGETS_PER_USER = 3
IMPOSTOR_ATTEMPTS_PER_TARGET = 5
OUTPUT_CSV = "../data/experimental_data_N8.csv"

def simulate_genuine_attempt(user_profile, session_num, attempt_num):
    fsr = FSRSimulator()
    mic = MicrophoneSimulator()
    ultra = UltrasonicSimulator()
    
    baseline = mic.measure_baseline(user_profile.baseline_noise)
    
    # Force pattern
    observed_force = []
    observed_hold = []
    observed_gap = []
    for i in range(3):
        force = fsr.read_press(user_profile.force_levels[i], user_profile.force_std)
        hold = fsr.read_hold_time(user_profile.hold_times[i], user_profile.hold_std)
        observed_force.append(force)
        observed_hold.append(hold)
        if i < 2:
            gap = fsr.read_gap_time(user_profile.gap_times[i], user_profile.gap_std)
            observed_gap.append(gap)
    
    # Voice digits
    observed_voice_digits = []
    for i in range(3):
        recognized = mic.recognize_digit(user_profile.voice_digits[i], user_profile.voice_error_rate)
        observed_voice_digits.append(recognized)
    
    # Spatial gaps
    observed_gap_distances = []
    for i in range(2):
        measured = ultra.measure_gap_distance(user_profile.gap_distances[i], user_profile.gap_distance_std)
        observed_gap_distances.append(measured)
    
    # Spatial position
    distances = ultra.measure_gesture(user_profile.mean_distance, user_profile.distance_std)
    mean_distance = int(np.mean(distances)) if distances else 0
    
    # Compute scores
    enrolled = {'force': user_profile.force_levels, 'hold': user_profile.hold_times, 'gap': user_profile.gap_times}
    observed = {'force': observed_force, 'hold': observed_hold, 'gap': observed_gap}
    
    s_m = compute_mechanical_score(observed, enrolled)
    s_v = compute_voice_score(observed_voice_digits, user_profile.voice_digits)
    s_g = compute_spatial_gap_score(observed_gap_distances, user_profile.gap_distances)
    s_u = compute_spatial_score(mean_distance, user_profile.mean_distance)
    
    # HYBRID FUSION: Sequential + Weighted AND-Check
    decision, details = apply_hybrid_fusion(s_m, s_v, s_g, s_u)
    
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
        'voice_0': observed_voice_digits[0],
        'voice_1': observed_voice_digits[1],
        'voice_2': observed_voice_digits[2],
        'gap_dist_0': observed_gap_distances[0] if len(observed_gap_distances) > 0 else 0,
        'gap_dist_1': observed_gap_distances[1] if len(observed_gap_distances) > 1 else 0,
        'baseline_noise': baseline,
        'mean_distance': mean_distance,
        'S_M': s_m,
        'S_V': s_v,
        'S_G': s_g,
        'S_U': s_u,
        'decision': 'ACCEPT' if decision else 'REJECT',
        'fusion_details': str(details)
    }

def simulate_impostor_attempt(impostor_id, target_profile):
    impostor = ImpostorModel(target_profile, observation_accuracy=0.70)
    attempt_data = impostor.generate_attempt()
    
    fsr = FSRSimulator()
    mic = MicrophoneSimulator()
    ultra = UltrasonicSimulator()
    
    baseline = mic.measure_baseline(target_profile.baseline_noise)
    
    # Force
    observed_force = []
    observed_hold = []
    observed_gap = []
    for i in range(3):
        force = fsr.read_press(int(attempt_data['force'][i]), attempt_data['force_std'])
        hold = fsr.read_hold_time(int(attempt_data['hold'][i]), attempt_data['hold_std'])
        observed_force.append(force)
        observed_hold.append(hold)
        if i < 2:
            gap = fsr.read_gap_time(int(attempt_data['gap'][i]), attempt_data['gap_std'])
            observed_gap.append(gap)
    
    # Voice
    observed_voice_digits = attempt_data['voice_digits']
    
    # Spatial gaps
    observed_gap_distances = []
    for i in range(2):
        measured = ultra.measure_gap_distance(int(attempt_data['gap_distances'][i]), attempt_data['gap_distance_std'])
        observed_gap_distances.append(measured)
    
    # Spatial position
    distances = ultra.measure_gesture(int(attempt_data['distance']), attempt_data['distance_std'])
    mean_distance = int(np.mean(distances)) if distances else 0
    
    # Compute scores against target
    enrolled = {'force': target_profile.force_levels, 'hold': target_profile.hold_times, 'gap': target_profile.gap_times}
    observed = {'force': observed_force, 'hold': observed_hold, 'gap': observed_gap}
    
    s_m = compute_mechanical_score(observed, enrolled)
    s_v = compute_voice_score(observed_voice_digits, target_profile.voice_digits)
    s_g = compute_spatial_gap_score(observed_gap_distances, target_profile.gap_distances)
    s_u = compute_spatial_score(mean_distance, target_profile.mean_distance)
    
    decision, details = apply_hybrid_fusion(s_m, s_v, s_g, s_u)
    
    return {
        'participant_id': f'P{impostor_id:02d}',
        'session': 99,
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
        'voice_0': observed_voice_digits[0],
        'voice_1': observed_voice_digits[1],
        'voice_2': observed_voice_digits[2],
        'gap_dist_0': observed_gap_distances[0] if len(observed_gap_distances) > 0 else 0,
        'gap_dist_1': observed_gap_distances[1] if len(observed_gap_distances) > 1 else 0,
        'baseline_noise': baseline,
        'mean_distance': mean_distance,
        'S_M': s_m,
        'S_V': s_v,
        'S_G': s_g,
        'S_U': s_u,
        'decision': 'ACCEPT' if decision else 'REJECT',
        'fusion_details': str(details)
    }

def generate_full_dataset():
    print("=" * 60)
    print("=== HYBRID AUTHENTICATION - SYNTHETIC DATASET GENERATOR ===")
    print("=== Sequential + Weighted AND-Check ===")
    print("=" * 60)
    print()
    
    print("Creating user profiles...")
    print("-" * 40)
    users = create_user_profiles(NUM_USERS)
    print(f"✓ {NUM_USERS} users created\n")
    
    all_attempts = []
    
    print("Generating genuine attempts...")
    print("-" * 40)
    for user in users:
        for session in range(1, 5):
            for attempt in range(1, 6):
                data = simulate_genuine_attempt(user, session, attempt)
                all_attempts.append(data)
    
    genuine_count = len(all_attempts)
    genuine_accepts = sum(1 for a in all_attempts if a['decision'] == 'ACCEPT')
    print(f"✓ {genuine_count} genuine attempts generated")
    print(f"  Genuine acceptance rate: {100*genuine_accepts/genuine_count:.1f}%")
    print()
    
    print("Generating impostor attempts...")
    print("-" * 40)
    impostor_count = 0
    for impostor_idx, impostor_user in enumerate(users):
        other_users = [u for u in users if u.user_id != impostor_user.user_id]
        targets = random.sample(other_users, IMPOSTOR_TARGETS_PER_USER)
        for target in targets:
            for attempt in range(IMPOSTOR_ATTEMPTS_PER_TARGET):
                data = simulate_impostor_attempt(impostor_user.user_id, target)
                all_attempts.append(data)
                impostor_count += 1
    
    print(f"✓ {impostor_count} impostor attempts generated")
    impostor_attempts = [a for a in all_attempts if not a['genuine']]
    impostor_rejects = sum(1 for a in impostor_attempts if a['decision'] == 'REJECT')
    print(f"  Impostor rejection rate: {100*impostor_rejects/impostor_count:.1f}%")
    print()
    
    df = pd.DataFrame(all_attempts)
    
    start_date = datetime(2025, 1, 15, 9, 0, 0)
    timestamps = []
    for _, row in df.iterrows():
        if row['genuine']:
            day_offset = [0, 2, 6, 13][row['session'] - 1]
            dt = start_date + timedelta(days=day_offset, minutes=row['attempt']*3)
        else:
            dt = start_date + timedelta(minutes=random.randint(0, 120))
        timestamps.append(int(dt.timestamp()))
    df['timestamp'] = timestamps
    
    column_order = [
        'participant_id', 'session', 'attempt', 'genuine', 'timestamp',
        'force_0', 'force_1', 'force_2',
        'hold_0', 'hold_1', 'hold_2',
        'gap_0', 'gap_1',
        'voice_0', 'voice_1', 'voice_2',
        'gap_dist_0', 'gap_dist_1',
        'baseline_noise', 'mean_distance',
        'S_M', 'S_V', 'S_G', 'S_U', 'decision'
    ]
    df = df[column_order]
    
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✓ Dataset saved to {OUTPUT_CSV}")
    print(f"  Total attempts: {len(df)}")
    print(f"  Genuine: {genuine_count}, Impostor: {impostor_count}")
    print()
    
    print("=" * 60)
    print("=== SUMMARY STATISTICS ===")
    print("=" * 60)
    print(f"False Accept Rate (FAR):  {100*(impostor_count - impostor_rejects)/impostor_count:.3f}%")
    print(f"False Reject Rate (FRR):  {100*(genuine_count - genuine_accepts)/genuine_count:.1f}%")
    print(f"Overall Accuracy:         {100*(genuine_accepts + impostor_rejects)/(genuine_count + impostor_count):.1f}%")
    print()
    
    print("Per-Layer Performance:")
    print("-" * 40)
    
    voice_accepts_genuine = sum(1 for a in all_attempts if a['genuine'] and a['S_V'] >= 45)
    voice_accepts_impostor = sum(1 for a in all_attempts if not a['genuine'] and a['S_V'] >= 45)
    print(f"  Voice (3 digits): FAR={100*voice_accepts_impostor/impostor_count:.1f}%, FRR={100*(genuine_count-voice_accepts_genuine)/genuine_count:.1f}%")
    
    gap_accepts_genuine = sum(1 for a in all_attempts if a['genuine'] and a['S_G'] >= 50)
    gap_accepts_impostor = sum(1 for a in all_attempts if not a['genuine'] and a['S_G'] >= 50)
    print(f"  Spatial Gap: FAR={100*gap_accepts_impostor/impostor_count:.1f}%, FRR={100*(genuine_count-gap_accepts_genuine)/genuine_count:.1f}%")
    
    mech_accepts_genuine = sum(1 for a in all_attempts if a['genuine'] and a['S_M'] >= 55)
    mech_accepts_impostor = sum(1 for a in all_attempts if not a['genuine'] and a['S_M'] >= 55)
    print(f"  Mechanical (FHG): FAR={100*mech_accepts_impostor/impostor_count:.1f}%, FRR={100*(genuine_count-mech_accepts_genuine)/genuine_count:.1f}%")
    
    spat_accepts_genuine = sum(1 for a in all_attempts if a['genuine'] and a['S_U'] >= 55)
    spat_accepts_impostor = sum(1 for a in all_attempts if not a['genuine'] and a['S_U'] >= 55)
    print(f"  Spatial Pos: FAR={100*spat_accepts_impostor/impostor_count:.1f}%, FRR={100*(genuine_count-spat_accepts_genuine)/genuine_count:.1f}%")
    print()
    
    print("=" * 60)
    print("✓ DATASET GENERATION COMPLETE!")
    print("=" * 60)
    
    return df

if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    df = generate_full_dataset()
    print("\nFirst 10 rows:")
    print("-" * 40)
    print(df.head(10).to_string())