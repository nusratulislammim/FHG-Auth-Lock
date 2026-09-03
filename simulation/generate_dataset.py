"""
generate_dataset.py - Generate N=50 synthetic dataset
WORKING VERSION
"""
import os
import random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from tqdm import tqdm

from sensor_models import (
    create_user_profiles, FSRSimulator, MicrophoneSimulator,
    UltrasonicSimulator, VoiceSimulator, SpeakerModel, ImpostorModel,
    compute_mechanical_score, compute_spatial_gap_score, apply_hybrid_fusion
)

# ============================================================
# CONFIGURATION
# ============================================================

NUM_USERS = 50
TRAIN_SPLIT = 0.60

OUTPUT_DIR = "../data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_CSV_TRAIN = os.path.join(OUTPUT_DIR, "train_data.csv")
OUTPUT_CSV_TEST  = os.path.join(OUTPUT_DIR, "test_data.csv")
OUTPUT_CSV_ALL   = os.path.join(OUTPUT_DIR, "experimental_data_N50.csv")

# ============================================================
# GENUINE ATTEMPT
# ============================================================

def simulate_genuine_attempt(user_profile, session_num: int, attempt_num: int):
    fsr   = FSRSimulator()
    mic   = MicrophoneSimulator()
    ultra = UltrasonicSimulator()

    baseline = mic.measure_baseline(user_profile.baseline_noise)

    observed_force, observed_hold, observed_gap = [], [], []

    for i in range(3):
        force = fsr.read_press(user_profile.force_levels[i], user_profile.force_std)
        hold  = fsr.read_hold_time(user_profile.hold_times[i], user_profile.hold_std)
        observed_force.append(force)
        observed_hold.append(hold)
        if i < 2:
            gap = fsr.read_gap_time(user_profile.gap_times[i], user_profile.gap_std)
            observed_gap.append(gap)

    voice_sim    = VoiceSimulator(user_profile.speaker)
    voice_scores, voice_digits, voice_dtw_costs = [], [], []

    for i in range(3):
        energy = baseline + 5 + observed_force[i] / 10
        score, dtw_cost = voice_sim.simulate_voice_attempt(
            true_digit   = user_profile.voice_digits[i],
            template     = user_profile.voice_template,
            energy_level = energy,
            digit_index  = i
        )
        voice_scores.append(score)
        voice_dtw_costs.append(dtw_cost)
        voice_digits.append(
            user_profile.voice_digits[i] if score >= 60
            else int(np.random.randint(0, 10))
        )

    s_v = int(np.mean(voice_scores))

    observed_gap_distances = []
    for i in range(2):
        measured = ultra.measure_gap_distance(
            user_profile.gap_distances[i], user_profile.gap_distance_std
        )
        observed_gap_distances.append(measured)

    enrolled = {
        "force": user_profile.force_levels,
        "hold":  user_profile.hold_times,
        "gap":   user_profile.gap_times
    }
    observed = {
        "force": observed_force,
        "hold":  observed_hold,
        "gap":   observed_gap
    }

    s_m = compute_mechanical_score(observed, enrolled)
    s_g = compute_spatial_gap_score(observed_gap_distances, user_profile.gap_distances)
    decision, details = apply_hybrid_fusion(s_m=s_m, s_v=s_v, s_g=s_g)

    return {
        "participant_id": f"P{user_profile.user_id:02d}",
        "session":        session_num,
        "attempt":        attempt_num,
        "genuine":        True,
        "target_id":      "",
        "force_1": observed_force[0], "force_2": observed_force[1], "force_3": observed_force[2],
        "hold_1":  observed_hold[0],  "hold_2":  observed_hold[1],  "hold_3":  observed_hold[2],
        "gap_1":   observed_gap[0],   "gap_2":   observed_gap[1],
        "voice_digit_1": voice_digits[0],
        "voice_digit_2": voice_digits[1],
        "voice_digit_3": voice_digits[2],
        "voice_dtw_1": voice_dtw_costs[0],
        "voice_dtw_2": voice_dtw_costs[1],
        "voice_dtw_3": voice_dtw_costs[2],
        "gap_dist_1": observed_gap_distances[0],
        "gap_dist_2": observed_gap_distances[1],
        "baseline": baseline,
        "S_M": s_m, "S_V": s_v, "S_G": s_g,
        "decision": "ACCEPT" if decision else "REJECT",
        "fusion_details": str(details)
    }

# ============================================================
# IMPOSTOR ATTEMPT
# ============================================================

def simulate_impostor_attempt(impostor_id: int, target_profile):
    impostor_speaker = SpeakerModel(impostor_id + 10000)
    impostor = ImpostorModel(target_profile, impostor_speaker, observation_accuracy=0.70)
    attempt_data = impostor.generate_attempt()

    fsr   = FSRSimulator()
    mic   = MicrophoneSimulator()
    ultra = UltrasonicSimulator()

    baseline = mic.measure_baseline(target_profile.baseline_noise)

    observed_force, observed_hold, observed_gap = [], [], []

    for i in range(3):
        force = fsr.read_press(int(attempt_data["force"][i]), attempt_data["force_std"])
        hold  = fsr.read_hold_time(int(attempt_data["hold"][i]), attempt_data["hold_std"])
        observed_force.append(force)
        observed_hold.append(hold)
        if i < 2:
            gap = fsr.read_gap_time(int(attempt_data["gap"][i]), attempt_data["gap_std"])
            observed_gap.append(gap)

    voice_sim    = VoiceSimulator(impostor_speaker)
    voice_scores, voice_digits, voice_dtw_costs = [], [], []

    for i in range(3):
        energy = baseline + 5 + observed_force[i] / 10
        attempted_digit = attempt_data["voice_digits"][i]
        score, dtw_cost = voice_sim.simulate_voice_attempt(
            true_digit   = attempted_digit,
            template     = target_profile.voice_template,
            energy_level = energy,
            digit_index  = i
        )
        voice_scores.append(score)
        voice_dtw_costs.append(dtw_cost)
        voice_digits.append(
            attempted_digit if score >= 55
            else int(np.random.randint(0, 10))
        )

    s_v = int(np.mean(voice_scores))

    observed_gap_distances = []
    for i in range(2):
        measured = ultra.measure_gap_distance(
            int(attempt_data["gap_distances"][i]),
            attempt_data["gap_distance_std"]
        )
        observed_gap_distances.append(measured)

    enrolled = {
        "force": target_profile.force_levels,
        "hold":  target_profile.hold_times,
        "gap":   target_profile.gap_times
    }
    observed = {
        "force": observed_force,
        "hold":  observed_hold,
        "gap":   observed_gap
    }

    s_m = compute_mechanical_score(observed, enrolled)
    s_g = compute_spatial_gap_score(observed_gap_distances, target_profile.gap_distances)
    decision, details = apply_hybrid_fusion(s_m=s_m, s_v=s_v, s_g=s_g)

    return {
        "participant_id": f"P{impostor_id:02d}",
        "session":        99,
        "attempt":        0,
        "genuine":        False,
        "target_id":      f"P{target_profile.user_id:02d}",
        "force_1": observed_force[0], "force_2": observed_force[1], "force_3": observed_force[2],
        "hold_1":  observed_hold[0],  "hold_2":  observed_hold[1],  "hold_3":  observed_hold[2],
        "gap_1":   observed_gap[0],   "gap_2":   observed_gap[1],
        "voice_digit_1": voice_digits[0],
        "voice_digit_2": voice_digits[1],
        "voice_digit_3": voice_digits[2],
        "voice_dtw_1": voice_dtw_costs[0],
        "voice_dtw_2": voice_dtw_costs[1],
        "voice_dtw_3": voice_dtw_costs[2],
        "gap_dist_1": observed_gap_distances[0],
        "gap_dist_2": observed_gap_distances[1],
        "baseline": baseline,
        "S_M": s_m, "S_V": s_v, "S_G": s_g,
        "decision": "ACCEPT" if decision else "REJECT",
        "fusion_details": str(details)
    }

# ============================================================
# TIMESTAMPS
# ============================================================

def add_timestamps(attempts):
    start_date = datetime(2025, 1, 15, 9, 0, 0)
    session_offsets = [0, 2, 6, 13, 20]
    for row in attempts:
        if row["genuine"]:
            session = int(row["session"])
            attempt = int(row["attempt"])
            day_offset = session_offsets[min(session - 1, 4)]
            dt = start_date + timedelta(days=day_offset, minutes=attempt * 3)
        else:
            dt = start_date + timedelta(minutes=random.randint(0, 120))
        row["timestamp"] = dt.isoformat(sep=" ")
    return attempts

# ============================================================
# MAIN
# ============================================================

def generate_full_dataset():
    print("\n" + "=" * 70)
    print("DATASET GENERATION (N=50)")
    print("=" * 70)

    print("\n[1/5] Creating user profiles...")
    users = create_user_profiles(NUM_USERS)

    random.shuffle(users)
    n_train     = int(NUM_USERS * TRAIN_SPLIT)
    train_users = users[:n_train]
    test_users  = users[n_train:]

    print(f"\nTraining users: {len(train_users)}")
    print(f"Testing users:  {len(test_users)}")

    all_attempts   = []
    train_attempts = []
    test_attempts  = []

    GENUINE_PER_USER  = 30   # 5 sessions × 6 attempts
    IMPOSTOR_PER_USER = 15   # 3 targets × 5 attempts
    total_train = len(train_users) * (GENUINE_PER_USER + IMPOSTOR_PER_USER)
    total_test  = len(test_users)  * (GENUINE_PER_USER + IMPOSTOR_PER_USER)

    print("\n[2/5] Generating training data...")
    with tqdm(total=total_train, desc="  Training", unit="attempt") as pbar:
        for user in train_users:
            for session in range(1, 6):
                for attempt in range(1, 7):
                    data = simulate_genuine_attempt(user, session, attempt)
                    train_attempts.append(data)
                    all_attempts.append(data)
                    pbar.update(1)

            others  = [u for u in train_users if u.user_id != user.user_id]
            targets = random.sample(others, min(3, len(others)))
            for target in targets:
                for _ in range(5):
                    data = simulate_impostor_attempt(user.user_id, target)
                    train_attempts.append(data)
                    all_attempts.append(data)
                    pbar.update(1)

    print("\n[3/5] Generating test data...")
    with tqdm(total=total_test, desc="  Testing", unit="attempt") as pbar:
        for user in test_users:
            for session in range(1, 6):
                for attempt in range(1, 7):
                    data = simulate_genuine_attempt(user, session, attempt)
                    test_attempts.append(data)
                    all_attempts.append(data)
                    pbar.update(1)

            others  = [u for u in test_users if u.user_id != user.user_id]
            targets = random.sample(others, min(3, len(others)))
            for target in targets:
                for _ in range(5):
                    data = simulate_impostor_attempt(user.user_id, target)
                    test_attempts.append(data)
                    all_attempts.append(data)
                    pbar.update(1)

    print("\n[4/5] Adding timestamps...")
    add_timestamps(train_attempts)
    add_timestamps(test_attempts)
    add_timestamps(all_attempts)

    print("\n[5/5] Saving datasets...")
    cols = [
        "participant_id", "target_id", "session", "attempt", "genuine", "timestamp",
        "force_1", "force_2", "force_3",
        "hold_1",  "hold_2",  "hold_3",
        "gap_1",   "gap_2",
        "voice_digit_1", "voice_digit_2", "voice_digit_3",
        "voice_dtw_1",   "voice_dtw_2",   "voice_dtw_3",
        "gap_dist_1", "gap_dist_2", "baseline",
        "S_M", "S_V", "S_G", "decision", "fusion_details"
    ]

    train_df = pd.DataFrame(train_attempts)[cols]
    test_df  = pd.DataFrame(test_attempts)[cols]
    all_df   = pd.DataFrame(all_attempts)[cols]

    train_df.to_csv(OUTPUT_CSV_TRAIN, index=False)
    test_df.to_csv(OUTPUT_CSV_TEST,   index=False)
    all_df.to_csv(OUTPUT_CSV_ALL,     index=False)

    print("\n" + "=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)
    print(f"Training: {len(train_df)} ({train_df['genuine'].sum()} genuine)")
    print(f"Testing:  {len(test_df)}  ({test_df['genuine'].sum()} genuine)")
    print(f"Total:    {len(all_df)}")
    print(f"\nSaved: {OUTPUT_CSV_TRAIN}")
    print(f"Saved: {OUTPUT_CSV_TEST}")
    print(f"Saved: {OUTPUT_CSV_ALL}\n")


if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    generate_full_dataset()