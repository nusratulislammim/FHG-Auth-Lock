"""
sensor_models.py - Realistic Sensor Simulation Models
Based on FSR-402, MAX4466, and HC-SR04 specifications

Authors: Nusratul Islam Mim, Firuze Tasnim Sneha, Al Musabbir, Dr. Md. Sujan Ali
Institution: JKKNIU, Bangladesh
"""

import numpy as np
import random

class UserProfile:
    """Represents a single user's FHG password characteristics"""
    def __init__(self, user_id):
        self.user_id = user_id
        np.random.seed(user_id * 42)
        
        # Force levels (ADC values 0-1023)
        self.force_levels = [
            np.random.randint(80, 180),
            np.random.randint(520, 720),
            np.random.randint(220, 420)
        ]
        
        # Hold times (ms)
        self.hold_times = [
            np.random.randint(150, 250),
            np.random.randint(380, 520),
            np.random.randint(220, 380)
        ]
        
        # Gap times (ms)
        self.gap_times = [
            np.random.randint(450, 580),
            np.random.randint(250, 380)
        ]
        
        # Spatial characteristics (mm)
        self.mean_distance = np.random.randint(150, 350)
        self.gesture_length = 20 + np.random.randint(-3, 3)
        
        # Acoustic baseline
        self.baseline_noise = np.random.randint(15, 35)
        
        # Intra-user variance
        self.force_std = np.random.uniform(0.25, 0.45)
        self.hold_std = np.random.uniform(60, 95)
        self.gap_std = np.random.uniform(70, 110)
        self.distance_std = np.random.uniform(15, 35)

class FSRSimulator:
    """Simulates FSR-402 force sensor"""
    def __init__(self):
        self.adc_resolution = 1024
        self.noise_factor = 0.03
    
    def read_press(self, true_force_adc, user_variance):
        user_noise = np.random.normal(0, user_variance * 50)
        sensor_noise = np.random.normal(0, true_force_adc * self.noise_factor)
        quantization = np.random.uniform(-0.5, 0.5)
        reading = true_force_adc + user_noise + sensor_noise + quantization
        return int(np.clip(reading, 0, 1023))
    
    def read_hold_time(self, true_hold_ms, user_variance):
        noise = np.random.normal(0, user_variance)
        return int(max(50, true_hold_ms + noise))
    
    def read_gap_time(self, true_gap_ms, user_variance):
        noise = np.random.normal(0, user_variance)
        return int(max(100, true_gap_ms + noise))

class MicrophoneSimulator:
    """Simulates MAX4466 electret microphone"""
    def __init__(self):
        self.sample_rate = 8000
    
    def measure_baseline(self, user_baseline, ambient_variation=5):
        variation = np.random.normal(0, ambient_variation)
        return int(max(5, user_baseline + variation))
    
    def measure_press_rms(self, baseline, press_force_adc):
        baseline = max(baseline, 1)
        press_force_adc = max(press_force_adc, 0)
        rms = baseline + 5 + (press_force_adc / 15)
        noise = np.random.normal(0, 2)
        return int(max(baseline + 2, rms + noise))
    
    def check_temporal_correlation(self, press_timestamp_ms):
        jitter = np.random.normal(0, 30)
        return abs(jitter) < 50

class UltrasonicSimulator:
    """Simulates HC-SR04 ultrasonic ranging sensor"""
    def __init__(self):
        self.min_range_mm = 20
        self.max_range_mm = 4000
        self.accuracy_mm = 3
    
    def read_distance(self, true_distance_mm):
        true_distance_mm = abs(true_distance_mm)
        noise = np.random.normal(0, self.accuracy_mm)
        temp_error = np.random.normal(0, true_distance_mm * 0.008)
        measured = true_distance_mm + noise + temp_error
        if np.random.random() < 0.01:
            return 0
        return int(np.clip(measured, self.min_range_mm, self.max_range_mm))
    
    def measure_gesture(self, user_mean_distance, user_std, num_samples=20):
        user_mean_distance = abs(user_mean_distance)
        user_std = abs(user_std)
        distances = []
        for _ in range(num_samples):
            true_dist = np.random.normal(user_mean_distance, user_std)
            measured = self.read_distance(true_dist)
            if measured > 0:
                distances.append(measured)
        return distances

class ImpostorModel:
    """Models impostor attack behavior"""
    def __init__(self, genuine_profile, observation_accuracy=0.7):
        self.genuine = genuine_profile
        self.accuracy = observation_accuracy
        
        self.estimated_force = [
            self._estimate_value(f, error_range=150)
            for f in genuine_profile.force_levels
        ]
        self.estimated_hold = [
            self._estimate_value(h, error_range=200)
            for h in genuine_profile.hold_times
        ]
        self.estimated_gap = [
            self._estimate_value(g, error_range=180)
            for g in genuine_profile.gap_times
        ]
        self.estimated_distance = self._estimate_value(
            genuine_profile.mean_distance, error_range=80
        )
    
    def _estimate_value(self, true_value, error_range):
        error = np.random.uniform(
            -error_range * (1 - self.accuracy),
            error_range * (1 - self.accuracy)
        )
        return true_value + error
    
    def generate_attempt(self):
        return {
            'force': self.estimated_force,
            'hold': self.estimated_hold,
            'gap': self.estimated_gap,
            'distance': abs(self.estimated_distance),
            'force_std': 0.6,
            'hold_std': 140,
            'gap_std': 160,
            'distance_std': 50
        }

# ========================================
# HELPER FUNCTIONS - FINAL TUNED VERSION
# ========================================

def create_user_profiles(num_users=8):
    """Create N synthetic user profiles"""
    profiles = []
    for i in range(num_users):
        profile = UserProfile(user_id=i+1)
        profiles.append(profile)
        print(f"User P{i+1:02d}:")
        print(f"  Force:    {profile.force_levels} (ADC)")
        print(f"  Hold:     {profile.hold_times} ms")
        print(f"  Gap:      {profile.gap_times} ms")
        print(f"  Distance: {profile.mean_distance} mm")
        print(f"  Variance: F±{profile.force_std:.2f}N, H±{profile.hold_std:.0f}ms")
        print()
    return profiles

def compute_mechanical_score(observed, enrolled_mean, delta_f=80, delta_t=150):
    """Compute mechanical similarity score with relaxed tolerances"""
    scores = []
    for i in range(3):
        dF = (observed['force'][i] - enrolled_mean['force'][i]) / delta_f
        dH = (observed['hold'][i] - enrolled_mean['hold'][i]) / delta_t
        dG = 0
        if i < 2 and i < len(observed['gap']):
            dG = (observed['gap'][i] - enrolled_mean['gap'][i]) / delta_t
        
        dist_sq = dF**2 + dH**2 + dG**2
        similarity = np.exp(-dist_sq)
        scores.append(similarity)
    
    return int(100 * np.mean(scores))

def compute_acoustic_score(press_forces, baseline, threshold_multiplier=2.0):
    """Compute acoustic liveness score - STRICTER VERSION"""
    if len(press_forces) == 0:
        return 0
    
    baseline = max(baseline, 1)
    valid_count = 0
    
    for force in press_forces:
        if force is None or force <= 0:
            continue
        
        rms = baseline + 5 + (force / 15)
        energy_ok = (rms > threshold_multiplier * baseline)
        
        if energy_ok:
            valid_count += 1
    
    score = int(100 * valid_count / len(press_forces))
    return score

def compute_spatial_score(observed_dist, enrolled_dist, tolerance_mm=30):
    """Compute spatial similarity score - STRICTER VERSION"""
    observed_dist = abs(observed_dist)
    enrolled_dist = abs(enrolled_dist)
    dist_diff = abs(observed_dist - enrolled_dist)
    similarity = np.exp(-dist_diff / tolerance_mm)
    return int(100 * similarity)

def apply_fusion_rule(s_m, s_a, s_u, thresh_m=70, thresh_a=50, thresh_u=70):
    """AND-rule fusion with balanced thresholds"""
    return (s_m > thresh_m) and (s_a > thresh_a) and (s_u > thresh_u)

if __name__ == "__main__":
    print("=== Sensor Model Test ===\n")
    user = UserProfile(user_id=1)
    
    fsr = FSRSimulator()
    print("FSR Test (5 readings):")
    for _ in range(5):
        reading = fsr.read_press(user.force_levels[0], user.force_std)
        print(f"  {reading} ADC")
    print()
    
    mic = MicrophoneSimulator()
    baseline = mic.measure_baseline(user.baseline_noise)
    print(f"Microphone baseline: {baseline} RMS")
    print()
    
    ultra = UltrasonicSimulator()
    distances = ultra.measure_gesture(user.mean_distance, user.distance_std)
    print(f"Ultrasonic: {len(distances)} samples, mean={np.mean(distances):.1f}mm")