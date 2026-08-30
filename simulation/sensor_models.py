"""
sensor_models.py - Realistic Sensor Simulation Models
Based on FSR-402, MAX4466, and HC-SR04 specifications

OPTIMIZED VERSION - Lower FRR with balanced security
"""

import numpy as np
import random

class UserProfile:
    """Represents a single user's authentication characteristics"""
    def __init__(self, user_id):
        self.user_id = user_id
        np.random.seed(user_id * 42)
        
        # ========================================
        # LAYER 1: FORCE PATTERN (FHG)
        # ========================================
        self.force_levels = [
            np.random.randint(80, 180),
            np.random.randint(520, 720),
            np.random.randint(220, 420)
        ]
        
        self.hold_times = [
            np.random.randint(150, 250),
            np.random.randint(380, 520),
            np.random.randint(220, 380)
        ]
        
        self.gap_times = [
            np.random.randint(450, 580),
            np.random.randint(250, 380)
        ]
        
        # ========================================
        # LAYER 2: VOICE DIGITS
        # ========================================
        self.voice_digits = [
            np.random.randint(0, 9),
            np.random.randint(0, 9),
            np.random.randint(0, 9)
        ]
        
        # ========================================
        # LAYER 3: SPATIAL GAP
        # ========================================
        self.gap_distances = [
            np.random.randint(20, 80),
            np.random.randint(20, 80)
        ]
        
        # ========================================
        # LAYER 4: SPATIAL POSITION
        # ========================================
        self.mean_distance = np.random.randint(150, 350)
        self.gesture_length = 20 + np.random.randint(-3, 3)
        
        # ========================================
        # ACOUSTIC BASELINE
        # ========================================
        self.baseline_noise = np.random.randint(15, 35)
        
        # ========================================
        # INTRA-USER VARIANCE
        # ========================================
        self.force_std = np.random.uniform(0.25, 0.45)
        self.hold_std = np.random.uniform(60, 95)
        self.gap_std = np.random.uniform(70, 110)
        self.distance_std = np.random.uniform(15, 35)
        self.voice_error_rate = np.random.uniform(0.02, 0.08)
        self.gap_distance_std = np.random.uniform(8, 18)  # Increased from 5-15

class FSRSimulator:
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
    def __init__(self):
        self.sample_rate = 8000
    
    def measure_baseline(self, user_baseline, ambient_variation=5):
        variation = np.random.normal(0, ambient_variation)
        return int(max(5, user_baseline + variation))
    
    def recognize_digit(self, true_digit, error_rate):
        if np.random.random() < error_rate:
            return np.random.randint(0, 9)
        return true_digit
    
    def measure_press_rms(self, baseline, press_force_adc):
        baseline = max(baseline, 1)
        press_force_adc = max(press_force_adc, 0)
        rms = baseline + 15 + (press_force_adc / 8)
        noise = np.random.normal(0, 3)
        return int(max(baseline + 5, rms + noise))
    
    def check_temporal_correlation(self, press_timestamp_ms):
        jitter = np.random.normal(0, 30)
        return abs(jitter) < 50

class UltrasonicSimulator:
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
    
    def measure_gap_distance(self, true_distance_mm, user_std):
        true_distance_mm = abs(true_distance_mm)
        user_noise = np.random.normal(0, user_std)
        return self.read_distance(true_distance_mm + user_noise)
    
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
    def __init__(self, genuine_profile, observation_accuracy=0.7):
        self.genuine = genuine_profile
        self.accuracy = observation_accuracy
        
        # Force estimates
        self.estimated_force = [self._estimate_value(f, 150) for f in genuine_profile.force_levels]
        self.estimated_hold = [self._estimate_value(h, 200) for h in genuine_profile.hold_times]
        self.estimated_gap = [self._estimate_value(g, 180) for g in genuine_profile.gap_times]
        
        # Voice estimates
        self.estimated_voice_digits = []
        for digit in genuine_profile.voice_digits:
            if np.random.random() < 0.3:
                self.estimated_voice_digits.append(np.random.randint(0, 9))
            else:
                self.estimated_voice_digits.append(digit)
        
        # Gap distance estimates
        self.estimated_gap_distances = [
            self._estimate_value(g, 30) for g in genuine_profile.gap_distances
        ]
        
        self.estimated_distance = self._estimate_value(genuine_profile.mean_distance, 80)
    
    def _estimate_value(self, true_value, error_range):
        error = np.random.uniform(-error_range * (1 - self.accuracy), error_range * (1 - self.accuracy))
        return true_value + error
    
    def generate_attempt(self):
        return {
            'force': self.estimated_force,
            'hold': self.estimated_hold,
            'gap': self.estimated_gap,
            'voice_digits': self.estimated_voice_digits,
            'gap_distances': self.estimated_gap_distances,
            'distance': abs(self.estimated_distance),
            'force_std': 0.6,
            'hold_std': 140,
            'gap_std': 160,
            'gap_distance_std': 30,  # Increased
            'distance_std': 50
        }

# ========================================
# SCORE COMPUTATION FUNCTIONS (OPTIMIZED)
# ========================================

def create_user_profiles(num_users=8):
    profiles = []
    for i in range(num_users):
        profile = UserProfile(user_id=i+1)
        profiles.append(profile)
        print(f"User P{i+1:02d}:")
        print(f"  Force:    {profile.force_levels} (ADC)")
        print(f"  Hold:     {profile.hold_times} ms")
        print(f"  Gap:      {profile.gap_times} ms")
        print(f"  Voice:    {profile.voice_digits} (digits)")
        print(f"  Gap Dist: {profile.gap_distances} mm")
        print(f"  Distance: {profile.mean_distance} mm")
        print()
    return profiles

def compute_mechanical_score(observed, enrolled_mean, delta_f=100, delta_t=200):
    """
    OPTIMIZED: More forgiving tolerance bands
    delta_f: 100 ADC (was 80)
    delta_t: 200 ms (was 150)
    """
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

def compute_voice_score(observed_digits, enrolled_digits):
    if len(observed_digits) != 3 or len(enrolled_digits) != 3:
        return 0
    correct = sum(1 for i in range(3) if observed_digits[i] == enrolled_digits[i])
    return int(100 * correct / 3)

def compute_spatial_gap_score(observed_gaps, enrolled_gaps, tolerance_mm=25):
    """
    OPTIMIZED: More forgiving tolerance
    tolerance_mm: 25mm (was 15mm)
    """
    if len(observed_gaps) != 2 or len(enrolled_gaps) != 2:
        return 0
    scores = []
    for i in range(2):
        dist_diff = abs(observed_gaps[i] - enrolled_gaps[i])
        similarity = np.exp(-dist_diff / tolerance_mm)
        scores.append(similarity)
    return int(100 * np.mean(scores))

def compute_spatial_score(observed_dist, enrolled_dist, tolerance_mm=40):
    """
    OPTIMIZED: More forgiving tolerance
    tolerance_mm: 40mm (was 30mm)
    """
    observed_dist = abs(observed_dist)
    enrolled_dist = abs(enrolled_dist)
    dist_diff = abs(observed_dist - enrolled_dist)
    similarity = np.exp(-dist_diff / tolerance_mm)
    return int(100 * similarity)

# ========================================
# FUSION FUNCTIONS (OPTIMIZED)
# ========================================

def apply_hybrid_fusion(s_m, s_v, s_g, s_u,
                        seq_thresh_m=50, seq_thresh_v=40, seq_thresh_g=45, seq_thresh_u=50,
                        weighted_threshold=53):
    """
    OPTIMIZED: Lower thresholds for better FRR
    Sequential thresholds: M=50, V=40, G=45, U=50 (was 55, 45, 50, 55)
    Weighted threshold: 53 (was 58)
    """
    # Step 1: Sequential early check
    if s_v < seq_thresh_v:
        return False, "Voice (early)"
    if s_g < seq_thresh_g:
        return False, "Spatial Gap (early)"
    if s_m < seq_thresh_m:
        return False, "Mechanical (early)"
    if s_u < seq_thresh_u:
        return False, "Spatial Position (early)"
    
    # Step 2: Weighted final check
    weights = {'S_M': 0.35, 'S_V': 0.15, 'S_G': 0.25, 'S_U': 0.25}
    s_total = (s_m * weights['S_M'] + 
               s_v * weights['S_V'] + 
               s_g * weights['S_G'] + 
               s_u * weights['S_U'])
    
    if s_total > weighted_threshold:
        return True, s_total
    else:
        return False, f"Weighted (score={s_total:.1f})"

def apply_simplified_fusion(s_m, s_v, s_g, s_u,
                            thresh_m=50, thresh_v=40, thresh_g=45, thresh_u=50):
    """
    OPTIMIZED: Simplified AND-fusion with lower thresholds
    """
    return (s_m > thresh_m) and (s_v > thresh_v) and (s_g > thresh_g) and (s_u > thresh_u)

if __name__ == "__main__":
    print("=== Sensor Model Test ===\n")
    user = UserProfile(user_id=1)
    
    print(f"User Profile:")
    print(f"  Force:    {user.force_levels}")
    print(f"  Hold:     {user.hold_times}")
    print(f"  Gap:      {user.gap_times}")
    print(f"  Voice:    {user.voice_digits}")
    print(f"  Gap Dist: {user.gap_distances}")
    print(f"  Distance: {user.mean_distance} mm")
    print()
    
    fsr = FSRSimulator()
    print("FSR Test (5 readings):")
    for _ in range(5):
        reading = fsr.read_press(user.force_levels[0], user.force_std)
        print(f"  {reading} ADC")
    print()
    
    mic = MicrophoneSimulator()
    baseline = mic.measure_baseline(user.baseline_noise)
    print(f"Microphone baseline: {baseline} RMS")
    print("Voice digit recognition test:")
    for i, digit in enumerate(user.voice_digits):
        recognized = mic.recognize_digit(digit, user.voice_error_rate)
        print(f"  Press {i+1}: Said {digit} → Recognized {recognized}")
    print()
    
    ultra = UltrasonicSimulator()
    print("Spatial gap test:")
    for i, gap_dist in enumerate(user.gap_distances):
        measured = ultra.measure_gap_distance(gap_dist, user.distance_std)
        print(f"  Gap {i+1}: True {gap_dist}mm → Measured {measured}mm")
    print()