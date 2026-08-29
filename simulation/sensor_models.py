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
        np.random.seed(user_id * 42)  # Reproducible randomness
        
        # Force levels (ADC values 0-1023)
        # Soft: <200, Medium: 200-500, Hard: >500
        self.force_levels = [
            np.random.randint(80, 180),      # Soft press
            np.random.randint(520, 720),     # Hard press
            np.random.randint(220, 420)      # Medium press
        ]
        
        # Hold times (ms)
        # Short: <200, Medium: 200-400, Long: >400
        self.hold_times = [
            np.random.randint(150, 250),     # Short hold
            np.random.randint(380, 520),     # Long hold
            np.random.randint(220, 380)      # Medium hold
        ]
        
        # Gap times (ms) - only N-1 gaps for N presses
        # Brief: <300, Normal: 300-600, Extended: >600
        self.gap_times = [
            np.random.randint(450, 580),     # Normal gap
            np.random.randint(250, 380)      # Normal-Brief gap
        ]
        
        # Spatial characteristics (mm)
        self.mean_distance = np.random.randint(150, 350)  # 15-35cm from sensor
        self.gesture_length = 20 + np.random.randint(-3, 3)  # ~20 samples at 10Hz
        
        # Acoustic baseline (depends on environment)
        self.baseline_noise = np.random.randint(15, 35)  # ADC RMS units
        
        # Intra-user variance (individual variability)
        self.force_std = np.random.uniform(0.25, 0.45)      # ±0.3N typical
        self.hold_std = np.random.uniform(60, 95)           # ±75ms typical
        self.gap_std = np.random.uniform(70, 110)           # ±85ms typical
        self.distance_std = np.random.uniform(15, 35)       # ±25mm typical

class FSRSimulator:
    """Simulates FSR-402 force sensor with realistic noise"""
    def __init__(self):
        # FSR-402 specs: 0.2N-20N range, ±5% accuracy, 1-5% noise
        self.adc_resolution = 1024
        self.noise_factor = 0.03  # 3% noise (from datasheet)
    
    def read_press(self, true_force_adc, user_variance):
        """
        Simulate a force reading with sensor noise + user variance
        
        Args:
            true_force_adc: User's intended force (ADC units)
            user_variance: User's typical force variation (ADC units)
        
        Returns:
            Simulated ADC reading (0-1023)
        """
        # User variance (Gaussian)
        user_noise = np.random.normal(0, user_variance * 50)  # Scale to ADC
        
        # Sensor noise (±3% of reading)
        sensor_noise = np.random.normal(0, true_force_adc * self.noise_factor)
        
        # Quantization noise (ADC discrete steps)
        quantization = np.random.uniform(-0.5, 0.5)
        
        reading = true_force_adc + user_noise + sensor_noise + quantization
        
        # Clamp to valid ADC range
        return int(np.clip(reading, 0, 1023))
    
    def read_hold_time(self, true_hold_ms, user_variance):
        """Simulate hold duration with user variance"""
        noise = np.random.normal(0, user_variance)
        return int(max(50, true_hold_ms + noise))  # Minimum 50ms hold
    
    def read_gap_time(self, true_gap_ms, user_variance):
        """Simulate inter-press gap with user variance"""
        noise = np.random.normal(0, user_variance)
        return int(max(100, true_gap_ms + noise))  # Minimum 100ms gap

class MicrophoneSimulator:
    """Simulates MAX4466 electret microphone with AGC"""
    def __init__(self):
        self.sample_rate = 8000  # 8kHz
        self.adc_resolution = 1024
    
    def measure_baseline(self, user_baseline, ambient_variation=5):
        """
        Measure ambient noise RMS
        
        Args:
            user_baseline: User's environment baseline (ADC RMS units)
            ambient_variation: Random environmental fluctuation
        
        Returns:
            RMS energy as integer
        """
        # Simulate 1-second averaging with some variation
        variation = np.random.normal(0, ambient_variation)
        return int(max(5, user_baseline + variation))
    
    def measure_press_rms(self, baseline, press_force_adc):
        """
        Simulate acoustic RMS during press event
        
        Press force correlates with acoustic energy (harder press = louder)
        
        Args:
            baseline: Ambient noise level
            press_force_adc: Force of press (ADC units)
        
        Returns:
            RMS energy during press
        """
        # Model: RMS = baseline + k*sqrt(force)
        # Harder presses generate more acoustic energy
        acoustic_energy = baseline + 0.15 * np.sqrt(press_force_adc)
        
        # Add microphone noise (±10%)
        noise = np.random.normal(0, acoustic_energy * 0.1)
        
        return int(max(baseline, acoustic_energy + noise))
    
    def check_temporal_correlation(self, press_timestamp_ms):
        """
        Simulate temporal alignment check
        In real system: acoustic peak within ±50ms of mechanical peak
        
        Returns:
            True if synchronized (95% success for genuine users)
        """
        # Genuine users: 95% temporal correlation success
        # Simulated jitter: ±30ms typical (within 50ms threshold)
        jitter = np.random.normal(0, 30)
        return abs(jitter) < 50

class UltrasonicSimulator:
    """Simulates HC-SR04 ultrasonic ranging sensor"""
    def __init__(self):
        self.min_range_mm = 20     # 2cm minimum
        self.max_range_mm = 4000   # 400cm maximum
        self.accuracy_mm = 3       # ±3mm (from datasheet)
    
    def read_distance(self, true_distance_mm):
        """
        Simulate distance measurement
        
        Args:
            true_distance_mm: Actual distance to hand
        
        Returns:
            Measured distance in mm
        """
        # Measurement noise (±3mm typical)
        noise = np.random.normal(0, self.accuracy_mm)
        
        # Temperature effect (±1% per 10°C deviation from 20°C)
        # Assume indoor: 18-28°C range → ±0.8% error
        temp_error = np.random.normal(0, true_distance_mm * 0.008)
        
        measured = true_distance_mm + noise + temp_error
        
        # Invalid readings (1% failure rate per datasheet)
        if np.random.random() < 0.01:
            return 0  # Timeout/no echo
        
        return int(np.clip(measured, self.min_range_mm, self.max_range_mm))
    
    def measure_gesture(self, user_mean_distance, user_std, num_samples=20):
        """
        Simulate gesture distance sampling (10Hz for 2s = 20 samples)
        
        Args:
            user_mean_distance: User's typical hand distance
            user_std: User's distance variation
            num_samples: Number of samples (default 20)
        
        Returns:
            List of distance measurements
        """
        distances = []
        for _ in range(num_samples):
            # User's hand moves slightly during gesture
            true_dist = np.random.normal(user_mean_distance, user_std)
            measured = self.read_distance(true_dist)
            if measured > 0:  # Valid reading
                distances.append(measured)
        
        return distances

class ImpostorModel:
    """Models impostor attack behavior after observation"""
    def __init__(self, genuine_profile, observation_accuracy=0.7):
        """
        Args:
            genuine_profile: The UserProfile being imitated
            observation_accuracy: How well impostor estimates values (0-1)
                0.5 = random guess
                0.7 = typical shoulder-surfing (from Tari2006)
                0.9 = video-assisted attack
        """
        self.genuine = genuine_profile
        self.accuracy = observation_accuracy
        
        # Impostor's estimate of genuine user's password
        # Force: partially observable (can see hand pressure, not exact newtons)
        self.estimated_force = [
            self._estimate_value(f, error_range=150) 
            for f in genuine_profile.force_levels
        ]
        
        # Hold time: observable with error (±200ms typical from Tari2006)
        self.estimated_hold = [
            self._estimate_value(h, error_range=200) 
            for h in genuine_profile.hold_times
        ]
        
        # Gap time: partially observable
        self.estimated_gap = [
            self._estimate_value(g, error_range=180) 
            for g in genuine_profile.gap_times
        ]
        
        # Spatial: mostly hidden (can see gesture direction, not exact distance)
        self.estimated_distance = self._estimate_value(
            genuine_profile.mean_distance, 
            error_range=80
        )
    
    def _estimate_value(self, true_value, error_range):
        """
        Impostor's noisy estimate of a genuine user's value
        
        Args:
            true_value: Actual value
            error_range: Maximum estimation error (uniform distribution)
        
        Returns:
            Estimated value with observation error
        """
        # Error based on observation accuracy
        error = np.random.uniform(
            -error_range * (1 - self.accuracy),
            error_range * (1 - self.accuracy)
        )
        return true_value + error
    
    def generate_attempt(self):
        """
        Generate an impostor authentication attempt
        
        Returns:
            Dictionary with estimated values (same structure as genuine attempt)
        """
        return {
            'force': self.estimated_force,
            'hold': self.estimated_hold,
            'gap': self.estimated_gap,
            'distance': self.estimated_distance,
            # Impostors have higher variance (less practiced)
            'force_std': 0.6,  # Higher than genuine (0.3-0.45)
            'hold_std': 140,
            'gap_std': 160,
            'distance_std': 50
        }

# ========================================
# HELPER FUNCTIONS
# ========================================

def create_user_profiles(num_users=8):
    """Create N synthetic user profiles with diverse FHG passwords"""
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

def compute_mechanical_score(observed, enrolled_mean, delta_f=50, delta_t=100):
    """
    Compute mechanical similarity score (Eq. 3 from paper)
    
    Args:
        observed: [force, hold, gap] for 3 presses
        enrolled_mean: Template mean values
        delta_f, delta_t: Tolerance bands
    
    Returns:
        Score 0-100 (integer)
    """
    scores = []
    for i in range(3):
        dF = (observed['force'][i] - enrolled_mean['force'][i]) / delta_f
        dH = (observed['hold'][i] - enrolled_mean['hold'][i]) / delta_t
        dG = 0
        if i < 2:
            dG = (observed['gap'][i] - enrolled_mean['gap'][i]) / delta_t
        
        # 3D distance
        dist_sq = dF**2 + dH**2 + dG**2
        
        # Gaussian kernel: exp(-d^2)
        similarity = np.exp(-dist_sq)
        scores.append(similarity)
    
    # Average score, scale to 0-100
    return int(100 * np.mean(scores))

def compute_acoustic_score(press_forces, baseline, threshold_multiplier=1.5):
    """
    Compute acoustic liveness score (Eq. 4 from paper)
    
    Args:
        press_forces: List of force values for each press
        baseline: Ambient noise baseline
        threshold_multiplier: Energy threshold (1.5x baseline)
    
    Returns:
        Score 0-100
    """
    mic_sim = MicrophoneSimulator()
    valid_count = 0
    
    for force in press_forces:
        rms = mic_sim.measure_press_rms(baseline, force)
        energy_ok = (rms > threshold_multiplier * baseline)
        temporal_ok = mic_sim.check_temporal_correlation(0)  # Timestamp unused in sim
        
        if energy_ok and temporal_ok:
            valid_count += 1
    
    # Score: percentage of valid presses
    return int(100 * valid_count / len(press_forces))

def compute_spatial_score(observed_dist, enrolled_dist, tolerance_mm=50):
    """
    Compute spatial similarity score (Eq. 5 from paper)
    
    Args:
        observed_dist: Mean distance during auth
        enrolled_dist: Enrolled mean distance
        tolerance_mm: Distance tolerance (50mm)
    
    Returns:
        Score 0-100
    """
    dist_diff = abs(observed_dist - enrolled_dist)
    
    # Exponential decay: exp(-|diff| / tolerance)
    similarity = np.exp(-dist_diff / tolerance_mm)
    
    return int(100 * similarity)

def apply_fusion_rule(s_m, s_a, s_u, thresh_m=70, thresh_a=66, thresh_u=70):
    """
    AND-rule fusion (Eq. 6 from paper)
    
    Returns:
        True if ACCEPT, False if REJECT
    """
    return (s_m > thresh_m) and (s_a > thresh_a) and (s_u > thresh_u)

if __name__ == "__main__":
    print("=== Sensor Model Test ===\n")
    
    # Create test user
    user = UserProfile(user_id=1)
    
    # Test FSR
    fsr = FSRSimulator()
    print("FSR Test (5 readings of same press):")
    for _ in range(5):
        reading = fsr.read_press(user.force_levels[0], user.force_std)
        print(f"  {reading} ADC")
    print()
    
    # Test Microphone
    mic = MicrophoneSimulator()
    baseline = mic.measure_baseline(user.baseline_noise)
    print(f"Microphone baseline: {baseline} RMS")
    print("Press acoustic energy:")
    for force in user.force_levels:
        rms = mic.measure_press_rms(baseline, force)
        print(f"  Force {force} → RMS {rms}")
    print()
    
    # Test Ultrasonic
    ultra = UltrasonicSimulator()
    distances = ultra.measure_gesture(user.mean_distance, user.distance_std)
    print(f"Ultrasonic gesture: {len(distances)} samples")
    print(f"  Mean: {np.mean(distances):.1f}mm, Std: {np.std(distances):.1f}mm")