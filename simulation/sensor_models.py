"""
sensor_models.py - WORKING VERSION
Voice divisor=0.3, Mechanical delta_f=50, delta_t=100
"""
import numpy as np
import random
from typing import List, Tuple, Dict, Any
from scipy.spatial.distance import euclidean
from scipy.fft import fft
import warnings

# ============================================================
# SPEAKER MODEL
# ============================================================

class SpeakerModel:
    """Speaker-specific voice characteristics."""
    
    def __init__(self, speaker_id: int):
        if speaker_id < 0:
            raise ValueError("speaker_id must be non-negative")
        
        rng = np.random.default_rng(speaker_id * 100)
        self.speaker_id = speaker_id
        
        # INCREASED variability for good separation
        self.pitch_base = float(rng.normal(120, 40))
        self.pitch_variability = float(rng.uniform(8, 20))
        
        self.formant_f1 = float(rng.normal(400, 120))
        self.formant_f2 = float(rng.normal(900, 200))
        self.formant_f3 = float(rng.normal(2200, 300))
        
        self.duration_factor = float(rng.normal(1.0, 0.15))
        self.energy_factor = float(rng.normal(1.0, 0.20))

# ============================================================
# VOICE TEMPLATE
# ============================================================

class VoiceTemplate:
    """Stores MFCC templates for enrolled digits."""
    
    def __init__(self, digits: List[int], speaker_model: SpeakerModel, num_recordings: int = 3):
        if not all(0 <= d <= 9 for d in digits):
            raise ValueError("All digits must be 0-9")
        
        self.digits = digits
        self.speaker = speaker_model
        self.num_recordings = num_recordings
        self.templates: List[List[np.ndarray]] = []
        
        voice_sim = VoiceSimulator(speaker_model)
        
        for digit in digits:
            digit_templates = []
            for _ in range(num_recordings):
                _, mfcc_seq = voice_sim.generate_waveform_with_frames(
                    digit=digit, energy_level=20, variation=0.08
                )
                digit_templates.append(mfcc_seq)
            self.templates.append(digit_templates)
    
    def get_templates(self, digit_index: int) -> List[np.ndarray]:
        if 0 <= digit_index < len(self.templates):
            return self.templates[digit_index]
        return self.templates[0]

# ============================================================
# VOICE SIMULATOR
# ============================================================

class VoiceSimulator:
    """Voice authentication with MFCC + DTW."""
    
    def __init__(self, speaker_model: SpeakerModel, sample_rate: int = 8000,
                 duration: float = 0.8, frame_size: int = 256,
                 hop_length: int = 128, num_filters: int = 13):
        self.speaker = speaker_model
        self.sample_rate = sample_rate
        self.base_duration = duration
        self.frame_size = frame_size
        self.hop_length = hop_length
        self.num_filters = num_filters

    def generate_waveform_with_frames(self, digit: int, energy_level: float, 
                                      variation: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        speaker = self.speaker
        base_pitch = speaker.pitch_base + digit * 2.0
        pitch = np.random.normal(base_pitch, speaker.pitch_variability)
        
        f1 = speaker.formant_f1 + digit * 3
        f2 = speaker.formant_f2 + digit * 5
        f3 = speaker.formant_f3 + digit * 2
        
        duration_factor = speaker.duration_factor * (1.0 + np.random.normal(0, 0.03)) * (1.0 + variation * 0.3)
        duration = np.clip(self.base_duration * duration_factor, 0.3, 1.2)
        
        num_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, num_samples, endpoint=False)
        
        waveform = np.zeros(num_samples)
        harmonics = [1, 2, 3, 4, 5]
        amplitudes = [1.0, 0.6, 0.4, 0.2, 0.1]
        
        for h, amp in zip(harmonics, amplitudes):
            waveform += amp * np.sin(2 * np.pi * pitch * h * t)
        
        formant_freqs = [f1, f2, f3]
        formant_amps = [0.4, 0.2, 0.1]
        
        for ff, fa in zip(formant_freqs, formant_amps):
            if 0 < ff < self.sample_rate / 2:
                waveform += fa * np.sin(2 * np.pi * ff * t)
        
        if digit in [0, 1, 2]:
            envelope = 1.0 + 0.3 * np.sin(2 * np.pi * 1.5 * t) * (1 - variation)
        elif digit in [3, 4, 5]:
            envelope = 1.0 + 0.4 * np.sin(2 * np.pi * 2.0 * t) * (1 - variation)
        else:
            envelope = 1.0 + 0.2 * np.sin(2 * np.pi * 1.0 * t) * (1 - variation)
        
        envelope *= speaker.energy_factor * (1.0 + variation * 0.2)
        waveform *= envelope
        
        current_rms = np.sqrt(np.mean(waveform ** 2) + 1e-6)
        if current_rms > 0:
            waveform *= (energy_level / current_rms)
        
        noise_level = 0.03 * energy_level * (1 + variation * 0.5)
        waveform += np.random.normal(0, noise_level, num_samples)
        
        mfcc_sequence = self.extract_mfcc_frames(waveform)
        return waveform, mfcc_sequence

    def extract_mfcc_frames(self, waveform: np.ndarray) -> np.ndarray:
        if len(waveform) < self.frame_size:
            waveform = np.pad(waveform, (0, self.frame_size - len(waveform)))
        
        num_frames = max(1, (len(waveform) - self.frame_size) // self.hop_length + 1)
        mfcc_sequence = []
        
        def hz_to_mel(hz):
            return 2595 * np.log10(1 + hz / 700)
        
        f_max = self.sample_rate / 2
        mel_min = hz_to_mel(0)
        mel_max = hz_to_mel(f_max)
        mel_points = np.linspace(mel_min, mel_max, self.num_filters + 2)
        freq_bins = np.linspace(0, f_max, self.frame_size // 2)
        mel_bins = hz_to_mel(freq_bins)
        
        for i in range(num_frames):
            start = i * self.hop_length
            end = start + self.frame_size
            if end > len(waveform):
                break
            
            frame = waveform[start:end] * np.hamming(self.frame_size)
            fft_vals = fft(frame)
            magnitude = np.abs(fft_vals[:len(fft_vals) // 2])
            
            mel_spectrum = np.zeros(self.num_filters)
            for j in range(self.num_filters):
                mel_left = mel_points[j]
                mel_center = mel_points[j + 1]
                mel_right = mel_points[j + 2]
                
                left_idx = np.argmin(np.abs(mel_bins - mel_left))
                center_idx = np.argmin(np.abs(mel_bins - mel_center))
                right_idx = np.argmin(np.abs(mel_bins - mel_right))
                
                for k in range(left_idx, right_idx + 1):
                    if k >= len(magnitude):
                        continue
                    if k <= center_idx:
                        weight = (mel_bins[k] - mel_left) / (mel_center - mel_left) if center_idx > left_idx else 1.0
                    else:
                        weight = (mel_right - mel_bins[k]) / (mel_right - mel_center) if right_idx > center_idx else 1.0
                    weight = np.clip(weight, 0.0, 1.0)
                    mel_spectrum[j] += weight * magnitude[k]
            
            mel_spectrum = np.log(mel_spectrum + 1e-6)
            mfcc = np.zeros(self.num_filters)
            n = np.arange(self.num_filters)
            for j in range(self.num_filters):
                mfcc[j] = np.sum(mel_spectrum * np.cos(np.pi * j * (n + 0.5) / self.num_filters))
            
            mfcc -= np.mean(mfcc)
            std = np.std(mfcc)
            if std > 0:
                mfcc /= std
            mfcc_sequence.append(mfcc)
        
        return np.asarray(mfcc_sequence, dtype=float)

    def compute_dtw_cost(self, observed_seq: np.ndarray, template_sequences: List[np.ndarray]) -> float:
        if len(observed_seq) == 0 or len(template_sequences) == 0:
            return float("inf")
        
        distances = []
        for template_seq in template_sequences:
            if len(template_seq) == 0:
                continue
            n, m = len(observed_seq), len(template_seq)
            dtw = np.full((n + 1, m + 1), np.inf)
            dtw[0, 0] = 0.0
            
            for i in range(1, n + 1):
                for j in range(1, m + 1):
                    cost = euclidean(observed_seq[i - 1], template_seq[j - 1])
                    dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
            
            distances.append(dtw[n, m] / (n + m))
        
        return float(np.median(distances)) if distances else float("inf")

    def simulate_voice_attempt(self, true_digit: int, template: VoiceTemplate,
                               energy_level: float, digit_index: int) -> Tuple[float, float]:
        variation = 0.05 + np.random.random() * 0.15
        _, observed_mfcc = self.generate_waveform_with_frames(
            digit=true_digit, energy_level=energy_level, variation=variation
        )
        template_sequences = template.get_templates(digit_index)
        dtw_cost = self.compute_dtw_cost(observed_mfcc, template_sequences)
        
        # WORKING DIVISOR = 0.3
        score = 100 * np.exp(-dtw_cost / 0.3)
        score = np.clip(score + np.random.normal(0, 3), 0, 100)
        return float(score), float(dtw_cost)

# ============================================================
# USER PROFILE
# ============================================================

class UserProfile:
    """Complete user profile."""
    
    def __init__(self, user_id: int):
        if user_id < 1:
            raise ValueError("user_id must be >= 1")
        
        self.user_id = user_id
        rng = np.random.default_rng(user_id * 42)
        
        self.force_levels = [int(rng.integers(80, 181)), int(rng.integers(520, 721)), int(rng.integers(220, 421))]
        self.hold_times = [int(rng.integers(150, 251)), int(rng.integers(380, 521)), int(rng.integers(220, 381))]
        self.gap_times = [int(rng.integers(450, 581)), int(rng.integers(250, 381))]
        
        self.speaker = SpeakerModel(user_id)
        self.voice_digits = [int(rng.integers(0, 10)), int(rng.integers(0, 10)), int(rng.integers(0, 10))]
        self.voice_template = VoiceTemplate(self.voice_digits, self.speaker, num_recordings=3)
        
        self.gap_distances = [int(rng.integers(20, 81)), int(rng.integers(20, 81))]
        self.baseline_noise = int(rng.integers(15, 36))
        
        self.force_std = float(rng.uniform(0.25, 0.45))
        self.hold_std = float(rng.uniform(60, 95))
        self.gap_std = float(rng.uniform(70, 110))
        self.distance_std = float(rng.uniform(15, 35))
        self.gap_distance_std = float(rng.uniform(8, 18))

# ============================================================
# SENSOR SIMULATORS
# ============================================================

class FSRSimulator:
    def __init__(self, noise_factor: float = 0.03):
        self.noise_factor = noise_factor
    
    def read_press(self, true_force_adc: int, user_variance: float) -> int:
        reading = true_force_adc + np.random.normal(0, user_variance * 50) + \
                  np.random.normal(0, abs(true_force_adc) * self.noise_factor) + \
                  np.random.uniform(-0.5, 0.5)
        return int(np.clip(reading, 0, 1023))
    
    def read_hold_time(self, true_hold_ms: int, user_variance: float) -> int:
        return int(max(50, true_hold_ms + np.random.normal(0, user_variance)))
    
    def read_gap_time(self, true_gap_ms: int, user_variance: float) -> int:
        return int(max(100, true_gap_ms + np.random.normal(0, user_variance)))

class MicrophoneSimulator:
    def __init__(self, sample_rate: int = 8000):
        self.sample_rate = sample_rate
    
    def measure_baseline(self, user_baseline: int, ambient_variation: float = 5) -> int:
        return int(max(5, user_baseline + np.random.normal(0, ambient_variation)))

class UltrasonicSimulator:
    def __init__(self, min_range_mm: int = 20, max_range_mm: int = 4000, accuracy_mm: float = 3):
        self.min_range_mm = min_range_mm
        self.max_range_mm = max_range_mm
        self.accuracy_mm = accuracy_mm
    
    def read_distance(self, true_distance_mm: float) -> int:
        measured = abs(true_distance_mm) + np.random.normal(0, self.accuracy_mm) + \
                   np.random.normal(0, abs(true_distance_mm) * 0.008)
        if np.random.random() < 0.01:
            return 0
        return int(np.clip(measured, self.min_range_mm, self.max_range_mm))
    
    def measure_gap_distance(self, true_distance_mm: float, user_std: float) -> int:
        return self.read_distance(abs(true_distance_mm) + np.random.normal(0, user_std))

# ============================================================
# IMPOSTOR MODEL
# ============================================================

class ImpostorModel:
    def __init__(self, genuine_profile: UserProfile, impostor_speaker: SpeakerModel, observation_accuracy: float = 0.70):
        self.genuine = genuine_profile
        self.impostor_speaker = impostor_speaker
        self.accuracy = observation_accuracy
        
        self.estimated_force = [self._estimate_value(f, 150) for f in genuine_profile.force_levels]
        self.estimated_hold = [self._estimate_value(h, 200) for h in genuine_profile.hold_times]  # FIXED
        self.estimated_gap = [self._estimate_value(g, 180) for g in genuine_profile.gap_times]
        
        # 50% impostor digit error rate
        self.estimated_voice_digits = []
        for digit in genuine_profile.voice_digits:
            if np.random.random() < 0.50:
                self.estimated_voice_digits.append(int(np.random.randint(0, 10)))
            else:
                self.estimated_voice_digits.append(int(digit))
        
        self.estimated_gap_distances = [self._estimate_value(g, 30) for g in genuine_profile.gap_distances]
    
    def _estimate_value(self, true_value: float, error_range: float) -> float:
        error = np.random.uniform(-error_range * (1 - self.accuracy), error_range * (1 - self.accuracy))
        return true_value + error
    
    def generate_attempt(self) -> Dict[str, Any]:
        return {
            "force": self.estimated_force,
            "hold": self.estimated_hold,
            "gap": self.estimated_gap,
            "voice_digits": self.estimated_voice_digits,
            "gap_distances": self.estimated_gap_distances,
            "force_std": 0.60,
            "hold_std": 140,
            "gap_std": 160,
            "gap_distance_std": 30
        }

# ============================================================
# SCORING FUNCTIONS
# ============================================================

def compute_mechanical_score(observed: Dict, enrolled_mean: Dict, 
                             delta_f: float = 50, delta_t: float = 100) -> int:
    """WORKING: delta_f=50, delta_t=100"""
    scores = []
    for i in range(3):
        dF = (observed["force"][i] - enrolled_mean["force"][i]) / delta_f
        dH = (observed["hold"][i] - enrolled_mean["hold"][i]) / delta_t
        dG = 0.0
        if i < 2 and i < len(observed["gap"]):
            dG = (observed["gap"][i] - enrolled_mean["gap"][i]) / delta_t
        dist_sq = dF ** 2 + dH ** 2 + dG ** 2
        scores.append(np.exp(-dist_sq))
    return int(np.clip(100 * np.mean(scores), 0, 100))

def compute_spatial_gap_score(observed_gaps: List[float], enrolled_gaps: List[float], 
                              tolerance_mm: float = 25) -> int:
    if len(observed_gaps) != 2 or len(enrolled_gaps) != 2:
        return 0
    scores = []
    for i in range(2):
        dist_diff = abs(observed_gaps[i] - enrolled_gaps[i])
        scores.append(np.exp(-dist_diff / tolerance_mm))
    return int(np.clip(100 * np.mean(scores), 0, 100))

def apply_hybrid_fusion(s_m: float, s_v: float, s_g: float,
                       seq_thresh_v: int = 34, seq_thresh_g: int = 38, seq_thresh_m: int = 62,
                       w_m: float = 0.45, w_v: float = 0.15, w_g: float = 0.40,
                       weighted_threshold: int = 60) -> Tuple[bool, str]:
    if s_v < seq_thresh_v:
        return False, f"Voice screening failed ({s_v:.1f} < {seq_thresh_v})"
    if s_g < seq_thresh_g:
        return False, f"Gap screening failed ({s_g:.1f} < {seq_thresh_g})"
    if s_m < seq_thresh_m:
        return False, f"Mechanical screening failed ({s_m:.1f} < {seq_thresh_m})"
    
    s_total = w_m * s_m + w_v * s_v + w_g * s_g
    if s_total >= weighted_threshold:
        return True, f"Accepted (score={s_total:.1f})"
    return False, f"Weighted check failed (score={s_total:.1f} < {weighted_threshold})"

def create_user_profiles(num_users: int = 50, verbose: bool = True) -> List[UserProfile]:
    profiles = []
    for i in range(num_users):
        profile = UserProfile(user_id=i + 1)
        profiles.append(profile)
        if verbose and i < 10:
            print(f"User P{i + 1:02d}: Force={profile.force_levels}, Voice={profile.voice_digits}")
    if verbose and num_users > 10:
        print(f"... and {num_users - 10} more users\n")
    return profiles