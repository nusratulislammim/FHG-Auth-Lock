"""
sensor_models.py - Improved Complete Version

THREE-LAYER AUTHENTICATION SYSTEM
=================================

Layers:
1. Voice
   - Speaker-specific synthetic acoustic model
   - MFCC frame extraction
   - Actual DTW matching
   - Voice model kept unchanged from working version

2. Spatial Gap
   - HC-SR04 radial hand-distance measurement
   - Absolute gap-distance similarity
   - Relative two-gap movement-pattern similarity

3. Mechanical FHG
   - Force
   - Hold time
   - Inter-press gap
   - Absolute FHG similarity
   - Relative force/hold/gap pattern similarity

Fusion:
    Voice -> Spatial Gap -> Mechanical -> Weighted Fusion

IMPORTANT:
This is a simulation model, not a physical sensor implementation.
Sensor parameters are intended as parametric approximations.
"""

import numpy as np
import random

from typing import List, Tuple, Dict, Any

from scipy.spatial.distance import euclidean
from scipy.fft import fft


# ============================================================
# SPEAKER MODEL
# ============================================================

class SpeakerModel:
    """
    Speaker-specific synthetic voice characteristics.
    """

    def __init__(self, speaker_id: int):

        if speaker_id < 0:
            raise ValueError(
                "speaker_id must be non-negative"
            )

        rng = np.random.default_rng(
            speaker_id * 100
        )

        self.speaker_id = speaker_id

        # ----------------------------------------------------
        # Speaker characteristics
        # ----------------------------------------------------

        self.pitch_base = float(
            rng.normal(120, 40)
        )

        self.pitch_variability = float(
            rng.uniform(8, 20)
        )

        self.formant_f1 = float(
            rng.normal(400, 120)
        )

        self.formant_f2 = float(
            rng.normal(900, 200)
        )

        self.formant_f3 = float(
            rng.normal(2200, 300)
        )

        self.duration_factor = float(
            rng.normal(1.0, 0.15)
        )

        self.energy_factor = float(
            rng.normal(1.0, 0.20)
        )


# ============================================================
# VOICE TEMPLATE
# ============================================================

class VoiceTemplate:
    """
    Stores MFCC templates for enrolled digits.
    """

    def __init__(
        self,
        digits: List[int],
        speaker_model: SpeakerModel,
        num_recordings: int = 3
    ):

        if not all(
            0 <= d <= 9
            for d in digits
        ):
            raise ValueError(
                "All digits must be 0-9"
            )

        self.digits = digits
        self.speaker = speaker_model
        self.num_recordings = num_recordings

        self.templates: List[
            List[np.ndarray]
        ] = []

        voice_sim = VoiceSimulator(
            speaker_model
        )

        # ----------------------------------------------------
        # Generate multiple enrollment recordings
        # ----------------------------------------------------

        for digit in digits:

            digit_templates = []

            for _ in range(num_recordings):

                _, mfcc_seq = (
                    voice_sim.generate_waveform_with_frames(
                        digit=digit,
                        energy_level=20,
                        variation=0.08
                    )
                )

                digit_templates.append(
                    mfcc_seq
                )

            self.templates.append(
                digit_templates
            )

    def get_templates(
        self,
        digit_index: int
    ) -> List[np.ndarray]:

        if (
            0 <= digit_index
            < len(self.templates)
        ):
            return self.templates[
                digit_index
            ]

        return self.templates[0]


# ============================================================
# VOICE SIMULATOR
# ============================================================

class VoiceSimulator:
    """
    Synthetic voice authentication model.

    Uses:
        - waveform generation
        - MFCC-like feature extraction
        - frame-level DTW
        - exponential similarity conversion
    """

    def __init__(
        self,
        speaker_model: SpeakerModel,
        sample_rate: int = 8000,
        duration: float = 0.8,
        frame_size: int = 256,
        hop_length: int = 128,
        num_filters: int = 13
    ):

        self.speaker = speaker_model

        self.sample_rate = sample_rate

        self.base_duration = duration

        self.frame_size = frame_size

        self.hop_length = hop_length

        self.num_filters = num_filters

    # --------------------------------------------------------
    # Waveform generation
    # --------------------------------------------------------

    def generate_waveform_with_frames(
        self,
        digit: int,
        energy_level: float,
        variation: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:

        speaker = self.speaker

        # ----------------------------------------------------
        # Pitch
        # ----------------------------------------------------

        base_pitch = (
            speaker.pitch_base
            + digit * 2.0
        )

        pitch = np.random.normal(
            base_pitch,
            speaker.pitch_variability
        )

        # ----------------------------------------------------
        # Formants
        # ----------------------------------------------------

        f1 = (
            speaker.formant_f1
            + digit * 3
        )

        f2 = (
            speaker.formant_f2
            + digit * 5
        )

        f3 = (
            speaker.formant_f3
            + digit * 2
        )

        # ----------------------------------------------------
        # Duration
        # ----------------------------------------------------

        duration_factor = (
            speaker.duration_factor
            *
            (
                1.0
                + np.random.normal(0, 0.03)
            )
            *
            (
                1.0
                + variation * 0.3
            )
        )

        duration = np.clip(
            self.base_duration
            * duration_factor,
            0.3,
            1.2
        )

        num_samples = int(
            self.sample_rate * duration
        )

        t = np.linspace(
            0,
            duration,
            num_samples,
            endpoint=False
        )

        # ----------------------------------------------------
        # Harmonic waveform
        # ----------------------------------------------------

        waveform = np.zeros(
            num_samples
        )

        harmonics = [
            1,
            2,
            3,
            4,
            5
        ]

        amplitudes = [
            1.0,
            0.6,
            0.4,
            0.2,
            0.1
        ]

        for h, amp in zip(
            harmonics,
            amplitudes
        ):

            waveform += (
                amp
                *
                np.sin(
                    2
                    * np.pi
                    * pitch
                    * h
                    * t
                )
            )

        # ----------------------------------------------------
        # Formants
        # ----------------------------------------------------

        formant_freqs = [
            f1,
            f2,
            f3
        ]

        formant_amps = [
            0.4,
            0.2,
            0.1
        ]

        for ff, fa in zip(
            formant_freqs,
            formant_amps
        ):

            if (
                0 < ff
                < self.sample_rate / 2
            ):

                waveform += (
                    fa
                    *
                    np.sin(
                        2
                        * np.pi
                        * ff
                        * t
                    )
                )

        # ----------------------------------------------------
        # Digit-dependent envelope
        # ----------------------------------------------------

        if digit in [0, 1, 2]:

            envelope = (
                1.0
                +
                0.3
                *
                np.sin(
                    2
                    * np.pi
                    * 1.5
                    * t
                )
                *
                (1 - variation)
            )

        elif digit in [3, 4, 5]:

            envelope = (
                1.0
                +
                0.4
                *
                np.sin(
                    2
                    * np.pi
                    * 2.0
                    * t
                )
                *
                (1 - variation)
            )

        else:

            envelope = (
                1.0
                +
                0.2
                *
                np.sin(
                    2
                    * np.pi
                    * 1.0
                    * t
                )
                *
                (1 - variation)
            )

        envelope *= (
            speaker.energy_factor
            *
            (
                1.0
                + variation * 0.2
            )
        )

        waveform *= envelope

        # ----------------------------------------------------
        # Normalize RMS
        # ----------------------------------------------------

        current_rms = np.sqrt(
            np.mean(
                waveform ** 2
            )
            + 1e-6
        )

        if current_rms > 0:

            waveform *= (
                energy_level
                / current_rms
            )

        # ----------------------------------------------------
        # Add acoustic noise
        # ----------------------------------------------------

        noise_level = (
            0.03
            * energy_level
            *
            (
                1
                + variation * 0.5
            )
        )

        waveform += np.random.normal(
            0,
            noise_level,
            num_samples
        )

        # ----------------------------------------------------
        # MFCC extraction
        # ----------------------------------------------------

        mfcc_sequence = (
            self.extract_mfcc_frames(
                waveform
            )
        )

        return (
            waveform,
            mfcc_sequence
        )

    # --------------------------------------------------------
    # MFCC-like frame extraction
    # --------------------------------------------------------

    def extract_mfcc_frames(
        self,
        waveform: np.ndarray
    ) -> np.ndarray:

        if len(waveform) < self.frame_size:

            waveform = np.pad(
                waveform,
                (
                    0,
                    self.frame_size
                    - len(waveform)
                )
            )

        num_frames = max(
            1,
            (
                len(waveform)
                - self.frame_size
            )
            // self.hop_length
            + 1
        )

        mfcc_sequence = []

        # ----------------------------------------------------
        # Mel conversion
        # ----------------------------------------------------

        def hz_to_mel(hz):

            return (
                2595
                *
                np.log10(
                    1
                    + hz / 700
                )
            )

        f_max = (
            self.sample_rate / 2
        )

        mel_min = hz_to_mel(0)

        mel_max = hz_to_mel(
            f_max
        )

        mel_points = np.linspace(
            mel_min,
            mel_max,
            self.num_filters + 2
        )

        freq_bins = np.linspace(
            0,
            f_max,
            self.frame_size // 2
        )

        mel_bins = hz_to_mel(
            freq_bins
        )

        # ----------------------------------------------------
        # Process frames
        # ----------------------------------------------------

        for i in range(num_frames):

            start = (
                i
                * self.hop_length
            )

            end = (
                start
                + self.frame_size
            )

            if end > len(waveform):
                break

            frame = (
                waveform[start:end]
                *
                np.hamming(
                    self.frame_size
                )
            )

            fft_vals = fft(
                frame
            )

            magnitude = np.abs(
                fft_vals[
                    :len(fft_vals) // 2
                ]
            )

            mel_spectrum = np.zeros(
                self.num_filters
            )

            # ------------------------------------------------
            # Triangular filter bank
            # ------------------------------------------------

            for j in range(
                self.num_filters
            ):

                mel_left = (
                    mel_points[j]
                )

                mel_center = (
                    mel_points[j + 1]
                )

                mel_right = (
                    mel_points[j + 2]
                )

                left_idx = np.argmin(
                    np.abs(
                        mel_bins
                        - mel_left
                    )
                )

                center_idx = np.argmin(
                    np.abs(
                        mel_bins
                        - mel_center
                    )
                )

                right_idx = np.argmin(
                    np.abs(
                        mel_bins
                        - mel_right
                    )
                )

                for k in range(
                    left_idx,
                    right_idx + 1
                ):

                    if (
                        k
                        >= len(magnitude)
                    ):
                        continue

                    if (
                        k
                        <= center_idx
                    ):

                        if (
                            center_idx
                            > left_idx
                        ):

                            weight = (
                                (
                                    mel_bins[k]
                                    - mel_left
                                )
                                /
                                (
                                    mel_center
                                    - mel_left
                                )
                            )

                        else:

                            weight = 1.0

                    else:

                        if (
                            right_idx
                            > center_idx
                        ):

                            weight = (
                                (
                                    mel_right
                                    - mel_bins[k]
                                )
                                /
                                (
                                    mel_right
                                    - mel_center
                                )
                            )

                        else:

                            weight = 1.0

                    weight = np.clip(
                        weight,
                        0.0,
                        1.0
                    )

                    mel_spectrum[j] += (
                        weight
                        * magnitude[k]
                    )

            # ------------------------------------------------
            # Log compression
            # ------------------------------------------------

            mel_spectrum = np.log(
                mel_spectrum
                + 1e-6
            )

            # ------------------------------------------------
            # DCT-like MFCC computation
            # ------------------------------------------------

            mfcc = np.zeros(
                self.num_filters
            )

            n = np.arange(
                self.num_filters
            )

            for j in range(
                self.num_filters
            ):

                mfcc[j] = np.sum(
                    mel_spectrum
                    *
                    np.cos(
                        np.pi
                        * j
                        * (n + 0.5)
                        / self.num_filters
                    )
                )

            # ------------------------------------------------
            # Cepstral normalization
            # ------------------------------------------------

            mfcc -= np.mean(
                mfcc
            )

            std = np.std(
                mfcc
            )

            if std > 0:

                mfcc /= std

            mfcc_sequence.append(
                mfcc
            )

        return np.asarray(
            mfcc_sequence,
            dtype=float
        )

    # --------------------------------------------------------
    # DTW
    # --------------------------------------------------------

    def compute_dtw_cost(
        self,
        observed_seq: np.ndarray,
        template_sequences: List[np.ndarray]
    ) -> float:

        if (
            len(observed_seq) == 0
            or len(template_sequences) == 0
        ):

            return float("inf")

        distances = []

        for template_seq in (
            template_sequences
        ):

            if len(template_seq) == 0:
                continue

            n = len(
                observed_seq
            )

            m = len(
                template_seq
            )

            dtw = np.full(
                (n + 1, m + 1),
                np.inf
            )

            dtw[0, 0] = 0.0

            for i in range(
                1,
                n + 1
            ):

                for j in range(
                    1,
                    m + 1
                ):

                    cost = euclidean(
                        observed_seq[
                            i - 1
                        ],
                        template_seq[
                            j - 1
                        ]
                    )

                    dtw[i, j] = (
                        cost
                        +
                        min(
                            dtw[i - 1, j],
                            dtw[i, j - 1],
                            dtw[i - 1, j - 1]
                        )
                    )

            distances.append(
                dtw[n, m]
                /
                (n + m)
            )

        if not distances:

            return float("inf")

        return float(
            np.median(
                distances
            )
        )

    # --------------------------------------------------------
    # Voice authentication attempt
    # --------------------------------------------------------

    def simulate_voice_attempt(
        self,
        true_digit: int,
        template: VoiceTemplate,
        energy_level: float,
        digit_index: int
    ) -> Tuple[float, float]:

        variation = (
            0.05
            + np.random.random()
            * 0.15
        )

        _, observed_mfcc = (
            self.generate_waveform_with_frames(
                digit=true_digit,
                energy_level=energy_level,
                variation=variation
            )
        )

        template_sequences = (
            template.get_templates(
                digit_index
            )
        )

        dtw_cost = (
            self.compute_dtw_cost(
                observed_mfcc,
                template_sequences
            )
        )

        # ----------------------------------------------------
        # Original working divisor retained
        # ----------------------------------------------------

        score = (
            100
            *
            np.exp(
                -dtw_cost / 0.3
            )
        )

        score = np.clip(
            score
            +
            np.random.normal(0, 3),
            0,
            100
        )

        return (
            float(score),
            float(dtw_cost)
        )


# ============================================================
# USER PROFILE
# ============================================================

class UserProfile:
    """
    Complete synthetic authentication profile.
    """

    def __init__(
        self,
        user_id: int
    ):

        if user_id < 1:

            raise ValueError(
                "user_id must be >= 1"
            )

        self.user_id = user_id

        rng = np.random.default_rng(
            user_id * 42
        )

        # ----------------------------------------------------
        # Mechanical FHG pattern
        # ----------------------------------------------------

        self.force_levels = [
            int(
                rng.integers(
                    80,
                    181
                )
            ),

            int(
                rng.integers(
                    520,
                    721
                )
            ),

            int(
                rng.integers(
                    220,
                    421
                )
            )
        ]

        self.hold_times = [
            int(
                rng.integers(
                    150,
                    251
                )
            ),

            int(
                rng.integers(
                    380,
                    521
                )
            ),

            int(
                rng.integers(
                    220,
                    381
                )
            )
        ]

        self.gap_times = [
            int(
                rng.integers(
                    450,
                    581
                )
            ),

            int(
                rng.integers(
                    250,
                    381
                )
            )
        ]

        # ----------------------------------------------------
        # Voice
        # ----------------------------------------------------

        self.speaker = SpeakerModel(
            user_id
        )

        self.voice_digits = [
            int(
                rng.integers(
                    0,
                    10
                )
            ),

            int(
                rng.integers(
                    0,
                    10
                )
            ),

            int(
                rng.integers(
                    0,
                    10
                )
            )
        ]

        self.voice_template = (
            VoiceTemplate(
                self.voice_digits,
                self.speaker,
                num_recordings=3
            )
        )

        # ----------------------------------------------------
        # Spatial gap distances
        # ----------------------------------------------------

        self.gap_distances = [
            int(
                rng.integers(
                    20,
                    81
                )
            ),

            int(
                rng.integers(
                    20,
                    81
                )
            )
        ]

        # ----------------------------------------------------
        # Acoustic baseline
        # ----------------------------------------------------

        self.baseline_noise = int(
            rng.integers(
                15,
                36
            )
        )

        # ----------------------------------------------------
        # User-specific variability
        # ----------------------------------------------------

        self.force_std = float(
            rng.uniform(
                0.25,
                0.45
            )
        )

        self.hold_std = float(
            rng.uniform(
                60,
                95
            )
        )

        self.gap_std = float(
            rng.uniform(
                70,
                110
            )
        )

        self.distance_std = float(
            rng.uniform(
                15,
                35
            )
        )

        self.gap_distance_std = float(
            rng.uniform(
                8,
                18
            )
        )


# ============================================================
# FSR SIMULATOR
# ============================================================

class FSRSimulator:
    """
    Parametric FSR-402 response simulator.

    The returned force value should be interpreted as an
    ADC-like relative response unless a physical calibration
    curve is established.
    """

    def __init__(
        self,
        noise_factor: float = 0.03
    ):

        self.noise_factor = (
            noise_factor
        )

    def read_press(
        self,
        true_force_adc: int,
        user_variance: float
    ) -> int:

        reading = (
            true_force_adc
            +
            np.random.normal(
                0,
                user_variance * 50
            )
            +
            np.random.normal(
                0,
                abs(true_force_adc)
                * self.noise_factor
            )
            +
            np.random.uniform(
                -0.5,
                0.5
            )
        )

        return int(
            np.clip(
                reading,
                0,
                1023
            )
        )

    def read_hold_time(
        self,
        true_hold_ms: int,
        user_variance: float
    ) -> int:

        return int(
            max(
                50,
                true_hold_ms
                +
                np.random.normal(
                    0,
                    user_variance
                )
            )
        )

    def read_gap_time(
        self,
        true_gap_ms: int,
        user_variance: float
    ) -> int:

        return int(
            max(
                100,
                true_gap_ms
                +
                np.random.normal(
                    0,
                    user_variance
                )
            )
        )


# ============================================================
# MICROPHONE SIMULATOR
# ============================================================

class MicrophoneSimulator:
    """
    Ambient-noise baseline simulator.
    """

    def __init__(
        self,
        sample_rate: int = 8000
    ):

        self.sample_rate = (
            sample_rate
        )

    def measure_baseline(
        self,
        user_baseline: int,
        ambient_variation: float = 5
    ) -> int:

        return int(
            max(
                5,
                user_baseline
                +
                np.random.normal(
                    0,
                    ambient_variation
                )
            )
        )


# ============================================================
# ULTRASONIC SIMULATOR
# ============================================================

class UltrasonicSimulator:
    """
    HC-SR04-like radial distance simulator.

    accuracy_mm:
        Nominal measurement noise term.

    An additional proportional error term is included.
    """

    def __init__(
        self,
        min_range_mm: int = 20,
        max_range_mm: int = 4000,
        accuracy_mm: float = 3
    ):

        self.min_range_mm = (
            min_range_mm
        )

        self.max_range_mm = (
            max_range_mm
        )

        self.accuracy_mm = (
            accuracy_mm
        )

    def read_distance(
        self,
        true_distance_mm: float
    ) -> int:

        measured = (
            abs(true_distance_mm)
            +
            np.random.normal(
                0,
                self.accuracy_mm
            )
            +
            np.random.normal(
                0,
                abs(true_distance_mm)
                * 0.008
            )
        )

        # ----------------------------------------------------
        # Approximate 1% zero-reading/failure event
        # ----------------------------------------------------

        if np.random.random() < 0.01:

            return 0

        return int(
            np.clip(
                measured,
                self.min_range_mm,
                self.max_range_mm
            )
        )

    def measure_gap_distance(
        self,
        true_distance_mm: float,
        user_std: float
    ) -> int:

        user_variation = (
            np.random.normal(
                0,
                user_std
            )
        )

        return self.read_distance(
            abs(true_distance_mm)
            +
            user_variation
        )


# ============================================================
# IMPOSTOR MODEL
# ============================================================

class ImpostorModel:
    """
    Observation-based impostor model.

    The impostor attempts to reproduce the target user's
    observed authentication behavior with imperfect accuracy.
    """

    def __init__(
        self,
        genuine_profile: UserProfile,
        impostor_speaker: SpeakerModel,
        observation_accuracy: float = 0.70
    ):

        self.genuine = (
            genuine_profile
        )

        self.impostor_speaker = (
            impostor_speaker
        )

        self.accuracy = (
            observation_accuracy
        )

        # ----------------------------------------------------
        # Estimate target mechanical behavior
        # ----------------------------------------------------

        self.estimated_force = [
            self._estimate_value(
                f,
                150
            )
            for f in
            genuine_profile.force_levels
        ]

        self.estimated_hold = [
            self._estimate_value(
                h,
                200
            )
            for h in
            genuine_profile.hold_times
        ]

        self.estimated_gap = [
            self._estimate_value(
                g,
                180
            )
            for g in
            genuine_profile.gap_times
        ]

        # ----------------------------------------------------
        # Estimate voice digits
        #
        # 50% attempted digit error rate.
        # ----------------------------------------------------

        self.estimated_voice_digits = []

        for digit in (
            genuine_profile.voice_digits
        ):

            if np.random.random() < 0.50:

                self.estimated_voice_digits.append(
                    int(
                        np.random.randint(
                            0,
                            10
                        )
                    )
                )

            else:

                self.estimated_voice_digits.append(
                    int(digit)
                )

        # ----------------------------------------------------
        # Estimate spatial distances
        # ----------------------------------------------------

        self.estimated_gap_distances = [
            self._estimate_value(
                g,
                30
            )
            for g in
            genuine_profile.gap_distances
        ]

    def _estimate_value(
        self,
        true_value: float,
        error_range: float
    ) -> float:

        error = np.random.uniform(
            -error_range
            * (1 - self.accuracy),

            error_range
            * (1 - self.accuracy)
        )

        return (
            true_value
            + error
        )

    def generate_attempt(
        self
    ) -> Dict[str, Any]:

        return {

            "force":
                self.estimated_force,

            "hold":
                self.estimated_hold,

            "gap":
                self.estimated_gap,

            "voice_digits":
                self.estimated_voice_digits,

            "gap_distances":
                self.estimated_gap_distances,

            # ------------------------------------------------
            # Attack execution variability
            # ------------------------------------------------

            "force_std":
                0.60,

            "hold_std":
                140,

            "gap_std":
                160,

            "gap_distance_std":
                30
        }


# ============================================================
# HELPER FUNCTIONS FOR SCORING
# ============================================================

def _safe_sum(
    values: np.ndarray
) -> float:

    value = float(
        np.sum(
            np.abs(values)
        )
    )

    return max(
        value,
        1e-9
    )


# ============================================================
# IMPROVED MECHANICAL FHG SCORE
# ============================================================

def compute_mechanical_score(
    observed: Dict,
    enrolled_mean: Dict,
    delta_f: float = 50,
    delta_t: float = 100
) -> int:
    """
    Improved Force-Hold-Gap authentication score.

    The score combines two types of evidence:

    A. Absolute FHG similarity
       - Force agreement
       - Hold-time agreement
       - Inter-press gap agreement

    B. Relative pattern similarity
       - Relative force pattern
       - Relative hold-time pattern
       - Relative gap pattern

    This retains the original FHG structure while adding
    pattern-shape information.

    Parameters
    ----------
    observed:
        Dictionary containing:
            force -> 3 values
            hold  -> 3 values
            gap   -> 2 values

    enrolled_mean:
        Enrolled target pattern.

    delta_f:
        Force normalization tolerance.

    delta_t:
        Hold/gap normalization tolerance.

    Returns
    -------
    int
        Score from 0 to 100.
    """

    # --------------------------------------------------------
    # Convert to arrays
    # --------------------------------------------------------

    obs_force = np.asarray(
        observed["force"],
        dtype=float
    )

    obs_hold = np.asarray(
        observed["hold"],
        dtype=float
    )

    obs_gap = np.asarray(
        observed["gap"],
        dtype=float
    )

    ref_force = np.asarray(
        enrolled_mean["force"],
        dtype=float
    )

    ref_hold = np.asarray(
        enrolled_mean["hold"],
        dtype=float
    )

    ref_gap = np.asarray(
        enrolled_mean["gap"],
        dtype=float
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if (
        len(obs_force) != 3
        or len(ref_force) != 3
    ):
        return 0

    if (
        len(obs_hold) != 3
        or len(ref_hold) != 3
    ):
        return 0

    if (
        len(obs_gap) != 2
        or len(ref_gap) != 2
    ):
        return 0

    # ========================================================
    # PART A: ABSOLUTE FHG SIMILARITY
    # ========================================================

    absolute_scores = []

    for i in range(3):

        # ----------------------------------------------------
        # Force difference
        # ----------------------------------------------------

        dF = (
            obs_force[i]
            -
            ref_force[i]
        ) / delta_f

        # ----------------------------------------------------
        # Hold difference
        # ----------------------------------------------------

        dH = (
            obs_hold[i]
            -
            ref_hold[i]
        ) / delta_t

        # ----------------------------------------------------
        # Gap difference
        #
        # Only two inter-press gaps exist.
        # ----------------------------------------------------

        dG = 0.0

        if i < 2:

            dG = (
                obs_gap[i]
                -
                ref_gap[i]
            ) / delta_t

        # ----------------------------------------------------
        # Combined squared distance
        # ----------------------------------------------------

        dist_sq = (
            dF ** 2
            +
            dH ** 2
            +
            dG ** 2
        )

        similarity = np.exp(
            -dist_sq
        )

        absolute_scores.append(
            similarity
        )

    absolute_similarity = float(
        np.mean(
            absolute_scores
        )
    )

    # ========================================================
    # PART B: RELATIVE FORCE PATTERN
    # ========================================================

    obs_force_sum = _safe_sum(
        obs_force
    )

    ref_force_sum = _safe_sum(
        ref_force
    )

    obs_force_ratio = (
        obs_force
        /
        obs_force_sum
    )

    ref_force_ratio = (
        ref_force
        /
        ref_force_sum
    )

    force_shape_error = float(
        np.mean(
            np.abs(
                obs_force_ratio
                -
                ref_force_ratio
            )
        )
    )

    # ========================================================
    # RELATIVE HOLD PATTERN
    # ========================================================

    obs_hold_sum = _safe_sum(
        obs_hold
    )

    ref_hold_sum = _safe_sum(
        ref_hold
    )

    obs_hold_ratio = (
        obs_hold
        /
        obs_hold_sum
    )

    ref_hold_ratio = (
        ref_hold
        /
        ref_hold_sum
    )

    hold_shape_error = float(
        np.mean(
            np.abs(
                obs_hold_ratio
                -
                ref_hold_ratio
            )
        )
    )

    # ========================================================
    # RELATIVE GAP PATTERN
    # ========================================================

    obs_gap_sum = _safe_sum(
        obs_gap
    )

    ref_gap_sum = _safe_sum(
        ref_gap
    )

    obs_gap_ratio = (
        obs_gap
        /
        obs_gap_sum
    )

    ref_gap_ratio = (
        ref_gap
        /
        ref_gap_sum
    )

    gap_shape_error = float(
        np.mean(
            np.abs(
                obs_gap_ratio
                -
                ref_gap_ratio
            )
        )
    )

    # ========================================================
    # COMBINED PATTERN ERROR
    # ========================================================

    shape_error = (
        0.50 * force_shape_error
        +
        0.30 * hold_shape_error
        +
        0.20 * gap_shape_error
    )

    # --------------------------------------------------------
    # Convert pattern error to similarity
    # --------------------------------------------------------

    shape_similarity = np.exp(
        -8.0 * shape_error
    )

    # ========================================================
    # FINAL MECHANICAL SCORE
    # ========================================================

    final_similarity = (
        0.75 * absolute_similarity
        +
        0.25 * shape_similarity
    )

    score = int(
        np.clip(
            100.0 * final_similarity,
            0.0,
            100.0
        )
    )

    return score


# ============================================================
# IMPROVED SPATIAL GAP SCORE
# ============================================================

def compute_spatial_gap_score(
    observed_gaps: List[float],
    enrolled_gaps: List[float],
    tolerance_mm: float = 25
) -> int:
    """
    Improved Spatial Gap authentication score.

    The layer continues to use only two ultrasonic
    hand-distance measurements.

    Two properties are evaluated:

    1. Absolute gap-distance agreement.
    2. Relative two-gap movement pattern.

    The HC-SR04 measurement remains a radial distance
    measurement rather than an exact hand-displacement
    measurement.
    """

    observed = np.asarray(
        observed_gaps,
        dtype=float
    )

    enrolled = np.asarray(
        enrolled_gaps,
        dtype=float
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if (
        len(observed) != 2
        or len(enrolled) != 2
    ):
        return 0

    tolerance_mm = max(
        float(tolerance_mm),
        1e-9
    )

    # ========================================================
    # PART A: ABSOLUTE DISTANCE AGREEMENT
    # ========================================================

    distance_errors = (
        np.abs(
            observed
            -
            enrolled
        )
        /
        tolerance_mm
    )

    absolute_error = float(
        np.mean(
            distance_errors
        )
    )

    absolute_similarity = np.exp(
        -absolute_error
    )

    # ========================================================
    # PART B: RELATIVE TWO-GAP PATTERN
    # ========================================================

    observed_sum = _safe_sum(
        observed
    )

    enrolled_sum = _safe_sum(
        enrolled
    )

    observed_ratio = (
        observed
        /
        observed_sum
    )

    enrolled_ratio = (
        enrolled
        /
        enrolled_sum
    )

    pattern_error = float(
        np.mean(
            np.abs(
                observed_ratio
                -
                enrolled_ratio
            )
        )
    )

    pattern_similarity = np.exp(
        -8.0 * pattern_error
    )

    # ========================================================
    # FINAL SPATIAL SCORE
    # ========================================================

    final_similarity = (
        0.75 * absolute_similarity
        +
        0.25 * pattern_similarity
    )

    return int(
        np.clip(
            100.0 * final_similarity,
            0.0,
            100.0
        )
    )


# ============================================================
# HYBRID FUSION
# ============================================================

def apply_hybrid_fusion(
    s_m: float,
    s_v: float,
    s_g: float,

    seq_thresh_v: int = 34,
    seq_thresh_g: int = 38,
    seq_thresh_m: int = 62,

    w_m: float = 0.45,
    w_v: float = 0.15,
    w_g: float = 0.40,

    weighted_threshold: int = 60
) -> Tuple[bool, str]:
    """
    Two-stage hybrid fusion.

    Stage 1:
        Sequential screening.

    Stage 2:
        Weighted score decision.

    NOTE:
    These defaults are retained for compatibility with the
    existing dataset generator. The optimization scripts
    should recompute the final decision using their selected
    parameters.
    """

    # ========================================================
    # STAGE 1: SEQUENTIAL SCREENING
    # ========================================================

    if s_v < seq_thresh_v:

        return (
            False,
            f"Voice screening failed "
            f"({s_v:.1f} < {seq_thresh_v})"
        )

    if s_g < seq_thresh_g:

        return (
            False,
            f"Gap screening failed "
            f"({s_g:.1f} < {seq_thresh_g})"
        )

    if s_m < seq_thresh_m:

        return (
            False,
            f"Mechanical screening failed "
            f"({s_m:.1f} < {seq_thresh_m})"
        )

    # ========================================================
    # STAGE 2: WEIGHTED FUSION
    # ========================================================

    s_total = (
        w_m * s_m
        +
        w_v * s_v
        +
        w_g * s_g
    )

    # ========================================================
    # FINAL DECISION
    # ========================================================

    if s_total >= weighted_threshold:

        return (
            True,
            f"Accepted "
            f"(score={s_total:.1f})"
        )

    return (
        False,
        f"Weighted check failed "
        f"(score={s_total:.1f} "
        f"< {weighted_threshold})"
    )


# ============================================================
# USER PROFILE GENERATION
# ============================================================

def create_user_profiles(
    num_users: int = 50,
    verbose: bool = True
) -> List[UserProfile]:
    """
    Create synthetic user profiles.
    """

    profiles = []

    for i in range(num_users):

        profile = UserProfile(
            user_id=i + 1
        )

        profiles.append(
            profile
        )

        if (
            verbose
            and i < 10
        ):

            print(
                f"User P{i + 1:02d}: "
                f"Force={profile.force_levels}, "
                f"Voice={profile.voice_digits}"
            )

    if (
        verbose
        and num_users > 10
    ):

        print(
            f"... and "
            f"{num_users - 10} "
            f"more users\n"
        )

    return profiles


# ============================================================
# OPTIONAL SELF-TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("sensor_models.py SELF-TEST")
    print("=" * 60)

    # --------------------------------------------------------
    # Create two profiles
    # --------------------------------------------------------

    profiles = create_user_profiles(
        num_users=2,
        verbose=True
    )

    user1 = profiles[0]
    user2 = profiles[1]

    # --------------------------------------------------------
    # Genuine-like mechanical observation
    # --------------------------------------------------------

    fsr = FSRSimulator()

    observed_force = [
        fsr.read_press(
            user1.force_levels[i],
            user1.force_std
        )
        for i in range(3)
    ]

    observed_hold = [
        fsr.read_hold_time(
            user1.hold_times[i],
            user1.hold_std
        )
        for i in range(3)
    ]

    observed_gap = [
        fsr.read_gap_time(
            user1.gap_times[i],
            user1.gap_std
        )
        for i in range(2)
    ]

    observed_mechanical = {
        "force": observed_force,
        "hold": observed_hold,
        "gap": observed_gap
    }

    enrolled_mechanical = {
        "force": user1.force_levels,
        "hold": user1.hold_times,
        "gap": user1.gap_times
    }

    mechanical_score = (
        compute_mechanical_score(
            observed_mechanical,
            enrolled_mechanical
        )
    )

    # --------------------------------------------------------
    # Genuine-like spatial observation
    # --------------------------------------------------------

    ultrasonic = UltrasonicSimulator()

    observed_spatial = [
        ultrasonic.measure_gap_distance(
            user1.gap_distances[i],
            user1.gap_distance_std
        )
        for i in range(2)
    ]

    spatial_score = (
        compute_spatial_gap_score(
            observed_spatial,
            user1.gap_distances
        )
    )

    # --------------------------------------------------------
    # Voice test
    # --------------------------------------------------------

    voice_sim = VoiceSimulator(
        user1.speaker
    )

    voice_score, dtw_cost = (
        voice_sim.simulate_voice_attempt(
            true_digit=user1.voice_digits[0],
            template=user1.voice_template,
            energy_level=20,
            digit_index=0
        )
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n" + "-" * 60)
    print("SELF-TEST RESULTS")
    print("-" * 60)

    print(
        f"Mechanical score : "
        f"{mechanical_score}"
    )

    print(
        f"Spatial score    : "
        f"{spatial_score}"
    )

    print(
        f"Voice score      : "
        f"{voice_score:.2f}"
    )

    print(
        f"Voice DTW cost   : "
        f"{dtw_cost:.4f}"
    )

    print("-" * 60)
    print("SELF-TEST COMPLETE")
    print("-" * 60)