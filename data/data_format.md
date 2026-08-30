# Dataset Format Documentation

## File: experimental_data_N8.csv

### Overview
- **Total records:** 280 authentication attempts
- **Genuine attempts:** 160 (8 users × 20 attempts)
- **Impostor attempts:** 120 (8 users × 3 targets × 5 attempts)
- **Data source:** Simulation using empirically-derived sensor noise models
- **System:** 4-Layer Hybrid Authentication (Voice + Spatial Gap + Mechanical + Spatial Position)

### Column Descriptions

| Column | Type | Description |
|--------|------|-------------|
| `participant_id` | string | User identifier (P01-P08) |
| `session` | int | Session number (1-4, 99=impostor) |
| `attempt` | int | Attempt number within session |
| `genuine` | bool | TRUE = genuine, FALSE = impostor |
| `timestamp` | int | Unix epoch timestamp |
| `force_0, force_1, force_2` | int | Peak force for presses 1-3 (ADC 0-1023) |
| `hold_0, hold_1, hold_2` | int | Hold duration for presses 1-3 (ms) |
| `gap_0, gap_1` | int | Gap between presses 1-2 and 2-3 (ms) |
| `voice_0, voice_1, voice_2` | int | Recognized digits (0-9) for presses 1-3 |
| `gap_dist_0, gap_dist_1` | int | Hand movement distance during gaps 1-2 (mm) |
| `baseline_noise` | int | Ambient acoustic noise (RMS ADC units) |
| `mean_distance` | int | Mean ultrasonic hand distance (mm) |
| `S_V` | int | Voice layer score (0-100) |
| `S_G` | int | Spatial Gap layer score (0-100) |
| `S_M` | int | Mechanical layer score (0-100) |
| `S_U` | int | Spatial Position layer score (0-100) |
| `decision` | string | ACCEPT or REJECT |

### Performance Summary (Optimal Thresholds)
- **FAR:** 8.33%
- **FRR:** 7.50%
- **EER:** 7.92%
- **Accuracy:** 92.08%
- **Optimal Thresholds:** τ_V=60, τ_G=30, τ_M=55, τ_U=60