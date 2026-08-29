# Dataset Format Documentation

## File: experimental_data_N8.csv

### Overview
- **Total records:** 280 authentication attempts
- **Genuine attempts:** 160 (8 users × 20 attempts)
- **Impostor attempts:** 120 (8 users × 3 targets × 5 attempts)
- **Data source:** Simulation using empirically-derived sensor noise models
- **Date range:** 2025-01-15 to 2025-01-28 (simulated)

### Column Descriptions

| Column | Type | Description |
|--------|------|-------------|
| `participant_id` | string | User identifier (P01-P08) |
| `session` | int | Session number (1-4, 99=impostor) |
| `attempt` | int | Attempt number within session |
| `genuine` | bool | TRUE = genuine, FALSE = impostor |
| `timestamp` | int | Unix epoch timestamp |
| `force_0` | int | Peak force for press 1 (ADC 0-1023) |
| `force_1` | int | Peak force for press 2 (ADC 0-1023) |
| `force_2` | int | Peak force for press 3 (ADC 0-1023) |
| `hold_0` | int | Hold duration for press 1 (ms) |
| `hold_1` | int | Hold duration for press 2 (ms) |
| `hold_2` | int | Hold duration for press 3 (ms) |
| `gap_0` | int | Gap between press 1 and 2 (ms) |
| `gap_1` | int | Gap between press 2 and 3 (ms) |
| `baseline_noise` | int | Ambient acoustic noise (RMS ADC units) |
| `mean_distance` | int | Mean ultrasonic distance (mm) |
| `S_M` | int | Mechanical layer score (0-100) |
| `S_A` | int | Acoustic layer score (0-100) |
| `S_U` | int | Spatial layer score (0-100) |
| `decision` | string | ACCEPT or REJECT |