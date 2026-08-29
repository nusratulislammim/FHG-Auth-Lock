# Dataset Format Documentation

## File: `experimental_data_N8.csv`

### Overview
- **Total records:** 280 authentication attempts
- **Genuine attempts:** 160 (8 users × 20 attempts across 4 sessions)
- **Impostor attempts:** 120 (8 users × 3 targets × 5 attempts each)
- **Data source:** Simulation using empirically-derived sensor noise models
- **Date range:** 2025-01-15 to 2025-01-28 (simulated timestamps)

### Column Descriptions

| Column | Type | Range | Description |
|--------|------|-------|-------------|
| `participant_id` | string | P01–P08 | User identifier |
| `session` | int | 1–4, 99 | Session number (1=day 1, 2=day 3, 3=day 7, 4=day 14; 99=impostor) |
| `attempt` | int | 1–5 | Attempt number within session |
| `genuine` | bool | TRUE/FALSE | TRUE = genuine user, FALSE = impostor |
| `timestamp` | int | Unix epoch | Simulated timestamp of authentication attempt |
| `force_0` | int | 0–1023 | Peak force (ADC) for press 1 |
| `force_1` | int | 0–1023 | Peak force (ADC) for press 2 |
| `force_2` | int | 0–1023 | Peak force (ADC) for press 3 |
| `hold_0` | int | 50–800 ms | Hold duration for press 1 |
| `hold_1` | int | 50–800 ms | Hold duration for press 2 |
| `hold_2` | int | 50–800 ms | Hold duration for press 3 |
| `gap_0` | int | 100–1000 ms | Gap between press 1 and 2 |
| `gap_1` | int | 100–1000 ms | Gap between press 2 and 3 |
| `baseline_noise` | int | 5–50 | Ambient acoustic noise (RMS ADC units) |
| `mean_distance` | int | 100–500 mm | Mean ultrasonic distance during gesture |
| `S_M` | int | 0–100 | Mechanical layer similarity score |
| `S_A` | int | 0–100 | Acoustic layer liveness score |
| `S_U` | int | 0–100 | Spatial layer similarity score |
| `decision` | string | ACCEPT/REJECT | Final authentication decision (AND-rule fusion) |

### Usage Examples

**Python (Pandas):**
```python
import pandas as pd

# Load data
df = pd.read_csv("experimental_data_N8.csv")

# Filter genuine attempts
genuine = df[df['genuine'] == True]

# Compute FAR
impostor = df[df['genuine'] == False]
false_accepts = impostor[impostor['decision'] == 'ACCEPT']
FAR = 100 * len(false_accepts) / len(impostor)
print(f"FAR: {FAR:.3f}%")