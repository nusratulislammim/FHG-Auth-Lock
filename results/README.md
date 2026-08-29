# Results Directory

This directory contains all analysis outputs from the simulation.

## Generated Files

| File | Description | Use |
|------|-------------|-----|
| `table_III_error_rates.tex` | LaTeX table of error rates | Copy into paper Section V-B |
| `table_IX_correlation.tex` | LaTeX correlation matrix | Copy into paper Section V-B |
| `figure_3_roc_curves.png` | ROC curves comparison | Insert as Figure 3 in paper |
| `statistical_summary.txt` | Detailed statistical analysis | Reference for discussion |
| `score_distributions.png` | Histograms of scores | Supplementary/Poster |
| `user_consistency.png` | Per-user consistency | Supplementary/Poster |
| `temporal_stability.png` | Stability across sessions | Supplementary/Poster |
| `confusion_matrix.png` | Confusion matrix | Supplementary/Poster |

## Re-generating Results

To regenerate all results:

```bash
cd ../simulation
python analyze_results.py
python visualize_data.py


---

## ✅ VERIFY FINAL STRUCTURE

After running all commands, your structure should look like this:

```powershell
# Check the structure
tree /F

# Output should show:
FHG-Auth-Lock/
├── simulation/
│   ├── sensor_models.py
│   ├── generate_dataset.py
│   ├── analyze_results.py
│   ├── visualize_data.py
│   └── requirements.txt
├── data/
│   ├── experimental_data_N8.csv
│   └── data_format.md
├── results/
│   ├── table_III_error_rates.tex
│   ├── table_IX_correlation.tex
│   ├── figure_3_roc_curves.png
│   ├── statistical_summary.txt
│   ├── score_distributions.png
│   ├── user_consistency.png
│   ├── temporal_stability.png
│   └── confusion_matrix.png
├── firmware/
│   └── FHG_Auth_Lock.ino
├── docs/
│   └── assembly_guide.md
├── README.md
├── LICENSE
└── .gitignore