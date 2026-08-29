# FHG Authentication Lock: Low-Cost Offline Multi-Modal Physical Authentication

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Paper](https://img.shields.io/badge/paper-ICCIT%202026-green.svg)](https://iccit.org.bd)

> **A fully offline, sub-$30 physical authentication lock fusing mechanical, acoustic, and ultrasonic sensing on a 16 MHz ATmega328P microcontroller.**

## 📋 Overview

This repository contains the design, simulation, and planned hardware implementation of a multi-modal physical authentication system that combines:
- **Mechanical sensing** (FSR-402 force sensor) → Force–Hold–Gap (FHG) password
- **Acoustic sensing** (MAX4466 microphone) → Liveness verification via temporal correlation
- **Spatial sensing** (HC-SR04 ultrasonic) → Gesture distance profiling

**Key Features:**
- ✅ **Fully offline** (no network/cloud dependency)
- ✅ **Low-cost** ($27 bill of materials)
- ✅ **Resource-efficient** (512 bytes SRAM, 3.1 KB flash on ATmega328P)
- ✅ **100× FAR improvement** (0.025% vs. 2.5% single-layer, simulated)
- ✅ **Revocable secrets** (FHG passwords redefinable, not like biometrics)

**Target Applications:**
- Rural health clinics (medicine storage)
- Small workshops (tool access control)
- Community centers (equipment lockers)
- Agricultural cooperatives (storage facilities)
- Offline deployments in connectivity-limited regions

---

## 📁 Repository Structure
FHG-Auth-Lock/
├── README.md # This file
├── LICENSE # MIT License
├── .gitignore # Git ignore file
│
├── simulation/ # Python simulation
│ ├── sensor_models.py # Realistic sensor simulators
│ ├── generate_dataset.py # Generate N=8 synthetic dataset
│ ├── analyze_results.py # Statistical analysis & tables
│ ├── visualize_data.py # Supplementary visualizations
│ └── requirements.txt # Python dependencies
│
├── data/ # Experimental data
│ ├── data_format.md # Column descriptions
│ └── experimental_data_N8.csv # Synthetic dataset (280 attempts)
│
├── firmware/ # Arduino firmware
│ └── FHG_Auth_Lock.ino # Main sketch
│
├── docs/ # Documentation
│ └── assembly_guide.md # Hardware assembly instructions
│
└── results/ # Analysis outputs
└── README.md # Results summary

text

---

## 🚀 Quick Start (Simulation)

### Prerequisites
- Python 3.8+
- NumPy, Pandas, Matplotlib, SciPy, Seaborn

### Installation

```bash
# Clone repository
git clone https://github.com/[your-username]/FHG-Auth-Lock.git
cd FHG-Auth-Lock/simulation

# Install dependencies
pip install -r requirements.txt