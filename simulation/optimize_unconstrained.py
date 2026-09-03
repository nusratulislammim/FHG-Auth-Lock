"""
optimize_unconstrained.py

Unconstrained fusion optimization.

IMPORTANT:
- Uses TRAINING SET ONLY.
- Test set is never used for optimization.
- Voice is not artificially weakened.
- Weights are allowed to vary freely.
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm


TRAIN_FILE = "../data/train_data.csv"
OUTPUT_FILE = "../results/optimal_unconstrained.txt"

os.makedirs("../results", exist_ok=True)


# ============================================================
# LOAD TRAINING DATA
# ============================================================

df = pd.read_csv(TRAIN_FILE)

genuine = df[df["genuine"] == True]
impostor = df[df["genuine"] == False]

print("Training genuine :", len(genuine))
print("Training impostor:", len(impostor))


# ============================================================
# ARRAYS
# ============================================================

G_V = genuine["S_V"].to_numpy(dtype=float)
G_G = genuine["S_G"].to_numpy(dtype=float)
G_M = genuine["S_M"].to_numpy(dtype=float)

I_V = impostor["S_V"].to_numpy(dtype=float)
I_G = impostor["S_G"].to_numpy(dtype=float)
I_M = impostor["S_M"].to_numpy(dtype=float)


# ============================================================
# GRID
# ============================================================

TAU_V_VALUES = range(5, 71, 5)
TAU_G_VALUES = range(5, 61, 5)
TAU_M_VALUES = range(5, 61, 5)
TAU_T_VALUES = range(20, 81, 5)

WEIGHT_VALUES = np.arange(0.05, 0.55, 0.05)


# ============================================================
# FUSION EVALUATION
# ============================================================

def evaluate_configuration(
    tau_v,
    tau_g,
    tau_m,
    tau_t,
    w_v,
    w_g,
    w_m
):

    # --------------------------------------------------------
    # Genuine
    # --------------------------------------------------------
    g_screen = (
        (G_V >= tau_v) &
        (G_G >= tau_g) &
        (G_M >= tau_m)
    )

    g_score = (
        w_v * G_V +
        w_g * G_G +
        w_m * G_M
    )

    g_accept = g_screen & (g_score >= tau_t)

    fr = 1.0 - np.mean(g_accept)

    # --------------------------------------------------------
    # Impostor
    # --------------------------------------------------------
    i_screen = (
        (I_V >= tau_v) &
        (I_G >= tau_g) &
        (I_M >= tau_m)
    )

    i_score = (
        w_v * I_V +
        w_g * I_G +
        w_m * I_M
    )

    i_accept = i_screen & (i_score >= tau_t)

    far = np.mean(i_accept)

    aer = (far + fr) / 2.0

    return far, fr, aer


# ============================================================
# OPTIMIZATION
# ============================================================

best = None

configs = 0

for tau_v in tqdm(TAU_V_VALUES, desc="Optimizing"):

    for tau_g in TAU_G_VALUES:

        for tau_m in TAU_M_VALUES:

            for tau_t in TAU_T_VALUES:

                for w_v in WEIGHT_VALUES:

                    for w_g in WEIGHT_VALUES:

                        w_m = round(1.0 - w_v - w_g, 2)

                        # All weights must be positive.
                        if w_m < 0.05 or w_m > 0.90:
                            continue

                        # Floating-point safety.
                        if abs(
                            w_v + w_g + w_m - 1.0
                        ) > 1e-9:
                            continue

                        configs += 1

                        far, frr, aer = evaluate_configuration(
                            tau_v,
                            tau_g,
                            tau_m,
                            tau_t,
                            w_v,
                            w_g,
                            w_m
                        )

                        candidate = (
                            aer,
                            far,
                            frr,
                            tau_v,
                            tau_g,
                            tau_m,
                            tau_t,
                            w_v,
                            w_g,
                            w_m
                        )

                        # ------------------------------------------------
                        # Primary objective:
                        # minimum AER
                        #
                        # Tie-breaker:
                        # lower FAR
                        # ------------------------------------------------
                        if best is None:

                            best = candidate

                        else:

                            if candidate[:3] < best[:3]:
                                best = candidate


# ============================================================
# RESULT
# ============================================================

(
    aer,
    far,
    frr,
    tau_v,
    tau_g,
    tau_m,
    tau_t,
    w_v,
    w_g,
    w_m
) = best


print("\n" + "=" * 60)
print("UNCONSTRAINED OPTIMIZATION")
print("=" * 60)

print(f"tau_V = {tau_v}")
print(f"tau_G = {tau_g}")
print(f"tau_M = {tau_m}")
print(f"tau_T = {tau_t}")

print(f"w_V = {w_v:.2f}")
print(f"w_G = {w_g:.2f}")
print(f"w_M = {w_m:.2f}")

print(f"Training FAR = {far * 100:.4f}%")
print(f"Training FRR = {frr * 100:.4f}%")
print(f"Training AER = {aer * 100:.4f}%")

print(f"Configurations evaluated = {configs}")


# ============================================================
# SAVE
# ============================================================

with open(OUTPUT_FILE, "w") as f:

    f.write("UNCONSTRAINED FUSION OPTIMIZATION\n")
    f.write("=" * 50 + "\n\n")

    f.write(f"tau_V = {tau_v}\n")
    f.write(f"tau_G = {tau_g}\n")
    f.write(f"tau_M = {tau_m}\n")
    f.write(f"tau_T = {tau_t}\n\n")

    f.write(f"w_V = {w_v:.4f}\n")
    f.write(f"w_G = {w_g:.4f}\n")
    f.write(f"w_M = {w_m:.4f}\n\n")

    f.write(f"Training FAR = {far * 100:.4f}%\n")
    f.write(f"Training FRR = {frr * 100:.4f}%\n")
    f.write(f"Training AER = {aer * 100:.4f}%\n")

print("\nSaved:", OUTPUT_FILE)