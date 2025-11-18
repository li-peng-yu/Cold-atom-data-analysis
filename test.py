# Compute k (length per unit) and v0 from position-coefficient data when total time is 0.03 s across 13 samples.

import numpy as np
import matplotlib.pyplot as plt

# Given data
x = np.array([
    92.2312, 113.3891, 140.4001, 173.2465, 211.6426, 256.343,
    306.6195, 362.5244, 424.2219, 492.2367, 566.2968, 645.4582, 730.6947
], dtype=float)

n = len(x)
T_total = 0.024  # seconds (total duration across all 13 points)
dt = T_total / (n - 1)  # spacing
t = np.arange(n) * dt

# Fit model: x = c + a*t + b*t^2
A = np.vstack([np.ones_like(t), t, t**2]).T
coeffs, *_ = np.linalg.lstsq(A, x, rcond=None)
c, a, b = coeffs

# Known gravity (can be adjusted)
g = 9.8  # m/s^2

# Convert to scale and initial velocity using: x = c + (v0/k)*t + (g/(2k))*t^2
k = g / (2 * b)  # meters per unit
v0 = a * k       # m/s

# Reconstruct fit
x_fit = c + a*t + b*t**2

print(f"Samples: {n}, total time: {T_total:.6f} s, dt: {dt:.6f} s")
print(f"Fitted parameters: c = {c:.6f}, a = {a:.6f}, b = {b:.6f}")
print(f"Per-unit length k = {k:.9f} m/unit")
print(f"Initial velocity v0 = {v0:.9f} m/s")

# Plot data and fit for sanity check
plt.figure()
plt.scatter(t, x, label="Data")
plt.plot(t, x_fit, label="Quadratic fit")
plt.xlabel("Time (s)")
plt.ylabel("Position coefficient (units)")
plt.legend()
plt.title("Position Coefficient vs Time (Total 0.03 s)")
plt.show()
