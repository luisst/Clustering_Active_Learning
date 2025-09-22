import numpy as np
import matplotlib.pyplot as plt


def hump_curve(k, a=0.3, m=6, b=40, H=1.0):
    """
    Generate a non-negative right-skewed hump-shaped curve.
    
    Parameters
    ----------
    k : int or array-like
        Input value(s). k should be >= 0.
    a : float
        Controls steepness of the rise before the peak.
    m : float
        Location of the peak (mode).
    b : float
        Controls the rate of decay after the peak.
    H : float
        Peak height (scaling factor).

    Returns
    -------
    y : float or ndarray
        Value(s) of the curve at k.
    """
    k = np.asarray(k, dtype=float)
    y = np.zeros_like(k, dtype=float)
    
    mask = k > 0
    
    # Modified formula to ensure peak is always at k=m
    # Use different behavior before and after the peak
    before_peak = (k <= m) & mask
    after_peak = (k > m) & mask
    
    # Before peak: power function
    if np.any(before_peak):
        y[before_peak] = H * (k[before_peak] / m) ** a
    
    # After peak: exponential decay
    if np.any(after_peak):
        y[after_peak] = H * np.exp(-(k[after_peak] - m) / b)
    
    return y

# Example usage
k_vals = np.arange(0, 91)  # integers from 0 to 90
y_vals = hump_curve(k_vals, a=0.3, m=6, b=40, H=1.0)

plt.figure(figsize=(8,5))
plt.plot(k_vals, y_vals, marker='o')
plt.title("Custom Skewed Hump Curve")
plt.xlabel("k")
plt.ylabel("y(k)")
plt.grid(True)
plt.show()

