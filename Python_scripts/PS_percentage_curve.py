import numpy as np
import matplotlib.pyplot as plt

def membership_curve(w, slope_plateau=0.01, sharpness=1.5, H=1.0):
    """
    Generate a curve in [0,100]:
      - Near 0 until ~40
      - Rise until ~60
      - Consistent linear slope 60-85
      - Gradual drop to almost 0 by 95%
    
    Parameters
    ----------
    w : float or array-like
        Input values in [0,100].
    slope_plateau : float
        Linear slope for the plateau region (rise per unit w).
    sharpness : float
        Controls how sharp the rise/drop transitions are (lower = sharper).
    H : float
        Peak height scaling.
    """
    w = np.asarray(w, dtype=float)
    
    # Smooth step-up from 40 to 60
    rise = 1 / (1 + np.exp(-(w-60)/sharpness))   # sigmoid centered at 50
    
    # More gradual step-down starting from 85, nearly zero by 95%
    fall = 1 / (1 + np.exp((w-95)/2.0))    # gentler sigmoid centered at 95 
    
    # Base curve from rise and fall
    base = rise * fall

    # Create consistent linear slope in plateau region (60-100)
    plateau_mask = (w >= 60) & (w <= 100)
    slope_adjustment = np.ones_like(w)
    
    # Add linear slope only in plateau region
    slope_adjustment[plateau_mask] = 1 + slope_plateau * (w[plateau_mask] - 60)
    
    y = H * base * slope_adjustment
    
    # substract a constant of 0.25
    y -= 0.25

    return y

# Example usage
w_vals = np.linspace(0, 100, 500)
y_vals = membership_curve(w_vals, slope_plateau=0.01, sharpness=1.5, H=1.0)

plt.figure(figsize=(8,5))
plt.plot(w_vals, y_vals)
plt.title("Custom Membership Curve with Gradual Drop")
plt.xlabel("w (%)")
plt.ylabel("y(w)")
plt.grid(True)
plt.axvline(x=60, color='r', linestyle='--', alpha=0.5, label='Plateau start')
plt.axvline(x=85, color='r', linestyle='--', alpha=0.5, label='Plateau end')
plt.axvline(x=95, color='g', linestyle='--', alpha=0.5, label='Near zero')
plt.legend()
plt.show()