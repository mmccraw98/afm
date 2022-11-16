import numpy as np
import pandas as pd
from .misc import get_window_size, get_elbow_min
from .math_funcs import sma

def try_cut_contact_fd(f, h, z, n_pi_rot=0, min_size=10):
    j0 = np.argmax(z)
    j = np.argmax(f[:j0])
    i = get_elbow_min(h, f, np.pi * n_pi_rot)
    if j - i > min_size:
        f = f[i:j] - f[i]
        h = h[i:j] - h[i]
        z = z[i:j] - z[i]
    return f, h, z

def try_smooth_and_offset_fd(f, h, pct_smooth=0.01):
    win_size = get_window_size(pct_smooth, f.size)
    if win_size > 1:
        f = sma(f, win_size)
        h = sma(h, win_size)
    return f - f[0], h - h[0]

def format_ramplike_fd_for_z_transform(f, h, z, n_pi_rot=0, min_size=10, pct_smooth=0.01):
    f, h, z = try_cut_contact_fd(f, h, z, n_pi_rot=n_pi_rot, min_size=min_size)
    f, h = try_smooth_and_offset_fd(f, h, pct_smooth=pct_smooth)
    return pd.DataFrame({'f': abs(f), 'h': abs(h)})

def get_f_h_z(df, k):
    f = df.Defl.values * k
    h = (df.ZSnsr - df.Defl).values
    z = df.ZSnsr.values
    return f, h, z

def get_contact_points(f, h, z):
    j = np.argmax(f[: np.argmax(z)])
    i = np.argmin(f[: j])
    return i, j