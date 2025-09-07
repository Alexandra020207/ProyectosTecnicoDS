import numpy as np
import cv2
from math import sqrt
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color

def _srgb_to_linear(v):
    v = v/255.0
    return np.where(v <= 0.04045, v/12.92, ((v+0.055)/1.055)**2.4)

def relative_luminance_rgb(rgb):
    r_lin, g_lin, b_lin = _srgb_to_linear(np.array(rgb, dtype=np.float32))
    return 0.2126*r_lin + 0.7152*g_lin + 0.0722*b_lin

def contrast_ratio(rgb1, rgb2):
    L1 = relative_luminance_rgb(rgb1)
    L2 = relative_luminance_rgb(rgb2)
    L1, L2 = max(L1, L2), min(L1, L2)
    return (L1 + 0.05) / (L2 + 0.05)

def rgb_to_lab_tuple(rgb):
    sr = sRGBColor(rgb[0], rgb[1], rgb[2], is_upscaled=True)
    lab = convert_color(sr, LabColor)
    return (lab.lab_l, lab.lab_a, lab.lab_b)

def lab_chroma(lab):
    _, a, b = lab
    return sqrt(a*a + b*b)

def hsv_tuple(rgb):
    bgr = np.array([[rgb[::-1]]], dtype=np.uint8)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0,0]
    H = float(hsv[0]) * 2.0
    S = hsv[1]/255.0
    V = hsv[2]/255.0
    return H, S, V

def is_neutral(rgb):
    lab = rgb_to_lab_tuple(rgb)
    if lab_chroma(lab) < 12:
        return True
    _, S, _ = hsv_tuple(rgb)
    return S < 0.18

def hue_distance_deg(h1, h2):
    d = abs(h1 - h2)
    return d if d <= 180 else 360 - d

# -------- Daltonismo (simulación y score de contraste) --------
_M = {
    "deuteranopia": np.array([[0.367, 0.861, -0.228],
                              [0.280, 0.673,  0.047],
                              [0.000, 0.142,  0.858]], dtype=np.float32),
    "protanopia":   np.array([[0.152, 1.053, -0.205],
                              [0.115, 0.786,  0.099],
                              [0.000, 0.046,  0.954]], dtype=np.float32),
    "tritanopia":   np.array([[1.256, -0.077, -0.179],
                              [-0.078,  0.931,  0.147],
                              [0.000,  0.280,  0.720]], dtype=np.float32),
}

def simulate_cvd(bgr: np.ndarray, mode: str) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    R,G,B = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    mat = _M.get(mode)
    if mat is None: return bgr
    Rp = mat[0,0]*R + mat[0,1]*G + mat[0,2]*B
    Gp = mat[1,0]*R + mat[1,1]*G + mat[1,2]*B
    Bp = mat[2,0]*R + mat[2,1]*G + mat[2,2]*B
    out = np.stack([Rp,Gp,Bp], axis=-1).clip(0,1)
    out = (out*255.0).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

def _lin(v):
    v = v/255.0
    return np.where(v<=0.04045, v/12.92, ((v+0.055)/1.055)**2.4)

def contrast_ratio_rgb(rgb1, rgb2):
    L1 = 0.2126*_lin(rgb1[0]) + 0.7152*_lin(rgb1[1]) + 0.0722*_lin(rgb1[2])
    L2 = 0.2126*_lin(rgb2[0]) + 0.7152*_lin(rgb2[1]) + 0.0722*_lin(rgb2[2])
    L1,L2 = max(L1,L2), min(L1,L2)
    return (L1+0.05)/(L2+0.05)

def hex_to_rgb(hex_color: str):
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2],16) for i in (0,2,4))

def cvd_accessible_score(hex_color: str) -> float:
    try: r,g,b = hex_to_rgb(hex_color)
    except Exception: return 0.0
    base = np.array([r,g,b], dtype=np.float32)
    white, black = np.array([255,255,255]), np.array([0,0,0])
    c_w = contrast_ratio_rgb(base, white)
    c_b = contrast_ratio_rgb(base, black)
    s = 0.5*min(c_w/3.0, 1.0) + 0.5*min(c_b/3.0, 1.0)  # umbral 3 ~ WCAG AA
    return float(s*100.0)
