# core/features.py
# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2

try:
    import pillow_avif
except Exception:
    pass
try:
    import pillow_heif
    try:
        pillow_heif.register_heif_opener()
    except Exception:
        pass
except Exception:
    pass

from PIL import Image
from sklearn.cluster import KMeans

from .color_utils import (
    hsv_tuple,
    is_neutral,
    relative_luminance_rgb,
    hue_distance_deg,
    rgb_to_lab_tuple,
    contrast_ratio,
)

# -----------------------------
# Lectura de imágenes (con fallback a Pillow y soporte AVIF/HEIF)
# -----------------------------
def read_bgr(path: str) -> np.ndarray:
    """
    Lee una imagen y devuelve BGR (OpenCV). Soporta JPG/PNG por OpenCV y
    cae a Pillow para AVIF/HEIF u otros formatos.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No existe el archivo: {path}")

    # 1) OpenCV primero (rápido y soporta JPG/PNG/WebP, etc.)
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is not None:
        return img

    # 2) Fallback con Pillow (activa AVIF/HEIF si hay plugin)
    try:
        pil = Image.open(path).convert("RGB")
        arr = np.array(pil)  # RGB
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    except Exception as e:
        raise FileNotFoundError(
            f"No se pudo leer la imagen: {path}. "
            f"Usa JPG/PNG o instala 'pillow-avif-plugin' / 'pillow-heif'. Detalle: {e}"
        )


# -----------------------------
# Utilidades de segmentación / paletas
# -----------------------------
def _mask_background(hsv_img: np.ndarray) -> np.ndarray:
    """
    Máscara simple que quita blancos muy claros y negros muy oscuros (posible fondo).
    Devuelve máscara binaria (uint8 {0,1}).
    """
    S = hsv_img[:, :, 1] / 255.0
    V = hsv_img[:, :, 2] / 255.0
    not_whitey = ~((V > 0.90) & (S < 0.15))
    not_blacky = ~(V < 0.08)
    return (not_whitey & not_blacky).astype(np.uint8)


def edge_density_score(img_bgr: np.ndarray) -> float:
    """
    Densidad de bordes Canny normalizada [0..1] como proxy de patrón/estampado.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 180)
    return float(edges.mean() / 255.0)


def _prepare_pixels_for_kmeans(img_bgr: np.ndarray) -> np.ndarray:
    """
    Devuelve arreglo de colores en RGB (float32) tras filtrar fondo cuando posible.
    Si la máscara queda vacía, usa toda la imagen. Si hay muy pocos píxeles,
    asegura al menos unos cientos haciendo un resize pequeño para estabilidad.
    """
    img = img_bgr
    h, w = img.shape[:2]

    # Si es muy chica, reescalar un poco para tener más muestras.
    if min(h, w) < 40:
        scale = 40.0 / float(min(h, w))
        nh, nw = int(round(h * scale)), int(round(w * scale))
        if nh > 0 and nw > 0:
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = _mask_background(hsv)

    ys, xs = np.where(mask > 0)
    if len(ys) < 200:
        flat = img.reshape(-1, 3)
    else:
        flat = img[ys, xs].reshape(-1, 3)

    # BGR -> RGB float32
    data_rgb = flat[:, ::-1].astype(np.float32)
    return data_rgb


def dominant_colors_bgr(img_bgr: np.ndarray, k: int = 4, min_area_ratio: float = 0.10) -> np.ndarray:
    """
    Estima k colores dominantes (en RGB uint8) tras filtrar fondo y extremos.
    - Ajusta k según muestras y número de colores únicos.
    - Filtra blancos muy claros y negros muy oscuros (si hay alternativas).
    - Siempre retorna >= 1 color.
    """
    data_rgb = _prepare_pixels_for_kmeans(img_bgr)
    if data_rgb.size == 0:
        # fallback improbable: usa el color medio de la imagen
        mean_bgr = cv2.mean(img_bgr)[:3]  # B, G, R
        mean_rgb = np.array(mean_bgr[::-1], dtype=np.uint8)
        return mean_rgb.reshape(1, 3)

    # Limitar k por número de colores únicos disponibles
    uniques = np.unique(data_rgb.astype(np.uint8), axis=0)
    unique_count = int(uniques.shape[0])
    k_eff = int(max(1, min(k, unique_count, max(2, data_rgb.shape[0] // 5000))))

    if k_eff == 1:
        centers = uniques[:1].astype(np.uint8)
        counts = np.array([data_rgb.shape[0]], dtype=np.int64)
    else:
        # KMeans con estabilidad
        kmeans = KMeans(n_clusters=k_eff, n_init=10, random_state=0)
        labels = kmeans.fit_predict(data_rgb)
        centers = kmeans.cluster_centers_.astype(np.uint8)
        counts = np.bincount(labels, minlength=k_eff)

    # Ordenar por frecuencia
    order = np.argsort(counts)[::-1]
    centers = centers[order]
    counts = counts[order]
    total = int(counts.sum()) if counts.sum() > 0 else 1

    # Quedarme con clusters relevantes (pero siempre al menos el primero)
    keep_idx = [i for i, c in enumerate(counts) if (c / total >= min_area_ratio) or i == 0]
    centers = centers[keep_idx]

    # Filtra extremos: blancos muy claros (alto V, baja S) y negros muy oscuros
    filtered_c = []
    for rgb in centers:
        H, S, V = hsv_tuple(rgb)
        if (V > 0.92 and S < 0.10) or (V < 0.08):
            continue
        filtered_c.append(rgb)

    if len(filtered_c) == 0:
        filtered_c = [centers[0]]

    return np.array(filtered_c, dtype=np.uint8)


def palette_stats(pal: np.ndarray) -> dict:
    """
    Estadísticos de una paleta (RGB uint8):
      - Promedios de H, S, V, luminancia (L)
      - Fracción de colores "neutros"
    """
    pal = np.array(pal, dtype=np.uint8)
    if pal.ndim == 1:
        pal = pal.reshape(1, 3)

    Hs, Ss, Vs, Ls = [], [], [], []
    neutrals = 0
    for rgb in pal:
        H, S, V = hsv_tuple(rgb)
        Hs.append(H)
        Ss.append(S)
        Vs.append(V)
        Ls.append(relative_luminance_rgb(rgb))
        if is_neutral(rgb):
            neutrals += 1

    return {
        "H_mean": float(np.mean(Hs)) if Hs else 0.0,
        "S_mean": float(np.mean(Ss)) if Ss else 0.0,
        "V_mean": float(np.mean(Vs)) if Vs else 0.0,
        "L_mean": float(np.mean(Ls)) if Ls else 0.0,
        "frac_neutral": (neutrals / len(pal)) if len(pal) > 0 else 0.0,
    }


def pair_features(imgA_bgr: np.ndarray, imgB_bgr: np.ndarray, k_colors: int = 4):
    """
    Extrae features para un par de prendas (A y B) y devuelve (feats, meta).
    Garantiza que exista siempre un color principal por prenda.
    """
    palA = dominant_colors_bgr(imgA_bgr, k=k_colors)
    palB = dominant_colors_bgr(imgB_bgr, k=k_colors)

    # Asegurar al menos un color por prenda
    if palA.size == 0:
        mean_bgr = cv2.mean(imgA_bgr)[:3]
        palA = np.array([[mean_bgr[2], mean_bgr[1], mean_bgr[0]]], dtype=np.uint8)
    if palB.size == 0:
        mean_bgr = cv2.mean(imgB_bgr)[:3]
        palB = np.array([[mean_bgr[2], mean_bgr[1], mean_bgr[0]]], dtype=np.uint8)

    a_main = palA[0]
    b_main = palB[0]

    # Distancias entre todas las combinaciones de la paleta A×B
    dH_list, dE_list = [], []
    for ra in palA:
        for rb in palB:
            H1, _, _ = hsv_tuple(ra)
            H2, _, _ = hsv_tuple(rb)
            dH_list.append(hue_distance_deg(H1, H2))

            la = rgb_to_lab_tuple(ra)
            lb = rgb_to_lab_tuple(rb)
            dE_list.append(((la[0] - lb[0]) ** 2 + (la[1] - lb[1]) ** 2 + (la[2] - lb[2]) ** 2) ** 0.5)

    dH_min = float(np.min(dH_list)) if dH_list else 0.0
    dH_mean = float(np.mean(dH_list)) if dH_list else 0.0
    dE_min = float(np.min(dE_list)) if dE_list else 0.0
    dE_mean = float(np.mean(dE_list)) if dE_list else 0.0

    cr_main = contrast_ratio(a_main, b_main)

    pA = edge_density_score(imgA_bgr)
    pB = edge_density_score(imgB_bgr)

    neutralA = 1.0 if is_neutral(a_main) else 0.0
    neutralB = 1.0 if is_neutral(b_main) else 0.0

    _, sA, vA = hsv_tuple(a_main)
    _, sB, vB = hsv_tuple(b_main)
    lA = relative_luminance_rgb(a_main)
    lB = relative_luminance_rgb(b_main)

    stA = palette_stats(palA)
    stB = palette_stats(palB)

    feats = {
        "contrast_main": cr_main,
        "patternA": pA,
        "patternB": pB,
        "neutralA": neutralA,
        "neutralB": neutralB,
        "satA": sA,
        "satB": sB,
        "valA": vA,
        "valB": vB,
        "lumA": lA,
        "lumB": lB,
        "lum_diff": abs(lA - lB),
        "dH_min": dH_min,
        "dH_mean": dH_mean,
        "dE_min": dE_min,
        "dE_mean": dE_mean,
        "A_H_mean": stA["H_mean"],
        "A_S_mean": stA["S_mean"],
        "A_V_mean": stA["V_mean"],
        "A_L_mean": stA["L_mean"],
        "A_frac_neutral": stA["frac_neutral"],
        "B_H_mean": stB["H_mean"],
        "B_S_mean": stB["S_mean"],
        "B_V_mean": stB["V_mean"],
        "B_L_mean": stB["L_mean"],
        "B_frac_neutral": stB["frac_neutral"],
    }

    meta = {
        "a_main_rgb": [int(x) for x in a_main],
        "b_main_rgb": [int(x) for x in b_main],
    }
    return feats, meta


def explain_pair(feats: dict, meta: dict):
    """
    Explicación textual sencilla basada en las features calculadas.
    """
    H1, _, _ = hsv_tuple(meta["a_main_rgb"])
    H2, _, _ = hsv_tuple(meta["b_main_rgb"])
    dH = hue_distance_deg(H1, H2)

    cr = feats["contrast_main"]
    patA, patB = feats["patternA"], feats["patternB"]
    neuA, neuB = bool(feats["neutralA"]), bool(feats["neutralB"])
    satA, satB = feats["satA"], feats["satB"]
    lum_diff = feats["lum_diff"]

    notes = []
    if neuA or neuB:
        notes.append("Al menos una prenda es neutra (negro/blanco/gris/beige).")
    if dH < 20:
        notes.append("Colores análogos (muy parecidos).")
    elif 110 <= dH <= 150:
        notes.append("Colores casi complementarios (alto contraste de tono).")
    else:
        notes.append(f"Diferencia de tono ≈ {int(dH)}°.")
    if cr < 1.25:
        notes.append("Muy poco contraste de luminosidad; puede verse plano.")
    elif cr > 3.0:
        notes.append("Contraste alto de luminosidad; look marcado.")
    if patA > 0.22 and patB > 0.22:
        notes.append("Dos patrones fuertes compiten.")
    elif (patA > 0.22) ^ (patB > 0.22):
        notes.append("Un patrón con prenda lisa (regla segura).")
    if max(satA, satB) > 0.75 and min(satA, satB) < 0.25:
        notes.append("Una prenda muy saturada con otra muy apagada.")

    recs = []
    if patA > 0.22 and patB > 0.22:
        recs.append("Usa solo un patrón y deja la otra prenda lisa.")
    if dH < 15 and cr < 1.25:
        recs.append("Aumenta contraste: aclara/oscurece una prenda o añade un neutro.")
    if not (neuA or neuB) and 100 <= dH <= 160 and cr < 1.4:
        recs.append("Para complementarios, sube contraste o saturación de una prenda.")
    if lum_diff < 0.08:
        recs.append("La diferencia de brillo es muy baja; usa un tono más claro/oscuro.")

    return " ".join(notes), (" ".join(recs) if recs else "Combina bien tal como está.")
