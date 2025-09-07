# core/detection.py
# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2

HAS_DINO = True
gdino = None
GD_DEVICE = "cpu"

# Intento de import; si falla, apagamos DINO
try:
    import torch
    from groundingdino.util.inference import load_model, load_image, predict, annotate
    import groundingdino  # para ubicar el config dentro del paquete
    GD_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    HAS_DINO = False

# -------- Parámetros anti-"zoom" (ajustables por env) --------
def _get_float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return default

# Más padding por defecto para evitar recortes muy estrechos
PAD_FRAC = _get_float_env("GDINO_PAD_FRAC", 0.30)        # 30% de padding alrededor de la caja
# Rechaza cajas pequeñas -> usar imagen completa
MIN_REL_AREA = _get_float_env("GDINO_MIN_REL_AREA", 0.20) # si la caja < 20% del área -> full image
# Además, si alguna dimensión ocupa menos del 30% de la original, también usamos full image
MIN_REL_SIDE = _get_float_env("GDINO_MIN_REL_SIDE", 0.30)

# -------- Utilidades internas --------
def _xyxy_to_pixels(box, w, h):
    """
    Convierte coordenadas [x0,y0,x1,y1] (normalizadas o absolutas) a píxeles enteros seguros.
    """
    b = np.array(box, dtype=float)
    if b.max() <= 1.0:
        b = b * np.array([w, h, w, h], dtype=float)
    x0, y0, x1, y1 = b
    x0 = max(0, min(w-1, int(round(x0))))
    y0 = max(0, min(h-1, int(round(y0))))
    x1 = max(0, min(w-1, int(round(x1))))
    y1 = max(0, min(h-1, int(round(y1))))
    if x1 <= x0 or y1 <= y0:
        return 0, 0, w-1, h-1
    return x0, y0, x1, y1

def _expand_and_clip(x0, y0, x1, y1, w, h, frac=0.30):
    """
    Expande la caja un % de su tamaño y recorta a los límites.
    """
    bw = x1 - x0
    bh = y1 - y0
    padw = int(round(bw * frac))
    padh = int(round(bh * frac))
    nx0 = max(0, x0 - padw)
    ny0 = max(0, y0 - padh)
    nx1 = min(w-1, x1 + padw)
    ny1 = min(h-1, y1 + padh)
    if nx1 <= nx0 or ny1 <= ny0:
        return x0, y0, x1, y1
    return nx0, ny0, nx1, ny1

def _phrase_to_es_label(phrase: str) -> str:
    p = (phrase or "").lower()
    if any(k in p for k in ["pant", "jean", "trouser", "short"]):  return "Pantalón"
    if "skirt" in p:  return "Falda"
    if "dress" in p:  return "Vestido"
    if any(k in p for k in ["shirt","t-shirt","blouse","top","sweater","hoodie","jacket","coat"]): return "Camisa"
    return "Prenda"

def _find_default_cfg():
    env_cfg = os.environ.get("GDINO_CFG")
    if env_cfg and os.path.isfile(env_cfg):
        return env_cfg
    if not HAS_DINO:
        return None
    base = os.path.dirname(groundingdino.__file__)
    cfg = os.path.join(base, "config", "GroundingDINO_SwinT_OGC.py")
    return cfg if os.path.isfile(cfg) else None

def _ensure_ckpt():
    env_ckpt = os.environ.get("GDINO_CKPT")
    if env_ckpt and os.path.isfile(env_ckpt):
        return env_ckpt

    weights_dir = "weights"
    os.makedirs(weights_dir, exist_ok=True)
    ckpt = os.path.join(weights_dir, "groundingdino_swint_ogc.pth")
    if os.path.isfile(ckpt):
        return ckpt

    try:
        import urllib.request
        url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
        print("Descargando pesos de GroundingDINO… (una sola vez)")
        urllib.request.urlretrieve(url, ckpt)
        print("✓ Pesos descargados en:", ckpt)
        return ckpt
    except Exception as e:
        print("No se pudieron descargar los pesos de GroundingDINO:", e)
        return None

def maybe_load_model():
    global gdino, HAS_DINO, GD_DEVICE
    if not HAS_DINO:
        return None

    cfg = _find_default_cfg()
    ckpt = _ensure_ckpt()
    if not cfg or not os.path.isfile(cfg) or not ckpt or not os.path.isfile(ckpt):
        HAS_DINO = False
        print("GroundingDINO desactivado (faltan config o pesos). Se usará la imagen completa.")
        return None

    if gdino is None:
        gdino = load_model(cfg, ckpt)
        GD_DEVICE = "cuda" if ("torch" in globals() and torch.cuda.is_available()) else "cpu"
    return gdino

# ----------------- Detección + recorte con padding -----------------
def detect_and_crop(
    img_path,
    text_prompt=("shirt, t-shirt, blouse, top, sweater, hoodie, jacket, coat, "
                 "pants, jeans, trousers, shorts, skirt, dress"),
    box_threshold=0.35,
    text_threshold=0.25,
    pick="largest"
):
    """
    Devuelve (crop_bgr, annotated_rgb, etiqueta_es).
    - Si DINO no está disponible o falla, devuelve (imagen completa, None, None).
    - Aplica padding a la caja para evitar "zoom" excesivo.
    - Si la caja es demasiado pequeña o angosta vs. la imagen, usa la imagen completa.
    """
    if not HAS_DINO:
        return cv2.imread(img_path), None, None

    model = maybe_load_model()
    if model is None:
        return cv2.imread(img_path), None, None

    image_source, image_tensor = load_image(img_path)
    h, w = image_source.shape[:2]

    try:
        boxes, logits, phrases = predict(
            model=model, image=image_tensor, caption=text_prompt,
            box_threshold=box_threshold, text_threshold=text_threshold, device=GD_DEVICE
        )
    except Exception as e:
        print("Fallo en GroundingDINO.predict:", e)
        return cv2.imread(img_path), None, None

    if boxes is None or len(boxes) == 0:
        return cv2.imread(img_path), None, None

    # Elegir caja
    if pick == "best":
        idx = int(np.argmax(logits))
    else:
        areas = []
        for b in boxes:
            x0, y0, x1, y1 = _xyxy_to_pixels(b, w, h)
            areas.append((x1-x0)*(y1-y0))
        idx = int(np.argmax(areas))

    x0, y0, x1, y1 = _xyxy_to_pixels(boxes[idx], w, h)

    # Padding anti-zoom
    x0, y0, x1, y1 = _expand_and_clip(x0, y0, x1, y1, w, h, frac=PAD_FRAC)

    # Reglas de fallback a imagen completa
    bw, bh = (x1 - x0), (y1 - y0)
    rel_area = (bw * bh) / float(max(1, w*h))
    rel_w = bw / float(w)
    rel_h = bh / float(h)
    if (rel_area < MIN_REL_AREA) or (rel_w < MIN_REL_SIDE) or (rel_h < MIN_REL_SIDE):
        # Caja demasiado pequeña o angosta: usar full image
        x0, y0, x1, y1 = 0, 0, w-1, h-1

    bgr_full = cv2.imread(img_path)
    crop = bgr_full[y0:y1, x0:x1].copy()
    try:
        annotated = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    except Exception:
        annotated = None
    label_es = _phrase_to_es_label(phrases[idx] if phrases is not None and len(phrases) > idx else "")

    return crop, annotated, label_es
