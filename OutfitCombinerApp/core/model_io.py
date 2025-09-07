# core/model_io.py
import os, time, numpy as np, pandas as pd
from joblib import load, dump

DATASET_CSV = os.path.join("data", "combina_dataset.csv")
MODEL_PATH  = os.path.join("models", "combina_model.joblib")

# Columnas base de features usadas por el modelo
BASE_FEATURES = [
    "contrast_main","patternA","patternB","neutralA","neutralB",
    "satA","satB","valA","valB","lumA","lumB","lum_diff",
    "dH_min","dH_mean","dE_min","dE_mean",
    "A_H_mean","A_S_mean","A_V_mean","A_L_mean","A_frac_neutral",
    "B_H_mean","B_S_mean","B_V_mean","B_L_mean","B_frac_neutral"
]

# Colores principales (RGB) siempre presentes
CORE_COLOR_COLS = [
    "a_main_r","a_main_g","a_main_b","b_main_r","b_main_g","b_main_b"
]

# Metadatos extra que queremos persistir
EXTRA_COLS = [
    "pred","pred_proba","explanation","timestamp","mode","source",
    "detA_label","detB_label","imgA_path","imgB_path","hue_pair_note",
    # Usuario que hizo la predicción
    "user_id","user_name","user_email",
    # Para UI (chips/nombres) y patrones
    "a_color_hex","b_color_hex","a_color_name","b_color_name",
    "a_pattern","b_pattern"
]

def ensure_folders():
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("images", exist_ok=True)
    os.makedirs("assets", exist_ok=True)

def ensure_csv(path=DATASET_CSV):
    """
    Crea el CSV si no existe o agrega las columnas faltantes sin perder datos.
    """
    ensure_folders()
    base_cols = ["label"] + BASE_FEATURES + CORE_COLOR_COLS + EXTRA_COLS
    if not os.path.exists(path):
        pd.DataFrame([], columns=base_cols).to_csv(path, index=False)
    else:
        df = pd.read_csv(path)
        # Añadir cualquier columna faltante con NaN
        for c in base_cols:
            if c not in df.columns:
                df[c] = np.nan
        # También elimina duplicados de nombres de columna si se diera el caso
        df = df.loc[:, ~df.columns.duplicated()]
        df.to_csv(path, index=False)

def append_row(row: dict, path=DATASET_CSV):
    ensure_csv(path)
    df = pd.read_csv(path)
    # Asegura que todas las claves existan como columnas
    for k in row.keys():
        if k not in df.columns:
            df[k] = np.nan
    df.loc[len(df)] = row
    df.to_csv(path, index=False)

def load_bundle():
    bundle = load(MODEL_PATH)
    model = bundle["model"]
    feature_cols = bundle["features"]
    return model, feature_cols
