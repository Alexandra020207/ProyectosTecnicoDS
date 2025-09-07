# app.py
# --- (opcional) silenciar avisos ruidosos ---
import os, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")
# --------------------------------------------

import io, time, numpy as np, cv2, traceback, math
import flet as ft
from PIL import Image

from ui.theme import (
    PRIMARY, ACCENT, CARD_BG, SURFACE,
    TITLE_STYLE, SUBTITLE_STYLE, LABEL_STYLE, BIG_RESULT,
    elevated_style, outlined_style
)

# --- Core existentes (tu lógica del modelo) ---
from core.features import read_bgr, pair_features, explain_pair
from core.model_io import ensure_folders, ensure_csv, append_row, load_bundle, DATASET_CSV
from core.detection import detect_and_crop, HAS_DINO
from core.color_utils import hsv_tuple, hue_distance_deg, contrast_ratio

# --- DB helpers (MariaDB/MySQL) ---
from db.mysql_conn import (
    get_connection,
    register_user, login_user,
    catalogo_list, catalogo_add, catalogo_delete
)

# ================== Helpers UI/Color ==================
def base64_from_bytes(b: bytes) -> str:
    import base64 as b64
    return b64.b64encode(b).decode("utf-8")

def pil_preview_from_bgr(bgr, max_w=520):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    w, h = pil.size
    if w > max_w:
        s = max_w / w
        pil = pil.resize((int(w*s), int(h*s)))
    buf = io.BytesIO(); pil.save(buf, format="PNG")
    return buf.getvalue()

def rgb_to_hex(rgb):
    r,g,b = int(rgb[0]), int(rgb[1]), int(rgb[2])
    return f"#{r:02X}{g:02X}{b:02X}"

def hex_to_rgb(hexs: str):
    h = (hexs or "").lstrip("#")
    if len(h) != 6: return (0,0,0)
    return tuple(int(h[i:i+2], 16) for i in (0,2,4))

def basic_color_name(rgb):
    H, S, V = hsv_tuple(rgb)
    if V < 0.12: return "negro"
    if V > 0.88 and S < 0.12: return "blanco"
    if S < 0.15: return "gris"
    if 0 <= H < 15 or 345 <= H < 360:  return "rojo"
    if 15 <= H < 35:   return "naranja"
    if 35 <= H < 55:   return "amarillo"
    if 55 <= H < 85:   return "lima"
    if 85 <= H < 150:  return "verde"
    if 150 <= H < 190: return "cian"
    if 190 <= H < 225: return "azul"
    if 225 <= H < 270: return "índigo"
    if 270 <= H < 295: return "morado"
    if 295 <= H < 330: return "magenta"
    if 330 <= H < 345: return "rosado"
    return "color"

# ================== Detección de patrón (robusta) ==================
def _prep_gray(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
    except Exception:
        pass
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    return gray

def _fg_mask(img_bgr):
    """Máscara de prenda: quita fondos blancos/lavados (alto V, baja S)."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    S = hsv[:,:,1] / 255.0
    V = hsv[:,:,2] / 255.0
    not_white = ~((V > 0.92) & (S < 0.12))
    mask = (not_white.astype(np.uint8)) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    return mask

def _edge_density(gray, mask=None):
    if mask is not None:
        gray = cv2.bitwise_and(gray, gray, mask=mask)
    edges = cv2.Canny(gray, 80, 180)
    if mask is None:
        return edges, float(edges.mean() / 255.0)
    fg = max(1, int(np.count_nonzero(mask)))
    ed = float(np.count_nonzero(edges)) / float(fg)
    return edges, ed

def _make_blob_detector(min_area, max_area,
                        min_circ=0.55, min_inertia=0.15, min_conv=0.70):
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 10
    params.maxThreshold = 220
    params.thresholdStep = 10
    params.filterByArea = True
    params.minArea = float(min_area)
    params.maxArea = float(max_area)
    params.filterByCircularity = True
    params.minCircularity = float(min_circ)
    params.filterByInertia = True
    params.minInertiaRatio = float(min_inertia)
    params.filterByConvexity = True
    params.minConvexity = float(min_conv)
    params.filterByColor = False
    params.minDistBetweenBlobs = 10
    ver_major = int(cv2.__version__.split(".")[0])
    return (cv2.SimpleBlobDetector_create(params)
            if ver_major >= 3 else cv2.SimpleBlobDetector(params))

def _blob_stats(gray, h, w, mask=None):
    """
    Cuenta blobs circulares en ambas polaridades y devuelve:
      densidad_normalizada_por_Mpx, mediana_diametro_px
    (método 1: SimpleBlobDetector/Hough de OpenCV)
    """
    if mask is not None:
        g1 = cv2.bitwise_and(gray, gray, mask=mask)
        g2 = cv2.bitwise_and(255-gray, 255-gray, mask=mask)
        area = max(1, int(np.count_nonzero(mask)))
    else:
        g1, g2 = gray, (255-gray)
        area = h*w

    # áreas relativas a la zona de prenda (no a toda la imagen)
    min_area = max(25, int(0.00005 * area))   # 0.005%
    max_area = int(0.10 * area)               # 10%
    detector = _make_blob_detector(min_area, max_area)

    best_cnt, best_sizes = 0, []
    for img in (g1, g2):
        kps = detector.detect(img)
        cnt = len(kps)
        if cnt > best_cnt:
            best_cnt = cnt
            best_sizes = [kp.size for kp in kps]  # size = diámetro en px

    mp = max(area / 1_000_000.0, 0.2)  # normalización
    dens = best_cnt / mp
    med_size = float(np.median(best_sizes)) if best_sizes else 0.0
    return dens, med_size

def _hough_circle_density(gray, h, w, mask=None):
    """Refuerzo con HoughCircles (en zona de prenda si hay máscara)."""
    if mask is not None:
        gray = cv2.bitwise_and(gray, gray, mask=mask)
        area = max(1, int(np.count_nonzero(mask)))
    else:
        area = h*w
    mn = min(h, w)
    minR = max(3, int(0.015 * mn))
    maxR = max(minR + 2, int(0.22 * mn))
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2,
        minDist=max(12, int(0.08 * mn)),
        param1=100, param2=18,
        minRadius=minR, maxRadius=maxR
    )
    cnt = 0 if circles is None else circles.shape[1]
    mp = max(area / 1_000_000.0, 0.2)
    return cnt / mp

def _detect_grid_lines(edges, h, w, mask=None):
    """Heurística de cuadros / rayas: líneas H y V fuertes en la prenda."""
    if mask is not None:
        edges = cv2.bitwise_and(edges, edges, mask=mask)

    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, h//20)))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, w//20), 1))
    v_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, v_kernel)
    h_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, h_kernel)

    if mask is not None:
        area = max(1, int(np.count_nonzero(mask)))
    else:
        area = float(h*w)

    v_score = float(np.count_nonzero(v_lines)) / float(area)
    h_score = float(np.count_nonzero(h_lines)) / float(area)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=120,
                            minLineLength=max(20, int(min(h, w)*0.20)),
                            maxLineGap=int(min(h, w)*0.05))
    v_cnt = h_cnt = 0
    if lines is not None:
        for l in lines[:,0,:]:
            x1,y1,x2,y2 = l
            ang = abs(math.degrees(math.atan2(y2-y1, x2-x1)))
            if ang < 12:     h_cnt += 1
            elif ang > 78:   v_cnt += 1

    mp = max(((area) / 1_000_000.0), 0.2)
    return h_score, v_score, (h_cnt/mp), (v_cnt/mp)

def _round_blob_density(gray, h, w, mask=None):
    """
    Método 2 para lunares: contornos -> circularidad.
    Devuelve (densidad_normalizada, mediana_diametro_px) de blobs MUY redondos.
    """
    if mask is not None:
        g = cv2.bitwise_and(gray, gray, mask=mask)
        area = max(1, int(np.count_nonzero(mask)))
    else:
        g = gray
        area = h*w

    # Umbral adaptativo + inversión para resaltar puntos oscuros sobre claros y viceversa
    thr = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 35, 5)
    if mask is not None:
        thr = cv2.bitwise_and(thr, thr, mask=mask)

    cnts,_ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = max(25, int(0.00005 * area))
    max_area = int(0.10 * area)
    good = 0
    diam = []
    for c in cnts:
        A = cv2.contourArea(c)
        if A < min_area or A > max_area:
            continue
        P = cv2.arcLength(c, True)
        if P <= 0:
            continue
        circ = 4.0*math.pi*A/(P*P)  # 1 = círculo perfecto
        if circ >= 0.70:
            good += 1
            # diámetro equivalente
            d = 2.0*math.sqrt(A/math.pi)
            diam.append(d)

    mp = max(area / 1_000_000.0, 0.2)
    dens = good / mp
    med_diam = float(np.median(diam)) if diam else 0.0
    return dens, med_diam

def detect_pattern(img_bgr: np.ndarray) -> str:
    """
    'liso'     : muy pocos bordes en la prenda.
    'lunares'  : blobs/círculos suficientes y de tamaño razonable (evita micro-puntos y rayas/pliegues).
    'cuadros'  : líneas H y V significativas.
    'estampado': textura apreciable sin cumplir lo anterior.
    """
    try:
        h, w = img_bgr.shape[:2]
        gray = _prep_gray(img_bgr)
        mask = _fg_mask(img_bgr)  # prenda (quita fondo blanco)

        edges, edge_dens = _edge_density(gray, mask=mask)

        # 1) liso muy claro
        if edge_dens < 0.025:
            return "liso"

        # 2) estadística de puntos / círculos
        dot_dens1, med_diam1 = _blob_stats(gray, h, w, mask=mask)
        dot_dens2, med_diam2 = _round_blob_density(gray, h, w, mask=mask)
        hough_dens = _hough_circle_density(gray, h, w, mask=mask)
        dot_density = max(dot_dens1, dot_dens2, hough_dens)
        med_diam   = max(med_diam1, med_diam2)  # ser conservadores

        # 3) líneas (para cuadros, rayas y para descartar falsos lunares)
        h_score, v_score, h_cnt, v_cnt = _detect_grid_lines(edges, h, w, mask=mask)
        anisotropy = (max(h_cnt, v_cnt) / max(1e-6, min(h_cnt, v_cnt))) if (h_cnt>0 or v_cnt>0) else 1.0

        # Umbrales de lunares estrictos:
        min_polka_diam = max(6.0, 0.02 * min(h, w))                 # 2% del lado menor
        max_polka_diam = max(min_polka_diam + 2.0, 0.12 * min(h, w))

        if (
            (dot_density > 7.0) and
            (min_polka_diam <= med_diam <= max_polka_diam) and
            (edge_dens < 0.10) and
            ((h_cnt + v_cnt) < 12) and
            (anisotropy < 2.0)
        ):
            return "lunares"

        # 4) cuadros (ambas direcciones con líneas)
        if (h_score > 0.003 and v_score > 0.003) and (h_cnt > 10 and v_cnt > 10):
            return "cuadros"

        # 5) estampado genérico si hay textura
        if edge_dens >= 0.045:
            return "estampado"
    except Exception:
        pass
    return "liso"

# -------- Logout modal (reutilizable) --------
def show_logout_confirm(page: ft.Page, on_yes):
    def _close(_e=None):
        try:
            page.close(dlg)
        except Exception:
            dlg.open = False
            page.update()

    dlg = ft.AlertDialog(
        modal=True,
        title=ft.Text("Cerrar sesión", style=TITLE_STYLE),
        content=ft.Text("¿Estás seguro de que deseas cerrar sesión?", style=LABEL_STYLE),
        actions=[
            ft.TextButton("No", on_click=_close),
            ft.FilledButton("Sí", on_click=lambda e: ( _close(), on_yes() )),
        ],
        actions_alignment=ft.MainAxisAlignment.END,
    )
    try:
        page.open(dlg)
    except Exception:
        page.dialog = dlg
        dlg.open = True
        page.update()

# ============ AUTH: Login / Registro con toggle ============
def build_auth_view(page: ft.Page, db, cur, set_route):
    def snack(msg: str): page.open(ft.SnackBar(ft.Text(msg)))

    def clear_login():
        login_email.value = ""; login_pass.value = ""; page.update()

    def clear_register():
        reg_user.value = ""; reg_email.value = ""; reg_pass.value = ""; page.update()

    # ---------- Login ----------
    login_email = ft.TextField(label="Correo", width=360)
    login_pass  = ft.TextField(label="Contraseña", width=360, password=True, can_reveal_password=True)

    def do_login(e):
        try:
            ok, msg, user = login_user(cur, login_email.value.strip(), login_pass.value.strip())
            if not ok:
                snack(msg or "No se pudo iniciar sesión."); return
            clear_login(); clear_register()
            set_route("admin" if user["role"]=="admin" else "user", user)
        except Exception as ex:
            print("[LOGIN ERROR]", ex); traceback.print_exc()
            snack(f"Error inesperado: {ex}")

    login_btn = ft.ElevatedButton("Iniciar sesión", on_click=do_login, style=elevated_style(), width=360)
    login_email.on_submit = do_login
    login_pass.on_submit  = do_login

    login_footer = ft.Row(
        [ft.Text("¿No tienes cuenta?", style=SUBTITLE_STYLE),
         ft.TextButton("Crear cuenta", on_click=lambda e: switch_to_register())],
        alignment=ft.MainAxisAlignment.CENTER
    )

    login_card = ft.Container(
        bgcolor=CARD_BG, border_radius=20, padding=20, width=420,
        content=ft.Column(
            [ft.Text("Acceso", style=TITLE_STYLE),
             login_email, login_pass, login_btn,
             ft.Container(height=6), login_footer],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=12
        )
    )

    # ---------- Registro ----------
    reg_user  = ft.TextField(label="Nombre de usuario", width=360)
    reg_email = ft.TextField(label="Correo", width=360)
    reg_pass  = ft.TextField(label="Contraseña", width=360, password=True, can_reveal_password=True)

    def do_register(e):
        try:
            ok, msg = register_user(db, cur, reg_user.value.strip(), reg_email.value.strip(), reg_pass.value.strip())
            if ok:
                snack(msg or "Usuario creado.")
                clear_register()
                switch_to_login()
            else:
                snack(msg or "No se pudo registrar.")
        except Exception as ex:
            print("[REGISTER ERROR]", ex); traceback.print_exc()
            snack(f"Error inesperado: {ex}")

    reg_btn = ft.FilledButton("Crear cuenta", on_click=do_register, style=elevated_style(), width=360)
    reg_user.on_submit  = do_register
    reg_email.on_submit = do_register
    reg_pass.on_submit  = do_register

    reg_footer = ft.Row(
        [ft.Text("¿Ya tienes cuenta?", style=SUBTITLE_STYLE),
         ft.TextButton("Inicia sesión", on_click=lambda e: switch_to_login())],
        alignment=ft.MainAxisAlignment.CENTER
    )

    reg_card = ft.Container(
        bgcolor=CARD_BG, border_radius=20, padding=20, width=420,
        content=ft.Column(
            [ft.Text("Registro", style=TITLE_STYLE),
             reg_user, reg_email, reg_pass, reg_btn,
             ft.Container(height=6), reg_footer],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=12
        )
    )

    card_holder = ft.Container(content=login_card)

    def switch_to_register():
        clear_login()
        card_holder.content = reg_card; page.update()

    def switch_to_login():
        clear_register()
        card_holder.content = login_card; page.update()

    wrapper = ft.Container(
        expand=True, bgcolor=SURFACE,
        content=ft.Column(
            [ft.Text("OutfitCombiner", style=TITLE_STYLE),
             ft.Container(height=12), card_holder],
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=16
        )
    )
    return wrapper

# ============ ADMIN VIEW (sidebar: Catálogo / Actividad) ============
def build_admin_view(page: ft.Page, db, cur, user, set_route):
    # ---- panel Catálogo ----
    def make_catalog_panel():
        nombre = ft.TextField(label="Nombre prenda", width=240)
        color_hex = ft.TextField(label="Color HEX (#10b981)", width=160)
        es_neutro = ft.Checkbox(label="Es neutro (blanco/negro/gris/beige)")
        recomendado_para = ft.Dropdown(
            label="Recomendado para",
            options=[ft.dropdown.Option(x) for x in
                     ["ninguno","protanopia","deuteranopia","tritanopia","acromatopsia"]],
            value="ninguno", width=200
        )
        descripcion = ft.TextField(label="Descripción", multiline=True, min_lines=2, max_lines=4, width=420)
        status = ft.Text("", style=SUBTITLE_STYLE)

        table = ft.DataTable(
            columns=[
                ft.DataColumn(ft.Text("ID")),
                ft.DataColumn(ft.Text("Nombre")),
                ft.DataColumn(ft.Text("Color")),
                ft.DataColumn(ft.Text("Neutro")),
                ft.DataColumn(ft.Text("Recomendado")),
                ft.DataColumn(ft.Text("Descripción")),
                ft.DataColumn(ft.Text("Acciones")),
            ],
            rows=[], width=1100
        )

        def refresh_table():
            rows = []
            for (iid, nom, hexcol, neutro, reco, desc) in catalogo_list(cur):
                color_box = ft.Container(width=28, height=18, bgcolor=hexcol, border_radius=6)
                del_btn = ft.IconButton(icon=ft.Icons.DELETE_OUTLINE, tooltip="Eliminar",
                                        on_click=lambda e, _id=iid: do_delete(_id))
                rows.append(
                    ft.DataRow(cells=[
                        ft.DataCell(ft.Text(str(iid))),
                        ft.DataCell(ft.Text(nom)),
                        ft.DataCell(ft.Row([color_box, ft.Text(hexcol, style=SUBTITLE_STYLE)], spacing=8)),
                        ft.DataCell(ft.Text("Sí" if neutro else "No")),
                        ft.DataCell(ft.Text(reco)),
                        ft.DataCell(ft.Text(desc or "")),
                        ft.DataCell(del_btn),
                    ])
                )
            table.rows = rows; page.update()

        def do_add(e):
            ok, msg = catalogo_add(
                db, cur,
                nombre.value.strip(),
                (color_hex.value.strip() or "#000000"),
                es_neutro.value,
                recomendado_para.value,
                descripcion.value.strip()
            )
            status.value = msg
            if ok:
                nombre.value=""; color_hex.value=""; es_neutro.value=False
                recomendado_para.value="ninguno"; descripcion.value=""
                refresh_table()
            page.update()

        def do_delete(item_id: int):
            ok, msg = catalogo_delete(db, cur, item_id)
            status.value = msg; refresh_table()

        add_btn = ft.FilledButton("Agregar al catálogo", on_click=do_add, style=elevated_style())
        refresh_table()

        table_scroller = ft.Container(
            content=ft.Column([ft.Row([table], alignment=ft.MainAxisAlignment.CENTER)], height=380, scroll=ft.ScrollMode.AUTO),
            alignment=ft.alignment.center)

        return ft.Container(
            bgcolor=CARD_BG, border_radius=20, padding=16, width=1200,
            content=ft.Column([
                ft.Row([nombre, color_hex, es_neutro, recomendado_para], spacing=12),
                descripcion,
                ft.Row([add_btn, status], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                ft.Container(height=10),
                table_scroller
            ], spacing=10, horizontal_alignment=ft.CrossAxisAlignment.CENTER)
        )

    # ---- panel Actividad ----
    def make_activity_panel():
        import pandas as pd
        rows = []
        total = ok_rate = avg_conf = 0.0

        def _clean_str(val, default="-"):
            if val is None: return default
            try:
                if isinstance(val, float) and np.isnan(val): return default
            except Exception: pass
            s = str(val).strip()
            return default if s=="" or s.lower()=="nan" else s

        def confirm_delete(idx_csv: int):
            def _close(_e=None):
                try: page.close(dlg)
                except Exception: dlg.open=False; page.update()
            def _do(_e=None):
                _close()
                do_delete_prediction(idx_csv)
            dlg = ft.AlertDialog(
                modal=True,
                title=ft.Text("Eliminar predicción", style=TITLE_STYLE),
                content=ft.Text("¿Seguro que deseas eliminar esta predicción?", style=LABEL_STYLE),
                actions=[ft.TextButton("Cancelar", on_click=_close),
                         ft.TextButton("Eliminar", style=ft.ButtonStyle(color=ft.Colors.RED_400), on_click=_do)],
                actions_alignment=ft.MainAxisAlignment.END,
            )
            try: page.open(dlg)
            except Exception:
                page.dialog = dlg; dlg.open=True; page.update()

        def do_delete_prediction(csv_index: int):
            try:
                df_full = pd.read_csv(DATASET_CSV)
                if 0 <= csv_index < len(df_full):
                    df_full = df_full.drop(index=csv_index).reset_index(drop=True)
                    df_full.to_csv(DATASET_CSV, index=False)
                    page.open(ft.SnackBar(ft.Text("Predicción eliminada.")))
                else:
                    page.open(ft.SnackBar(ft.Text("No se encontró la fila a eliminar.")))
            except Exception as ex:
                print("[DELETE PRED] Error:", ex); traceback.print_exc()
                page.open(ft.SnackBar(ft.Text(f"Error al eliminar: {ex}")))
            content_holder.content = make_activity_panel(); page.update()

        if os.path.isfile(DATASET_CSV):
            try:
                df = pd.read_csv(DATASET_CSV)
                if len(df) > 0:
                    df = df.tail(300).iloc[::-1]
                    total = float(len(df))
                    if "pred" in df.columns: ok_rate = float(df["pred"].mean()*100.0)
                    if "pred_proba" in df.columns: avg_conf = float(df["pred_proba"].mean())
                    for idx, r in df.iterrows():
                        ts = int(r.get("timestamp", 0)) if not np.isnan(r.get("timestamp", np.nan)) else 0
                        fecha = time.strftime("%Y-%m-%d %H:%M", time.localtime(ts)) if ts>0 else "-"
                        uname = _clean_str(r.get("user_name"))
                        a = _clean_str(r.get("detA_label"), "A")
                        b = _clean_str(r.get("detB_label"), "B")
                        a_fn = os.path.basename(str(r.get("imgA_path") or "")) or "-"
                        b_fn = os.path.basename(str(r.get("imgB_path") or "")) or "-"
                        pred = int(r.get("pred", 0)) if not np.isnan(r.get("pred", np.nan)) else 0
                        proba = float(r.get("pred_proba", 0.0)) if not np.isnan(r.get("pred_proba", np.nan)) else 0.0
                        dh = int(r.get("hue_pair_note", 0)) if not np.isnan(r.get("hue_pair_note", np.nan)) else 0
                        a_hex = (r.get("a_color_hex") or "").upper() if isinstance(r.get("a_color_hex"), str) else ""
                        b_hex = (r.get("b_color_hex") or "").upper() if isinstance(r.get("b_color_hex"), str) else ""
                        a_name = _clean_str(r.get("a_color_name"))
                        b_name = _clean_str(r.get("b_color_name"))
                        a_pat  = _clean_str(r.get("a_pattern"))
                        b_pat  = _clean_str(r.get("b_pattern"))
                        if (b_pat=="-") and isinstance(r.get("imgB_path"), str) and os.path.isfile(r.get("imgB_path")):
                            try: b_pat = detect_pattern(read_bgr(r.get("imgB_path")))
                            except Exception: pass

                        rows.append((idx, fecha, uname,
                                    f"{a} ({a_fn})", (a_name, a_hex), a_pat,
                                    f"{b} ({b_fn})", (b_name, b_hex), b_pat,
                                    "✅" if pred==1 else "❌",
                                    f"{proba:.2f}", f"{dh}°"))
            except Exception as ex:
                print("[ACTIVIDAD] Error leyendo CSV:", ex); traceback.print_exc()

        stats = ft.Row([
            ft.Text(f"Registros: {int(total)}", style=SUBTITLE_STYLE),
            ft.Text(f"Aciertos: {ok_rate:.1f}%", style=SUBTITLE_STYLE),
            ft.Text(f"Conf. media: {avg_conf:.2f}", style=SUBTITLE_STYLE),
        ], spacing=18)

        data_rows = []
        for row in rows:
            (csv_idx, fecha, uname, A, (a_name, a_hex), a_pat,
             B, (b_name, b_hex), b_pat, res, conf, dh) = row
            a_chip = ft.Container(width=20, height=14, bgcolor=a_hex or "#333333", border_radius=4)
            b_chip = ft.Container(width=20, height=14, bgcolor=b_hex or "#333333", border_radius=4)
            del_btn = ft.TextButton("Eliminar", on_click=lambda e, _idx=csv_idx: confirm_delete(_idx))
            data_rows.append(
                ft.DataRow(cells=[
                    ft.DataCell(ft.Text(fecha)),
                    ft.DataCell(ft.Text(uname)),
                    ft.DataCell(ft.Text(A)),
                    ft.DataCell(ft.Row([a_chip, ft.Text(f"{a_name} ({a_hex or '-'})")], spacing=8)),
                    ft.DataCell(ft.Text(a_pat)),
                    ft.DataCell(ft.Text(B)),
                    ft.DataCell(ft.Row([b_chip, ft.Text(f"{b_name} ({b_hex or '-'})")], spacing=8)),
                    ft.DataCell(ft.Text(b_pat)),
                    ft.DataCell(ft.Text(res)),
                    ft.DataCell(ft.Text(conf)),
                    ft.DataCell(ft.Text(dh)),
                    ft.DataCell(del_btn),
                ])
            )

        columns = [
            ft.DataColumn(ft.Text("Fecha")),
            ft.DataColumn(ft.Text("Usuario")),
            ft.DataColumn(ft.Text("Prenda A")),
            ft.DataColumn(ft.Text("Color A")),
            ft.DataColumn(ft.Text("Patrón A")),
            ft.DataColumn(ft.Text("Prenda B")),
            ft.DataColumn(ft.Text("Color B")),
            ft.DataColumn(ft.Text("Patrón B")),
            ft.DataColumn(ft.Text("Resultado")),
            ft.DataColumn(ft.Text("Conf.")),
            ft.DataColumn(ft.Text("ΔHue°")),
            ft.DataColumn(ft.Text("Acciones")),
        ]

        table = ft.DataTable(columns=columns, rows=data_rows, width=1600)

        if not rows:
            empty = ft.Text("Sin actividad reciente.", style=SUBTITLE_STYLE)
            return ft.Container(bgcolor=CARD_BG, border_radius=20, padding=16, width=1220,
                                content=ft.Column([stats, ft.Divider(), empty], spacing=10,
                                                  horizontal_alignment=ft.CrossAxisAlignment.CENTER))

        # Scroll vertical y horizontal
        table_scroller = ft.Container(
            content=ft.Column(
                controls=[ft.Row([table], alignment=ft.MainAxisAlignment.START, scroll=ft.ScrollMode.AUTO)],
                height=520,
                scroll=ft.ScrollMode.AUTO
            ),
            alignment=ft.alignment.center,
            expand=True
        )

        return ft.Container(bgcolor=CARD_BG, border_radius=20, padding=16, width=1220,
                            content=ft.Column([stats, ft.Divider(), table_scroller], spacing=10,
                                              horizontal_alignment=ft.CrossAxisAlignment.CENTER))

    # ---- header con logout (modal) ----
    def logout_action():
        set_route("auth", None)

    topbar = ft.Row(
        [ft.Text("Vista Admin", style=TITLE_STYLE), ft.Container(expand=True),
         ft.OutlinedButton("Cerrar sesión",
                           on_click=lambda e: show_logout_confirm(page, logout_action),
                           style=outlined_style())],
        alignment=ft.MainAxisAlignment.START
    )

    # ---- Sidebar + contenido ----
    content_holder = ft.Container(content=make_catalog_panel(), expand=True)

    def on_nav_change(e):
        if e.control.selected_index == 0:
            content_holder.content = make_catalog_panel()
        else:
            content_holder.content = make_activity_panel()
        page.update()

    rail = ft.NavigationRail(
        selected_index=0,
        label_type=ft.NavigationRailLabelType.ALL,
        destinations=[
            ft.NavigationRailDestination(icon=ft.Icons.PALETTE_OUTLINED, selected_icon=ft.Icons.PALETTE, label="Catálogo"),
            ft.NavigationRailDestination(icon=ft.Icons.HISTORY, selected_icon=ft.Icons.HISTORY_TOGGLE_OFF, label="Actividad"),
        ],
        on_change=on_nav_change,
        min_width=72,
        min_extended_width=200,
    )

    body = ft.Column([
        topbar,
        ft.Container(height=12),
        ft.Row([rail, ft.Container(width=16), content_holder], expand=True)
    ], expand=True)

    return body

# ============ USER VIEW ============
def build_user_view(page: ft.Page, db, cur, user, set_route):
    ensure_folders(); ensure_csv(DATASET_CSV)

    state = {"pathA": None, "pathB": None, "cropA": None, "cropB": None, "labA": None, "labB": None,
             "model": None, "feature_cols": None}

    try:
        model, feature_cols = load_bundle()
        state["model"], state["feature_cols"] = model, feature_cols
        model_badge = ft.Text("Modelo cargado ✓", size=12, color=ACCENT)
    except Exception as e:
        model_badge = ft.Text(f"Modelo no encontrado: {e}", size=12, color=ft.Colors.RED_300)

    imgA = ft.Image(src="assets/placeholder_a.png", width=520, height=360, fit=ft.ImageFit.CONTAIN, border_radius=16)
    imgB = ft.Image(src="assets/placeholder_b.png", width=520, height=360, fit=ft.ImageFit.CONTAIN, border_radius=16)
    a_label = ft.Text("Prenda A", style=SUBTITLE_STYLE)
    b_label = ft.Text("Prenda B", style=SUBTITLE_STYLE)

    # Color/patrón UI (placeholders)
    a_chip = ft.Container(width=18, height=18, bgcolor="#222222", border_radius=4)
    b_chip = ft.Container(width=18, height=18, bgcolor="#222222", border_radius=4)
    a_color_txt = ft.Text("Color: —", style=SUBTITLE_STYLE)
    b_color_txt = ft.Text("Color: —", style=SUBTITLE_STYLE)
    a_pattern_txt = ft.Text("Patrón: —", style=SUBTITLE_STYLE)
    b_pattern_txt = ft.Text("Patrón: —", style=SUBTITLE_STYLE)

    result_text = ft.Text("Sube dos prendas para evaluar", style=BIG_RESULT)
    reason_text = ft.Text("", style=SUBTITLE_STYLE)
    rec_text    = ft.Text("", style=SUBTITLE_STYLE)

    # --- Modo daltonismo + sugerencias del catálogo ---
    cvd_mode_dd = ft.Dropdown(
        label="Modo daltonismo",
        options=[ft.dropdown.Option(x) for x in ["ninguno","protanopia","deuteranopia","tritanopia","acromatopsia"]],
        value="ninguno",
        width=220
    )
    suggest_holder = ft.Container()

    def snack(msg: str): page.open(ft.SnackBar(ft.Text(msg)))

    def on_pick_a(res: ft.FilePickerResultEvent):
        if not res or not res.files: return
        try:
            p = res.files[0].path
            state["pathA"] = p
            bgr = read_bgr(p)
            imgA.src_base64 = base64_from_bytes(pil_preview_from_bgr(bgr)); imgA.update()
        except Exception as ex:
            snack(f"Error prenda A: {ex}")

    def on_pick_b(res: ft.FilePickerResultEvent):
        if not res or not res.files: return
        try:
            p = res.files[0].path
            state["pathB"] = p
            bgr = read_bgr(p)
            imgB.src_base64 = base64_from_bytes(pil_preview_from_bgr(bgr)); imgB.update()
        except Exception as ex:
            snack(f"Error prenda B: {ex}")

    fp_a = ft.FilePicker(on_result=on_pick_a); fp_b = ft.FilePicker(on_result=on_pick_b)
    page.overlay.append(fp_a); page.overlay.append(fp_b); page.update()

    loadA_btn = ft.ElevatedButton("Cargar prenda A", on_click=lambda e: fp_a.pick_files(allow_multiple=False), style=elevated_style())
    loadB_btn = ft.ElevatedButton("Cargar prenda B", on_click=lambda e: fp_b.pick_files(allow_multiple=False), style=elevated_style())

    def reset_ui():
        for k in ["pathA","pathB","cropA","cropB","labA","labB"]: state[k]=None
        imgA.src="assets/placeholder_a.png"; imgA.src_base64=None; imgA.update()
        imgB.src="assets/placeholder_b.png"; imgB.src_base64=None; imgB.update()
        a_label.value="Prenda A"; b_label.value="Prenda B"; a_label.update(); b_label.update()
        a_chip.bgcolor="#222222"; b_chip.bgcolor="#222222"; a_chip.update(); b_chip.update()
        a_color_txt.value="Color: —"; b_color_txt.value="Color: —"
        a_pattern_txt.value="Patrón: —"; b_pattern_txt.value="Patrón: —"
        a_color_txt.update(); b_color_txt.update(); a_pattern_txt.update(); b_pattern_txt.update()
        result_text.value="Sube dos prendas para evaluar"; reason_text.value=""; rec_text.value=""
        suggest_holder.content=None; suggest_holder.update()
        result_text.update(); reason_text.update(); rec_text.update()

    def remove_last_and_reset():
        import pandas as pd
        try:
            if os.path.isfile(DATASET_CSV):
                df = pd.read_csv(DATASET_CSV)
                if len(df)>0:
                    df.iloc[:-1].to_csv(DATASET_CSV, index=False)
        except Exception as e:
            snack(f"No se pudo eliminar: {e}")
        reset_ui()

    def open_clear_modal(e):
        def _close(_e=None):
            try:
                page.close(dlg)
            except Exception:
                dlg.open = False
                page.update()

        dlg = ft.AlertDialog(
            modal=True,
            title=ft.Text("Limpiar datos", style=TITLE_STYLE),
            content=ft.Text("¿Deseas limpiar la pantalla o también borrar la última predicción guardada?", style=LABEL_STYLE),
            actions=[
                ft.TextButton("Cancelar", on_click=_close),
                ft.TextButton("Solo pantalla", on_click=lambda _e: (reset_ui(), _close())),
                ft.TextButton("Eliminar última predicción y limpiar",
                              style=ft.ButtonStyle(color=ft.Colors.RED_400),
                              on_click=lambda _e: (remove_last_and_reset(), _close())),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )
        try:
            page.open(dlg)
        except Exception:
            page.dialog = dlg
            dlg.open = True
            page.update()

    clear_btn = ft.OutlinedButton("Limpiar datos", on_click=open_clear_modal, style=outlined_style())

    # ---------- Sugerencias desde catálogo ----------
    def score_catalog_color(base_rgb, cand_rgb, es_neutro, reco_para, mode_sel):
        H1,_,_ = hsv_tuple(base_rgb); H2,_,_ = hsv_tuple(cand_rgb)
        dH = abs(H1 - H2); dH = dH if dH <= 180 else 360 - dH
        hue_score = max(0.0, 1.0 - abs(dH - 120.0)/120.0)  # [0..1]
        cr = contrast_ratio(base_rgb, cand_rgb)
        contrast_score = min(cr/3.0, 1.0)
        neutral_bonus = 0.15 if es_neutro else 0.0
        mode_bonus = 0.20 if (mode_sel and reco_para==mode_sel) else (0.08 if reco_para=="ninguno" else 0.0)
        return 0.55*hue_score + 0.35*contrast_score + neutral_bonus + mode_bonus

    def get_catalog_suggestions(base_rgb, mode_sel: str, topk=3):
        items = catalogo_list(cur)  # (id, nombre, color_hex, es_neutro, recomendado_para, descripcion)
        scored = []
        for iid, nombre, hexcol, es_neutro, reco, desc in items:
            cand_rgb = hex_to_rgb(hexcol)
            s = score_catalog_color(base_rgb, cand_rgb, bool(es_neutro), reco, mode_sel or "ninguno")
            scored.append((s, iid, nombre, hexcol, es_neutro, reco, desc, cand_rgb))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:topk]

    def render_suggestions(base_rgb, mode_sel):
        try:
            sugs = get_catalog_suggestions(base_rgb, mode_sel, topk=3)
        except Exception as ex:
            print("[CATALOGO] Error:", ex); sugs=[]
        if not sugs:
            suggest_holder.content = ft.Text("No hay elementos en el catálogo o no se encontraron sugerencias.", style=SUBTITLE_STYLE)
            suggest_holder.update(); return

        rows = []
        for s, iid, nombre, hexcol, es_neutro, reco, desc, cand_rgb in sugs:
            chip = ft.Container(width=20, height=14, bgcolor=hexcol, border_radius=4)
            meta = []
            if es_neutro: meta.append("neutro")
            if reco and reco!="ninguno": meta.append(f"recomendado: {reco}")
            meta_txt = (" • " + " • ".join(meta)) if meta else ""
            rows.append(ft.Row([chip, ft.Text(f"{nombre} {hexcol}{meta_txt}")], spacing=8))

        suggest_holder.content = ft.Container(
            bgcolor=CARD_BG, border_radius=12, padding=12,
            content=ft.Column(
                [ft.Text("Sugerencias del catálogo", style=SUBTITLE_STYLE)]
                + rows,
                spacing=6
            )
        )
        suggest_holder.update()

    # ---------- Wrapper seguro para DINO ----------
    def safe_detect(path: str, default_label: str):
        """Intenta DINO; si falla o devuelve recorte vacío, cae a read_bgr()."""
        try:
            if HAS_DINO:
                crop, _, lab = detect_and_crop(path)
                if crop is None or getattr(crop, "size", 0) == 0:
                    raise ValueError("Detección vacía")
                return crop, (lab or default_label)
        except Exception as ex:
            print("[DINO] Falló, uso fallback:", ex)
        return read_bgr(path), default_label

    # ---------- Predicción ----------
    def run_predict(e):
        result_text.value="Procesando…"; reason_text.value=""; rec_text.value=""; suggest_holder.content=None; suggest_holder.update(); page.update()
        if state["model"] is None or state["feature_cols"] is None:
            result_text.value="Falta el modelo (models/combina_model.joblib)."; page.update(); return
        if not state["pathA"] or not state["pathB"]:
            result_text.value="Carga dos imágenes primero."; page.update(); return

        # Detección robusta
        cropA, labA = safe_detect(state["pathA"], "Prenda A")
        cropB, labB = safe_detect(state["pathB"], "Prenda B")

        state.update({"cropA":cropA,"cropB":cropB,"labA":labA,"labB":labB})
        imgA.src_base64 = base64_from_bytes(pil_preview_from_bgr(cropA))
        imgB.src_base64 = base64_from_bytes(pil_preview_from_bgr(cropB))
        a_label.value = labA or "Prenda A"; b_label.value = labB or "Prenda B"
        imgA.update(); imgB.update(); a_label.update(); b_label.update()

        # features + meta
        feats, meta = pair_features(cropA, cropB, k_colors=4)
        H1,_,_ = hsv_tuple(meta["a_main_rgb"]); H2,_,_ = hsv_tuple(meta["b_main_rgb"])
        dH = hue_distance_deg(H1, H2)

        # nombre/hex
        a_hex = rgb_to_hex(meta["a_main_rgb"])
        b_hex = rgb_to_hex(meta["b_main_rgb"])
        a_name = basic_color_name(meta["a_main_rgb"])
        b_name = basic_color_name(meta["b_main_rgb"])

        # patrón
        a_pat = detect_pattern(cropA)
        b_pat = detect_pattern(cropB)

        # UI
        a_chip.bgcolor = a_hex; b_chip.bgcolor = b_hex
        a_color_txt.value = f"Color: {a_name} ({a_hex})"
        b_color_txt.value = f"Color: {b_name} ({b_hex})"
        a_pattern_txt.value = f"Patrón: {a_pat}"
        b_pattern_txt.value = f"Patrón: {b_pat}"
        a_chip.update(); b_chip.update()
        a_color_txt.update(); b_color_txt.update()
        a_pattern_txt.update(); b_pattern_txt.update()

        # predicción
        x = np.array([[feats[c] for c in state["feature_cols"]]], dtype=np.float32)
        model = state["model"]
        try:
            proba = float(model.predict_proba(x)[0,1])
        except Exception:
            try: proba = float(model.decision_function(x))
            except Exception: proba = float(model.predict(x))
        pred = int(model.predict(x)[0])

        notes, recs = explain_pair(feats, meta)
        result_text.value = ("COMBINAN ✅" if pred==1 else "NO COMBINAN ❌") + f"  |  Confianza: {proba:.2f}"
        reason_text.value = "Por qué: " + notes
        rec_text.value = ("Recomendación: " if pred==1 else "Cómo mejorarlo: ") + (recs or "")
        page.update()

        # Si NO combinan, sugerimos colores del catálogo para reemplazar B (usamos el color A como base)
        if pred == 0:
            render_suggestions(tuple(meta["a_main_rgb"]), cvd_mode_dd.value)

        # guardar fila
        row = {
            "label": np.nan, "hue_pair_note": int(dH), **feats,
            "a_main_r": meta["a_main_rgb"][0], "a_main_g": meta["a_main_rgb"][1], "a_main_b": meta["a_main_rgb"][2],
            "b_main_r": meta["b_main_rgb"][0], "b_main_g": meta["b_main_rgb"][1], "b_main_b": meta["b_main_rgb"][2],
            "imgA_path": state["pathA"], "imgB_path": state["pathB"],
            "pred": pred, "pred_proba": proba, "explanation": (notes + " | " + (recs or "")),
            "timestamp": int(time.time()),
            "mode": "predict+dino" if HAS_DINO else "predict",
            "source": "flet_ui", "detA_label": state["labA"], "detB_label": state["labB"],
            "user_id": (user.get("id") if isinstance(user, dict) else None),
            "user_name": (user.get("username") if isinstance(user, dict) else None),
            "user_email": (user.get("email") if isinstance(user, dict) else None),
            "a_color_hex": a_hex, "b_color_hex": b_hex,
            "a_color_name": a_name, "b_color_name": b_name,
            "a_pattern": a_pat, "b_pattern": b_pat,
            "cvd_mode": cvd_mode_dd.value,
        }
        append_row(row, DATASET_CSV)

    predict_btn = ft.FilledButton("Evaluar combinación", on_click=run_predict, style=elevated_style())

    def logout_action():
        set_route("auth", None)

    header = ft.Row(
        [ft.Text(f"OutfitCombiner — Hola, {user['username']}", style=TITLE_STYLE),
         ft.Container(expand=True),
         cvd_mode_dd,
         ft.OutlinedButton("Cerrar sesión",
                           on_click=lambda e: show_logout_confirm(page, logout_action),
                           style=outlined_style())],
        alignment=ft.MainAxisAlignment.START
    )

    # Tarjetas con color/patrón
    left_card = ft.Container(
        bgcolor=CARD_BG, border_radius=20, padding=16, width=560,
        content=ft.Column(
            [
                a_label,
                imgA,
                ft.Row([a_chip, a_color_txt], spacing=8),
                a_pattern_txt,
                loadA_btn
            ],
            spacing=12, horizontal_alignment=ft.CrossAxisAlignment.CENTER
        )
    )
    right_card = ft.Container(
        bgcolor=CARD_BG, border_radius=20, padding=16, width=560,
        content=ft.Column(
            [
                b_label,
                imgB,
                ft.Row([b_chip, b_color_txt], spacing=8),
                b_pattern_txt,
                loadB_btn
            ],
            spacing=12, horizontal_alignment=ft.CrossAxisAlignment.CENTER
        )
    )

    result_card = ft.Container(
        bgcolor=CARD_BG, border_radius=20, padding=20,
        content=ft.Column([
            ft.Row([ft.Text("Resultado", style=TITLE_STYLE), ft.Container(expand=True), model_badge]),
            result_text, reason_text, rec_text,
            suggest_holder,
            ft.Row([predict_btn, clear_btn], alignment=ft.MainAxisAlignment.END, spacing=8)
        ], spacing=12)
    )

    # Scroll vertical para que siempre puedas llegar al botón
    body = ft.Column(
        [header, ft.Container(height=12),
         ft.Row([left_card, right_card], spacing=16),
         ft.Container(height=12), result_card],
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        scroll=ft.ScrollMode.AUTO,
        expand=True
    )
    return body

# ============ APP (router simple) ============
def main(page: ft.Page):
    page.title = "OutfitCombiner — Flet"
    page.window_min_width = 1000; page.window_width = 1200; page.window_height = 780
    page.padding = 16; page.bgcolor = SURFACE; page.theme_mode = ft.ThemeMode.DARK

    # Conexión DB
    try:
        db, cur = get_connection()
    except Exception as e:
        err = f"No se pudo conectar a MariaDB/MySQL: {e}"
        print(err); traceback.print_exc()
        page.add(ft.Container(padding=20, content=ft.Text(err, color=ft.Colors.RED_300)))
        return

    session = {"user": None, "route": "auth"}  # auth | admin | user

    def set_route(route: str, user_obj):
        session["route"] = route
        session["user"] = user_obj
        render()

    def render():
        page.controls.clear()
        if session["route"] == "auth":
            page.add(build_auth_view(page, db, cur, set_route))
        elif session["route"] == "admin" and session["user"] and session["user"]["role"] == "admin":
            page.add(build_admin_view(page, db, cur, session["user"], set_route))
        else:
            page.add(build_user_view(page, db, cur, session["user"] or {"username":"Invitado"}, set_route))
        page.update()

    render()

if __name__ == "__main__":
    ft.app(target=main)
