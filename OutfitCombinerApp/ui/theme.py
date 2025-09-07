# ui/theme.py
import flet as ft

PRIMARY = "#111827"   # gris casi negro
ACCENT  = "#10b981"   # verde menta
CARD_BG = "#1f2937"   # gris carbón
MUTED   = "#9ca3af"   # gris texto secundario
SURFACE = "#0b1220"   # fondo general

TITLE_STYLE    = ft.TextStyle(size=22, weight=ft.FontWeight.W_700, color=ft.Colors.WHITE)
SUBTITLE_STYLE = ft.TextStyle(size=13, color=MUTED)
LABEL_STYLE    = ft.TextStyle(size=12, color=MUTED)
BIG_RESULT     = ft.TextStyle(size=22, weight=ft.FontWeight.W_800, color=ft.Colors.WHITE)

def elevated_style():
    return ft.ButtonStyle(bgcolor=ACCENT, color="white")

def outlined_style():
    return ft.ButtonStyle(color=ACCENT)
