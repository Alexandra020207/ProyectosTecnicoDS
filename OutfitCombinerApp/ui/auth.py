# ui/auth.py
import flet as ft
from db.mysql_conn import login_user, register_user
from ui.theme import CARD_BG, SURFACE, ACCENT, MUTED


def make_auth_view(page: ft.Page, on_success, db, cursor) -> ft.Control:
    page.bgcolor = SURFACE

    # -------- Login (correo + contraseña) --------
    email_in = ft.TextField(
        label="Correo",
        autofocus=True,
        prefix_icon=ft.Icons.MAIL_OUTLINE,
        expand=True,
    )
    pass_in = ft.TextField(
        label="Contraseña",
        password=True,
        can_reveal_password=True,
        prefix_icon=ft.Icons.LOCK_OUTLINE,
        expand=True,
    )
    login_msg = ft.Text("", size=12, color=ft.Colors.RED_300)

    def do_login(e):
        ok, err, user = login_user(cursor, email_in.value.strip(), pass_in.value.strip())
        if ok:
            on_success(user)
        else:
            login_msg.value = err
            page.update()

    login_btn = ft.FilledButton("Entrar", on_click=do_login)

    login_card = ft.Container(
        bgcolor=CARD_BG,
        padding=20,
        border_radius=16,
        width=440,
        content=ft.Column(
            [
                ft.Text("Iniciar sesión", size=18, weight=ft.FontWeight.W_700),
                ft.Row([email_in], expand=True),
                ft.Row([pass_in], expand=True),
                login_btn,
                login_msg,
                ft.Divider(),
                ft.Text("Admin por defecto: admin@local / admin123", size=12, color=MUTED),
            ],
            spacing=12,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
        ),
    )

    # -------- Registro (usuario + correo + contraseña) --------
    reg_user = ft.TextField(label="Nombre de usuario", prefix_icon=ft.Icons.PERSON_OUTLINE, expand=True)
    reg_email = ft.TextField(label="Correo", prefix_icon=ft.Icons.MAIL_OUTLINE, expand=True)
    reg_pass = ft.TextField(label="Contraseña", password=True, can_reveal_password=True, prefix_icon=ft.Icons.LOCK_OUTLINE, expand=True)
    reg_msg = ft.Text("", size=12, color=ft.Colors.RED_300)

    def do_register(e):
        ok, msg = register_user(db, cursor, reg_user.value.strip(), reg_email.value.strip(), reg_pass.value.strip())
        if ok:
            page.open(ft.SnackBar(ft.Text(msg)))
            tabs.selected_index = 0
            page.update()
        else:
            reg_msg.value = msg
            page.update()

    reg_btn = ft.OutlinedButton("Crear cuenta", on_click=do_register)

    register_card = ft.Container(
        bgcolor=CARD_BG,
        padding=20,
        border_radius=16,
        width=440,
        content=ft.Column(
            [
                ft.Text("Crear cuenta", size=18, weight=ft.FontWeight.W_700),
                ft.Row([reg_user], expand=True),
                ft.Row([reg_email], expand=True),
                ft.Row([reg_pass], expand=True),
                reg_btn,
                reg_msg,
            ],
            spacing=12,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
        ),
    )

    # -------- Tabs en centro --------
    tabs = ft.Tabs(
        selected_index=0,
        tabs=[
            ft.Tab(text="Entrar", content=login_card),
            ft.Tab(text="Registrarse", content=register_card),
        ],
        expand=0,
    )

    wrapper = ft.Container(
        content=ft.Column(
            [
                ft.Text("OutfitCombiner", size=22, weight=ft.FontWeight.W_800, color=ft.Colors.WHITE),
                ft.Text("Accede para continuar", size=12, color=MUTED),
                ft.Container(height=10),
                tabs,
            ],
            spacing=12,
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        ),
        padding=20,
    )

    # Centrado total
    return ft.Container(
        expand=True,
        content=ft.Row(
            [wrapper],
            alignment=ft.MainAxisAlignment.CENTER,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        ),
    )
