# ui/admin.py
import flet as ft
from db.mysql_conn import catalogo_add, catalogo_list, catalogo_delete
from ui.theme import CARD_BG, MUTED


def make_admin_view(page: ft.Page, user: dict, db, cursor, on_go_model, on_logout) -> ft.Control:
    """
    Vista Admin (diseño anterior) + modal de confirmación al cerrar sesión.
    on_go_model: callback para ir a la vista de usuario
    on_logout:   callback para volver al login (cerrar sesión)
    """

    # ---------- Modal cerrar sesión ----------
    def ask_logout(_e=None):
        dlg = ft.AlertDialog(
            modal=True,
            title=ft.Text("Cerrar sesión"),
            content=ft.Text("¿Estás seguro de que deseas cerrar sesión?"),
            actions=[
                ft.TextButton("No", on_click=lambda __e: close()),
                ft.FilledButton("Sí", on_click=lambda __e: (close(), on_logout())),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )

        def close():
            dlg.open = False
            page.update()

        page.dialog = dlg
        dlg.open = True
        page.update()

    # ---------- Formulario de alta al catálogo ----------
    nombre_in = ft.TextField(label="Nombre del item", expand=True)
    color_in = ft.TextField(label="Color HEX (ej. #AABBCC)", width=180)
    tipo_dd = ft.Dropdown(
        label="Tipo de daltonismo",
        options=[
            ft.dropdown.Option("ninguno"),
            ft.dropdown.Option("protanopia"),
            ft.dropdown.Option("deuteranopia"),
            ft.dropdown.Option("tritanopia"),
            ft.dropdown.Option("acromatopsia"),
        ],
        value="ninguno",
        width=200,
    )
    neutro_sw = ft.Switch(label="¿Es color neutro?", value=False)
    desc_in = ft.TextField(label="Descripción", multiline=True, min_lines=2, max_lines=3, expand=True)

    table = ft.DataTable(
        columns=[
            ft.DataColumn(ft.Text("ID")),
            ft.DataColumn(ft.Text("Nombre")),
            ft.DataColumn(ft.Text("Color")),
            ft.DataColumn(ft.Text("Daltonismo")),
            ft.DataColumn(ft.Text("Neutro")),
            ft.DataColumn(ft.Text("Descripción")),
            ft.DataColumn(ft.Text("Acciones")),
        ],
        rows=[],
    )

    def refresh_table():
        rows = catalogo_list(cursor)
        table.rows.clear()
        for r in rows:
            # db.mysql_conn.catalago_list devuelve:
            # (id, nombre, color_hex, es_neutro, recomendado_para, descripcion)
            rid, nombre, color_hex, es_neutro, recomendado, descripcion = r

            def do_delete(_e, _rid=rid):
                # confirmación de borrado
                dlg = ft.AlertDialog(
                    modal=True,
                    title=ft.Text("Eliminar"),
                    content=ft.Text(f"¿Eliminar el item #{_rid}?"),
                    actions=[
                        ft.TextButton("Cancelar", on_click=lambda __e: close()),
                        ft.TextButton(
                            "Eliminar",
                            style=ft.ButtonStyle(color=ft.Colors.RED_400),
                            on_click=lambda __e: (exec_delete(_rid), close()),
                        ),
                    ],
                    actions_alignment=ft.MainAxisAlignment.END,
                )

                def close():
                    dlg.open = False
                    page.update()

                page.dialog = dlg
                dlg.open = True
                page.update()

            def exec_delete(_id):
                ok, msg = catalogo_delete(db, cursor, _id)
                page.open(ft.SnackBar(ft.Text(msg)))
                if ok:
                    refresh_table()

            color_chip = ft.Container(width=24, height=16, bgcolor=color_hex, border_radius=6)
            table.rows.append(
                ft.DataRow(
                    cells=[
                        ft.DataCell(ft.Text(str(rid))),
                        ft.DataCell(ft.Text(nombre)),
                        ft.DataCell(ft.Row([color_chip, ft.Text(color_hex)], spacing=8)),
                        ft.DataCell(ft.Text(recomendado)),
                        ft.DataCell(ft.Text("Sí" if es_neutro else "No")),
                        ft.DataCell(ft.Text(descripcion or "-")),
                        ft.DataCell(ft.IconButton(icon=ft.Icons.DELETE_OUTLINE, on_click=do_delete)),
                    ]
                )
            )
        page.update()

    def add_item(_e):
        ok, msg = catalogo_add(
            db,
            cursor,
            nombre_in.value.strip(),
            color_in.value.strip(),
            neutro_sw.value,
            tipo_dd.value,
            desc_in.value.strip(),
        )
        page.open(ft.SnackBar(ft.Text(msg)))
        if ok:
            nombre_in.value = ""
            color_in.value = ""
            neutro_sw.value = False
            tipo_dd.value = "ninguno"
            desc_in.value = ""
            page.update()
            refresh_table()

    add_btn = ft.FilledButton("Agregar al catálogo", on_click=add_item)
    go_model_btn = ft.OutlinedButton("Ir al modelo (usuario)", on_click=lambda e: on_go_model())
    logout_btn = ft.OutlinedButton("Cerrar sesión", on_click=ask_logout)

    header = ft.Row(
        [
            ft.Text("Panel de administración", size=20, weight=ft.FontWeight.W_700),
            ft.Container(expand=True),
            ft.Text(f"{user['username']} • {user['email']}", size=12, color=MUTED),
            go_model_btn,
            logout_btn,
        ],
        alignment=ft.MainAxisAlignment.START,
    )

    form_card = ft.Container(
        bgcolor=CARD_BG,
        border_radius=16,
        padding=16,
        content=ft.Column(
            [
                ft.Text("Catálogo accesible (daltonismo)", size=16, weight=ft.FontWeight.W_700),
                ft.Row([nombre_in, color_in, tipo_dd], wrap=True, spacing=10),
                ft.Row([neutro_sw, desc_in], wrap=True, spacing=10),
                add_btn,
                ft.Text(
                    "Sugerencia: prioriza neutros y buen contraste para protanopia/deuteranopia.",
                    size=12,
                    color=MUTED,
                ),
            ],
            spacing=10,
        ),
    )

    table_card = ft.Container(
        bgcolor=CARD_BG,
        border_radius=16,
        padding=16,
        content=ft.Column([ft.Text("Items en catálogo", size=16, weight=ft.FontWeight.W_700), table], spacing=10),
    )

    body = ft.Column([header, form_card, table_card], spacing=16, expand=True)
    refresh_table()
    return body
