# db/mysql_conn.py
# -*- coding: utf-8 -*-
import sys
import traceback

# Intento usar primero mariadb (MariaDB Connector/Python)
PARAMSTYLE = "qmark"  # placeholders "?"
try:
    import mariadb as dbapi
except Exception:
    # Fallback a mysql-connector si no está mariadb
    import mysql.connector as dbapi
    PARAMSTYLE = "format"  # placeholders "%s"

DB_HOST = "localhost"
DB_PORT = 3306
DB_USER = "root"
DB_PASSWORD = ""      
DB_NAME = "outfitcombiner"

def _q(sql: str) -> str:
    """Convierte placeholders si estamos en mysql-connector."""
    if PARAMSTYLE == "format":
        return sql.replace("?", "%s")
    return sql

def get_raw_server_connection():
    if PARAMSTYLE == "qmark":
        return dbapi.connect(host=DB_HOST, user=DB_USER, password=DB_PASSWORD, port=DB_PORT)
    else:
        return dbapi.connect(host=DB_HOST, user=DB_USER, password=DB_PASSWORD, port=DB_PORT)

def get_connection():
    """
    Devuelve (db, cursor). Crea BD y tablas si no existen.
    Hace seed del admin (admin@local / admin123).
    """
    # 1) Crear BD si no existe (conexión al servidor)
    try:
        srv = get_raw_server_connection()
        cur = srv.cursor()
        cur.execute("CREATE DATABASE IF NOT EXISTS outfitcombiner")
        srv.commit() if hasattr(srv, "commit") else None
        cur.close()
        srv.close()
    except Exception as e:
        print("[DB] No pude crear la BD:", e)
        traceback.print_exc()
        raise

    # 2) Conectar a la BD
    if PARAMSTYLE == "qmark":
        db = dbapi.connect(host=DB_HOST, user=DB_USER, password=DB_PASSWORD, port=DB_PORT, database=DB_NAME)
    else:
        db = dbapi.connect(host=DB_HOST, user=DB_USER, password=DB_PASSWORD, port=DB_PORT, database=DB_NAME)

    cursor = db.cursor()

    # 3) Crear tablas
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS usuarios (
          id INT AUTO_INCREMENT PRIMARY KEY,
          username VARCHAR(50) NOT NULL,
          email VARCHAR(120) NOT NULL UNIQUE,
          password VARCHAR(128) NOT NULL,
          role ENUM('admin','user') NOT NULL DEFAULT 'user',
          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS catalogo_items (
          id INT AUTO_INCREMENT PRIMARY KEY,
          nombre VARCHAR(120) NOT NULL,
          color_hex CHAR(7) NOT NULL,
          es_neutro TINYINT(1) NOT NULL DEFAULT 0,
          recomendado_para ENUM('ninguno','protanopia','deuteranopia','tritanopia','acromatopsia')
            NOT NULL DEFAULT 'ninguno',
          descripcion TEXT,
          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    db.commit()

    # 4) Seed admin
    try:
        cursor.execute(_q("SELECT id FROM usuarios WHERE email=? LIMIT 1"), ("admin@local",))
        row = cursor.fetchone()
        if not row:
            cursor.execute(
                _q("INSERT INTO usuarios (username, email, password, role) VALUES (?, ?, ?, ?)"),
                ("admin", "admin@local", "admin123", "admin")
            )
            db.commit()
            print("[DB] Admin seed creado: admin@local / admin123")
        else:
            print("[DB] Admin ya existe (admin@local).")
    except Exception as e:
        print("[DB] Error creando seed admin:", e)
        traceback.print_exc()

    return db, cursor

# ---------- Helpers de Auth ----------

def register_user(db, cur, username: str, email: str, password: str):
    if not username or not email or not password:
        return False, "Completa usuario, correo y contraseña."
    try:
        cur.execute(_q("INSERT INTO usuarios (username, email, password, role) VALUES (?, ?, ?, ?)"),
                    (username, email, password, "user"))
        db.commit()
        return True, "Usuario creado. Ahora inicia sesión."
    except Exception as err:
        return False, f"Error al registrar: {err}"

def login_user(cur, email: str, password: str):
    try:
        cur.execute(_q("SELECT id, username, email, role FROM usuarios WHERE email=? AND password=? LIMIT 1"),
                    (email, password))
        row = cur.fetchone()
        if row:
            uid, uname, mail, role = row[0], row[1], row[2], row[3]
            return True, "", {"id": uid, "username": uname, "email": mail, "role": role}
        return False, "Correo o contraseña incorrectos.", None
    except Exception as err:
        return False, f"Error en login: {err}", None

# ---------- Helpers Catálogo ----------

def catalogo_list(cur):
    cur.execute("SELECT id, nombre, color_hex, es_neutro, recomendado_para, descripcion FROM catalogo_items ORDER BY id DESC")
    return cur.fetchall()

def catalogo_add(db, cur, nombre, color_hex, es_neutro, recomendado_para, descripcion):
    try:
        cur.execute(
            _q("INSERT INTO catalogo_items (nombre, color_hex, es_neutro, recomendado_para, descripcion) VALUES (?, ?, ?, ?, ?)"),
            (nombre, color_hex, int(bool(es_neutro)), recomendado_para, descripcion)
        )
        db.commit()
        return True, "Elemento agregado."
    except Exception as e:
        return False, f"Error al agregar: {e}"

def catalogo_delete(db, cur, item_id: int):
    try:
        cur.execute(_q("DELETE FROM catalogo_items WHERE id=?"), (item_id,))
        db.commit()
        return True, "Eliminado."
    except Exception as e:
        return False, f"Error al eliminar: {e}"
