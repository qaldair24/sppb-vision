"""Interfaz enfocada exclusivamente en la recolección 5xSTS cámara + IMU L5."""

from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import messagebox

from configuracion_v4 import (
    ALTURA_SILLA_PREDETERMINADA_CM,
    ESTADO_VERSION,
    VERSION_SISTEMA,
)


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def _id_valido(valor: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9_-]{1,32}", valor))


def abrir_resultados() -> None:
    participante = entrada_participante.get().strip()
    carpeta = os.path.join(PROJECT_DIR, "resultados", participante)
    if not _id_valido(participante):
        messagebox.showerror("ID inválido", "Use únicamente letras, números, guion o guion bajo.")
        return
    if not os.path.isdir(carpeta):
        messagebox.showinfo("Sin resultados", "Todavía no hay resultados para este ID.")
        return
    if platform.system() == "Darwin":
        subprocess.Popen(["open", carpeta])
    elif os.name == "nt":
        os.startfile(carpeta)
    else:
        subprocess.Popen(["xdg-open", carpeta])


def ejecutar() -> None:
    participante = entrada_participante.get().strip()
    ensayo = entrada_ensayo.get().strip()
    altura_silla = entrada_silla.get().strip()

    if not _id_valido(participante):
        messagebox.showerror(
            "ID inválido",
            "Ingrese un código anónimo de 1 a 32 caracteres. "
            "Use letras, números, guion o guion bajo; por ejemplo P001.",
        )
        return
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,16}", ensayo):
        messagebox.showerror("Ensayo inválido", "Use un código como T1, T2 o RETEST.")
        return
    try:
        altura = float(altura_silla.replace(",", "."))
    except ValueError:
        messagebox.showerror("Altura inválida", "Ingrese la altura de la silla en centímetros.")
        return
    if not 30.0 <= altura <= 60.0:
        messagebox.showerror("Altura inválida", "La altura debe estar entre 30 y 60 cm.")
        return
    if not usar_imu.get():
        messagebox.showerror(
            "IMU requerido",
            "La recolección definitiva del estudio requiere cámara + IMU lumbar L5. "
            "Active el IMU antes de iniciar.",
        )
        return

    boton_ejecutar.config(state="disabled")
    texto_salida.delete("1.0", tk.END)
    texto_salida.insert(
        tk.END,
        "Iniciando recolección. No cierre esta ventana mientras la cámara esté activa.\n",
    )

    entorno = os.environ.copy()
    entorno["SPPB_PROJECT_DIR"] = PROJECT_DIR
    entorno["SPPB_VOICE"] = "1" if usar_voz.get() else "0"
    entorno["SPPB_IMU"] = "1" if usar_imu.get() else "0"
    entorno["SPPB_TRIAL_ID"] = ensayo
    entorno["SPPB_CHAIR_HEIGHT_CM"] = f"{altura:.1f}"
    entrada = f"{participante}\n2\n3\n"

    def proceso() -> None:
        try:
            resultado = subprocess.run(
                [sys.executable, os.path.join(PROJECT_DIR, "main.py")],
                input=entrada,
                text=True,
                capture_output=True,
                cwd=PROJECT_DIR,
                env=entorno,
            )
            salida = resultado.stdout
            if resultado.stderr:
                salida += "\n\nMENSAJES TÉCNICOS:\n" + resultado.stderr
            ventana.after(0, lambda: texto_salida.insert(tk.END, salida))
            if resultado.returncode == 0:
                ventana.after(
                    0,
                    lambda: messagebox.showinfo(
                        "Ensayo guardado",
                        "La prueba terminó y sus archivos fueron guardados.",
                    ),
                )
            else:
                ventana.after(
                    0,
                    lambda: messagebox.showerror(
                        "Ensayo incompleto",
                        "Revise los mensajes técnicos antes de repetir la prueba.",
                    ),
                )
        except Exception as error:
            ventana.after(0, lambda: messagebox.showerror("Error", str(error)))
        finally:
            ventana.after(0, lambda: boton_ejecutar.config(state="normal"))

    threading.Thread(target=proceso, daemon=True).start()


ventana = tk.Tk()
ventana.title(f"5xSTS-VISION v{VERSION_SISTEMA}")
ventana.geometry("720x610")

tk.Label(
    ventana,
    text="5xSTS-VISION",
    font=("Arial", 22, "bold"),
    fg="#0B3F8A",
).pack(pady=(18, 2))
tk.Label(
    ventana,
    text="Recolección multimodal: cámara monocular + IMU lumbar L5",
    font=("Arial", 11),
).pack(pady=(0, 4))
tk.Label(
    ventana,
    text=f"Versión {VERSION_SISTEMA} · {ESTADO_VERSION}",
    font=("Arial", 9, "bold"),
    fg="#8B1A1A",
).pack(pady=(0, 16))

formulario = tk.Frame(ventana)
formulario.pack(pady=5)

tk.Label(formulario, text="ID anónimo del participante:").grid(row=0, column=0, sticky="e", padx=8, pady=7)
entrada_participante = tk.Entry(formulario, width=28)
entrada_participante.grid(row=0, column=1, padx=8, pady=7)

tk.Label(formulario, text="Código del ensayo:").grid(row=1, column=0, sticky="e", padx=8, pady=7)
entrada_ensayo = tk.Entry(formulario, width=28)
entrada_ensayo.insert(0, "T1")
entrada_ensayo.grid(row=1, column=1, padx=8, pady=7)

tk.Label(formulario, text="Altura de la silla (cm):").grid(row=2, column=0, sticky="e", padx=8, pady=7)
entrada_silla = tk.Entry(formulario, width=28)
entrada_silla.insert(0, f"{ALTURA_SILLA_PREDETERMINADA_CM:.1f}")
entrada_silla.grid(row=2, column=1, padx=8, pady=7)

usar_imu = tk.BooleanVar(value=True)
tk.Checkbutton(
    formulario,
    text="Usar IMU lumbar L5 (requerido para el análisis multimodal)",
    variable=usar_imu,
).grid(row=3, column=0, columnspan=2, pady=7)

usar_voz = tk.BooleanVar(value=True)
tk.Checkbutton(
    formulario,
    text="Instrucciones por voz en español",
    variable=usar_voz,
).grid(row=4, column=0, columnspan=2, pady=4)

tk.Label(
    ventana,
    text=(
        "Antes de iniciar: cámara lateral fija, cuerpo completo visible, "
        "silla estable sin apoyabrazos, pies apoyados y brazos cruzados."
    ),
    wraplength=620,
    justify="center",
    fg="#333333",
).pack(pady=12)

boton_ejecutar = tk.Button(
    ventana,
    text="INICIAR ENSAYO 5xSTS",
    command=ejecutar,
    width=28,
    bg="#0B6E4F",
    fg="white",
    font=("Arial", 12, "bold"),
)
boton_ejecutar.pack(pady=8)

tk.Button(
    ventana,
    text="Abrir resultados del participante",
    command=abrir_resultados,
    width=28,
).pack(pady=4)

texto_salida = tk.Text(ventana, height=12, width=82)
texto_salida.pack(padx=16, pady=14)

ventana.mainloop()
