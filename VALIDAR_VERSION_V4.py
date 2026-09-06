"""Comprobación de instalación y reglas congeladas de V4.1 para Mac M1."""

from __future__ import annotations

import importlib
import platform
import sys
from importlib.metadata import version

import pandas as pd

from configuracion_v4 import VERSION_SISTEMA
from metricas_sts import (
    construir_repeticiones,
    fusionar_eventos,
    subpuntaje_silla_sppb,
)


def main() -> None:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(f"Se requiere Python 3.12; se encontró {platform.python_version()}.")
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError(
            f"Se requiere macOS ARM64; se encontró {platform.system()} {platform.machine()}."
        )

    paquetes = {
        "mediapipe": ("mediapipe", "0.10.21"),
        "numpy": ("numpy", "1.26.4"),
        "opencv-contrib-python": ("cv2", "4.11.0.86"),
        "pandas": ("pandas", "2.2.3"),
        "matplotlib": ("matplotlib", "3.10.1"),
        "scipy": ("scipy", "1.15.2"),
        "reportlab": ("reportlab", "4.3.1"),
        "Pillow": ("PIL", "11.1.0"),
    }
    for paquete, (modulo, esperada) in paquetes.items():
        importlib.import_module(modulo)
        encontrada = version(paquete)
        if encontrada != esperada:
            raise RuntimeError(
                f"Versión incorrecta de {paquete}: {encontrada}; se requiere {esperada}."
            )

    eventos = []
    pares = [
        ("GO", 0.0),
        ("MO1", 0.5), ("R1", 1.4), ("D1", 1.6), ("S1", 2.3),
        ("MO2", 2.5), ("R2", 3.4), ("D2", 3.6), ("S2", 4.3),
        ("MO3", 4.5), ("R3", 5.4), ("D3", 5.6), ("S3", 6.3),
        ("MO4", 6.5), ("R4", 7.4), ("D4", 7.6), ("S4", 8.3),
        ("MO5", 8.5), ("R5", 9.4),
    ]
    for evento, tiempo in pares:
        eventos.append({"evento": evento, "tiempo_desde_go_s": tiempo})

    datos = pd.DataFrame(
        {
            "tiempo_s": [indice / 10 for indice in range(100)],
            "angulo_rodilla": [100 + indice % 70 for indice in range(100)],
            "angulo_tronco": [80 + indice % 40 for indice in range(100)],
        }
    )
    repeticiones = construir_repeticiones(eventos, datos)
    assert len(repeticiones) == 5
    assert bool(repeticiones["repeticion_valida"].all())
    assert subpuntaje_silla_sppb(9.4, 5, True) == 4
    assert subpuntaje_silla_sppb(9.4, 4, False) is None
    assert len(fusionar_eventos(eventos, pd.DataFrame())) == 19

    print(f"5xSTS-VISION V{VERSION_SISTEMA}: VALIDACIÓN TÉCNICA SUPERADA")
    print("Dependencias principales: OK")
    print("Eventos GO/MO/R/D/S: OK")
    print("Métricas por repetición: OK")
    print("Subpuntaje de silla 0–4: OK")
    print("SPPB total: no calculado (correcto)")


if __name__ == "__main__":
    main()
