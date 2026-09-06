"""Reprocesa la última prueba IMU de un paciente sin repetir la captura."""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import pandas as pd

from analisis_imu_sts import analizar_imu_sts
from configuracion_v4 import FRECUENCIA_IMU_HZ, TOLERANCIA_CONCORDANCIA_S
from metricas_sts import fusionar_eventos


def archivo_mas_reciente(patron: str) -> str:
    candidatos = glob.glob(patron)
    if not candidatos:
        raise FileNotFoundError(f"No se encontró: {patron}")
    return max(candidatos, key=os.path.getmtime)


def main() -> None:
    proyecto = Path(__file__).resolve().parent
    paciente = (
        sys.argv[1].strip()
        if len(sys.argv) > 1
        else input("ID del paciente: ").strip()
    )
    carpeta = proyecto / "resultados" / paciente
    ruta_imu = archivo_mas_reciente(
        str(carpeta / "imu_l5_sit_to_stand*.csv")
    )
    ruta_eventos = archivo_mas_reciente(
        str(carpeta / "eventos_imu_l5_sit_to_stand*.csv")
    )
    ruta_eventos_camara = archivo_mas_reciente(
        str(carpeta / "eventos_camara_sit_to_stand*.csv")
    )

    muestras = pd.read_csv(ruta_imu, encoding="utf-8-sig")
    eventos_imu = pd.read_csv(ruta_eventos, encoding="utf-8-sig")
    eventos_camara = pd.read_csv(ruta_eventos_camara, encoding="utf-8-sig")
    referencias = pd.DataFrame(
        {
            "evento": "camera_" + eventos_camara["evento"].astype(str),
            "ms_prueba": pd.to_numeric(
                eventos_camara["tiempo_desde_go_s"],
                errors="coerce",
            )
            * 1000.0,
        }
    )
    marcas_go = eventos_imu[eventos_imu["evento"].astype(str) == "camera_GO"]
    origen_tiempo_s = (
        float(marcas_go.iloc[0]["ms_prueba"]) / 1000.0
        if not marcas_go.empty
        else 0.0
    )
    resultado = analizar_imu_sts(
        muestras,
        referencias,
        frecuencia_hz=FRECUENCIA_IMU_HZ,
        tolerancia_s=TOLERANCIA_CONCORDANCIA_S,
        origen_tiempo_s=origen_tiempo_s,
    )

    sufijo = Path(ruta_imu).stem.replace("imu_l5_", "")
    resultado["detecciones"].to_csv(
        carpeta / f"detecciones_imu_l5_{sufijo}.csv",
        index=False,
    )
    resultado["comparacion"].to_csv(
        carpeta / f"concordancia_camara_imu_{sufijo}.csv",
        index=False,
    )
    resultado["resumen"].to_csv(
        carpeta / f"resumen_reprocesamiento_imu_{sufijo}.csv",
        index=False,
    )
    fusionar_eventos(
        eventos_camara,
        resultado["detecciones"],
        tolerancia_s=TOLERANCIA_CONCORDANCIA_S,
    ).to_csv(
        carpeta / f"eventos_fusion_{sufijo}.csv",
        index=False,
    )

    resumen = resultado["resumen"].iloc[0]
    print("\nCONCORDANCIA TÉCNICA CÁMARA–IMU")
    print("--------------------------------")
    print(
        "Levantamientos detectados: "
        f"{int(resumen['levantamientos_imu_detectados'])}/5"
    )
    print(f"Precision: {resumen['precision']:.3f}")
    print(f"Recall: {resumen['recall']:.3f}")
    print(f"F1-score: {resumen['f1_score']:.3f}")
    print(f"MAE temporal: {resumen['mae_eventos_s']:.3f} s")
    print(f"Pérdida de muestras: {resumen['perdida_muestras_pct']:.2f}%")
    print(
        "Nota: estas cifras comparan los dos sensores; no sustituyen "
        "una referencia manual independiente."
    )
    print(f"\nArchivos guardados en: {carpeta}")


if __name__ == "__main__":
    main()
