"""Funciones puras para eventos y métricas del Five Times Sit-to-Stand."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd


ORDEN_EVENTOS = (
    "GO",
    "MO1", "R1", "D1", "S1",
    "MO2", "R2", "D2", "S2",
    "MO3", "R3", "D3", "S3",
    "MO4", "R4", "D4", "S4",
    "MO5", "R5",
)


def subpuntaje_silla_sppb(
    tiempo_clinico_s: float | None,
    repeticiones_validas: int,
    prueba_completada: bool,
) -> int | None:
    """Calcula únicamente el subpuntaje de silla (0–4), nunca el SPPB total."""
    if not prueba_completada or repeticiones_validas < 5:
        return None
    if tiempo_clinico_s is None or not math.isfinite(float(tiempo_clinico_s)):
        return None
    tiempo = float(tiempo_clinico_s)
    if tiempo > 60.0:
        return 0
    if tiempo <= 11.19:
        return 4
    if tiempo <= 13.69:
        return 3
    if tiempo <= 16.69:
        return 2
    return 1


def _mapa_eventos(eventos: Iterable[dict] | pd.DataFrame) -> dict[str, float]:
    df = eventos.copy() if isinstance(eventos, pd.DataFrame) else pd.DataFrame(eventos)
    if df.empty or "evento" not in df.columns:
        return {}
    columna_tiempo = (
        "tiempo_desde_go_s"
        if "tiempo_desde_go_s" in df.columns
        else "tiempo_prueba_s"
    )
    if columna_tiempo not in df.columns:
        return {}
    tiempos = pd.to_numeric(df[columna_tiempo], errors="coerce")
    salida: dict[str, float] = {}
    for evento, tiempo in zip(df["evento"].astype(str), tiempos):
        if math.isfinite(float(tiempo)):
            salida[evento.replace("camera_", "").replace("imu_", "")] = float(tiempo)
    return salida


def construir_repeticiones(
    eventos: Iterable[dict] | pd.DataFrame,
    datos_camara: pd.DataFrame,
) -> pd.DataFrame:
    """Construye las fases usando MO/R/D/S, sin confundir pausas con movimiento."""
    mapa = _mapa_eventos(eventos)
    filas: list[dict] = []

    def angulo_en(instante: float | None, columna: str) -> float:
        if (
            instante is None
            or datos_camara.empty
            or "tiempo_s" not in datos_camara.columns
            or columna not in datos_camara.columns
        ):
            return math.nan
        tiempos = pd.to_numeric(datos_camara["tiempo_s"], errors="coerce").to_numpy()
        valores = pd.to_numeric(datos_camara[columna], errors="coerce").to_numpy()
        validos = np.isfinite(tiempos) & np.isfinite(valores)
        if not np.any(validos):
            return math.nan
        indices = np.where(validos)[0]
        indice = indices[int(np.argmin(np.abs(tiempos[validos] - instante)))]
        return float(valores[indice])

    for numero in range(1, 6):
        mo = mapa.get(f"MO{numero}")
        r = mapa.get(f"R{numero}")
        d = mapa.get(f"D{numero}") if numero < 5 else None
        s = mapa.get(f"S{numero}") if numero < 5 else None
        siguiente_mo = mapa.get(f"MO{numero + 1}") if numero < 5 else None

        subida = r - mo if mo is not None and r is not None else math.nan
        descenso = s - d if d is not None and s is not None else math.nan
        pausa_pie = d - r if r is not None and d is not None else math.nan
        pausa_sentado = (
            siguiente_mo - s
            if s is not None and siguiente_mo is not None
            else math.nan
        )
        ciclo = (
            siguiente_mo - mo
            if mo is not None and siguiente_mo is not None
            else math.nan
        )
        evento_completo = mo is not None and r is not None and (
            numero == 5 or (d is not None and s is not None)
        )

        filas.append(
            {
                "repeticion": numero,
                "MO_s": mo,
                "R_s": r,
                "D_s": d,
                "S_s": s,
                "duracion_subida_s": subida,
                "pausa_de_pie_s": pausa_pie,
                "duracion_descenso_s": descenso,
                "pausa_sentado_s": pausa_sentado,
                "duracion_ciclo_MO_MO_s": ciclo,
                # Alias conservados para los reportes y gráficas de V3.
                "inicio_sentado_s": mo,
                "de_pie_s": r,
                "fin_sentado_s": s,
                "duracion_s": (
                    ciclo
                    if numero < 5
                    else subida
                ),
                "duracion_ciclo_s": ciclo,
                "angulo_rodilla_MO_grados": angulo_en(mo, "angulo_rodilla"),
                "angulo_rodilla_R_grados": angulo_en(r, "angulo_rodilla"),
                "angulo_rodilla_D_grados": angulo_en(d, "angulo_rodilla"),
                "angulo_rodilla_S_grados": angulo_en(s, "angulo_rodilla"),
                "angulo_tronco_MO_grados": angulo_en(mo, "angulo_tronco"),
                "angulo_tronco_R_grados": angulo_en(r, "angulo_tronco"),
                "angulo_tronco_D_grados": angulo_en(d, "angulo_tronco"),
                "angulo_tronco_S_grados": angulo_en(s, "angulo_tronco"),
                "angulo_sentado_inicio": angulo_en(
                    mo,
                    "angulo_rodilla",
                ),
                "angulo_de_pie": angulo_en(r, "angulo_rodilla"),
                "angulo_sentado_fin": angulo_en(s, "angulo_rodilla"),
                "criterio_validacion": (
                    "MO-R-D-S completos"
                    if numero < 5
                    else "MO5-R5: quinta extensión completa"
                ),
                "repeticion_valida": bool(evento_completo),
            }
        )
    return pd.DataFrame(filas)


def fusionar_eventos(
    eventos_camara: Iterable[dict] | pd.DataFrame,
    detecciones_imu: Iterable[dict] | pd.DataFrame,
    tolerancia_s: float = 0.50,
) -> pd.DataFrame:
    """Fusión determinista: promedio si hay acuerdo; conserva dato único; rechaza conflicto."""
    camara = _mapa_eventos(eventos_camara)
    imu_df = (
        detecciones_imu.copy()
        if isinstance(detecciones_imu, pd.DataFrame)
        else pd.DataFrame(detecciones_imu)
    )
    imu = _mapa_eventos(imu_df)

    # Los comienzos de transición del IMU se almacenan como columnas de la
    # detección de llegada R/S.
    if not imu_df.empty:
        for _, fila in imu_df.iterrows():
            evento_fin = str(fila.get("evento", "")).replace("imu_", "")
            inicio = pd.to_numeric(
                pd.Series([fila.get("tiempo_inicio_transicion_s")]),
                errors="coerce",
            ).iloc[0]
            if math.isfinite(float(inicio)):
                if evento_fin.startswith("R"):
                    imu[f"MO{evento_fin[1:]}"] = float(inicio)
                elif evento_fin.startswith("S"):
                    imu[f"D{evento_fin[1:]}"] = float(inicio)

    filas: list[dict] = []
    for evento in ORDEN_EVENTOS:
        tiempo_camara = camara.get(evento)
        tiempo_imu = imu.get(evento)
        if evento == "GO":
            diferencia = math.nan
            tiempo_fusion = tiempo_camara
            estado = "referencia_software"
        elif tiempo_camara is not None and tiempo_imu is not None:
            diferencia = tiempo_imu - tiempo_camara
            if abs(diferencia) <= tolerancia_s:
                tiempo_fusion = (tiempo_camara + tiempo_imu) / 2.0
                estado = "acuerdo_camara_imu"
            else:
                tiempo_fusion = math.nan
                estado = "conflicto_rechazado"
        elif tiempo_camara is not None:
            diferencia = math.nan
            tiempo_fusion = tiempo_camara
            estado = "solo_camara"
        elif tiempo_imu is not None:
            diferencia = math.nan
            tiempo_fusion = tiempo_imu
            estado = "solo_imu"
        else:
            diferencia = math.nan
            tiempo_fusion = math.nan
            estado = "no_detectado"

        filas.append(
            {
                "evento": evento,
                "tiempo_camara_s": tiempo_camara,
                "tiempo_imu_s": tiempo_imu,
                "diferencia_imu_menos_camara_s": diferencia,
                "tiempo_fusion_s": tiempo_fusion,
                "estado_fusion": estado,
                "tolerancia_s": tolerancia_s,
            }
        )
    return pd.DataFrame(filas)


def calidad_camara(
    datos_camara: pd.DataFrame,
    timeline: pd.DataFrame,
) -> dict:
    """Resume frecuencia real, huecos y disponibilidad de pose durante la prueba."""
    if datos_camara.empty:
        return {
            "frames_prueba": 0,
            "fps_camara_real": math.nan,
            "hueco_camara_max_ms": math.nan,
            "frames_pose_detectada_pct": 0.0,
        }

    tiempos = pd.to_numeric(datos_camara["tiempo_s"], errors="coerce").dropna()
    if len(tiempos) > 1:
        diferencias = np.diff(tiempos.to_numpy())
        diferencias_positivas = diferencias[diferencias > 0]
        if len(diferencias_positivas):
            fps_real = 1.0 / float(np.median(diferencias_positivas))
            hueco_max = float(np.max(diferencias_positivas)) * 1000.0
        else:
            fps_real = math.nan
            hueco_max = math.nan
    else:
        fps_real = math.nan
        hueco_max = math.nan

    pose_detectada = pd.to_numeric(
        datos_camara.get("pose_detectada", pd.Series(dtype=float)),
        errors="coerce",
    )
    pose_pct = (
        100.0 * float(pose_detectada.fillna(0).mean())
        if not pose_detectada.empty
        else math.nan
    )
    return {
        "frames_prueba": int(len(datos_camara)),
        "frames_video_total": int(len(timeline)),
        "fps_camara_real": fps_real,
        "hueco_camara_max_ms": hueco_max,
        "frames_pose_detectada_pct": pose_pct,
    }
