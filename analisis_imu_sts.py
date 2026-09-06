"""Detección dinámica y validación del IMU lumbar para Five Times Sit-to-Stand.

La detección no consulta la cámara. Localiza cada transición por el patrón
dinámico del giro lumbar ``gz`` y confirma que exista una excursión simultánea
de la aceleración ``ax``. Las marcas de cámara se usan únicamente después, para
calcular concordancia temporal.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, sosfiltfilt


COLUMNAS_IMU = (
    "ax_m_s2",
    "ay_m_s2",
    "az_m_s2",
    "gx_rad_s",
    "gy_rad_s",
    "gz_rad_s",
)

COLUMNAS_DETECCION = (
    "evento",
    "tipo",
    "numero",
    "estado_resultante",
    "ms_prueba",
    "tiempo_prueba_s",
    "ms_inicio_transicion",
    "tiempo_inicio_transicion_s",
    "duracion_transicion_s",
    "tiempo_desde_movimiento_s",
    "pico_gz_rad_s",
    "valle_gz_rad_s",
    "excursion_ax_m_s2",
    "confianza",
    "criterio",
)

COLUMNAS_COMPARACION = (
    "evento",
    "evento_imu",
    "tiempo_camara_s",
    "tiempo_imu_s",
    "error_temporal_s",
    "error_absoluto_s",
    "coincide_en_tolerancia",
    "tolerancia_s",
)


def _filtrar(señal: np.ndarray, frecuencia_hz: float, corte_hz: float) -> np.ndarray:
    if len(señal) < 20:
        return señal.astype(float, copy=True)
    sos = butter(
        3,
        corte_hz / (frecuencia_hz / 2.0),
        btype="lowpass",
        output="sos",
    )
    return sosfiltfilt(sos, señal)


def _primer_movimiento(
    tiempo_s: np.ndarray,
    gz_filtrado: np.ndarray,
    primer_pico_indice: int | None,
    frecuencia_hz: float,
) -> float:
    """Estima el comienzo del primer movimiento sin usar la cámara."""
    if primer_pico_indice is None:
        return float(tiempo_s[0])

    pico = float(gz_filtrado[primer_pico_indice])
    umbral = max(0.15, 0.18 * pico)
    inicio_busqueda = max(
        0,
        primer_pico_indice - int(2.0 * frecuencia_hz),
    )
    inicio = primer_pico_indice
    for indice in range(primer_pico_indice, inicio_busqueda, -1):
        if gz_filtrado[indice] <= umbral:
            inicio = indice
            break
    return float(tiempo_s[inicio])


def _calidad_paquetes(df: pd.DataFrame) -> dict:
    secuencias = pd.to_numeric(df.get("seq"), errors="coerce").dropna()
    tiempos_ms = pd.to_numeric(df.get("ms_esp32"), errors="coerce").dropna()
    if secuencias.empty:
        esperadas = len(df)
    else:
        esperadas = int(secuencias.max() - secuencias.min() + 1)
    recibidas = int(len(df))
    perdidas = max(0, esperadas - recibidas)
    perdida_pct = 100.0 * perdidas / esperadas if esperadas else 0.0

    if len(tiempos_ms) > 1:
        duracion_s = float(tiempos_ms.max() - tiempos_ms.min()) / 1000.0
        frecuencia_real = (len(tiempos_ms) - 1) / duracion_s if duracion_s else math.nan
        hueco_max_ms = float(np.diff(np.sort(tiempos_ms.to_numpy())).max())
    else:
        frecuencia_real = math.nan
        hueco_max_ms = math.nan

    return {
        "muestras_recibidas": recibidas,
        "muestras_esperadas": esperadas,
        "muestras_perdidas": perdidas,
        "perdida_muestras_pct": perdida_pct,
        "frecuencia_real_hz": frecuencia_real,
        "hueco_maximo_ms": hueco_max_ms,
    }


def _detectar_transiciones(
    tiempo_s: np.ndarray,
    ax_filtrado: np.ndarray,
    gz_filtrado: np.ndarray,
    frecuencia_hz: float,
    max_eventos: int = 9,
) -> tuple[list[dict], dict]:
    """Localiza hasta nueve transiciones completas por el patrón +gz/-gz."""
    amplitud_positiva = max(0.0, float(np.quantile(gz_filtrado, 0.995)))
    amplitud_negativa = abs(min(0.0, float(np.quantile(gz_filtrado, 0.005))))
    umbral_pico = max(0.25, 0.22 * amplitud_positiva)
    prominencia = max(0.30, 0.18 * amplitud_positiva)
    umbral_retorno = -max(0.28, 0.28 * amplitud_negativa)

    picos, propiedades = find_peaks(
        gz_filtrado,
        height=umbral_pico,
        prominence=prominencia,
        distance=max(1, int(round(0.55 * frecuencia_hz))),
    )

    candidatos: list[dict] = []
    for posicion, pico_indice in enumerate(picos):
        inicio_indice = int(pico_indice)
        umbral_inicio = max(0.08, 0.12 * amplitud_positiva)
        limite_inicio = max(
            0,
            int(pico_indice) - int(round(2.5 * frecuencia_hz)),
        )
        for indice in range(int(pico_indice), limite_inicio, -1):
            if abs(float(gz_filtrado[indice])) <= umbral_inicio:
                inicio_indice = indice
                break

        inicio_valle = pico_indice + max(2, int(round(0.10 * frecuencia_hz)))
        limite_por_tiempo = pico_indice + int(round(2.0 * frecuencia_hz))
        limite_por_siguiente = (
            picos[posicion + 1] - max(1, int(round(0.04 * frecuencia_hz)))
            if posicion + 1 < len(picos)
            else len(tiempo_s) - 1
        )
        fin_valle = min(len(tiempo_s) - 1, limite_por_tiempo, limite_por_siguiente)
        if fin_valle <= inicio_valle:
            continue

        indices_valle = np.arange(inicio_valle, fin_valle + 1)
        valle_indice = int(indices_valle[np.argmin(gz_filtrado[indices_valle])])
        valle = float(gz_filtrado[valle_indice])
        if valle > -max(0.25, 0.14 * amplitud_negativa):
            continue

        evento_indice = valle_indice
        for indice in range(valle_indice + 1, fin_valle + 1):
            if (
                gz_filtrado[indice] >= umbral_retorno
                and gz_filtrado[indice - 1] < umbral_retorno
            ):
                evento_indice = indice
                break

        inicio_ax = max(0, pico_indice - int(round(0.30 * frecuencia_hz)))
        fin_ax = min(
            len(ax_filtrado) - 1,
            valle_indice + int(round(0.35 * frecuencia_hz)),
        )
        excursion_ax = float(np.ptp(ax_filtrado[inicio_ax : fin_ax + 1]))
        candidatos.append(
            {
                "pico_indice": int(pico_indice),
                "inicio_indice": inicio_indice,
                "valle_indice": valle_indice,
                "evento_indice": evento_indice,
                "pico_gz_rad_s": float(gz_filtrado[pico_indice]),
                "valle_gz_rad_s": valle,
                "excursion_ax_m_s2": excursion_ax,
            }
        )

    if candidatos:
        excursion_mediana = float(
            np.median([c["excursion_ax_m_s2"] for c in candidatos])
        )
        umbral_excursion_ax = max(0.80, 0.65 * excursion_mediana)
        candidatos = [
            candidato
            for candidato in candidatos
            if candidato["excursion_ax_m_s2"] >= umbral_excursion_ax
        ]
    else:
        umbral_excursion_ax = 0.80

    # Si una recolocación genera un pico extra, se conservan los nueve patrones
    # con mayor evidencia conjunta de giro y aceleración, sin alterar su orden.
    if len(candidatos) > max_eventos:
        ax_escala = max(
            1.0,
            float(np.median([c["excursion_ax_m_s2"] for c in candidatos])),
        )
        gz_escala = max(0.25, amplitud_positiva)
        for candidato in candidatos:
            candidato["_puntaje"] = (
                candidato["pico_gz_rad_s"] / gz_escala
                + abs(candidato["valle_gz_rad_s"]) / max(0.25, amplitud_negativa)
                + candidato["excursion_ax_m_s2"] / ax_escala
            )
        candidatos = sorted(
            sorted(candidatos, key=lambda fila: fila["_puntaje"], reverse=True)[
                :max_eventos
            ],
            key=lambda fila: fila["evento_indice"],
        )

    eventos: list[dict] = []
    ax_referencia = max(
        1.0,
        float(np.median([c["excursion_ax_m_s2"] for c in candidatos]))
        if candidatos
        else 1.0,
    )
    for indice, candidato in enumerate(candidatos[:max_eventos]):
        es_levantamiento = indice % 2 == 0
        numero = indice // 2 + 1
        tipo = "R" if es_levantamiento else "S"
        instante = float(tiempo_s[candidato["evento_indice"]])
        inicio_transicion = float(tiempo_s[candidato["inicio_indice"]])
        confianza = float(
            np.clip(
                0.50
                * candidato["pico_gz_rad_s"] / max(umbral_pico, 1e-6)
                + 0.30
                * abs(candidato["valle_gz_rad_s"])
                / max(abs(umbral_retorno), 1e-6)
                + 0.20 * candidato["excursion_ax_m_s2"] / ax_referencia,
                0.0,
                5.0,
            )
            / 5.0
        )
        eventos.append(
            {
                "evento": f"imu_{tipo}{numero}",
                "tipo": tipo,
                "numero": numero,
                "estado_resultante": "de_pie" if es_levantamiento else "sentado",
                "ms_prueba": int(round(instante * 1000.0)),
                "tiempo_prueba_s": instante,
                "ms_inicio_transicion": int(
                    round(inicio_transicion * 1000.0)
                ),
                "tiempo_inicio_transicion_s": inicio_transicion,
                "duracion_transicion_s": instante - inicio_transicion,
                "pico_gz_rad_s": candidato["pico_gz_rad_s"],
                "valle_gz_rad_s": candidato["valle_gz_rad_s"],
                "excursion_ax_m_s2": candidato["excursion_ax_m_s2"],
                "confianza": confianza,
                "criterio": (
                    "patrón dinámico +gz/-gz confirmado por excursión de ax"
                ),
            }
        )

    parametros = {
        "amplitud_positiva_gz_rad_s": amplitud_positiva,
        "amplitud_negativa_gz_rad_s": amplitud_negativa,
        "umbral_pico_gz_rad_s": umbral_pico,
        "prominencia_gz_rad_s": prominencia,
        "umbral_retorno_gz_rad_s": umbral_retorno,
        "umbral_excursion_ax_m_s2": umbral_excursion_ax,
        "candidatos_dinamicos": len(candidatos),
    }
    return eventos, parametros


def _alinear_secuencias(
    referencias: list[tuple[str, float]],
    detecciones: list[tuple[str, float]],
    tolerancia_s: float,
) -> tuple[dict[int, int], set[int]]:
    """Alineación monótona que evita desplazar R3→R2 si falta un evento."""
    n, m = len(referencias), len(detecciones)
    dp: list[list[tuple[int, float]]] = [
        [(0, 0.0) for _ in range(m + 1)] for _ in range(n + 1)
    ]
    accion = [["" for _ in range(m + 1)] for _ in range(n + 1)]

    def mejor(
        actual: tuple[int, float],
        candidato: tuple[int, float],
    ) -> bool:
        return candidato[0] > actual[0] or (
            candidato[0] == actual[0] and candidato[1] < actual[1]
        )

    for i in range(n + 1):
        for j in range(m + 1):
            if i < n and mejor(dp[i + 1][j], dp[i][j]):
                dp[i + 1][j] = dp[i][j]
                accion[i + 1][j] = "omitir_ref"
            if j < m and mejor(dp[i][j + 1], dp[i][j]):
                dp[i][j + 1] = dp[i][j]
                accion[i][j + 1] = "omitir_det"
            if i < n and j < m:
                error = abs(referencias[i][1] - detecciones[j][1])
                if error <= tolerancia_s:
                    candidato = (dp[i][j][0] + 1, dp[i][j][1] + error)
                    if mejor(dp[i + 1][j + 1], candidato):
                        dp[i + 1][j + 1] = candidato
                        accion[i + 1][j + 1] = "unir"

    pares: dict[int, int] = {}
    usados: set[int] = set()
    i, j = n, m
    while i > 0 or j > 0:
        paso = accion[i][j]
        if paso == "unir":
            pares[i - 1] = j - 1
            usados.add(j - 1)
            i -= 1
            j -= 1
        elif paso == "omitir_ref":
            i -= 1
        elif paso == "omitir_det":
            j -= 1
        elif i > 0:
            i -= 1
        elif j > 0:
            j -= 1
    return pares, usados


def _comparar_eventos(
    referencias: pd.DataFrame,
    detecciones: pd.DataFrame,
    tolerancia_s: float,
) -> tuple[pd.DataFrame, dict]:
    if referencias.empty or not {"evento", "ms_prueba"}.issubset(referencias.columns):
        return pd.DataFrame(columns=COLUMNAS_COMPARACION), {
            "vp": 0,
            "fp": 0,
            "fn": 0,
            "total_referencia": 0,
        }

    refs = referencias[
        referencias["evento"].astype(str).str.match(r"camera_[RS]\d+$")
    ].copy()
    refs["tipo"] = refs["evento"].astype(str).str.extract(r"camera_([RS])")
    refs["tiempo"] = pd.to_numeric(refs["ms_prueba"], errors="coerce") / 1000.0
    refs = refs.dropna(subset=["tipo", "tiempo"]).sort_values("tiempo")

    filas: list[dict] = []
    indices_usados_globales: set[int] = set()
    for tipo in ("R", "S"):
        refs_tipo = [
            (str(fila["evento"]).replace("camera_", ""), float(fila["tiempo"]))
            for _, fila in refs[refs["tipo"] == tipo].iterrows()
        ]
        det_tipo_df = detecciones[detecciones["tipo"] == tipo].sort_values(
            "tiempo_prueba_s"
        )
        det_indices = list(det_tipo_df.index)
        det_tipo = [
            (str(fila["evento"]), float(fila["tiempo_prueba_s"]))
            for _, fila in det_tipo_df.iterrows()
        ]
        pares, usados_locales = _alinear_secuencias(
            refs_tipo,
            det_tipo,
            tolerancia_s,
        )
        indices_usados_globales.update(det_indices[j] for j in usados_locales)

        for indice_ref, (evento_ref, tiempo_ref) in enumerate(refs_tipo):
            indice_det = pares.get(indice_ref)
            if indice_det is None:
                tiempo_imu = math.nan
                evento_imu = ""
                error = math.nan
                coincide = False
            else:
                evento_imu, tiempo_imu = det_tipo[indice_det]
                error = tiempo_imu - tiempo_ref
                coincide = True
            filas.append(
                {
                    "evento": evento_ref,
                    "evento_imu": evento_imu,
                    "tiempo_camara_s": tiempo_ref,
                    "tiempo_imu_s": tiempo_imu,
                    "error_temporal_s": error,
                    "error_absoluto_s": (
                        abs(error) if math.isfinite(error) else math.nan
                    ),
                    "coincide_en_tolerancia": coincide,
                    "tolerancia_s": tolerancia_s,
                }
            )

    comparacion = pd.DataFrame(filas, columns=COLUMNAS_COMPARACION)
    vp = int(comparacion["coincide_en_tolerancia"].sum())
    fp = max(0, len(detecciones) - len(indices_usados_globales))
    fn = max(0, len(refs) - vp)
    return comparacion, {
        "vp": vp,
        "fp": fp,
        "fn": fn,
        "total_referencia": len(refs),
    }


def analizar_imu_sts(
    muestras: Iterable[dict] | pd.DataFrame,
    eventos_camara: Iterable[dict] | pd.DataFrame = (),
    frecuencia_hz: float = 50.0,
    tolerancia_s: float = 0.50,
    duracion_minima_s: float = 0.75,
    origen_tiempo_s: float = 0.0,
) -> dict:
    """Detecta R1–R5/S1–S4 y, después, los compara contra la cámara."""
    df = muestras.copy() if isinstance(muestras, pd.DataFrame) else pd.DataFrame(muestras)
    referencias = (
        eventos_camara.copy()
        if isinstance(eventos_camara, pd.DataFrame)
        else pd.DataFrame(eventos_camara)
    )

    requeridas = {"ms_prueba", *COLUMNAS_IMU}
    faltantes = requeridas.difference(df.columns)
    if faltantes:
        raise ValueError(
            "Faltan columnas IMU requeridas: " + ", ".join(sorted(faltantes))
        )

    if "grabando" in df.columns:
        df = df[pd.to_numeric(df["grabando"], errors="coerce").fillna(0) == 1]
    df = df.copy()
    for columna in ("seq", "ms_prueba", "ms_esp32", *COLUMNAS_IMU):
        if columna in df.columns:
            df[columna] = pd.to_numeric(df[columna], errors="coerce")
    df = df.dropna(subset=["ms_prueba", *COLUMNAS_IMU])
    if "seq" in df.columns:
        df = df.drop_duplicates(subset=["seq"], keep="last")
    df = df.sort_values("ms_prueba")

    if len(df) < max(20, int(frecuencia_hz * duracion_minima_s)):
        raise ValueError("La grabación IMU todavía es demasiado corta para analizar.")

    calidad = _calidad_paquetes(df)
    tiempo_original = (
        df["ms_prueba"].to_numpy(dtype=float) / 1000.0
        - float(origen_tiempo_s)
    )
    inicio = float(tiempo_original[0])
    fin = float(tiempo_original[-1])
    tiempo = np.arange(inicio, fin + 0.5 / frecuencia_hz, 1.0 / frecuencia_hz)

    señal = pd.DataFrame({"tiempo_prueba_s": tiempo})
    for columna in COLUMNAS_IMU:
        señal[columna] = np.interp(
            tiempo,
            tiempo_original,
            df[columna].to_numpy(dtype=float),
        )

    ax = _filtrar(señal["ax_m_s2"].to_numpy(), frecuencia_hz, 3.0)
    ay = _filtrar(señal["ay_m_s2"].to_numpy(), frecuencia_hz, 3.0)
    gz = _filtrar(señal["gz_rad_s"].to_numpy(), frecuencia_hz, 3.0)
    señal["ax_filtrado_m_s2"] = ax
    señal["ay_filtrado_m_s2"] = ay
    señal["gz_filtrado_rad_s"] = gz

    eventos, parametros = _detectar_transiciones(
        tiempo,
        ax,
        gz,
        frecuencia_hz,
    )
    primer_pico = None
    if eventos:
        primer_evento_s = float(eventos[0]["tiempo_prueba_s"])
        anteriores = np.where(tiempo <= primer_evento_s)[0]
        if len(anteriores):
            primer_pico = int(anteriores[np.argmax(gz[anteriores])])
    movimiento_inicio_s = _primer_movimiento(
        tiempo,
        gz,
        primer_pico,
        frecuencia_hz,
    )

    for evento in eventos:
        evento["tiempo_desde_movimiento_s"] = (
            float(evento["tiempo_prueba_s"]) - movimiento_inicio_s
        )
    df_detecciones = pd.DataFrame(eventos, columns=COLUMNAS_DETECCION)

    df_comparacion, conteos = _comparar_eventos(
        referencias,
        df_detecciones,
        tolerancia_s,
    )
    vp, fp, fn = conteos["vp"], conteos["fp"], conteos["fn"]
    precision = vp / (vp + fp) if vp + fp else math.nan
    recall = vp / (vp + fn) if vp + fn else math.nan
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if math.isfinite(precision)
        and math.isfinite(recall)
        and precision + recall > 0
        else math.nan
    )

    r5 = df_detecciones[df_detecciones["evento"] == "imu_R5"]
    tiempo_clinico_imu = (
        float(r5.iloc[0]["tiempo_prueba_s"]) if not r5.empty else math.nan
    )
    tiempo_efectivo_imu = (
        tiempo_clinico_imu - movimiento_inicio_s
        if math.isfinite(tiempo_clinico_imu)
        else math.nan
    )

    resumen = {
        **calidad,
        **parametros,
        "metodo_detector_imu": "patron_dinamico_gz_confirmado_ax",
        "origen_tiempo_imu_s": float(origen_tiempo_s),
        "movimiento_inicio_imu_s": movimiento_inicio_s,
        "eventos_imu_detectados": len(df_detecciones),
        "levantamientos_imu_detectados": int(
            (df_detecciones["tipo"] == "R").sum()
        ),
        "regresos_sentado_imu_detectados": int(
            (df_detecciones["tipo"] == "S").sum()
        ),
        "vp": vp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "mae_eventos_s": (
            float(
                df_comparacion.loc[
                    df_comparacion["coincide_en_tolerancia"],
                    "error_absoluto_s",
                ].mean()
            )
            if vp
            else math.nan
        ),
        "tiempo_clinico_imu_s": tiempo_clinico_imu,
        "tiempo_efectivo_imu_s": tiempo_efectivo_imu,
        "tolerancia_eventos_s": tolerancia_s,
        "metrica_comparacion": (
            "concordancia_camara_imu_no_referencia_manual"
        ),
    }

    if not referencias.empty and {"evento", "ms_prueba"}.issubset(referencias.columns):
        tiempos_ref = {
            str(fila["evento"]): float(fila["ms_prueba"]) / 1000.0
            for _, fila in referencias.iterrows()
        }
        camara_r5 = tiempos_ref.get("camera_R5", math.nan)
        camara_movimiento = tiempos_ref.get("camera_MO1", math.nan)
        resumen["tiempo_clinico_camara_s"] = camara_r5
        resumen["tiempo_efectivo_camara_s"] = (
            camara_r5 - camara_movimiento
            if math.isfinite(camara_r5) and math.isfinite(camara_movimiento)
            else math.nan
        )
        resumen["error_tiempo_clinico_s"] = (
            tiempo_clinico_imu - camara_r5
            if math.isfinite(tiempo_clinico_imu) and math.isfinite(camara_r5)
            else math.nan
        )
        resumen["error_tiempo_efectivo_s"] = (
            tiempo_efectivo_imu - resumen["tiempo_efectivo_camara_s"]
            if math.isfinite(tiempo_efectivo_imu)
            and math.isfinite(resumen["tiempo_efectivo_camara_s"])
            else math.nan
        )

    return {
        "senal": señal,
        "detecciones": df_detecciones,
        "comparacion": df_comparacion,
        "resumen": pd.DataFrame([resumen]),
    }
