"""Panel OpenCV para supervisar el IMU lumbar mientras se ejecuta 5xSTS."""

from __future__ import annotations

import math
import time

import cv2
import numpy as np
import pandas as pd

from analisis_imu_sts import analizar_imu_sts


class MonitorIMUEnVivo:
    """Mantiene una vista ligera de ax, gz, conexión y eventos independientes."""

    def __init__(self, frecuencia_hz: float = 50.0) -> None:
        self.frecuencia_hz = frecuencia_hz
        self._ultimo_analisis = 0.0
        self._resultado = None
        self._mensaje = "Esperando inicio de la prueba"

    def reiniciar(self) -> None:
        self._ultimo_analisis = 0.0
        self._resultado = None
        self._mensaje = "Esperando inicio de la prueba"

    @staticmethod
    def _calidad(df: pd.DataFrame) -> tuple[float, float]:
        if len(df) < 2:
            return math.nan, 0.0
        ms = pd.to_numeric(df.get("ms_esp32"), errors="coerce").dropna()
        secuencias = pd.to_numeric(df.get("seq"), errors="coerce").dropna()
        frecuencia = math.nan
        if len(ms) > 1 and ms.iloc[-1] > ms.iloc[0]:
            frecuencia = (len(ms) - 1) / ((ms.iloc[-1] - ms.iloc[0]) / 1000.0)
        perdida = 0.0
        if len(secuencias) > 1:
            esperadas = int(secuencias.max() - secuencias.min() + 1)
            perdida = 100.0 * max(0, esperadas - len(secuencias)) / esperadas
        return frecuencia, perdida

    def _actualizar_analisis(self, df: pd.DataFrame) -> None:
        ahora = time.perf_counter()
        if ahora - self._ultimo_analisis < 0.25:
            return
        self._ultimo_analisis = ahora
        try:
            self._resultado = analizar_imu_sts(
                df,
                frecuencia_hz=self.frecuencia_hz,
                duracion_minima_s=0.75,
            )
            self._mensaje = "Detector dinamico activo"
        except ValueError as exc:
            self._mensaje = str(exc)

    @staticmethod
    def _trazar(
        panel: np.ndarray,
        tiempos: np.ndarray,
        valores: np.ndarray,
        rectangulo: tuple[int, int, int, int],
        color: tuple[int, int, int],
        etiqueta: str,
    ) -> None:
        x0, y0, x1, y1 = rectangulo
        cv2.rectangle(panel, (x0, y0), (x1, y1), (70, 76, 86), 1)
        cv2.putText(
            panel,
            etiqueta,
            (x0 + 8, y0 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            1,
            cv2.LINE_AA,
        )
        if len(tiempos) < 2:
            return
        minimo = float(np.quantile(valores, 0.02))
        maximo = float(np.quantile(valores, 0.98))
        margen = max(0.15, 0.12 * (maximo - minimo))
        minimo -= margen
        maximo += margen
        if maximo <= minimo:
            maximo = minimo + 1.0
        xs = x0 + (tiempos - tiempos[0]) / max(
            tiempos[-1] - tiempos[0],
            1e-6,
        ) * (x1 - x0)
        ys = y1 - (valores - minimo) / (maximo - minimo) * (y1 - y0)
        puntos = np.column_stack([xs, ys]).astype(np.int32)
        cv2.polylines(panel, [puntos], False, color, 2, cv2.LINE_AA)

    def construir(
        self,
        muestras: list[dict],
        conectado: bool,
        grabando: bool,
        error: str = "",
    ) -> np.ndarray:
        panel = np.full((540, 920, 3), (24, 29, 38), dtype=np.uint8)
        cv2.putText(
            panel,
            "MONITOR IMU LUMBAR L5",
            (28, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (245, 245, 245),
            2,
            cv2.LINE_AA,
        )

        df = pd.DataFrame(muestras)
        if not df.empty and "grabando" in df.columns:
            df_grabando = df[
                pd.to_numeric(df["grabando"], errors="coerce").fillna(0) == 1
            ].copy()
        else:
            df_grabando = df
        if grabando and not df_grabando.empty:
            self._actualizar_analisis(df_grabando)

        frecuencia, perdida = self._calidad(df_grabando)
        detecciones = (
            self._resultado["detecciones"]
            if self._resultado is not None
            else pd.DataFrame()
        )
        repeticiones = (
            int((detecciones["tipo"] == "R").sum())
            if not detecciones.empty
            else 0
        )
        estado = "SENTADO"
        if not grabando:
            estado = "CALIBRANDO / LISTO"
        elif not detecciones.empty:
            estado = (
                "DE PIE"
                if str(detecciones.iloc[-1]["tipo"]) == "R"
                else "SENTADO"
            )

        # Mientras todavía no se completa la transición, informa el sentido
        # esperado según el último estado estable.
        if grabando and not df_grabando.empty and "gz_rad_s" in df_grabando:
            gz_reciente = pd.to_numeric(
                df_grabando["gz_rad_s"],
                errors="coerce",
            ).dropna().tail(8)
            if len(gz_reciente) and abs(float(gz_reciente.mean())) > 0.35:
                estado = "SUBIENDO" if estado == "SENTADO" else "BAJANDO"

        color_conexion = (80, 210, 120) if conectado else (70, 80, 230)
        conexion = "CONECTADO" if conectado else "SIN DATOS"
        cv2.circle(panel, (40, 78), 8, color_conexion, -1)
        cv2.putText(
            panel,
            f"{conexion}   |   {frecuencia:.1f} Hz"
            if math.isfinite(frecuencia)
            else conexion,
            (58, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (225, 230, 238),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            f"Paquetes perdidos: {perdida:.2f}%   |   Estado: {estado}",
            (365, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (225, 230, 238),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            f"Repeticiones IMU: {repeticiones}/5",
            (28, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (190, 145, 255),
            2,
            cv2.LINE_AA,
        )

        if not df_grabando.empty and {
            "ms_prueba",
            "ax_m_s2",
            "gz_rad_s",
        }.issubset(df_grabando.columns):
            vista = df_grabando.copy()
            vista["t"] = pd.to_numeric(vista["ms_prueba"], errors="coerce") / 1000.0
            vista["ax"] = pd.to_numeric(vista["ax_m_s2"], errors="coerce")
            vista["gz"] = pd.to_numeric(vista["gz_rad_s"], errors="coerce")
            vista = vista.dropna(subset=["t", "ax", "gz"])
            if not vista.empty:
                vista = vista[vista["t"] >= vista["t"].max() - 8.0]
                t = vista["t"].to_numpy(dtype=float)
                self._trazar(
                    panel,
                    t,
                    vista["ax"].to_numpy(dtype=float),
                    (28, 150, 892, 315),
                    (70, 170, 255),
                    "ax (m/s2) - postura y aceleracion",
                )
                self._trazar(
                    panel,
                    t,
                    vista["gz"].to_numpy(dtype=float),
                    (28, 335, 892, 500),
                    (210, 115, 255),
                    "gz (rad/s) - giro del tronco",
                )
        else:
            cv2.putText(
                panel,
                "La grafica aparecera cuando comience la adquisicion.",
                (135, 300),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (160, 170, 185),
                1,
                cv2.LINE_AA,
            )

        mensaje = error or self._mensaje
        if mensaje:
            cv2.putText(
                panel,
                mensaje[:105],
                (28, 528),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                (155, 165, 180) if not error else (80, 120, 245),
                1,
                cv2.LINE_AA,
            )
        return panel
