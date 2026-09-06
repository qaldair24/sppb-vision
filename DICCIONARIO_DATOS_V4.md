# Diccionario mínimo de datos — 5xSTS-VISION V4

## Resultado principal por ensayo

| Campo | Unidad | Descripción |
|---|---:|---|
| `repeticiones_camara` | conteo | Repeticiones completas detectadas |
| `prueba_completada` | lógico | Cinco repeticiones y tiempo clínico disponible |
| `tiempo_clinico_camara_s` | s | Tiempo desde GO hasta R5 |
| `tiempo_efectivo_camara_s` | s | Tiempo desde MO1 hasta R5 |
| `tiempo_reaccion_s` | s | Tiempo desde GO hasta MO1 |
| `puntaje_sts` | 0–4 | Subpuntaje de silla, solo si la prueba es completa |
| `sppb_total_calculado` | lógico | Siempre `False` en esta versión |

## Calidad técnica

| Campo | Unidad | Descripción |
|---|---:|---|
| `fps_camara_real` | Hz | Frecuencia mediana real de cuadros procesados |
| `hueco_camara_max_ms` | ms | Mayor intervalo entre cuadros |
| `frames_pose_detectada_pct` | % | Cuadros con pose disponible |
| `frecuencia_real_hz` | Hz | Frecuencia real del IMU |
| `perdida_muestras_pct` | % | Paquetes IMU perdidos |
| `hueco_maximo_ms` | ms | Mayor intervalo entre muestras IMU |

## Fusión

`eventos_fusion_...csv` usa una regla fija:

- `acuerdo_camara_imu`: ambos sensores detectan el evento dentro de ±0.50 s;
  se usa el promedio de sus tiempos.
- `solo_camara`: únicamente la cámara detecta el evento.
- `solo_imu`: únicamente el IMU detecta el evento.
- `conflicto_rechazado`: ambos lo detectan, pero difieren más de ±0.50 s; no se
  asigna tiempo fusionado.
- `no_detectado`: ninguno lo detecta.

Esta salida sigue siendo automática. No reemplaza la anotación manual de
referencia necesaria para sensibilidad, precisión, F1, MAE, RMSE, ICC o
Bland–Altman.

## Reproducibilidad del entorno

El archivo `metadata_...json` registra, para cada ensayo, la versión de
Python, macOS, arquitectura del procesador y versiones efectivamente instaladas
de MediaPipe, NumPy, OpenCV, pandas, Matplotlib, SciPy, ReportLab y Pillow.
