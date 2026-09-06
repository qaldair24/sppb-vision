"""Configuración congelada para la recolección del estudio 5xSTS.

Los valores de este archivo forman parte de la versión del algoritmo. No deben
modificarse durante la recolección definitiva. Cualquier cambio requiere una
versión nueva y debe documentarse antes de incluir más ensayos.
"""

VERSION_SISTEMA = "4.1.0-MAC-M1"
VERSION_ALGORITMO = "5xSTS-CAM-IMU-L5-2026.07"
ESTADO_VERSION = "CONGELADA_PARA_RECOLECCION_MACOS"

REPETICIONES_OBJETIVO = 5
FRECUENCIA_IMU_HZ = 50.0
TOLERANCIA_CONCORDANCIA_S = 0.50

UMBRAL_SENTADO_BASE_GRADOS = 115.0
UMBRAL_DE_PIE_MINIMO_GRADOS = 150.0
CONFIRMACION_POSTURA_S = 0.25
HISTERESIS_INICIO_MOVIMIENTO_GRADOS = 8.0

ESTABILIDAD_INICIAL_S = 2.0
ESTABILIDAD_ANUNCIO_S = 1.0
CUENTA_REGRESIVA_S = 3.0

ALTURA_SILLA_PREDETERMINADA_CM = 43.0
