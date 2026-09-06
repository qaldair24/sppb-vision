import cv2
import json
import mediapipe as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import platform
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from importlib.metadata import PackageNotFoundError, version as version_paquete
from scipy.signal import find_peaks
from analisis_imu_sts import analizar_imu_sts
from configuracion_v4 import (
    CONFIRMACION_POSTURA_S,
    CUENTA_REGRESIVA_S,
    ESTABILIDAD_ANUNCIO_S,
    ESTABILIDAD_INICIAL_S,
    FRECUENCIA_IMU_HZ,
    HISTERESIS_INICIO_MOVIMIENTO_GRADOS,
    REPETICIONES_OBJETIVO,
    TOLERANCIA_CONCORDANCIA_S,
    UMBRAL_DE_PIE_MINIMO_GRADOS,
    UMBRAL_SENTADO_BASE_GRADOS,
    VERSION_ALGORITMO,
    VERSION_SISTEMA,
)
from imu_lumbar import ErrorIMU, IMULumbar
from metricas_sts import (
    calidad_camara,
    construir_repeticiones,
    fusionar_eventos,
    subpuntaje_silla_sppb,
)
from monitor_imu_en_vivo import MonitorIMUEnVivo
from voz_espanol import preparar_texto_para_voz

if not hasattr(mp, "solutions"):
    version_mediapipe = getattr(mp, "__version__", "desconocida")
    ruta_mediapipe = getattr(mp, "__file__", "desconocida")
    raise RuntimeError(
        "MediaPipe incompatible: se detectó la versión "
        f"{version_mediapipe}, que no incluye 'mp.solutions'. "
        "Cierre el programa y ejecute 'INSTALAR_MAC_M1.command' para instalar "
        "MediaPipe 0.10.21. Módulo cargado desde: "
        f"{ruta_mediapipe}"
    )

PROJECT_DIR = os.environ.get(
    "SPPB_PROJECT_DIR",
    os.path.dirname(os.path.abspath(__file__))
)
BASE_VIDEO_FOLDER = os.path.join(PROJECT_DIR, "videos")
BASE_OUTPUT_FOLDER = os.path.join(PROJECT_DIR, "resultados")


def _version_instalada(nombre):
    try:
        return version_paquete(nombre)
    except PackageNotFoundError:
        return "no_instalado"

print("\n=================================")
print("SPPB-VISION")
print("Evaluación funcional automática")
print("=================================")

paciente_id = input("Ingrese ID del paciente: ").strip()
trial_id = os.environ.get("SPPB_TRIAL_ID", "T1").strip() or "T1"
chair_height_cm = float(os.environ.get("SPPB_CHAIR_HEIGHT_CM", "43.0"))
if not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", paciente_id):
    raise ValueError(
        "ID de participante inválido. Use letras, números, guion o guion bajo."
    )
if not re.fullmatch(r"[A-Za-z0-9_-]{1,16}", trial_id):
    raise ValueError(
        "ID de ensayo inválido. Use letras, números, guion o guion bajo."
    )
if not 30.0 <= chair_height_cm <= 60.0:
    raise ValueError("La altura de la silla debe estar entre 30 y 60 cm.")

print("\nSeleccione modo de análisis:")
print("1 - Video grabado")
print("2 - Cámara en vivo")

modo = input("Opción: ")

print("\nSeleccione prueba:")
print("1 - Equilibrio")
print("2 - Marcha 3 metros")
print("3 - Sit to Stand")
print("4 - Evaluación completa")
print("5 - Recoger objeto")
print("6 - Sentadilla")

opcion = input("Opción: ")

if opcion == "1":
    ejercicio = "equilibrio"
elif opcion == "2":
    ejercicio = "marcha"
elif opcion == "3":
    ejercicio = "sit_to_stand"
elif opcion == "4":
    ejercicio = "evaluacion_completa"
elif opcion == "5":
    ejercicio = "recoger_objeto"
elif opcion == "6":
    ejercicio = "sentadilla"
else:
    raise ValueError("Opción no válida.")

OUTPUT_DIR = os.path.join(BASE_OUTPUT_FOLDER, paciente_id)
os.makedirs(OUTPUT_DIR, exist_ok=True)

if modo == "1":
    if ejercicio == "evaluacion_completa":
        raise ValueError("Por ahora ejecutemos una prueba a la vez.")

    VIDEO_FOLDER = os.path.join(BASE_VIDEO_FOLDER, paciente_id)

    video_files = [
        f for f in os.listdir(VIDEO_FOLDER)
        if f.lower().endswith(".mp4") and ejercicio in f.lower()
    ]

    if not video_files:
        raise FileNotFoundError(
            f"No se encontró video para {ejercicio} en: {VIDEO_FOLDER}"
        )

    print(f"\nVideos encontrados para {ejercicio}:")
    for i, video in enumerate(video_files):
        print(f"{i+1}. {video}")

    video_opcion = int(input("Seleccione el número de video: ")) - 1
    VIDEO_PATH = os.path.join(VIDEO_FOLDER, video_files[video_opcion])
    nombre_video = os.path.splitext(video_files[video_opcion])[0]

elif modo == "2":
    VIDEO_PATH = 0
    sello_intento = time.strftime("%Y%m%d_%H%M%S")
    nombre_video = (
        f"{paciente_id}_{trial_id}_{ejercicio}_en_vivo_{sello_intento}"
    )

else:
    raise ValueError("Modo no válido.")

OUTPUT_CSV = os.path.join(OUTPUT_DIR, f"resultados_{ejercicio}_{nombre_video}.csv")
OUTPUT_CSV_REPS = os.path.join(OUTPUT_DIR, f"repeticiones_{ejercicio}_{nombre_video}.csv")
OUTPUT_GRAPH = os.path.join(OUTPUT_DIR, f"grafica_{ejercicio}_{nombre_video}.png")
OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, f"video_anotado_{ejercicio}_{nombre_video}.mp4")
OUTPUT_VIDEO_ORIGINAL = os.path.join(
    OUTPUT_DIR,
    f"video_original_{ejercicio}_{nombre_video}.mp4",
)
OUTPUT_EVENTOS_CAMARA = os.path.join(
    OUTPUT_DIR,
    f"eventos_camara_{ejercicio}_{nombre_video}.csv",
)
OUTPUT_TIMELINE_VIDEO = os.path.join(
    OUTPUT_DIR,
    f"timeline_video_{ejercicio}_{nombre_video}.csv",
)
OUTPUT_METADATA = os.path.join(
    OUTPUT_DIR,
    f"metadata_{ejercicio}_{nombre_video}.json",
)
OUTPUT_GRAPH_EVENTOS = os.path.join(OUTPUT_DIR, f"grafica_eventos_{ejercicio}_{nombre_video}.png")
OUTPUT_REPORTE = os.path.join(OUTPUT_DIR, f"reporte_final_{paciente_id}.csv")
OUTPUT_IMU_CSV = os.path.join(OUTPUT_DIR, f"imu_l5_{ejercicio}_{nombre_video}.csv")
OUTPUT_IMU_EVENTOS = os.path.join(
    OUTPUT_DIR,
    f"eventos_imu_l5_{ejercicio}_{nombre_video}.csv",
)
OUTPUT_GRAPH_IMU = os.path.join(
    OUTPUT_DIR,
    f"grafica_sincronizada_camara_imu_{nombre_video}.png",
)
OUTPUT_IMU_DETECCIONES = os.path.join(
    OUTPUT_DIR,
    f"detecciones_imu_l5_{ejercicio}_{nombre_video}.csv",
)
OUTPUT_IMU_CONCORDANCIA = os.path.join(
    OUTPUT_DIR,
    f"concordancia_camara_imu_{ejercicio}_{nombre_video}.csv",
)
OUTPUT_EVENTOS_FUSION = os.path.join(
    OUTPUT_DIR,
    f"eventos_fusion_{ejercicio}_{nombre_video}.csv",
)
OUTPUT_RESUMEN_STS = os.path.join(
    OUTPUT_DIR,
    f"resumen_sts_{nombre_video}.csv",
)

# =========================
# CONFIGURACIÓN MEDIAPIPE
# =========================

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# =========================
# FUNCIÓN PARA CALCULAR ÁNGULOS
# =========================

def calcular_angulo(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    norma_ba = np.linalg.norm(ba)
    norma_bc = np.linalg.norm(bc)

    if norma_ba == 0 or norma_bc == 0:
        return np.nan

    coseno = np.dot(ba, bc) / (norma_ba * norma_bc)
    coseno = np.clip(coseno, -1.0, 1.0)

    angulo = np.degrees(np.arccos(coseno))
    return angulo


class AsistenteVoz:
    """Texto a voz no bloqueante y sin repetir mensajes continuamente."""

    def __init__(self, habilitado=True):
        self.habilitado = habilitado
        # None significa que el motor todavía se está inicializando. Esto evita
        # perder la primera instrucción por una condición de carrera.
        self.disponible = None if habilitado else False
        self.cola = queue.Queue()
        self.ultimo_mensaje = None
        self.ultimo_instante = 0.0
        self.instante_por_clave = {}
        self.ocupado = threading.Event()

        if not habilitado:
            return

        reproductor_disponible = (
            shutil.which("say") is not None
            if platform.system() == "Darwin"
            else True
        )
        try:
            if platform.system() != "Darwin":
                import pyttsx3  # noqa: F401
            if not reproductor_disponible:
                raise RuntimeError("No se encontró el comando de voz de macOS.")
            self.disponible = True
            threading.Thread(target=self._procesar_cola, daemon=True).start()
        except (ImportError, RuntimeError):
            self.disponible = False
            print(
                "Aviso: la voz está desactivada porque el reproductor "
                "del sistema no está disponible."
            )

    def _procesar_cola(self):
        try:
            reproductor = os.path.join(PROJECT_DIR, "REPRODUCIR_VOZ.py")

            while True:
                elemento = self.cola.get()
                if elemento is None:
                    self.cola.task_done()
                    break

                clave, texto = elemento
                texto_preparado = preparar_texto_para_voz(texto)
                reproducido = False
                self.ocupado.set()

                # Cada instrucción se reproduce en un proceso independiente
                # para no bloquear el bucle de captura.
                for intento in range(2):
                    try:
                        print(f"[VOZ] Reproduciendo: {texto_preparado}")
                        resultado = subprocess.run(
                            [sys.executable, reproductor, texto_preparado],
                            capture_output=True,
                            text=True,
                            timeout=30,
                            creationflags=getattr(
                                subprocess, "CREATE_NO_WINDOW", 0
                            ),
                        )

                        if resultado.returncode != 0:
                            detalle = (
                                resultado.stderr.strip()
                                or resultado.stdout.strip()
                                or f"código {resultado.returncode}"
                            )
                            raise RuntimeError(detalle)

                        reproducido = True
                        self.disponible = True
                        break
                    except Exception as error:
                        print(
                            f"[VOZ] Intento {intento + 1} fallido para "
                            f"'{clave}': {error}"
                        )
                        time.sleep(0.15)

                if not reproducido and self.disponible is not False:
                    print(
                        f"[VOZ] No se pudo reproducir la instrucción: {texto}"
                    )

                self.ocupado.clear()
                self.cola.task_done()
        except Exception as error:
            self.ocupado.clear()
            self.disponible = False
            print(f"Aviso: no se pudo inicializar la voz: {error}")

    def decir(
        self,
        clave,
        texto,
        repetir_despues_s=6.0,
        conservar_pendientes=False,
    ):
        if not self.habilitado or self.disponible is False:
            return

        ahora = time.monotonic()
        primera_vez = clave not in self.instante_por_clave
        ultimo_instante_clave = self.instante_por_clave.get(clave, 0.0)
        puede_repetir = (
            ahora - ultimo_instante_clave >= repetir_despues_s
        )

        if primera_vez or puede_repetir:
            # Las correcciones de postura sustituyen instrucciones obsoletas.
            # Los eventos importantes (inicio, conteo y final) se conservan.
            if not conservar_pendientes:
                while not self.cola.empty():
                    try:
                        self.cola.get_nowait()
                        self.cola.task_done()
                    except queue.Empty:
                        break

            self.cola.put((clave, texto))
            self.ultimo_mensaje = clave
            self.ultimo_instante = ahora
            self.instante_por_clave[clave] = ahora

    def tiene_audio_pendiente(self):
        return self.ocupado.is_set() or not self.cola.empty()

    def emitir_senal_inicio(self):
        """Emite un tono breve sin bloquear el bucle de captura."""
        if platform.system() == "Darwin":
            sonido = "/System/Library/Sounds/Glass.aiff"
            try:
                subprocess.Popen(
                    ["afplay", sonido],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return time.perf_counter()
            except OSError as error:
                print(f"[VOZ] No se pudo emitir el tono de inicio: {error}")

        if os.name == "nt":
            instante = time.perf_counter()

            def sonar_windows():
                try:
                    import winsound

                    winsound.Beep(1200, 180)
                except Exception as error:
                    print(f"[VOZ] No se pudo emitir el tono de inicio: {error}")

            threading.Thread(target=sonar_windows, daemon=True).start()
            return instante

        print("\a", end="", flush=True)
        return time.perf_counter()

    def cerrar(self):
        if self.disponible:
            self.cola.put(None)


def evaluar_preparacion_sts(landmarks, angulo_rodilla_actual):
    """Valida las condiciones mínimas para iniciar Sit-to-Stand."""

    if landmarks is None or np.isnan(angulo_rodilla_actual):
        return False, "No se detecta a la persona", "persona_no_detectada"

    indices_lado = {
        "izquierdo": [
            mp_pose.PoseLandmark.LEFT_SHOULDER.value,
            mp_pose.PoseLandmark.LEFT_HIP.value,
            mp_pose.PoseLandmark.LEFT_KNEE.value,
            mp_pose.PoseLandmark.LEFT_ANKLE.value,
        ],
        "derecho": [
            mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
            mp_pose.PoseLandmark.RIGHT_HIP.value,
            mp_pose.PoseLandmark.RIGHT_KNEE.value,
            mp_pose.PoseLandmark.RIGHT_ANKLE.value,
        ],
    }

    visibilidad_lados = {
        nombre: np.mean([landmarks[i].visibility for i in indices])
        for nombre, indices in indices_lado.items()
    }
    mejor_lado = max(visibilidad_lados, key=visibilidad_lados.get)
    puntos_lado = [landmarks[i] for i in indices_lado[mejor_lado]]

    if min(p.visibility for p in puntos_lado) < 0.45:
        return (
            False,
            "Gire un poco: hombro, cadera, rodilla y tobillo deben verse",
            "landmarks_ocultos",
        )

    nariz = landmarks[mp_pose.PoseLandmark.NOSE.value]
    puntos_encuadre = puntos_lado + [nariz]
    fuera_de_cuadro = any(
        p.x < 0.03 or p.x > 0.97 or p.y < 0.02 or p.y > 0.98
        for p in puntos_encuadre
    )

    if fuera_de_cuadro:
        return False, "Alejese: el cuerpo completo debe verse", "cuerpo_incompleto"

    altura_visible = max(p.y for p in puntos_encuadre) - min(
        p.y for p in puntos_encuadre
    )
    if altura_visible < 0.42:
        return False, "Acerquese un poco a la camara", "persona_muy_lejos"

    hombro_izq = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    hombro_der = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    cadera_izq = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    cadera_der = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

    centro_hombros = np.array([
        (hombro_izq.x + hombro_der.x) / 2,
        (hombro_izq.y + hombro_der.y) / 2,
    ])
    centro_caderas = np.array([
        (cadera_izq.x + cadera_der.x) / 2,
        (cadera_izq.y + cadera_der.y) / 2,
    ])
    longitud_tronco = max(np.linalg.norm(centro_hombros - centro_caderas), 0.05)
    apertura_frontal = (
        abs(hombro_izq.x - hombro_der.x) + abs(cadera_izq.x - cadera_der.x)
    ) / (2 * longitud_tronco)

    # MediaPipe conserva cierta separación entre hombros incluso de perfil.
    # El límite anterior (0.65) bloqueaba posturas laterales válidas.
    if apertura_frontal > 1.05:
        return False, "Coloquese de lado frente a la camara", "vista_no_lateral"

    if angulo_rodilla_actual > 135:
        return False, "Sientese completamente para comenzar", "no_sentado"

    if angulo_rodilla_actual < 50:
        return False, "Ajuste los pies y la posicion en la silla", "sentado_inestable"

    return True, "Posicion correcta. Mantengase asi", "posicion_correcta"


def dibujar_mensaje_guia(frame, mensaje, valido=False, cuenta_regresiva=None):
    """Dibuja un panel legible para el asistente de posicionamiento."""

    alto, ancho = frame.shape[:2]
    color = (40, 170, 40) if valido else (35, 80, 210)
    cv2.rectangle(frame, (15, alto - 135), (ancho - 15, alto - 55), (20, 20, 20), -1)
    cv2.rectangle(frame, (15, alto - 135), (ancho - 15, alto - 55), color, 3)

    texto = mensaje
    if len(texto) > 62:
        corte = texto.rfind(" ", 0, 62)
        corte = 62 if corte == -1 else corte
        lineas = [texto[:corte], texto[corte + 1:]]
    else:
        lineas = [texto]

    y = alto - 105
    for linea in lineas:
        cv2.putText(
            frame,
            linea,
            (30, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2,
        )
        y += 28

    if cuenta_regresiva is not None:
        texto_cuenta = str(cuenta_regresiva)
        escala = 3.0
        grosor = 6
        (w_texto, h_texto), _ = cv2.getTextSize(
            texto_cuenta, cv2.FONT_HERSHEY_SIMPLEX, escala, grosor
        )
        cv2.putText(
            frame,
            texto_cuenta,
            ((ancho - w_texto) // 2, (alto + h_texto) // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            escala,
            (0, 255, 255),
            grosor,
        )


# =========================
# ABRIR VIDEO / CÁMARA
# =========================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise FileNotFoundError(f"No se pudo abrir el video/cámara: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)

if fps == 0:
    fps = 30

# El VideoWriter se crea después de leer el primer frame válido.
# Esto evita errores cuando OpenCV rota el video y cambian ancho/alto.
out = None
out_original = None

datos = []
timeline_video = []
eventos_camara_local = []
frame_id = 0
video_frame_id = 0

# =========================
# CONTROL DE PRUEBA EN VIVO
# =========================

modo_en_vivo = modo == "2"

prueba_iniciada = False
prueba_finalizada = False

tiempo_inicio_prueba = None
tiempo_fin_prueba = None
tiempo_prueba = 0.0

estado_prueba = "Esperando inicio"

# =========================
# CONTROL EQUILIBRIO SPPB EN VIVO
# =========================

posiciones_equilibrio = [
    "paralelo",
    "semitandem",
    "tandem"
]

nombres_posiciones = {
    "paralelo": "Paralelo / pies juntos",
    "semitandem": "Semi-tandem",
    "tandem": "Tandem"
}

indice_posicion_equilibrio = 0
tiempo_inicio_posicion = None
tiempo_posicion_actual = 0.0

tiempos_equilibrio = {
    "paralelo": 0.0,
    "semitandem": 0.0,
    "tandem": 0.0
}

posicion_equilibrio_actual = ""
fase_lista_para_iniciar = True

# =========================
# CONTROL SIT-TO-STAND EN VIVO
# =========================

sts_repeticiones_objetivo = REPETICIONES_OBJETIVO
sts_repeticiones_en_vivo = 0
sts_estado_postura = "Esperando sentado"
sts_listo_para_contar = False
sts_ultimo_tiempo_rep = -999.0
sts_tiempos_reps = []
sts_ultima_postura_estable = "Sentado"
sts_ciclo_de_pie_confirmado = False
sts_candidato_de_pie_desde = None
sts_candidato_sentado_desde = None
sts_confirmacion_postura_s = CONFIRMACION_POSTURA_S
sts_evento_de_pie_frame = 0
sts_evento_repeticion_frame = 0

# Umbrales prácticos para detectar sentado/de pie con ángulo de rodilla.
# Puedes ajustarlos si tu cámara o postura cambia mucho.
sts_umbral_sentado = UMBRAL_SENTADO_BASE_GRADOS
sts_umbral_de_pie = UMBRAL_DE_PIE_MINIMO_GRADOS
sts_cooldown_s = 0.8

# Asistente guiado de la prueba en vivo.
sts_fase_guia = "posicionamiento"
sts_mensaje_guia = "Coloquese de lado, sientese y muestre el cuerpo completo"
sts_clave_guia = "instruccion_inicial"
sts_posicion_valida = False
sts_inicio_estabilidad = None
sts_estabilidad_requerida_s = ESTABILIDAD_INICIAL_S
sts_estabilidad_anuncio_s = ESTABILIDAD_ANUNCIO_S
sts_posicion_correcta_anunciada = False
sts_clave_voz_candidata = None
sts_inicio_voz_candidata = None
sts_retraso_correccion_voz_s = 2.0
sts_intervalo_correccion_voz_s = 15.0
sts_inicio_cuenta_regresiva = None
sts_cuenta_visible = None
sts_ultimo_numero_hablado = None
sts_historial_angulos_sentado = deque(maxlen=max(15, int(fps)))
sts_angulo_base_sentado = None
sts_umbral_sentado_adaptativo = sts_umbral_sentado
sts_umbral_de_pie_adaptativo = sts_umbral_de_pie
sts_control_calidad = "Pendiente"
sts_tiempo_comando_inicio = None
sts_tiempo_movimiento_inicio = None
sts_tiempo_clinico_final = np.nan
sts_tiempo_efectivo_final = np.nan
sts_inicio_subida_marcado = False
sts_inicio_descenso_marcado = False

voz_habilitada = os.environ.get("SPPB_VOICE", "1") == "1"
asistente_voz = AsistenteVoz(
    habilitado=modo_en_vivo and ejercicio == "sit_to_stand" and voz_habilitada
)

guardar_frame_final_sts = False

# El IMU es un módulo opcional. Si está desactivado, todo el flujo de cámara
# permanece igual que en la versión estable.
imu_habilitada = (
    os.environ.get("SPPB_IMU", "0") == "1"
    and modo_en_vivo
    and ejercicio == "sit_to_stand"
)
imu_l5 = IMULumbar() if imu_habilitada else None
monitor_imu = MonitorIMUEnVivo() if imu_habilitada else None
imu_preparando = False
imu_preparado = False
imu_error = ""
imu_proximo_reintento = 0.0


def preparar_imu_en_segundo_plano():
    global imu_preparando, imu_preparado, imu_error, imu_proximo_reintento
    if imu_l5 is None:
        return
    imu_preparando = True
    imu_error = ""
    try:
        imu_l5.preparar(
            paciente_id,
            trial_id,
            VERSION_ALGORITMO,
        )
        imu_preparado = True
    except ErrorIMU as exc:
        imu_preparado = False
        imu_error = str(exc)
        imu_proximo_reintento = time.perf_counter() + 2.0
    finally:
        imu_preparando = False


def marcar_imu(etiqueta):
    if imu_l5 is None or not imu_l5.grabando:
        return
    try:
        imu_l5.marcar(etiqueta)
    except ErrorIMU as exc:
        # La cámara no se detiene si se pierde un paquete o la red IMU.
        global imu_error
        imu_error = str(exc)


def registrar_evento_camara(etiqueta, instante_sistema=None):
    """Registra una marca local aunque el IMU esté desactivado."""
    instante = (
        time.perf_counter()
        if instante_sistema is None
        else float(instante_sistema)
    )
    if any(fila["evento"] == etiqueta for fila in eventos_camara_local):
        return
    if etiqueta == "GO":
        tiempo_desde_go = 0.0
    elif sts_tiempo_comando_inicio is None:
        tiempo_desde_go = np.nan
    else:
        tiempo_desde_go = instante - sts_tiempo_comando_inicio

    eventos_camara_local.append(
        {
            "evento": etiqueta,
            "tiempo_desde_go_s": tiempo_desde_go,
            "tiempo_monotonic_s": instante,
            "frame_datos": frame_id,
            "frame_video": video_frame_id,
            "angulo_rodilla_grados": angulo_rodilla,
            "angulo_tronco_grados": angulo_tronco,
            "fuente": "camara_umbral_estable",
            "version_algoritmo": VERSION_ALGORITMO,
        }
    )
    marcar_imu(f"camera_{etiqueta}")


if modo_en_vivo:
    print("\nMODO CÁMARA EN VIVO")
    print("Controles:")
    print("i = iniciar prueba o iniciar posición actual")
    print("f = finalizar prueba")
    print("n = siguiente posición de equilibrio")
    print("q = salir")
    if ejercicio == "equilibrio":
        print("\nEquilibrio SPPB:")
        print("1. Paralelo / pies juntos")
        print("2. Semi-tándem")
        print("3. Tándem")
        print("Cada posición se cronometra hasta 10 segundos.")
    if ejercicio == "sit_to_stand":
        print("\nSit-to-Stand SPPB:")
        print("- El paciente inicia sentado.")
        print("- El asistente verificará la posición inicial.")
        print("- El inicio es automático después de la cuenta regresiva.")
        print("- El sistema cuenta automáticamente 5 levantadas.")
        print("- Al llegar a 5/5, finaliza automáticamente.")
        print("- Presiona r para reiniciar el asistente.")
        if voz_habilitada:
            print("- Instrucciones por voz activadas.")
        if imu_habilitada:
            print("- IMU lumbar L5 activado.")
            print("- Conecte la computadora a la red ESP32_SPPB.")
            try:
                imu_l5.iniciar_receptor()
            except ErrorIMU as exc:
                imu_error = str(exc)
        asistente_voz.decir(
            "instruccion_inicial",
            "Colóquese de lado frente a la cámara. "
            "Siéntese, apoye los pies y cruce los brazos sobre el pecho. "
            "Cuando escuche el tono, levántese y siéntese cinco veces "
            "lo más rápido posible.",
            repetir_despues_s=10.0,
        )
else:
    prueba_iniciada = True
    estado_prueba = "Video grabado"

# =========================
# PROCESAMIENTO DEL VIDEO
# =========================

with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # =========================
        # ORIENTACIÓN DEL VIDEO
        # =========================
        # En cámara en vivo NO se rota automáticamente.
        # En video grabado sí se corrige si viene horizontal.
        if modo == "1":
            if ejercicio == "marcha":
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif ejercicio not in ["equilibrio"]:
                if frame.shape[1] > frame.shape[0]:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        frame_original = frame.copy()

        # Crear VideoWriter con el tamaño real del frame ya corregido
        if out is None:
            height, width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(
                OUTPUT_VIDEO,
                fourcc,
                fps,
                (width, height)
            )
            out_original = cv2.VideoWriter(
                OUTPUT_VIDEO_ORIGINAL,
                fourcc,
                fps,
                (width, height),
            )

        # =========================
        # TIEMPO
        # =========================

        tiempo_sistema = time.perf_counter()

        if modo_en_vivo:
            if prueba_iniciada and not prueba_finalizada:
                tiempo_prueba = tiempo_sistema - tiempo_inicio_prueba
            elif prueba_finalizada and tiempo_inicio_prueba is not None and tiempo_fin_prueba is not None:
                tiempo_prueba = tiempo_fin_prueba - tiempo_inicio_prueba
            else:
                tiempo_prueba = 0.0

            tiempo = tiempo_prueba
        else:
            tiempo = frame_id / fps

        guardar_frame_final_sts = False
        sts_evento_de_pie_frame = 0
        sts_evento_repeticion_frame = 0

        # =========================
        # CONTROL AUTOMÁTICO DE EQUILIBRIO EN VIVO
        # =========================

        if modo_en_vivo and ejercicio == "equilibrio":

            posicion_equilibrio_actual = posiciones_equilibrio[indice_posicion_equilibrio]

            if prueba_iniciada and not prueba_finalizada and tiempo_inicio_posicion is not None:

                tiempo_posicion_actual = tiempo_sistema - tiempo_inicio_posicion

                if tiempo_posicion_actual >= 10.0:
                    tiempo_posicion_actual = 10.0
                    tiempos_equilibrio[posicion_equilibrio_actual] = 10.0

                    # Pausar para que el evaluador cambie la posición.
                    if indice_posicion_equilibrio < len(posiciones_equilibrio) - 1:
                        indice_posicion_equilibrio += 1
                        posicion_equilibrio_actual = posiciones_equilibrio[indice_posicion_equilibrio]
                        prueba_iniciada = False
                        fase_lista_para_iniciar = True
                        tiempo_inicio_posicion = None
                        tiempo_posicion_actual = 0.0
                        estado_prueba = (
                            "Cambie a "
                            + nombres_posiciones[posicion_equilibrio_actual]
                            + " y presione i"
                        )
                    else:
                        prueba_finalizada = True
                        tiempo_fin_prueba = tiempo_sistema
                        estado_prueba = "Equilibrio finalizado"
                else:
                    tiempos_equilibrio[posicion_equilibrio_actual] = tiempo_posicion_actual

        imagen_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultados = pose.process(imagen_rgb)

        angulo_rodilla = np.nan
        angulo_tronco = np.nan
        inclinacion_tronco_vertical = np.nan
        clasificacion_frame = "No detectado"
        centro_cadera_x = np.nan
        centro_cadera_y = np.nan
        centro_hombros_x = np.nan
        centro_hombros_y = np.nan
        tobillo_izq_x = np.nan
        tobillo_izq_y = np.nan
        tobillo_der_x = np.nan
        tobillo_der_y = np.nan
        distancia_tobillos = np.nan
        landmarks_actuales = None

        if resultados.pose_landmarks:
            lm = resultados.pose_landmarks.landmark
            landmarks_actuales = lm

            # =========================
            # LANDMARKS DEL LADO MÁS VISIBLE
            # =========================

            indices_izq = [
                mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                mp_pose.PoseLandmark.LEFT_HIP.value,
                mp_pose.PoseLandmark.LEFT_KNEE.value,
                mp_pose.PoseLandmark.LEFT_ANKLE.value,
            ]
            indices_der = [
                mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                mp_pose.PoseLandmark.RIGHT_HIP.value,
                mp_pose.PoseLandmark.RIGHT_KNEE.value,
                mp_pose.PoseLandmark.RIGHT_ANKLE.value,
            ]

            visibilidad_izq = np.mean([lm[i].visibility for i in indices_izq])
            visibilidad_der = np.mean([lm[i].visibility for i in indices_der])
            indices_activos = indices_izq if visibilidad_izq >= visibilidad_der else indices_der

            hombro = [lm[indices_activos[0]].x, lm[indices_activos[0]].y]
            cadera = [lm[indices_activos[1]].x, lm[indices_activos[1]].y]
            rodilla = [lm[indices_activos[2]].x, lm[indices_activos[2]].y]
            tobillo = [lm[indices_activos[3]].x, lm[indices_activos[3]].y]

            cadera_izq = [
                lm[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                lm[mp_pose.PoseLandmark.LEFT_HIP.value].y
            ]

            cadera_der = [
                lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y
            ]

            hombro_izq = [
                lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
            ]

            hombro_der = [
                lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
            ]

            centro_cadera_x = (cadera_izq[0] + cadera_der[0]) / 2
            centro_cadera_y = (cadera_izq[1] + cadera_der[1]) / 2

            centro_hombros_x = (hombro_izq[0] + hombro_der[0]) / 2
            centro_hombros_y = (hombro_izq[1] + hombro_der[1]) / 2

            tobillo_izq = [
                lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].y
            ]

            tobillo_der = [
                lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y
            ]

            distancia_tobillos = abs(tobillo_izq[0] - tobillo_der[0])
            tobillo_izq_x = tobillo_izq[0]
            tobillo_izq_y = tobillo_izq[1]
            tobillo_der_x = tobillo_der[0]
            tobillo_der_y = tobillo_der[1]

            # =========================
            # CÁLCULO DE ÁNGULOS
            # =========================

            angulo_rodilla = calcular_angulo(cadera, rodilla, tobillo)
            angulo_tronco = calcular_angulo(hombro, cadera, rodilla)
            vector_tronco = np.asarray(hombro) - np.asarray(cadera)
            inclinacion_tronco_vertical = np.degrees(
                np.arctan2(
                    abs(float(vector_tronco[0])),
                    max(abs(float(vector_tronco[1])), 1e-6),
                )
            )

            # =========================
            # CLASIFICACIÓN POR FRAME
            # =========================

            if angulo_rodilla < 100:
                clasificacion_frame = "Rodilla flexionada"
            elif angulo_rodilla >= 150 and inclinacion_tronco_vertical <= 30:
                clasificacion_frame = "Extension de pie"
            elif (
                100 <= angulo_rodilla < 150
                and inclinacion_tronco_vertical > 45
            ):
                clasificacion_frame = "Inclinacion de tronco elevada"
            else:
                clasificacion_frame = "Transicion"

            # =========================
            # DIBUJAR LANDMARKS
            # =========================

            mp_drawing.draw_landmarks(
                frame,
                resultados.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

        # =========================
        # CONTROL AUTOMÁTICO SIT-TO-STAND EN VIVO
        # =========================

        if modo_en_vivo and ejercicio == "sit_to_stand":

            # 1) Verificación de preparación y encuadre.
            if sts_fase_guia == "posicionamiento":
                (
                    sts_posicion_valida,
                    sts_mensaje_guia,
                    sts_clave_guia,
                ) = evaluar_preparacion_sts(landmarks_actuales, angulo_rodilla)

                estado_prueba = "Verificando posicion inicial"
                sts_control_calidad = (
                    "Valido" if sts_posicion_valida else "Requiere ajuste"
                )

                if sts_posicion_valida:
                    sts_historial_angulos_sentado.append(angulo_rodilla)

                    if sts_inicio_estabilidad is None:
                        sts_inicio_estabilidad = tiempo_sistema

                    tiempo_estable = tiempo_sistema - sts_inicio_estabilidad

                    # La confirmación se anuncia únicamente cuando la postura
                    # lleva un tiempo estable. Así un frame aislado no dispara
                    # una instrucción prematura.
                    if (
                        tiempo_estable >= sts_estabilidad_anuncio_s
                        and not sts_posicion_correcta_anunciada
                    ):
                        sts_posicion_correcta_anunciada = True
                        asistente_voz.decir(
                            "posicion_correcta",
                            "Posición correcta. Manténgase así.",
                            repetir_despues_s=30.0,
                            conservar_pendientes=True,
                        )

                    sts_clave_voz_candidata = None
                    sts_inicio_voz_candidata = None

                    # El sensor se calibra únicamente cuando la persona ya
                    # permanece sentada y quieta. La cámara sigue actualizando
                    # la imagen mientras la calibración ocurre en otro hilo.
                    if (
                        imu_habilitada
                        and not imu_preparado
                        and not imu_preparando
                        and tiempo_estable >= 1.0
                        and time.perf_counter() >= imu_proximo_reintento
                    ):
                        threading.Thread(
                            target=preparar_imu_en_segundo_plano,
                            daemon=True,
                        ).start()

                    imu_listo_para_cuenta = (
                        not imu_habilitada or imu_preparado
                    )

                    if imu_habilitada and not imu_preparado:
                        if imu_preparando:
                            sts_mensaje_guia = (
                                "Mantengase quieto: calibrando IMU lumbar"
                            )
                            estado_prueba = "Calibrando IMU lumbar L5"
                        elif imu_error:
                            sts_mensaje_guia = (
                                "No se pudo calibrar el IMU. Mantengase quieto "
                                "para reintentar"
                            )
                            estado_prueba = f"IMU: {imu_error}"

                    if (
                        tiempo_estable >= sts_estabilidad_requerida_s
                        and imu_listo_para_cuenta
                        and not asistente_voz.tiene_audio_pendiente()
                    ):
                        sts_fase_guia = "cuenta_regresiva"
                        sts_inicio_cuenta_regresiva = tiempo_sistema
                        sts_ultimo_numero_hablado = None
                        sts_angulo_base_sentado = float(
                            np.median(sts_historial_angulos_sentado)
                        )
                        sts_umbral_sentado_adaptativo = min(
                            140.0,
                            sts_angulo_base_sentado + 12.0,
                        )
                        # Nunca aceptar como "de pie" una flexión parcial.
                        # El valor adaptativo conserva tolerancia entre personas,
                        # pero exige al menos 150° de extensión de rodilla.
                        sts_umbral_de_pie_adaptativo = max(
                            150.0,
                            min(160.0, sts_angulo_base_sentado + 40.0),
                        )
                        estado_prueba = "Preparado"
                else:
                    sts_inicio_estabilidad = None
                    sts_posicion_correcta_anunciada = False
                    sts_historial_angulos_sentado.clear()

                    # El texto visual cambia inmediatamente, pero la voz espera
                    # a que la misma corrección permanezca estable. Esto evita
                    # repetir mensajes cuando MediaPipe alterna entre estados
                    # mientras la persona todavía se está colocando.
                    if sts_clave_guia != sts_clave_voz_candidata:
                        sts_clave_voz_candidata = sts_clave_guia
                        sts_inicio_voz_candidata = tiempo_sistema
                    elif (
                        sts_inicio_voz_candidata is not None
                        and tiempo_sistema - sts_inicio_voz_candidata
                        >= sts_retraso_correccion_voz_s
                    ):
                        asistente_voz.decir(
                            sts_clave_guia,
                            sts_mensaje_guia,
                            repetir_despues_s=sts_intervalo_correccion_voz_s,
                        )

            # 2) Cuenta regresiva; se cancela si cambia la postura.
            elif sts_fase_guia == "cuenta_regresiva":
                posicion_aun_valida, mensaje_actual, clave_actual = (
                    evaluar_preparacion_sts(landmarks_actuales, angulo_rodilla)
                )

                if not posicion_aun_valida:
                    sts_fase_guia = "posicionamiento"
                    sts_inicio_estabilidad = None
                    sts_inicio_cuenta_regresiva = None
                    sts_cuenta_visible = None
                    sts_mensaje_guia = mensaje_actual
                    sts_clave_guia = clave_actual
                    sts_posicion_correcta_anunciada = False
                    estado_prueba = "Cuenta cancelada: ajuste la posicion"
                    asistente_voz.decir(
                        "cuenta_cancelada",
                        "Espere. Ajuste nuevamente la posición inicial.",
                        repetir_despues_s=5.0,
                    )
                else:
                    transcurrido_cuenta = tiempo_sistema - sts_inicio_cuenta_regresiva
                    sts_cuenta_visible = max(1, 3 - int(transcurrido_cuenta))
                    sts_mensaje_guia = "Mantenga la posicion. Inicio automatico"
                    sts_posicion_valida = True

                    if transcurrido_cuenta >= CUENTA_REGRESIVA_S:
                        if imu_l5 is not None:
                            try:
                                imu_l5.iniciar()
                            except ErrorIMU as exc:
                                imu_error = str(exc)
                                imu_preparado = False
                                sts_fase_guia = "posicionamiento"
                                sts_inicio_estabilidad = None
                                sts_inicio_cuenta_regresiva = None
                                sts_cuenta_visible = None
                                sts_mensaje_guia = (
                                    "No se pudo iniciar el IMU. "
                                    "Mantengase quieto para reintentar"
                                )
                                estado_prueba = f"IMU: {imu_error}"
                                continue
                        sts_tiempo_comando_inicio = (
                            asistente_voz.emitir_senal_inicio()
                        )
                        tiempo_inicio_prueba = sts_tiempo_comando_inicio
                        tiempo_fin_prueba = None
                        tiempo_prueba = 0.0
                        prueba_iniciada = True
                        prueba_finalizada = False
                        registrar_evento_camara(
                            "GO",
                            sts_tiempo_comando_inicio,
                        )
                        sts_fase_guia = "esperando_movimiento"
                        sts_cuenta_visible = None
                        sts_mensaje_guia = (
                            "COMIENCE: levantese y sientese cinco veces"
                        )
                        estado_prueba = "Esperando primer movimiento"

            # 3) El tiempo clínico ya corre desde GO; aquí se marca MO1.
            elif sts_fase_guia == "esperando_movimiento":
                sts_posicion_valida = True
                movimiento_iniciado = (
                    not np.isnan(angulo_rodilla)
                    and sts_angulo_base_sentado is not None
                    and angulo_rodilla
                    >= sts_angulo_base_sentado
                    + HISTERESIS_INICIO_MOVIMIENTO_GRADOS
                )

                if movimiento_iniciado:
                    sts_tiempo_movimiento_inicio = tiempo_sistema
                    registrar_evento_camara("MO1", tiempo_sistema)
                    sts_fase_guia = "evaluacion"
                    sts_repeticiones_en_vivo = 0
                    sts_estado_postura = "Subiendo"
                    sts_ultima_postura_estable = "Sentado"
                    sts_listo_para_contar = False
                    sts_ultimo_tiempo_rep = -999.0
                    sts_tiempos_reps = []
                    sts_ciclo_de_pie_confirmado = False
                    sts_candidato_de_pie_desde = None
                    sts_candidato_sentado_desde = None
                    sts_inicio_subida_marcado = True
                    sts_inicio_descenso_marcado = False
                    sts_control_calidad = "Prueba iniciada correctamente"
                    sts_mensaje_guia = "Prueba en curso"
                    estado_prueba = "Sit-to-Stand en curso"

            # 4) Conteo automático durante la evaluación.
            #
            # Una repetición válida debe seguir esta secuencia:
            # sentado -> de pie confirmado -> sentado confirmado.
            # En el protocolo Five Times Sit-to-Stand, la quinta termina al
            # alcanzar y mantener la posición completamente erguida.
            elif sts_fase_guia == "evaluacion" and not prueba_finalizada:
                sts_posicion_valida = True

                if np.isnan(angulo_rodilla):
                    sts_candidato_de_pie_desde = None
                    sts_candidato_sentado_desde = None
                    sts_mensaje_guia = "Vuelva al encuadre: no se detecta la postura"
                    asistente_voz.decir(
                        "postura_perdida",
                        "No se detecta bien la postura. Manténgase dentro de la imagen.",
                        repetir_despues_s=5.0,
                    )
                elif angulo_rodilla <= sts_umbral_sentado_adaptativo:
                    sts_estado_postura = "Sentado"
                    sts_candidato_de_pie_desde = None

                    if sts_candidato_sentado_desde is None:
                        sts_candidato_sentado_desde = tiempo_sistema

                    sentado_estable = (
                        tiempo_sistema - sts_candidato_sentado_desde
                        >= sts_confirmacion_postura_s
                    )

                    if sentado_estable and sts_ciclo_de_pie_confirmado:
                        # Las repeticiones 1 a 4 se consolidan al regresar a
                        # sentado. La quinta ya habrá finalizado de pie.
                        sts_repeticiones_en_vivo += 1
                        sts_evento_repeticion_frame = (
                            sts_repeticiones_en_vivo
                        )
                        sts_ultimo_tiempo_rep = tiempo_prueba
                        sts_tiempos_reps.append(tiempo_prueba)
                        sts_ciclo_de_pie_confirmado = False
                        sts_ultima_postura_estable = "Sentado"
                        registrar_evento_camara(
                            f"S{sts_repeticiones_en_vivo}",
                            tiempo_sistema,
                        )
                        sts_inicio_subida_marcado = False
                        sts_inicio_descenso_marcado = False
                        sts_mensaje_guia = (
                            f"Repeticion {sts_repeticiones_en_vivo} "
                            f"de {sts_repeticiones_objetivo}"
                        )
                        estado_prueba = (
                            f"Sit-to-Stand {sts_repeticiones_en_vivo}/"
                            f"{sts_repeticiones_objetivo}"
                        )
                        asistente_voz.decir(
                            f"repeticion_{sts_repeticiones_en_vivo}",
                            ["Uno.", "Dos.", "Tres.", "Cuatro."][
                                sts_repeticiones_en_vivo - 1
                            ],
                            repetir_despues_s=30.0,
                            conservar_pendientes=True,
                        )
                    elif sentado_estable:
                        sts_ultima_postura_estable = "Sentado"
                        sts_mensaje_guia = (
                            "Levantese completamente para iniciar "
                            "la siguiente repeticion"
                        )
                    else:
                        sts_mensaje_guia = "Sientese completamente"

                elif angulo_rodilla >= sts_umbral_de_pie_adaptativo:
                    sts_estado_postura = "De pie"
                    sts_candidato_sentado_desde = None

                    if sts_candidato_de_pie_desde is None:
                        sts_candidato_de_pie_desde = tiempo_sistema

                    de_pie_estable = (
                        tiempo_sistema - sts_candidato_de_pie_desde
                        >= sts_confirmacion_postura_s
                    )

                    if de_pie_estable and not sts_ciclo_de_pie_confirmado:
                        numero_ciclo = sts_repeticiones_en_vivo + 1
                        sts_ciclo_de_pie_confirmado = True
                        sts_evento_de_pie_frame = numero_ciclo
                        sts_ultima_postura_estable = "De pie"
                        registrar_evento_camara(
                            f"R{numero_ciclo}",
                            tiempo_sistema,
                        )
                        sts_inicio_descenso_marcado = False

                        if numero_ciclo < sts_repeticiones_objetivo:
                            sts_mensaje_guia = (
                                "Posicion de pie confirmada. "
                                "Sientese completamente"
                            )
                        else:
                            # El cronómetro SPPB se detiene cuando se completa
                            # la extensión de la quinta levantada.
                            sts_repeticiones_en_vivo = numero_ciclo
                            sts_evento_repeticion_frame = numero_ciclo
                            sts_ultimo_tiempo_rep = tiempo_prueba
                            sts_tiempos_reps.append(tiempo_prueba)
                            prueba_finalizada = True
                            prueba_iniciada = False
                            sts_fase_guia = "finalizada"
                            tiempo_fin_prueba = tiempo_sistema
                            tiempo_prueba = (
                                tiempo_fin_prueba - tiempo_inicio_prueba
                            )
                            sts_tiempo_clinico_final = (
                                tiempo_fin_prueba - sts_tiempo_comando_inicio
                                if sts_tiempo_comando_inicio is not None
                                else np.nan
                            )
                            sts_tiempo_efectivo_final = (
                                tiempo_fin_prueba
                                - sts_tiempo_movimiento_inicio
                                if sts_tiempo_movimiento_inicio is not None
                                else np.nan
                            )
                            estado_prueba = "Sit-to-Stand finalizado"
                            sts_mensaje_guia = (
                                "Prueba finalizada. Puede descansar"
                            )
                            sts_control_calidad = "Completada"
                            guardar_frame_final_sts = True
                            asistente_voz.decir(
                                "repeticion_5",
                                "Cinco. Cinco repeticiones completadas. "
                                "Prueba finalizada. Puede descansar.",
                                repetir_despues_s=30.0,
                                conservar_pendientes=True,
                            )
                            if imu_l5 is not None:
                                try:
                                    imu_l5.detener()
                                except ErrorIMU as exc:
                                    imu_error = str(exc)
                    elif not de_pie_estable:
                        sts_mensaje_guia = (
                            "Mantengase completamente de pie"
                        )
                else:
                    sts_candidato_de_pie_desde = None
                    sts_candidato_sentado_desde = None
                    if sts_ultima_postura_estable == "Sentado":
                        sts_estado_postura = "Subiendo"
                        sts_mensaje_guia = "Levantese completamente"
                        if (
                            not sts_inicio_subida_marcado
                            and angulo_rodilla
                            >= sts_umbral_sentado_adaptativo
                            + HISTERESIS_INICIO_MOVIMIENTO_GRADOS
                        ):
                            numero_subida = sts_repeticiones_en_vivo + 1
                            registrar_evento_camara(
                                f"MO{numero_subida}",
                                tiempo_sistema,
                            )
                            sts_inicio_subida_marcado = True
                    else:
                        sts_estado_postura = "Bajando"
                        sts_mensaje_guia = "Sientese completamente"
                        if (
                            not sts_inicio_descenso_marcado
                            and sts_repeticiones_en_vivo < 4
                            and angulo_rodilla
                            <= sts_umbral_de_pie_adaptativo
                            - HISTERESIS_INICIO_MOVIMIENTO_GRADOS
                        ):
                            numero_descenso = sts_repeticiones_en_vivo + 1
                            registrar_evento_camara(
                                f"D{numero_descenso}",
                                tiempo_sistema,
                            )
                            sts_inicio_descenso_marcado = True

            elif sts_fase_guia == "finalizada":
                sts_posicion_valida = True
                sts_mensaje_guia = "Prueba finalizada. Puede descansar"

        # =========================
        # TEXTOS EN PANTALLA
        # =========================

        y_texto = 35

        cv2.putText(
            frame,
            f"SPPB-VISION | {ejercicio}",
            (30, y_texto),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2
        )

        y_texto += 35

        cv2.putText(
            frame,
            f"Estado: {estado_prueba}",
            (30, y_texto),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )

        y_texto += 35

        cv2.putText(
            frame,
            f"Tiempo prueba: {tiempo_prueba:.2f} s" if modo_en_vivo else f"Tiempo video: {tiempo:.2f} s",
            (30, y_texto),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

        y_texto += 35

        if imu_habilitada:
            if imu_l5 is not None and imu_l5.grabando:
                estado_imu_visible = "IMU L5: grabando"
                color_imu = (0, 255, 0)
            elif imu_preparado:
                estado_imu_visible = "IMU L5: calibrado"
                color_imu = (0, 255, 255)
            elif imu_preparando:
                estado_imu_visible = "IMU L5: calibrando"
                color_imu = (0, 255, 255)
            elif imu_error:
                estado_imu_visible = "IMU L5: esperando reconexion"
                color_imu = (0, 128, 255)
            else:
                estado_imu_visible = "IMU L5: esperando postura estable"
                color_imu = (0, 255, 255)

            cv2.putText(
                frame,
                estado_imu_visible,
                (30, y_texto),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                color_imu,
                2,
            )
            y_texto += 35

        if ejercicio == "equilibrio":
            posicion_equilibrio_actual = posiciones_equilibrio[indice_posicion_equilibrio]
            nombre_visible = nombres_posiciones[posicion_equilibrio_actual]

            cv2.putText(
                frame,
                f"Posicion: {nombre_visible}",
                (30, y_texto),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2
            )

            y_texto += 35

            cv2.putText(
                frame,
                f"Tiempo posicion: {tiempo_posicion_actual:.2f} s",
                (30, y_texto),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2
            )

            y_texto += 35

            cv2.putText(
                frame,
                (
                    f"P:{tiempos_equilibrio['paralelo']:.1f}s | "
                    f"S:{tiempos_equilibrio['semitandem']:.1f}s | "
                    f"T:{tiempos_equilibrio['tandem']:.1f}s"
                ),
                (30, y_texto),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                2
            )

        else:
            if not np.isnan(angulo_rodilla):
                cv2.putText(
                    frame,
                    f"Rodilla: {angulo_rodilla:.1f} deg",
                    (30, y_texto),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2
                )

                y_texto += 35

            if not np.isnan(angulo_tronco):
                cv2.putText(
                    frame,
                    f"Tronco: {angulo_tronco:.1f} deg",
                    (30, y_texto),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2
                )

        if ejercicio == "sit_to_stand":
            y_texto += 35

            cv2.putText(
                frame,
                f"Reps: {sts_repeticiones_en_vivo}/{sts_repeticiones_objetivo}",
                (30, y_texto),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 0),
                2
            )

            y_texto += 35

            cv2.putText(
                frame,
                f"Postura: {sts_estado_postura}",
                (30, y_texto),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2
            )

            if modo_en_vivo:
                dibujar_mensaje_guia(
                    frame,
                    sts_mensaje_guia,
                    valido=sts_posicion_valida,
                    cuenta_regresiva=sts_cuenta_visible,
                )

        if modo_en_vivo:
            texto_controles = (
                "Inicio automatico | r: reiniciar | f: finalizar | q: salir"
                if ejercicio == "sit_to_stand"
                else "i: iniciar | f: finalizar | n: siguiente | q: salir"
            )
            cv2.putText(
                frame,
                texto_controles,
                (30, frame.shape[0] - 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                2
            )
        else:
            cv2.putText(
                frame,
                "q: salir",
                (30, frame.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

        # =========================
        # GUARDAR DATOS
        # =========================

        timeline_video.append(
            {
                "frame_video": video_frame_id,
                "tiempo_monotonic_s": tiempo_sistema,
                "tiempo_desde_GO_s": (
                    tiempo_sistema - sts_tiempo_comando_inicio
                    if sts_tiempo_comando_inicio is not None
                    else np.nan
                ),
                "pose_detectada": int(resultados.pose_landmarks is not None),
                "fase_guia": (
                    sts_fase_guia if ejercicio == "sit_to_stand" else ""
                ),
                "incluido_en_datos_prueba": int(
                    (prueba_iniciada and not prueba_finalizada)
                    or guardar_frame_final_sts
                ),
            }
        )

        guardar_frame = True

        if modo_en_vivo:
            # En vivo solo se guardan datos cuando la prueba está en curso.
            # En Sit-to-Stand también se guarda el frame exacto donde llega a 5/5.
            guardar_frame = (prueba_iniciada and not prueba_finalizada) or guardar_frame_final_sts

        if guardar_frame:
            datos.append({
                "frame": frame_id,
                "tiempo_s": tiempo,
                "angulo_rodilla": angulo_rodilla,
                "angulo_tronco": angulo_tronco,
                "pose_detectada": int(resultados.pose_landmarks is not None),
                "tiempo_monotonic_s": tiempo_sistema,
                "inclinacion_tronco_vertical": inclinacion_tronco_vertical,
                "centro_cadera_x": centro_cadera_x,
                "centro_cadera_y": centro_cadera_y,
                "centro_hombros_x": centro_hombros_x,
                "centro_hombros_y": centro_hombros_y,
                "clasificacion_frame": clasificacion_frame,
                "tobillo_izq_x": tobillo_izq_x,
                "tobillo_izq_y": tobillo_izq_y,
                "tobillo_der_x": tobillo_der_x,
                "tobillo_der_y": tobillo_der_y,
                "distancia_tobillos": distancia_tobillos,
                "estado_prueba": estado_prueba,
                "posicion_equilibrio": posicion_equilibrio_actual if ejercicio == "equilibrio" else "",
                "tiempo_posicion_s": tiempo_posicion_actual if ejercicio == "equilibrio" else np.nan,
                "sts_repeticion_en_vivo": sts_repeticiones_en_vivo if ejercicio == "sit_to_stand" else np.nan,
                "sts_estado_postura": sts_estado_postura if ejercicio == "sit_to_stand" else "",
                "sts_objetivo_reps": sts_repeticiones_objetivo if ejercicio == "sit_to_stand" else np.nan,
                "sts_fase_guia": sts_fase_guia if ejercicio == "sit_to_stand" else "",
                "sts_control_calidad": sts_control_calidad if ejercicio == "sit_to_stand" else "",
                "sts_mensaje_guia": sts_mensaje_guia if ejercicio == "sit_to_stand" else "",
                "sts_evento_de_pie": sts_evento_de_pie_frame if ejercicio == "sit_to_stand" else 0,
                "sts_evento_repeticion_completa": sts_evento_repeticion_frame if ejercicio == "sit_to_stand" else 0,
                "sts_umbral_sentado": sts_umbral_sentado_adaptativo if ejercicio == "sit_to_stand" else np.nan,
                "sts_umbral_de_pie": sts_umbral_de_pie_adaptativo if ejercicio == "sit_to_stand" else np.nan,
                "sts_tiempo_clinico_s": sts_tiempo_clinico_final if ejercicio == "sit_to_stand" else np.nan,
                "sts_tiempo_efectivo_s": sts_tiempo_efectivo_final if ejercicio == "sit_to_stand" and prueba_finalizada else np.nan,
                "version_sistema": VERSION_SISTEMA,
                "version_algoritmo": VERSION_ALGORITMO,
                "id_ensayo": trial_id,
                "altura_silla_cm": chair_height_cm,
            })

            frame_id += 1

        # Guardar video anotado completo
        out.write(frame)
        out_original.write(frame_original)
        video_frame_id += 1

        # Mostrar como si fuera en vivo
        cv2.imshow("SPPB-VISION | Analisis en vivo", frame)
        if monitor_imu is not None and imu_l5 is not None:
            muestras_monitor, _ = imu_l5.datos()
            panel_imu = monitor_imu.construir(
                muestras_monitor,
                conectado=imu_l5.recibiendo,
                grabando=imu_l5.grabando,
                error=imu_error,
            )
            cv2.imshow("SPPB-VISION | Monitor IMU lumbar L5", panel_imu)

        # =========================
        # TECLAS
        # =========================

        tecla = cv2.waitKey(1) & 0xFF

        if (
            tecla == ord("i")
            and modo_en_vivo
            and ejercicio != "sit_to_stand"
        ):

            if ejercicio == "equilibrio":
                posicion_equilibrio_actual = posiciones_equilibrio[indice_posicion_equilibrio]
                prueba_iniciada = True
                prueba_finalizada = False
                fase_lista_para_iniciar = False

                if tiempo_inicio_prueba is None:
                    tiempo_inicio_prueba = time.perf_counter()

                tiempo_inicio_posicion = time.perf_counter()
                tiempo_posicion_actual = 0.0

                estado_prueba = (
                    "En curso: "
                    + nombres_posiciones[posicion_equilibrio_actual]
                )

            else:
                prueba_iniciada = True
                prueba_finalizada = False
                tiempo_inicio_prueba = time.perf_counter()
                tiempo_fin_prueba = None
                tiempo_prueba = 0.0
                estado_prueba = "Prueba en curso"

        elif (
            tecla == ord("r")
            and modo_en_vivo
            and ejercicio == "sit_to_stand"
        ):
            prueba_iniciada = False
            prueba_finalizada = False
            tiempo_inicio_prueba = None
            tiempo_fin_prueba = None
            tiempo_prueba = 0.0
            sts_repeticiones_en_vivo = 0
            sts_estado_postura = "Esperando sentado"
            sts_listo_para_contar = False
            sts_ultimo_tiempo_rep = -999.0
            sts_tiempos_reps = []
            sts_ultima_postura_estable = "Sentado"
            sts_ciclo_de_pie_confirmado = False
            sts_candidato_de_pie_desde = None
            sts_candidato_sentado_desde = None
            sts_evento_de_pie_frame = 0
            sts_evento_repeticion_frame = 0
            sts_fase_guia = "posicionamiento"
            sts_mensaje_guia = (
                "Coloquese de lado, sientese y muestre el cuerpo completo"
            )
            sts_clave_guia = "instruccion_inicial"
            sts_posicion_valida = False
            sts_inicio_estabilidad = None
            sts_posicion_correcta_anunciada = False
            sts_clave_voz_candidata = None
            sts_inicio_voz_candidata = None
            sts_inicio_cuenta_regresiva = None
            sts_cuenta_visible = None
            sts_ultimo_numero_hablado = None
            sts_angulo_base_sentado = None
            sts_umbral_sentado_adaptativo = sts_umbral_sentado
            sts_umbral_de_pie_adaptativo = sts_umbral_de_pie
            sts_control_calidad = "Pendiente"
            sts_tiempo_comando_inicio = None
            sts_tiempo_movimiento_inicio = None
            sts_tiempo_clinico_final = np.nan
            sts_tiempo_efectivo_final = np.nan
            sts_inicio_subida_marcado = False
            sts_inicio_descenso_marcado = False
            sts_historial_angulos_sentado.clear()
            eventos_camara_local.clear()
            if imu_l5 is not None:
                try:
                    imu_l5.detener()
                    imu_l5.limpiar()
                except ErrorIMU as exc:
                    imu_error = str(exc)
                imu_preparando = False
                imu_preparado = False
                imu_proximo_reintento = time.perf_counter() + 1.0
            if monitor_imu is not None:
                monitor_imu.reiniciar()
            datos = []
            frame_id = 0
            estado_prueba = "Asistente reiniciado"
            asistente_voz.decir(
                "reinicio",
                "Prueba reiniciada. Colóquese de lado, siéntese "
                "y muestre el cuerpo completo.",
                repetir_despues_s=30.0,
            )

        elif tecla == ord("n") and modo_en_vivo and ejercicio == "equilibrio":

            posicion_equilibrio_actual = posiciones_equilibrio[indice_posicion_equilibrio]

            if tiempo_inicio_posicion is not None:
                tiempos_equilibrio[posicion_equilibrio_actual] = min(
                    10.0,
                    tiempo_posicion_actual
                )

            if indice_posicion_equilibrio < len(posiciones_equilibrio) - 1:
                indice_posicion_equilibrio += 1
                posicion_equilibrio_actual = posiciones_equilibrio[indice_posicion_equilibrio]
                prueba_iniciada = False
                tiempo_inicio_posicion = None
                tiempo_posicion_actual = 0.0
                estado_prueba = (
                    "Cambie a "
                    + nombres_posiciones[posicion_equilibrio_actual]
                    + " y presione i"
                )
            else:
                prueba_finalizada = True
                tiempo_fin_prueba = time.perf_counter()
                estado_prueba = "Equilibrio finalizado"

        elif tecla == ord("f") and modo_en_vivo:

            if ejercicio == "equilibrio":
                posicion_equilibrio_actual = posiciones_equilibrio[indice_posicion_equilibrio]

                if tiempo_inicio_posicion is not None:
                    tiempos_equilibrio[posicion_equilibrio_actual] = min(
                        10.0,
                        tiempo_posicion_actual
                    )

            prueba_finalizada = True
            prueba_iniciada = False
            tiempo_fin_prueba = time.perf_counter()
            estado_prueba = "Prueba finalizada"
            if (
                ejercicio == "sit_to_stand"
                and sts_tiempo_comando_inicio is not None
                and sts_repeticiones_en_vivo < sts_repeticiones_objetivo
            ):
                registrar_evento_camara("ABORT", tiempo_fin_prueba)
                sts_control_calidad = "Incompleta: finalización manual"
            if imu_l5 is not None:
                try:
                    imu_l5.detener()
                except ErrorIMU as exc:
                    imu_error = str(exc)

        elif tecla == ord("q"):
            if (
                ejercicio == "sit_to_stand"
                and sts_tiempo_comando_inicio is not None
                and sts_repeticiones_en_vivo < sts_repeticiones_objetivo
            ):
                registrar_evento_camara("ABORT", time.perf_counter())
            if imu_l5 is not None:
                try:
                    imu_l5.detener()
                except ErrorIMU as exc:
                    imu_error = str(exc)
            break

# =========================
# CERRAR VIDEO
# =========================

cap.release()

if out is not None:
    out.release()
if out_original is not None:
    out_original.release()

asistente_voz.cerrar()
cv2.destroyAllWindows()

# Guardado preventivo: los datos de cámara quedan escritos antes del análisis
# IMU y antes de cualquier generación posterior de gráficas o reportes.
df_preliminar = pd.DataFrame(datos)
if not df_preliminar.empty:
    df_preliminar.to_csv(OUTPUT_CSV, index=False)

df_eventos_camara = pd.DataFrame(
    eventos_camara_local,
    columns=[
        "evento",
        "tiempo_desde_go_s",
        "tiempo_monotonic_s",
        "frame_datos",
        "frame_video",
        "angulo_rodilla_grados",
        "angulo_tronco_grados",
        "fuente",
        "version_algoritmo",
    ],
)
df_eventos_camara.to_csv(OUTPUT_EVENTOS_CAMARA, index=False)
df_timeline_video = pd.DataFrame(timeline_video)
df_timeline_video.to_csv(OUTPUT_TIMELINE_VIDEO, index=False)

metadata_ensayo = {
    "id_participante": paciente_id,
    "id_ensayo": trial_id,
    "fecha_hora_local": sello_intento if modo_en_vivo else None,
    "prueba": ejercicio,
    "modo": "camara_en_vivo" if modo_en_vivo else "video_grabado",
    "version_sistema": VERSION_SISTEMA,
    "version_algoritmo": VERSION_ALGORITMO,
    "entorno_ejecucion": {
        "python": platform.python_version(),
        "sistema": platform.platform(),
        "arquitectura": platform.machine(),
        "paquetes": {
            nombre: _version_instalada(nombre)
            for nombre in (
                "mediapipe",
                "numpy",
                "opencv-contrib-python",
                "pandas",
                "matplotlib",
                "scipy",
                "reportlab",
                "Pillow",
            )
        },
    },
    "altura_silla_cm": chair_height_cm,
    "imu_l5_habilitado": imu_habilitada,
    "voz_habilitada": voz_habilitada,
    "repeticiones_objetivo": sts_repeticiones_objetivo,
    "prueba_completada_automaticamente": bool(
        ejercicio == "sit_to_stand"
        and sts_repeticiones_en_vivo == sts_repeticiones_objetivo
        and np.isfinite(sts_tiempo_clinico_final)
    ),
    "tiempo_clinico_s": (
        float(sts_tiempo_clinico_final)
        if np.isfinite(sts_tiempo_clinico_final)
        else None
    ),
    "tiempo_efectivo_s": (
        float(sts_tiempo_efectivo_final)
        if np.isfinite(sts_tiempo_efectivo_final)
        else None
    ),
    "nota": (
        "Resultado parcial: subprueba de silla 0-4. "
        "No representa un puntaje SPPB total."
    ),
}
with open(OUTPUT_METADATA, "w", encoding="utf-8") as archivo_metadata:
    json.dump(metadata_ensayo, archivo_metadata, ensure_ascii=False, indent=2)

# Cerrar y guardar la adquisición inercial, incluso si la prueba de cámara fue
# interrumpida. Esto permite diagnosticar conexión y sincronización.
imu_muestras = []
imu_eventos = []
resultado_imu_sts = None
if imu_l5 is not None:
    try:
        imu_l5.detener()
    except ErrorIMU as exc:
        imu_error = str(exc)
    time.sleep(0.15)
    imu_muestras, imu_eventos = imu_l5.datos()
    cantidad_muestras, cantidad_eventos = imu_l5.guardar(
        OUTPUT_IMU_CSV,
        OUTPUT_IMU_EVENTOS,
    )
    imu_l5.cerrar()
    print(
        f"\nIMU lumbar: {cantidad_muestras} muestras y "
        f"{cantidad_eventos} eventos recibidos."
    )
    if imu_error:
        print(f"Aviso IMU: {imu_error}")

    if ejercicio == "sit_to_stand" and imu_muestras:
        try:
            go_imu_s = 0.0
            for evento_imu in imu_eventos:
                if str(evento_imu.get("evento", "")) == "camera_GO":
                    go_imu_s = float(evento_imu.get("ms_prueba", 0.0)) / 1000.0
                    break
            referencias_locales = []
            for evento_camara in eventos_camara_local:
                referencias_locales.append(
                    {
                        "evento": f"camera_{evento_camara['evento']}",
                        "ms_prueba": (
                            float(evento_camara["tiempo_desde_go_s"]) * 1000.0
                        ),
                    }
                )
            resultado_imu_sts = analizar_imu_sts(
                imu_muestras,
                referencias_locales,
                frecuencia_hz=FRECUENCIA_IMU_HZ,
                tolerancia_s=TOLERANCIA_CONCORDANCIA_S,
                origen_tiempo_s=go_imu_s,
            )
            resultado_imu_sts["detecciones"].to_csv(
                OUTPUT_IMU_DETECCIONES,
                index=False,
            )
            resultado_imu_sts["comparacion"].to_csv(
                OUTPUT_IMU_CONCORDANCIA,
                index=False,
            )
            fusionar_eventos(
                eventos_camara_local,
                resultado_imu_sts["detecciones"],
                tolerancia_s=TOLERANCIA_CONCORDANCIA_S,
            ).to_csv(OUTPUT_EVENTOS_FUSION, index=False)
            resumen_imu = resultado_imu_sts["resumen"].iloc[0]
            print(
                "Detector IMU independiente: "
                f"{int(resumen_imu['levantamientos_imu_detectados'])}/5 "
                "levantamientos."
            )
            if not pd.isna(resumen_imu.get("f1_score", np.nan)):
                print(
                    "Concordancia técnica cámara–IMU: "
                    f"precision={resumen_imu['precision']:.3f}, "
                    f"recall={resumen_imu['recall']:.3f}, "
                    f"F1={resumen_imu['f1_score']:.3f}, "
                    f"MAE={resumen_imu['mae_eventos_s']:.3f} s."
                )
        except (ValueError, RuntimeError) as exc:
            print(f"No se pudo completar el detector IMU: {exc}")

# =========================
# GUARDAR CSV
# =========================

df = pd.DataFrame(datos)

if df.empty:
    print("\nNo se guardaron datos.")
    if modo_en_vivo and ejercicio == "sit_to_stand":
        print(
            "La prueba Sit-to-Stand no llegó a iniciar o fue cerrada "
            "antes de detectar movimiento."
        )
    elif modo_en_vivo:
        print("En modo cámara en vivo, recuerda presionar 'i' para iniciar.")
    raise SystemExit

df.to_csv(OUTPUT_CSV, index=False)

# =========================
# GRÁFICA SINCRONIZADA CÁMARA + IMU
# =========================

if ejercicio == "sit_to_stand" and imu_muestras:
    df_imu = pd.DataFrame(imu_muestras)
    df_eventos_imu = pd.DataFrame(imu_eventos)

    marcador_go = pd.DataFrame()
    if not df_eventos_imu.empty and "evento" in df_eventos_imu.columns:
        marcador_go = df_eventos_imu[
            df_eventos_imu["evento"] == "camera_GO"
        ]

    if not marcador_go.empty:
        inicio_imu_ms = float(marcador_go.iloc[0]["ms_prueba"])
        df_imu = df_imu[
            pd.to_numeric(df_imu["grabando"], errors="coerce").fillna(0) == 1
        ].copy()
        df_imu["tiempo_sincronizado_s"] = (
            pd.to_numeric(df_imu["ms_prueba"], errors="coerce") - inicio_imu_ms
        ) / 1000.0
        df_imu = df_imu[df_imu["tiempo_sincronizado_s"] >= -0.25]
        df_imu["aceleracion_magnitud_m_s2"] = np.sqrt(
            df_imu["ax_m_s2"] ** 2
            + df_imu["ay_m_s2"] ** 2
            + df_imu["az_m_s2"] ** 2
        )
        df_imu["giroscopio_magnitud_rad_s"] = np.sqrt(
            df_imu["gx_rad_s"] ** 2
            + df_imu["gy_rad_s"] ** 2
            + df_imu["gz_rad_s"] ** 2
        )

        if not df_imu.empty:
            figura, ejes = plt.subplots(
                3,
                1,
                figsize=(12, 8),
                sharex=True,
            )
            ejes[0].plot(
                df["tiempo_s"],
                df["angulo_rodilla"],
                color="#2563EB",
                label="Ángulo de rodilla (cámara)",
            )
            ejes[0].set_ylabel("Grados")
            ejes[0].legend(loc="upper right")
            ejes[0].grid(True, alpha=0.3)

            ejes[1].plot(
                df_imu["tiempo_sincronizado_s"],
                df_imu["aceleracion_magnitud_m_s2"],
                color="#DC2626",
                label="Magnitud de aceleración (IMU)",
            )
            ejes[1].set_ylabel("m/s²")
            ejes[1].legend(loc="upper right")
            ejes[1].grid(True, alpha=0.3)

            ejes[2].plot(
                df_imu["tiempo_sincronizado_s"],
                df_imu["giroscopio_magnitud_rad_s"],
                color="#059669",
                label="Magnitud de velocidad angular (IMU)",
            )
            ejes[2].set_ylabel("rad/s")
            ejes[2].set_xlabel(
                "Tiempo desde la señal GO (s)"
            )
            ejes[2].legend(loc="upper right")
            ejes[2].grid(True, alpha=0.3)

            if not df_eventos_imu.empty:
                for _, evento in df_eventos_imu.iterrows():
                    etiqueta = str(evento.get("evento", ""))
                    if not (
                        etiqueta.startswith("camera_MO")
                        or etiqueta.startswith("camera_R")
                        or etiqueta.startswith("camera_D")
                        or etiqueta.startswith("camera_S")
                    ):
                        continue
                    tiempo_evento = (
                        float(evento["ms_prueba"]) - inicio_imu_ms
                    ) / 1000.0
                    for eje in ejes:
                        eje.axvline(
                            tiempo_evento,
                            color="#6B7280",
                            linestyle="--",
                            linewidth=0.8,
                            alpha=0.55,
                        )
                    ejes[0].text(
                        tiempo_evento,
                        ejes[0].get_ylim()[1],
                        etiqueta.replace("camera_", ""),
                        fontsize=8,
                        ha="center",
                        va="bottom",
                    )

            if resultado_imu_sts is not None:
                for _, evento in resultado_imu_sts["detecciones"].iterrows():
                    tiempo_evento = float(evento["tiempo_prueba_s"])
                    for eje in ejes:
                        eje.axvline(
                            tiempo_evento,
                            color="#7C3AED",
                            linestyle=":",
                            linewidth=1.0,
                            alpha=0.75,
                        )
                    ejes[2].text(
                        tiempo_evento,
                        ejes[2].get_ylim()[1],
                        str(evento["evento"]).replace("imu_", ""),
                        color="#7C3AED",
                        fontsize=8,
                        ha="center",
                        va="bottom",
                    )

            figura.suptitle(
                "Sit-to-Stand: cámara (gris) + detector IMU (morado)"
            )
            figura.tight_layout()
            figura.savefig(OUTPUT_GRAPH_IMU, dpi=150)
            plt.close(figura)
            print(f"Gráfica cámara + IMU: {OUTPUT_GRAPH_IMU}")
    else:
        print(
            "No se generó la gráfica cámara + IMU porque no se recibió "
            "el marcador GO."
        )

# =========================
# GENERAR GRÁFICA
# =========================

# No usamos df.dropna() general porque en modo en vivo hay columnas
# específicas de cada prueba que quedan vacías para las demás.
if ejercicio == "equilibrio":
    df_limpio = df.dropna(subset=["tiempo_s", "centro_cadera_x", "centro_cadera_y"])
elif ejercicio == "marcha":
    df_limpio = df.dropna(subset=["tiempo_s", "distancia_tobillos"])
else:
    df_limpio = df.dropna(subset=["tiempo_s", "angulo_rodilla", "angulo_tronco"])

plt.figure(figsize=(10, 5))
plt.plot(df_limpio["tiempo_s"], df_limpio["angulo_rodilla"], label="Ángulo de rodilla")
plt.plot(df_limpio["tiempo_s"], df_limpio["angulo_tronco"], label="Ángulo de tronco")
plt.xlabel("Tiempo (s)")
plt.ylabel("Ángulo (grados)")
plt.title("Análisis biomecánico del movimiento")
plt.legend()
plt.grid(True)
plt.savefig(OUTPUT_GRAPH)
plt.close()

# =========================
# RESUMEN FINAL MEJORADO
# =========================

if not df_limpio.empty:
    tiempo_total = df_limpio["tiempo_s"].max()
    angulo_min_rodilla = df_limpio["angulo_rodilla"].min()
    angulo_min_tronco = df_limpio["angulo_tronco"].min()
    angulo_max_rodilla = df_limpio["angulo_rodilla"].max()
    angulo_max_tronco = df_limpio["angulo_tronco"].max()

    if ejercicio == "equilibrio":

        desplazamiento_lateral = (
            df_limpio["centro_cadera_x"].max()
            - df_limpio["centro_cadera_x"].min()
        )

        desplazamiento_vertical = (
            df_limpio["centro_cadera_y"].max()
            - df_limpio["centro_cadera_y"].min()
        )

        oscilacion_lateral_std = df_limpio["centro_cadera_x"].std()
        oscilacion_vertical_std = df_limpio["centro_cadera_y"].std()

        estabilidad_global = (
            oscilacion_lateral_std + oscilacion_vertical_std
        )

        # =========================
        # EQUILIBRIO SPPB: 3 POSICIONES
        # paralelo, semitandem y tandem
        # =========================

        tiempos_posiciones = {
            "paralelo": 0.0,
            "semitandem": 0.0,
            "tandem": 0.0
        }

        if "posicion_equilibrio" in df_limpio.columns:
            df_eq_pos = df_limpio[
                df_limpio["posicion_equilibrio"].isin(
                    ["paralelo", "semitandem", "tandem"]
                )
            ]

            if not df_eq_pos.empty and "tiempo_posicion_s" in df_eq_pos.columns:
                for posicion in tiempos_posiciones.keys():
                    df_pos = df_eq_pos[df_eq_pos["posicion_equilibrio"] == posicion]
                    if not df_pos.empty:
                        tiempos_posiciones[posicion] = min(
                            10.0,
                            float(df_pos["tiempo_posicion_s"].max())
                        )

        tiempo_paralelo = tiempos_posiciones["paralelo"]
        tiempo_semitandem = tiempos_posiciones["semitandem"]
        tiempo_tandem = tiempos_posiciones["tandem"]

        # Puntaje SPPB de equilibrio:
        # 0 = no mantiene pies juntos 10 s
        # 1 = pies juntos 10 s, pero semi-tandem < 10 s
        # 2 = semi-tandem 10 s, pero tandem < 3 s
        # 3 = tandem entre 3 y 9.99 s
        # 4 = tandem 10 s
        if tiempo_paralelo < 10:
            puntaje_equilibrio = 0
        elif tiempo_semitandem < 10:
            puntaje_equilibrio = 1
        elif tiempo_tandem < 3:
            puntaje_equilibrio = 2
        elif tiempo_tandem < 10:
            puntaje_equilibrio = 3
        else:
            puntaje_equilibrio = 4

        if estabilidad_global < 0.01:
            interpretacion = "Estabilidad alta"
        elif estabilidad_global < 0.03:
            interpretacion = "Estabilidad moderada"
        else:
            interpretacion = "Estabilidad baja"

        print("\n----- RESULTADOS DEL ANÁLISIS -----")
        print(f"Ejercicio: {ejercicio}")
        print(f"Tiempo total analizado: {tiempo_total:.2f} s")
        print(f"Desplazamiento lateral de cadera: {desplazamiento_lateral:.4f}")
        print(f"Desplazamiento vertical de cadera: {desplazamiento_vertical:.4f}")
        print(f"Oscilación lateral STD: {oscilacion_lateral_std:.4f}")
        print(f"Oscilación vertical STD: {oscilacion_vertical_std:.4f}")
        print(f"Índice de estabilidad global: {estabilidad_global:.4f}")
        print(f"Interpretación: {interpretacion}")

        print("\n----- EQUILIBRIO SPPB -----")
        print(f"Paralelo / pies juntos: {tiempo_paralelo:.2f} s")
        print(f"Semi-tándem: {tiempo_semitandem:.2f} s")
        print(f"Tándem: {tiempo_tandem:.2f} s")
        print(f"Puntaje equilibrio SPPB: {puntaje_equilibrio}/4")

        plt.figure(figsize=(10, 5))
        plt.plot(
            df_limpio["tiempo_s"],
            df_limpio["centro_cadera_x"],
            label="Centro cadera X"
        )
        plt.plot(
            df_limpio["tiempo_s"],
            df_limpio["centro_cadera_y"],
            label="Centro cadera Y"
        )
        plt.xlabel("Tiempo (s)")
        plt.ylabel("Posición normalizada")
        plt.title("Análisis de equilibrio - Centro de cadera")
        plt.legend()
        plt.grid(True)
        plt.savefig(OUTPUT_GRAPH)
        plt.close()

        print(f"Gráfica equilibrio: {OUTPUT_GRAPH}")
        print(f"CSV: {OUTPUT_CSV}")

        exit()


    # =====================================
    # ANÁLISIS DE MARCHA 3 METROS
    # =====================================

    if ejercicio == "marcha":

        distancia_marcha_m = 3.0

        tiempo_total_marcha = (
            df_limpio["tiempo_s"].max()
            - df_limpio["tiempo_s"].min()
        )

        señal_marcha = df_limpio["distancia_tobillos"].values
        tiempos_marcha = df_limpio["tiempo_s"].values

        señal_marcha_suavizada = pd.Series(señal_marcha).rolling(
            window=7,
            center=True
        ).mean()

        señal_marcha_suavizada = señal_marcha_suavizada.bfill().ffill().values

        pasos, _ = find_peaks(
            señal_marcha_suavizada,
            distance=6,
            prominence=0.005
        )

        pasos_totales = len(pasos)

        if tiempo_total_marcha > 0:
            cadencia = (pasos_totales / tiempo_total_marcha) * 60
            velocidad = distancia_marcha_m / tiempo_total_marcha
        else:
            cadencia = 0
            velocidad = 0

        variabilidad_pasos = np.std(señal_marcha_suavizada)

        if velocidad >= 0.83:
            puntaje_marcha = 4
        elif velocidad >= 0.65:
            puntaje_marcha = 3
        elif velocidad >= 0.46:
            puntaje_marcha = 2
        elif velocidad > 0:
            puntaje_marcha = 1
        else:
            puntaje_marcha = 0

        print("\n----- RESULTADOS DEL ANÁLISIS -----")
        print(f"Ejercicio: {ejercicio}")
        print(f"Tiempo total de marcha: {tiempo_total_marcha:.2f} s")
        print(f"Pasos detectados: {pasos_totales}")
        print(f"Cadencia estimada: {cadencia:.2f} pasos/min")
        print(f"Velocidad estimada: {velocidad:.2f} m/s")
        print(f"Variabilidad de separación de tobillos: {variabilidad_pasos:.4f}")
        print(f"Puntaje marcha estimado: {puntaje_marcha}/4")

        plt.figure(figsize=(10, 5))
        plt.plot(
            tiempos_marcha,
            señal_marcha_suavizada,
            label="Distancia entre tobillos"
        )

        plt.scatter(
            tiempos_marcha[pasos],
            señal_marcha_suavizada[pasos],
            marker="^",
            s=80,
            label="Pasos detectados"
        )

        for i, paso in enumerate(pasos):
            plt.text(
                tiempos_marcha[paso],
                señal_marcha_suavizada[paso],
                f"P{i+1}",
                fontsize=9
            )

        plt.xlabel("Tiempo (s)")
        plt.ylabel("Distancia normalizada entre tobillos")
        plt.title("Análisis de marcha - Detección de pasos")
        plt.legend()
        plt.grid(True)
        plt.savefig(OUTPUT_GRAPH)
        plt.close()

        print(f"Gráfica marcha: {OUTPUT_GRAPH}")
        print(f"CSV: {OUTPUT_CSV}")

        exit()


    # =====================================
    # DETECCIÓN DE REPETICIONES POR PICOS Y VALLES
    # =====================================

    if ejercicio == "sentadilla":
        columna_angulo = "angulo_rodilla"
        tipo_patron = "pico_valle_pico"

    elif ejercicio == "recoger_objeto":
        columna_angulo = "angulo_tronco"
        tipo_patron = "pico_valle_pico"

    elif ejercicio == "sit_to_stand":
        columna_angulo = "angulo_rodilla"
        tipo_patron = "valle_pico_valle"

    else:
        columna_angulo = "angulo_rodilla"
        tipo_patron = "pico_valle_pico"

    señal = df_limpio[columna_angulo].values
    tiempos = df_limpio["tiempo_s"].values

    señal_suavizada = pd.Series(señal).rolling(
        window=5,
        center=True
    ).mean()

    señal_suavizada = señal_suavizada.bfill().ffill().values

    if ejercicio == "recoger_objeto":
        distancia_minima = 25
        prominencia_minima = 8

    elif ejercicio == "sit_to_stand":
        distancia_minima = 20
        prominencia_minima = 10

    else:
        distancia_minima = 20
        prominencia_minima = 15

    picos, _ = find_peaks(
        señal_suavizada,
        distance=distancia_minima,
        prominence=prominencia_minima
    )

    valles, _ = find_peaks(
        -señal_suavizada,
        distance=distancia_minima,
        prominence=prominencia_minima
    )

    # =====================================
    # LIMPIAR PICOS FALSOS EN SIT-TO-STAND
    # =====================================

    if ejercicio == "sit_to_stand":
        if "sts_umbral_de_pie" in df_limpio.columns:
            umbrales_de_pie = pd.to_numeric(
                df_limpio["sts_umbral_de_pie"],
                errors="coerce",
            ).dropna()
            umbral_de_pie_grafica = (
                float(umbrales_de_pie.median())
                if not umbrales_de_pie.empty
                else 150.0
            )
        else:
            umbral_de_pie_grafica = 150.0

        if "sts_umbral_sentado" in df_limpio.columns:
            umbrales_sentado = pd.to_numeric(
                df_limpio["sts_umbral_sentado"],
                errors="coerce",
            ).dropna()
            umbral_sentado_grafica = (
                float(umbrales_sentado.median())
                if not umbrales_sentado.empty
                else 130.0
            )
        else:
            umbral_sentado_grafica = 130.0

        picos_validos = []

        for pico in picos:
            if señal_suavizada[pico] >= umbral_de_pie_grafica:
                picos_validos.append(pico)

        picos = np.array(picos_validos, dtype=int)
        valles = np.array(
            [
                valle
                for valle in valles
                if señal_suavizada[valle] <= umbral_sentado_grafica
            ],
            dtype=int,
        )

    repeticiones = 0
    tiempos_reps = []
    detalles_reps = []
    

    if ejercicio in ["sentadilla", "recoger_objeto"]:

        repeticiones = len(valles)

        for i, valle in enumerate(valles):

            picos_antes = picos[picos < valle]
            picos_despues = picos[picos > valle]

            if len(picos_antes) > 0:
                inicio = picos_antes[-1]
            else:
                inicio = 0

            if len(picos_despues) > 0:
                fin = picos_despues[0]
            else:
                fin = len(señal_suavizada) - 1

            tiempo_descenso = tiempos[valle] - tiempos[inicio]
            tiempo_ascenso = tiempos[fin] - tiempos[valle]

            angulo_inicio = señal_suavizada[inicio]
            angulo_valle = señal_suavizada[valle]
            angulo_fin = señal_suavizada[fin]

            if tiempo_descenso > 0:
                velocidad_descenso = abs(angulo_inicio - angulo_valle) / tiempo_descenso
            else:
                velocidad_descenso = np.nan

            if tiempo_ascenso > 0:
                velocidad_ascenso = abs(angulo_fin - angulo_valle) / tiempo_ascenso
            else:
                velocidad_ascenso = np.nan

            angulo_rodilla = df_limpio["angulo_rodilla"].values[valle]
            angulo_tronco = df_limpio["angulo_tronco"].values[valle]

            if ejercicio == "recoger_objeto":
                if angulo_rodilla > 130:
                    estrategia = "Predominio de tronco"
                elif angulo_rodilla > 100:
                    estrategia = "Estrategia mixta"
                else:
                    estrategia = "Predominio de piernas"
            else:
                estrategia = "No aplica"
            detalles_reps.append({
                "repeticion": i + 1,
                "inicio_s": tiempos[inicio],
                "tiempo_evento_s": tiempos[valle],
                "fin_s": tiempos[fin],
                "tiempo_descenso_s": tiempo_descenso,
                "tiempo_ascenso_s": tiempo_ascenso,
                "velocidad_descenso_grados_s": velocidad_descenso,
                "velocidad_ascenso_grados_s": velocidad_ascenso,
                "angulo_min_variable_usada": señal_suavizada[valle],
                "angulo_rodilla_en_evento": angulo_rodilla,
                "angulo_tronco_en_evento": angulo_tronco,
                "indice_tronco_rodilla": float(angulo_rodilla) / float(angulo_tronco),
                "estrategia_movimiento": estrategia
            })

        if len(valles) > 1:
            tiempos_reps = list(np.diff(tiempos[valles]))

    elif ejercicio == "sit_to_stand":

        # En cámara en vivo se priorizan los eventos confirmados durante la
        # prueba. Las repeticiones 1-4 requieren regreso sentado y la quinta
        # termina en la extensión completa, como indica el protocolo SPPB.
        usar_eventos_en_vivo = (
            modo_en_vivo
            and "sts_repeticion_en_vivo" in df_limpio.columns
            and df_limpio["sts_repeticion_en_vivo"].max() > 0
            and "sts_evento_de_pie" in df_limpio.columns
            and "sts_evento_repeticion_completa" in df_limpio.columns
        )

        if usar_eventos_en_vivo:
            df_reps_eventos = construir_repeticiones(
                eventos_camara_local,
                df_limpio,
            )
            detalles_reps = df_reps_eventos.to_dict("records")
            repeticiones = int(
                df_reps_eventos["repeticion_valida"].sum()
            )
            tiempos_reps = (
                pd.to_numeric(
                    df_reps_eventos["duracion_s"],
                    errors="coerce",
                )
                .dropna()
                .tolist()
            )
            picos = np.array(
                [
                    int(np.argmin(np.abs(tiempos - float(valor))))
                    for valor in df_reps_eventos["R_s"].dropna()
                ],
                dtype=int,
            )
            valles = np.array(
                [
                    int(np.argmin(np.abs(tiempos - float(valor))))
                    for valor in df_reps_eventos["S_s"].dropna()
                ],
                dtype=int,
            )

        else:
            # En video grabado también se exige un pico de extensión completa.
            # Las primeras cuatro repeticiones requieren un valle sentado
            # posterior; la quinta puede finalizar completamente de pie.
            picos_ordenados = sorted(list(picos))[:5]
            tiempo_anterior = tiempos[0]

            for pico in picos_ordenados:
                valles_antes = valles[valles < pico]
                valles_despues = valles[valles > pico]

                if len(valles_antes) > 0:
                    inicio = valles_antes[-1]
                else:
                    inicio = 0

                if len(valles_despues) > 0:
                    fin = valles_despues[0]
                else:
                    # Solo la quinta repetición puede terminar de pie.
                    if repeticiones < 4:
                        continue
                    fin = pico

                repeticiones += 1
                duracion_rep = tiempos[fin] - tiempo_anterior
                tiempo_anterior = tiempos[fin]
                tiempos_reps.append(duracion_rep)

                detalles_reps.append({
                    "repeticion": repeticiones,
                    "inicio_sentado_s": tiempos[inicio],
                    "de_pie_s": tiempos[pico],
                    "fin_sentado_s": (
                        tiempos[fin] if fin != pico else np.nan
                    ),
                    "duracion_s": duracion_rep,
                    "duracion_subida_s": tiempos[pico] - tiempos[inicio],
                    "duracion_descenso_s": (
                        tiempos[fin] - tiempos[pico]
                        if fin != pico
                        else np.nan
                    ),
                    "duracion_ciclo_s": (
                        tiempos[fin] - tiempos[inicio]
                        if fin != pico
                        else np.nan
                    ),
                    "angulo_sentado_inicio": señal_suavizada[inicio],
                    "angulo_de_pie": señal_suavizada[pico],
                    "angulo_sentado_fin": (
                        señal_suavizada[fin] if fin != pico else np.nan
                    ),
                    "criterio_validacion": (
                        "Sentado-De pie-Sentado"
                        if fin != pico
                        else "Quinta extensión completa (fin SPPB)"
                    ),
                })

    if tiempos_reps:
        tiempo_promedio_rep = sum(tiempos_reps) / len(tiempos_reps)
    else:
        tiempo_promedio_rep = 0

    print("\n----- RESULTADOS DEL ANÁLISIS -----")
    print(f"Ejercicio: {ejercicio}")
    print(f"Variable usada: {columna_angulo}")
    print(f"Tiempo total analizado: {tiempo_total:.2f} s")
    print(f"Repeticiones detectadas: {repeticiones}")
    print(f"Tiempo promedio por repetición: {tiempo_promedio_rep:.2f} s")
    print(f"Ángulo mínimo de rodilla: {angulo_min_rodilla:.2f}°")
    print(f"Ángulo máximo de rodilla: {angulo_max_rodilla:.2f}°")
    print(f"Ángulo mínimo de tronco: {angulo_min_tronco:.2f}°")
    print(f"Ángulo máximo de tronco: {angulo_max_tronco:.2f}°")
    if ejercicio == "sit_to_stand":
        print(f"Posiciones de pie confirmadas: {len(picos)}")
        print(f"Regresos a sentado confirmados: {len(valles)}")
    else:
        print(f"Picos detectados: {len(picos)}")
        print(f"Valles detectados: {len(valles)}")
    print("Índices de pie/picos:", picos)
    print("Índices sentado/valles:", valles)
    # =========================
    # GRÁFICA CON PICOS Y VALLES
    # =========================

    plt.figure(figsize=(13, 5.5))

    plt.plot(
        df_limpio["tiempo_s"],
        df_limpio["angulo_rodilla"],
        label="Ángulo de rodilla"
    )

    plt.plot(
        df_limpio["tiempo_s"],
        df_limpio["angulo_tronco"],
        label="Ángulo de cadera (hombro-cadera-rodilla)"
    )

    etiqueta_picos = (
        "Posición de pie confirmada"
        if ejercicio == "sit_to_stand"
        else "Picos detectados"
    )
    etiqueta_valles = (
        "Regreso sentado confirmado"
        if ejercicio == "sit_to_stand"
        else "Valles detectados"
    )

    # Marcar eventos confirmados sobre la variable usada.
    plt.scatter(
        tiempos[picos],
        señal_suavizada[picos],
        marker="^",
        s=80,
        label=etiqueta_picos
    )

    if len(valles) > 0:
        plt.scatter(
            tiempos[valles],
            señal_suavizada[valles],
            marker="v",
            s=80,
            label=etiqueta_valles
        )

    for i, valle in enumerate(valles):
        plt.text(
            tiempos[valle],
            señal_suavizada[valle],
            f"S{i+1}" if ejercicio == "sit_to_stand" else f"V{i+1}",
            fontsize=9
        )

    for i, pico in enumerate(picos):
        plt.text(
            tiempos[pico],
            señal_suavizada[pico],
            f"R{i+1}" if ejercicio == "sit_to_stand" else f"P{i+1}",
            fontsize=9
        )

    plt.xlabel("Tiempo (s)")
    plt.ylabel("Ángulo (grados)")
    plt.title(f"Eventos detectados - {ejercicio}")
    # La leyenda queda fuera de los datos para no ocultar S1 ni otros eventos.
    plt.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
    )
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_GRAPH_EVENTOS, bbox_inches="tight", dpi=150)
    plt.close()

    print(f"Gráfica con eventos: {OUTPUT_GRAPH_EVENTOS}")
    print("\nDetalle por repetición:")

    for rep in detalles_reps:

        print(
            f"\nRep {rep['repeticion']}"
        )

        if "tiempo_evento_s" in rep:
            print(
                f"Tiempo evento: "
                f"{rep['tiempo_evento_s']:.2f} s"
            )

        if "angulo_rodilla_en_evento" in rep:
            print(
                f"Rodilla: "
                f"{rep['angulo_rodilla_en_evento']:.2f}°"
            )

        if "angulo_tronco_en_evento" in rep:
            print(
                f"Tronco: "
                f"{rep['angulo_tronco_en_evento']:.2f}°"
            )

        if "tiempo_descenso_s" in rep:
            print(
                f"Tiempo descenso: "
                f"{rep['tiempo_descenso_s']:.2f} s"
            )

        if "tiempo_ascenso_s" in rep:
            print(
                f"Tiempo ascenso: "
                f"{rep['tiempo_ascenso_s']:.2f} s"
            )

        if "velocidad_descenso_grados_s" in rep:
            print(
                f"Velocidad descenso: "
                f"{rep['velocidad_descenso_grados_s']:.2f} °/s"
            )

        if "velocidad_ascenso_grados_s" in rep:
            print(
                f"Velocidad ascenso: "
                f"{rep['velocidad_ascenso_grados_s']:.2f} °/s"
            )

        if "indice_tronco_rodilla" in rep:
            print(
                f"Índice Tronco-Rodilla: "
                f"{rep['indice_tronco_rodilla']:.2f}"
            )

        if "estrategia_movimiento" in rep:
            print(
                f"Estrategia: "
                f"{rep['estrategia_movimiento']}"
            )

        if "inicio_sentado_s" in rep:
            print(f"Inicio sentado: {rep['inicio_sentado_s']:.2f} s")

        if "de_pie_s" in rep:
            print(f"De pie: {rep['de_pie_s']:.2f} s")

        if "fin_sentado_s" in rep:
            print(f"Fin sentado: {rep['fin_sentado_s']:.2f} s")

        if "duracion_s" in rep:
            print(f"Duración: {rep['duracion_s']:.2f} s")

        if "angulo_sentado_inicio" in rep:
            print(f"Ángulo sentado inicio: {rep['angulo_sentado_inicio']:.2f}°")

        if "angulo_de_pie" in rep:
            print(f"Ángulo de pie: {rep['angulo_de_pie']:.2f}°")

        if "angulo_sentado_fin" in rep:
            print(f"Ángulo sentado final: {rep['angulo_sentado_fin']:.2f}°")
    # =========================
    # GUARDAR MÉTRICAS POR REPETICIÓN
    # =========================

    df_reps = pd.DataFrame(detalles_reps)

    if not df_reps.empty:

        if "angulo_min_variable_usada" in df_reps.columns:

            promedio_angulo = df_reps["angulo_min_variable_usada"].mean()
            std_angulo = df_reps["angulo_min_variable_usada"].std()

            print(
                f"\nPromedio ángulo mínimo: "
                f"{promedio_angulo:.2f}°"
            )

            print(
                f"Variabilidad: "
                f"{std_angulo:.2f}°"
            )

    df_reps.to_csv(
        OUTPUT_CSV_REPS,
        index=False
    )

    print(f"\nCSV por repetición: {OUTPUT_CSV_REPS}")

    if ejercicio == "sit_to_stand" and not df_reps.empty:
        tiempo_efectivo_camara = (
            float(sts_tiempo_efectivo_final)
            if modo_en_vivo and np.isfinite(sts_tiempo_efectivo_final)
            else (
                float(df_reps["de_pie_s"].max())
                - float(df_reps["inicio_sentado_s"].min())
            )
        )
        tiempos_clinicos = pd.to_numeric(
            df_limpio.get(
                "sts_tiempo_clinico_s",
                pd.Series(dtype=float),
            ),
            errors="coerce",
        ).dropna()
        tiempo_clinico_camara = (
            float(tiempos_clinicos.iloc[-1])
            if not tiempos_clinicos.empty
            else np.nan
        )
        tiempo_para_puntaje = tiempo_clinico_camara
        repeticiones_validas = (
            int(df_reps["repeticion_valida"].fillna(False).astype(bool).sum())
            if "repeticion_valida" in df_reps.columns
            else int(repeticiones)
        )
        prueba_completa = (
            repeticiones_validas == REPETICIONES_OBJETIVO
            and np.isfinite(tiempo_clinico_camara)
        )
        puntaje_calculado = subpuntaje_silla_sppb(
            tiempo_clinico_camara,
            repeticiones_validas,
            prueba_completa,
        )
        puntaje_sts_resumen = (
            puntaje_calculado
            if puntaje_calculado is not None
            else np.nan
        )
        calidad_video = calidad_camara(
            df,
            df_timeline_video,
        )

        resumen_sts = {
            "version_sistema": VERSION_SISTEMA,
            "version_algoritmo": VERSION_ALGORITMO,
            "id_participante": paciente_id,
            "id_ensayo": trial_id,
            "altura_silla_cm": chair_height_cm,
            "repeticiones_camara": repeticiones_validas,
            "prueba_completada": prueba_completa,
            "tiempo_clinico_camara_s": tiempo_clinico_camara,
            "tiempo_efectivo_camara_s": tiempo_efectivo_camara,
            "tiempo_reaccion_s": (
                tiempo_clinico_camara - tiempo_efectivo_camara
                if not np.isnan(tiempo_clinico_camara)
                else np.nan
            ),
            "subida_promedio_s": float(
                pd.to_numeric(
                    df_reps["duracion_subida_s"],
                    errors="coerce",
                ).mean()
            ),
            "descenso_promedio_s": float(
                pd.to_numeric(
                    df_reps["duracion_descenso_s"],
                    errors="coerce",
                ).mean()
            ),
            "ciclo_completo_promedio_s": float(
                pd.to_numeric(
                    df_reps["duracion_ciclo_s"],
                    errors="coerce",
                ).mean()
            ),
            "puntaje_sts": puntaje_sts_resumen,
            "tipo_puntaje": "subprueba_silla_0_a_4",
            "sppb_total_calculado": False,
            "tiempo_usado_para_puntaje_s": tiempo_para_puntaje,
            "fuente_tiempo_puntaje": "clinico_desde_GO_sonoro",
            **calidad_video,
        }
        if resultado_imu_sts is not None:
            resumen_sts.update(
                resultado_imu_sts["resumen"].iloc[0].to_dict()
            )
        pd.DataFrame([resumen_sts]).to_csv(
            OUTPUT_RESUMEN_STS,
            index=False,
        )
        if resultado_imu_sts is None:
            fusionar_eventos(
                eventos_camara_local,
                pd.DataFrame(),
                tolerancia_s=TOLERANCIA_CONCORDANCIA_S,
            ).to_csv(OUTPUT_EVENTOS_FUSION, index=False)
        print(f"Resumen Sit-to-Stand: {OUTPUT_RESUMEN_STS}")

    print("\n==============================")
    print("RESUMEN DEL MOVIMIENTO")
    print("==============================")

    if not df_reps.empty:

        if "tiempo_descenso_s" in df_reps.columns:
            print(
                f"Tiempo descenso promedio: "
                f"{df_reps['tiempo_descenso_s'].mean():.2f} s"
            )

        if "tiempo_ascenso_s" in df_reps.columns:
            print(
                f"Tiempo ascenso promedio: "
                f"{df_reps['tiempo_ascenso_s'].mean():.2f} s"
            )

        if "velocidad_descenso_grados_s" in df_reps.columns:
            print(
                f"Velocidad descenso promedio: "
                f"{df_reps['velocidad_descenso_grados_s'].mean():.2f} °/s"
            )

        if "velocidad_ascenso_grados_s" in df_reps.columns:
            print(
                f"Velocidad ascenso promedio: "
                f"{df_reps['velocidad_ascenso_grados_s'].mean():.2f} °/s"
            )

        if "indice_tronco_rodilla" in df_reps.columns:
            print(
                f"Índice Tronco-Rodilla promedio: "
                f"{df_reps['indice_tronco_rodilla'].mean():.2f}"
            )

        if "estrategia_movimiento" in df_reps.columns:
            estrategias = df_reps["estrategia_movimiento"].value_counts()

            print("\nEstrategias utilizadas:")

            for estrategia, cantidad in estrategias.items():
                print(f"- {estrategia}: {cantidad} repeticiones")

            print(f"\nEstrategia dominante: {estrategias.idxmax()}")
        
        if ejercicio == "sit_to_stand" and "duracion_subida_s" in df_reps.columns:
            print(
                "Subida promedio: "
                f"{df_reps['duracion_subida_s'].mean():.2f} s"
            )
            print(
                "Descenso promedio: "
                f"{df_reps['duracion_descenso_s'].mean():.2f} s"
            )
            print(
                "Ciclo completo promedio (repeticiones 1-4): "
                f"{df_reps['duracion_ciclo_s'].mean():.2f} s"
            )
        elif "duracion_s" in df_reps.columns:
            print(
                "Duración promedio: "
                f"{df_reps['duracion_s'].mean():.2f} s"
            )

        if "angulo_de_pie" in df_reps.columns:
            print(
                f"Ángulo promedio de pie: "
                f"{df_reps['angulo_de_pie'].mean():.2f}°"
            )
        if ejercicio == "sit_to_stand" and "de_pie_s" in df_reps.columns:
            tiempo_total_sts = (
                df_reps["de_pie_s"].max()
                - df_reps["inicio_sentado_s"].min()
            )
            print(
                f"Tiempo efectivo Sit-to-Stand: "
                f"{tiempo_total_sts:.2f} s"
            )
