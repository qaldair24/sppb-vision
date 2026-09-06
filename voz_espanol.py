import importlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import unicodedata


TOKENS_ESPANOL = (
    "espanol",
    "spanish",
    "es-",
    "es_",
    "spa",
    "sabina",
    "helena",
    "dalia",
    "jorge",
    "laura",
    "pablo",
    "monica",
    "sofia",
    "carolina",
)

TOKENS_LATINOAMERICA = (
    "es-mx",
    "es_mx",
    "mexic",
    "sabina",
    "es-us",
    "es_us",
    "latino",
    "latin american",
    "dalia",
    "jorge",
    "es-hn",
    "es_hn",
    "hondur",
)


def seleccionar_voz_macos():
    """Devuelve una voz española disponible para el comando nativo say."""

    if sys.platform != "darwin" or shutil.which("say") is None:
        return None

    resultado = subprocess.run(
        ["say", "-v", "?"],
        capture_output=True,
        text=True,
        check=False,
    )
    if resultado.returncode != 0:
        return None

    voces = []
    for linea in resultado.stdout.splitlines():
        partes = linea.split()
        if len(partes) < 2:
            continue
        nombre, idioma = partes[0], partes[1].lower()
        if idioma.startswith("es_") or idioma.startswith("es-"):
            voces.append((nombre, idioma))

    prioridades = ("Monica", "Paulina", "Juan", "Jorge", "Diego")
    for preferida in prioridades:
        for nombre, _ in voces:
            if nombre.casefold() == preferida.casefold():
                return nombre

    return voces[0][0] if voces else None


def _error_es_cache_comtypes(error):
    """Reconoce el fallo de código SAPI generado parcialmente por comtypes."""

    revisados = set()
    actual = error

    while actual is not None and id(actual) not in revisados:
        revisados.add(id(actual))
        archivo = str(getattr(actual, "filename", "") or "").lower()
        mensaje = str(actual).lower()

        if (
            isinstance(actual, (IndentationError, SyntaxError))
            and (
                "comtypes" in archivo
                or "comtypes.gen" in mensaje
                or "speechlib.py" in archivo
            )
        ):
            return True

        actual = getattr(actual, "__cause__", None) or getattr(
            actual,
            "__context__",
            None,
        )

    return False


def limpiar_cache_comtypes():
    """Elimina únicamente wrappers COM regenerables; conserva el paquete."""

    if os.name != "nt":
        return []

    import comtypes

    carpeta_gen = Path(comtypes.__file__).resolve().parent / "gen"
    if not carpeta_gen.is_dir():
        return []

    eliminados = []
    for elemento in carpeta_gen.iterdir():
        if elemento.name == "__init__.py":
            continue

        try:
            if elemento.is_dir():
                shutil.rmtree(elemento)
            elif elemento.suffix.lower() in {".py", ".pyc"}:
                elemento.unlink()
            else:
                continue
            eliminados.append(str(elemento))
        except FileNotFoundError:
            pass

    for nombre in list(sys.modules):
        if nombre.startswith("comtypes.gen."):
            sys.modules.pop(nombre, None)

    importlib.invalidate_caches()
    return eliminados


def inicializar_motor_voz():
    """Inicializa SAPI5 y repara una caché comtypes corrupta una sola vez."""

    import pyttsx3

    driver = "sapi5" if os.name == "nt" else None

    try:
        return pyttsx3.init(driverName=driver)
    except Exception as error:
        if os.name != "nt" or not _error_es_cache_comtypes(error):
            raise

        limpiar_cache_comtypes()

        # El primer intento pudo dejar importaciones parciales en memoria.
        for nombre in list(sys.modules):
            if (
                nombre == "pyttsx3.drivers.sapi5"
                or nombre.startswith("comtypes.gen.")
            ):
                sys.modules.pop(nombre, None)

        importlib.invalidate_caches()
        return pyttsx3.init(driverName="sapi5")


def _texto_limpio(valor):
    if valor is None:
        return ""

    if isinstance(valor, bytes):
        valor = valor.decode("utf-8", errors="ignore")

    texto = unicodedata.normalize("NFKD", str(valor))
    texto = "".join(
        caracter for caracter in texto
        if not unicodedata.combining(caracter)
    )
    texto = "".join(
        caracter if caracter.isprintable() else " "
        for caracter in texto
    )
    return re.sub(r"\s+", " ", texto).strip().lower()


def descripcion_voz(voz):
    idiomas = " ".join(
        _texto_limpio(idioma)
        for idioma in getattr(voz, "languages", []) or []
    )
    partes = [
        _texto_limpio(getattr(voz, "name", "")),
        _texto_limpio(getattr(voz, "id", "")),
        idiomas,
    ]
    return " ".join(parte for parte in partes if parte)


def puntuar_voz_espanol(voz):
    descripcion = descripcion_voz(voz)
    puntuacion = 0

    if any(token in descripcion for token in TOKENS_ESPANOL):
        puntuacion += 100

    if any(token in descripcion for token in TOKENS_LATINOAMERICA):
        puntuacion += 35

    # Evita que una coincidencia accidental seleccione una voz inglesa.
    if (
        ("english" in descripcion or "en-us" in descripcion)
        and not any(token in descripcion for token in TOKENS_ESPANOL)
    ):
        puntuacion -= 100

    return puntuacion


def seleccionar_voz_espanol(motor):
    voces = motor.getProperty("voices") or []
    voz_preferida = os.environ.get("SPPB_VOICE_ID", "").strip()

    if voz_preferida:
        for voz in voces:
            if getattr(voz, "id", "") == voz_preferida:
                return voz

    candidatas = [
        (puntuar_voz_espanol(voz), indice, voz)
        for indice, voz in enumerate(voces)
    ]
    candidatas = [
        candidata for candidata in candidatas
        if candidata[0] >= 100
    ]

    if not candidatas:
        return None

    candidatas.sort(key=lambda item: (-item[0], item[1]))
    return candidatas[0][2]


def configurar_motor_espanol(motor, velocidad=145):
    voz = seleccionar_voz_espanol(motor)
    if voz is None:
        return None

    motor.setProperty("voice", voz.id)
    motor.setProperty("rate", velocidad)
    motor.setProperty("volume", 1.0)
    return voz


def preparar_texto_para_voz(texto):
    reemplazos = {
        "Coloquese": "Colóquese",
        "coloquese": "colóquese",
        "Alejese": "Aléjese",
        "alejese": "aléjese",
        "Acerquese": "Acérquese",
        "acerquese": "acérquese",
        "Sientese": "Siéntese",
        "sientese": "siéntese",
        "Levantese": "Levántese",
        "levantese": "levántese",
        "Mantengase": "Manténgase",
        "mantengase": "manténgase",
        "Camara": "Cámara",
        "camara": "cámara",
        "Posicion": "Posición",
        "posicion": "posición",
        "Repeticion": "Repetición",
        "repeticion": "repetición",
    }

    resultado = str(texto)
    for original, corregido in reemplazos.items():
        resultado = resultado.replace(original, corregido)
    return resultado


def listar_voces(motor):
    return [
        {
            "nombre": getattr(voz, "name", "Sin nombre"),
            "id": getattr(voz, "id", ""),
            "descripcion": descripcion_voz(voz),
            "es_espanol": puntuar_voz_espanol(voz) >= 100,
        }
        for voz in (motor.getProperty("voices") or [])
    ]
