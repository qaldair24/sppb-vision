"""Cliente del IMU lumbar ESP32-S3 para SPPB-VISION.

El ESP32 crea la red ESP32_SPPB y transmite muestras por UDP. Este módulo
mantiene la adquisición separada de MediaPipe para que el uso del IMU sea
opcional y la versión de cámara pueda funcionar sin el sensor.
"""

from __future__ import annotations

import csv
import io
import json
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


class ErrorIMU(RuntimeError):
    """Error de conexión, calibración o control del IMU."""


class IMULumbar:
    def __init__(
        self,
        host: str = "192.168.4.1",
        puerto_udp: int = 4212,
        timeout_http_s: float = 5.0,
    ) -> None:
        self.host = host
        self.puerto_udp = puerto_udp
        self.timeout_http_s = timeout_http_s
        self._socket: socket.socket | None = None
        self._hilo: threading.Thread | None = None
        self._detener_hilo = threading.Event()
        self._lock = threading.Lock()
        self._muestras: list[dict] = []
        self._eventos: list[dict] = []
        self._respaldo_muestras: list[dict] = []
        self._respaldo_eventos: list[dict] = []
        self._ultimo_paquete_pc = 0.0
        self._ultimo_error = ""
        self._grabando = False
        self._preparado = False

    @property
    def preparado(self) -> bool:
        return self._preparado

    @property
    def grabando(self) -> bool:
        return self._grabando

    @property
    def ultimo_error(self) -> str:
        return self._ultimo_error

    @property
    def recibiendo(self) -> bool:
        return (
            self._ultimo_paquete_pc > 0
            and time.perf_counter() - self._ultimo_paquete_pc < 1.0
        )

    def _url(self, ruta: str) -> str:
        return f"http://{self.host}{ruta}"

    def _solicitud(
        self,
        ruta: str,
        metodo: str = "GET",
        payload: dict | None = None,
        timeout_s: float | None = None,
    ):
        datos = None
        headers = {}
        if payload is not None:
            datos = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        peticion = urllib.request.Request(
            self._url(ruta),
            data=datos,
            headers=headers,
            method=metodo,
        )
        try:
            with urllib.request.urlopen(
                peticion,
                timeout=timeout_s or self.timeout_http_s,
            ) as respuesta:
                contenido = respuesta.read().decode("utf-8", errors="replace")
                tipo = respuesta.headers.get("Content-Type", "")
                if "application/json" in tipo:
                    return json.loads(contenido)
                return contenido
        except urllib.error.HTTPError as exc:
            detalle = exc.read().decode("utf-8", errors="replace")
            try:
                mensaje = json.loads(detalle).get("message", detalle)
            except json.JSONDecodeError:
                mensaje = detalle
            self._ultimo_error = str(mensaje or exc)
            raise ErrorIMU(self._ultimo_error) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            self._ultimo_error = str(exc)
            raise ErrorIMU(
                "No se pudo comunicar con el ESP32. "
                "Compruebe que la computadora esté conectada a ESP32_SPPB."
            ) from exc

    def iniciar_receptor(self) -> None:
        if self._hilo and self._hilo.is_alive():
            return

        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._socket.bind(("", self.puerto_udp))
            self._socket.settimeout(0.25)
        except OSError as exc:
            if self._socket is not None:
                self._socket.close()
                self._socket = None
            raise ErrorIMU(
                f"No se pudo abrir el puerto UDP {self.puerto_udp}: {exc}"
            ) from exc
        self._detener_hilo.clear()
        self._hilo = threading.Thread(
            target=self._recibir,
            name="SPPB-IMU-L5",
            daemon=True,
        )
        self._hilo.start()

    def _recibir(self) -> None:
        assert self._socket is not None
        while not self._detener_hilo.is_set():
            try:
                paquete, origen = self._socket.recvfrom(512)
            except socket.timeout:
                continue
            except OSError:
                break

            recibido_pc = time.perf_counter()
            texto = paquete.decode("utf-8", errors="replace").strip()
            partes = texto.split(",")
            if not partes:
                continue

            try:
                if partes[0] == "IMU" and len(partes) == 13:
                    fila = {
                        "seq": int(partes[1]),
                        "ms_esp32": int(partes[2]),
                        "ms_prueba": int(partes[3]),
                        "grabando": int(partes[4]),
                        "fase": partes[5],
                        "ax_m_s2": float(partes[6]),
                        "ay_m_s2": float(partes[7]),
                        "az_m_s2": float(partes[8]),
                        "gx_rad_s": float(partes[9]),
                        "gy_rad_s": float(partes[10]),
                        "gz_rad_s": float(partes[11]),
                        "temperatura_c": float(partes[12]),
                        "tiempo_recepcion_pc_s": recibido_pc,
                        "ip_origen": origen[0],
                    }
                    with self._lock:
                        self._muestras.append(fila)
                    self._ultimo_paquete_pc = recibido_pc

                elif partes[0] == "EVENT" and len(partes) >= 5:
                    evento = {
                        "evento": partes[1],
                        "ms_esp32": int(partes[2]),
                        "ms_prueba": int(partes[3]),
                        "fase": partes[4],
                        "tiempo_recepcion_pc_s": recibido_pc,
                    }
                    with self._lock:
                        self._eventos.append(evento)
                    self._ultimo_paquete_pc = recibido_pc
            except (TypeError, ValueError):
                # Se ignora únicamente el paquete incompleto; la adquisición sigue.
                continue

    def comprobar(self) -> dict:
        estado = self._solicitud("/status")
        if not isinstance(estado, dict):
            raise ErrorIMU("El ESP32 respondió con un estado no válido.")
        if not estado.get("mpu_found", False):
            raise ErrorIMU("El ESP32 no detecta el MPU6050 lumbar.")
        return estado

    def configurar(
        self,
        paciente_id: str,
        ensayo_id: str = "T1",
        version_algoritmo: str = "",
    ) -> None:
        self._solicitud(
            "/meta",
            metodo="POST",
            payload={
                "patient_id": paciente_id,
                "test_type": "chair",
                "repetition": ensayo_id,
                "placement": "L5",
                "notes": (
                    "5xSTS-VISION cámara + IMU lumbar "
                    f"{version_algoritmo}"
                ).strip(),
            },
        )

    def preparar(
        self,
        paciente_id: str,
        ensayo_id: str = "T1",
        version_algoritmo: str = "",
    ) -> None:
        self.iniciar_receptor()
        self.comprobar()
        self.configurar(paciente_id, ensayo_id, version_algoritmo)
        respuesta = self._solicitud(
            "/prepare",
            metodo="POST",
            timeout_s=12.0,
        )
        if isinstance(respuesta, dict):
            correcto = bool(respuesta.get("ok", False))
            mensaje = str(respuesta.get("message", ""))
        else:
            correcto = "OK" in str(respuesta).upper()
            mensaje = str(respuesta)
        if not correcto:
            raise ErrorIMU(mensaje or "No se pudo calibrar el IMU.")
        self._preparado = True

    def iniciar(self) -> None:
        if not self._preparado:
            raise ErrorIMU("El IMU todavía no está calibrado.")
        respuesta = self._solicitud(
            "/start?prepared=1",
            metodo="POST",
        )
        if isinstance(respuesta, dict) and not respuesta.get("ok", False):
            raise ErrorIMU(str(respuesta.get("message", "No pudo iniciar.")))
        self._grabando = True

    def marcar(self, etiqueta: str) -> None:
        if not self._grabando:
            return
        etiqueta_segura = urllib.parse.quote(etiqueta, safe="")
        self._solicitud(
            f"/mark?label={etiqueta_segura}",
            metodo="POST",
            timeout_s=2.0,
        )

    def detener(self) -> None:
        if not self._grabando:
            return
        try:
            self._solicitud("/stop", metodo="POST", timeout_s=3.0)
        finally:
            self._grabando = False
        try:
            self._descargar_respaldo()
        except ErrorIMU as exc:
            # La captura UDP sigue disponible si el respaldo no se descarga.
            self._ultimo_error = (
                "No se descargó el respaldo interno; se conservará UDP: "
                f"{exc}"
            )

    def _descargar_respaldo(self) -> None:
        contenido = self._solicitud("/download", timeout_s=8.0)
        lineas = [
            linea
            for linea in str(contenido).splitlines()
            if linea.strip() and not linea.startswith("#")
        ]
        if not lineas:
            return

        lector = csv.DictReader(io.StringIO("\n".join(lineas)))
        muestras: list[dict] = []
        eventos: list[dict] = []
        for fila in lector:
            etiqueta = str(fila.get("event", "") or "").strip()
            if etiqueta:
                eventos.append(
                    {
                        "evento": etiqueta,
                        "ms_esp32": int(fila["ms_esp32"]),
                        "ms_prueba": int(fila["ms_prueba"]),
                        "fase": fila.get("phase", ""),
                        "tiempo_recepcion_pc_s": "",
                        "fuente": "respaldo_esp32",
                    }
                )
                continue

            if not str(fila.get("ax_m_s2", "")).strip():
                continue
            muestras.append(
                {
                    "seq": int(fila["seq"]),
                    "ms_esp32": int(fila["ms_esp32"]),
                    "ms_prueba": int(fila["ms_prueba"]),
                    "grabando": 1,
                    "fase": fila.get("phase", ""),
                    "ax_m_s2": float(fila["ax_m_s2"]),
                    "ay_m_s2": float(fila["ay_m_s2"]),
                    "az_m_s2": float(fila["az_m_s2"]),
                    "gx_rad_s": float(fila["gx_rad_s"]),
                    "gy_rad_s": float(fila["gy_rad_s"]),
                    "gz_rad_s": float(fila["gz_rad_s"]),
                    "temperatura_c": float(fila["temp_c"]),
                    "tiempo_recepcion_pc_s": "",
                    "ip_origen": self.host,
                    "fuente": "respaldo_esp32",
                }
            )

        if muestras:
            with self._lock:
                self._respaldo_muestras = muestras
                self._respaldo_eventos = eventos

    def limpiar(self) -> None:
        self._solicitud("/clear", metodo="POST", timeout_s=3.0)
        with self._lock:
            self._muestras = []
            self._eventos = []
            self._respaldo_muestras = []
            self._respaldo_eventos = []
        self._preparado = False
        self._grabando = False

    def datos(self) -> tuple[list[dict], list[dict]]:
        with self._lock:
            if self._respaldo_muestras:
                return (
                    list(self._respaldo_muestras),
                    list(self._respaldo_eventos),
                )
            return list(self._muestras), list(self._eventos)

    def guardar(
        self,
        ruta_muestras: str | Path,
        ruta_eventos: str | Path,
    ) -> tuple[int, int]:
        muestras, eventos = self.datos()
        ruta_muestras = Path(ruta_muestras)
        ruta_eventos = Path(ruta_eventos)
        ruta_muestras.parent.mkdir(parents=True, exist_ok=True)

        if muestras:
            with ruta_muestras.open("w", newline="", encoding="utf-8-sig") as f:
                escritor = csv.DictWriter(f, fieldnames=list(muestras[0]))
                escritor.writeheader()
                escritor.writerows(muestras)

        if eventos:
            with ruta_eventos.open("w", newline="", encoding="utf-8-sig") as f:
                escritor = csv.DictWriter(f, fieldnames=list(eventos[0]))
                escritor.writeheader()
                escritor.writerows(eventos)

        return len(muestras), len(eventos)

    def cerrar(self) -> None:
        try:
            self.detener()
        except ErrorIMU:
            pass
        self._detener_hilo.set()
        if self._socket is not None:
            try:
                self._socket.close()
            except OSError:
                pass
        if self._hilo and self._hilo.is_alive():
            self._hilo.join(timeout=1.0)


def probar_conexion() -> None:
    imu = IMULumbar()
    print("Conectando al ESP32-S3...")
    try:
        imu.iniciar_receptor()
        estado = imu.comprobar()
        print("ESP32 encontrado.")
        print(f"MPU6050: {'OK' if estado.get('mpu_found') else 'NO'}")
        print(f"Frecuencia: {estado.get('sample_hz', '?')} Hz")
        print(f"Puerto de datos: {estado.get('stream_port', '?')}")
        print("Esperando muestras durante 3 segundos...")
        limite = time.perf_counter() + 3.0
        while time.perf_counter() < limite and not imu.recibiendo:
            time.sleep(0.1)
        if not imu.recibiendo:
            raise ErrorIMU(
                "El ESP32 responde, pero no llegan muestras UDP al puerto 4212."
            )
        muestras, _ = imu.datos()
        print(f"Recepción en vivo OK: {len(muestras)} muestras.")
        print("La comunicación IMU está lista para SPPB-VISION.")
    finally:
        imu.cerrar()


if __name__ == "__main__":
    try:
        probar_conexion()
    except ErrorIMU as exc:
        print(f"\nERROR: {exc}")
        raise SystemExit(1)
