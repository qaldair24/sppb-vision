# 5xSTS-VISION V4.1 para Mac M1

Versión congelada para recolectar el Five Times Sit-to-Stand con cámara
monocular e IMU lumbar en L5.

Evalúa únicamente la subprueba de silla. El resultado es parcial, de 0 a 4;
no calcula ni informa un puntaje SPPB total de 0 a 12.

## 1. Instalación en este Mac

Equipo preparado: MacBook Air M1, macOS Sonoma 14.6.1, Python 3.12.4 y 8 GB.

1. Descomprima el ZIP y mueva la carpeta completa a Documentos.
2. Abra la carpeta en Finder.
3. Haga clic derecho sobre INSTALAR_MAC_M1.command y elija Abrir.
4. Confirme Abrir si macOS muestra una advertencia.
5. Espere hasta leer INSTALACIÓN COMPLETADA.

El instalador crea un entorno privado llamado .venv. No modifica el Python
del sistema. No desactive Gatekeeper ni cambie la seguridad del Mac.

Si macOS bloquea un archivo, abra Ajustes del Sistema > Privacidad y seguridad
y use Abrir igualmente únicamente para este paquete.

## 2. Validación obligatoria antes de recolectar

Ejecute en este orden:

1. VALIDAR_VERSION_MAC.command
2. PROBAR_CAMARA_MAC.command
3. PROBAR_VOZ_MAC.command
4. Conecte el Mac a la red Wi-Fi ESP32_SPPB.
5. PROBAR_IMU_MAC.command

Cuando se solicite acceso a la cámara, permita Terminal o Python. Si no aparece
imagen, revise Ajustes del Sistema > Privacidad y seguridad > Cámara.

Para el IMU, acepte el acceso a la red local si macOS lo solicita. Mientras el
Mac esté conectado a ESP32_SPPB puede quedar temporalmente sin Internet.

Con 8 GB de memoria, cierre navegadores con muchas pestañas, videollamadas y
aplicaciones pesadas antes de iniciar un ensayo.

## 3. Inicio normal

Abra INICIAR_5XSTS_MAC.command. La interfaz permitirá registrar el participante,
el ensayo, la altura de la silla y el uso del IMU. No ejecute main.py
directamente para la recolección habitual.

## 4. Preparación del IMU

El firmware se conserva en MAESTRO_L5_STS/MAESTRO_L5_STS.ino.

- Tarjeta: ESP32-S3 Dev Module.
- IMU: MPU6050 local.
- SDA: GPIO 8.
- SCL: GPIO 9.
- Monitor serial: 115200 baudios.
- Red: ESP32_SPPB.
- Clave: 12345678.
- UDP: puerto 4212.
- Frecuencia nominal: 50 Hz.

Si el ESP32 ya tiene este firmware no necesita volver a cargarlo. Para
reprogramarlo en el Mac instale Arduino IDE y el paquete ESP32.

## 5. Montaje obligatorio

- Cámara fija en vista lateral, aproximadamente a la altura de la cadera.
- Cabeza, hombro, cadera, rodilla y tobillo visibles durante toda la prueba.
- Silla estable, preferiblemente sin apoyabrazos, con altura registrada.
- Pies apoyados en el suelo y brazos cruzados sobre el pecho.
- IMU centrado y firme en la zona lumbar, aproximadamente en L5.
- Buena iluminación y fondo con contraste.

No cambie orientación del sensor, altura de cámara ni silla entre ensayos sin
documentarlo.

## 6. Flujo de cada ensayo

1. Ingrese un ID anónimo, por ejemplo P001.
2. Ingrese el ensayo, por ejemplo T1, T2 o RETEST.
3. Confirme la altura real de la silla.
4. Confirme que el IMU está activo y el Mac conectado a ESP32_SPPB.
5. Presione INICIAR ENSAYO 5xSTS.
6. La persona permanece sentada y quieta durante la verificación.
7. El programa muestra la cuenta regresiva.
8. El sonido define GO.
9. La persona realiza cinco levantamientos sin usar los brazos.
10. La prueba termina completamente de pie en R5.

Teclas de seguridad:

- r: descartar el intento y reiniciar.
- f: finalizar manualmente como incompleto.
- q: cerrar; será incompleto si ya había iniciado.

## 7. Eventos congelados

| Evento | Definición operativa |
|---|---|
| GO | Emisión de la señal sonora de inicio |
| MOi | Inicio detectado de la subida i |
| Ri | Posición de pie i confirmada durante 0.25 s |
| Di | Inicio detectado del descenso i |
| Si | Posición sentada i confirmada durante 0.25 s |
| ABORT | Finalización manual antes de completar la prueba |

Secuencia esperada: GO, MO1, R1, D1, S1, hasta MO5, R5. La quinta repetición
termina en R5; no existe D5 ni S5.

## 8. Tiempos y resultado

- Tiempo clínico: R5 menos GO.
- Tiempo efectivo: R5 menos MO1.
- Subida i: Ri menos MOi.
- Descenso i: Si menos Di.
- Pausa de pie i: Di menos Ri.
- Pausa sentado i: MO siguiente menos Si.

El subpuntaje de silla se calcula solo con cinco repeticiones válidas. Una
prueba incompleta no recibe puntaje.

## 9. Archivos

Cada intento usa ID, ensayo, fecha y hora para evitar sobrescrituras. Se guardan
video original, video anotado, timeline, señales y eventos de cámara, señales y
detecciones del IMU, fusión, concordancia, métricas por repetición, resumen,
metadata y gráficas.

No renombre ni edite archivos dentro de resultados durante la recolección.
Copie periódicamente la carpeta resultados completa a dos ubicaciones seguras.

## 10. Regla científica

La diferencia entre cámara e IMU es concordancia técnica, no validación. El
evento S de cámara usa flexión de rodilla y el IMU usa dinámica lumbar, por lo
que pueden representar instantes biomecánicos distintos.

No modifique los archivos del algoritmo después de comenzar la muestra. Todo
cambio de umbral o regla requiere una versión nueva y separar los ensayos.
