# Checklist de aceptación antes del primer participante

La V4.1 para Mac M1 está congelada a nivel de código, pero debe superar una prueba de
aceptación con la cámara, computadora, silla e IMU reales que se usarán en el
estudio. Esta comprobación no se realiza con participantes definitivos.

## A. Instalación

- [ ] `INSTALAR_MAC_M1.command` termina sin errores.
- [ ] `VALIDAR_VERSION_MAC.command` muestra “VALIDACIÓN TÉCNICA SUPERADA”.
- [ ] `PROBAR_CAMARA_MAC.command` muestra imagen fluida de la cámara elegida.
- [ ] macOS tiene permitido el acceso a Cámara para Terminal o Python.
- [ ] El Mac está conectado a `ESP32_SPPB`.
- [ ] `PROBAR_IMU_MAC.command` recibe muestras cercanas a 50 Hz.
- [ ] macOS tiene permitido el acceso a la red local si lo solicitó.
- [ ] La voz en español funciona o se decide trabajar sin instrucciones de voz.
- [ ] El tono GO se escucha claramente aunque la voz esté desactivada.

## B. Montaje congelado

- [ ] Se eligió una única silla y se midió su altura.
- [ ] Se marcó en el suelo la posición de las patas de la silla.
- [ ] Se marcó la posición y altura del trípode.
- [ ] Se documentó el lado de grabación.
- [ ] Se documentó la orientación del IMU.
- [ ] Se verificó que el cuerpo completo permanece visible.

## C. Tres ensayos técnicos

Realizar con un voluntario técnico usando el ID `PTEST`:

1. `NORMAL`: velocidad habitual.
2. `LENTO`: ejecución deliberadamente lenta.
3. `PAUSA`: pausa breve sentado o de pie.

En cada ensayo comprobar:

- [ ] Se escucha un único tono GO.
- [ ] El cronómetro visual comienza con el tono, no con la instrucción previa.
- [ ] El sistema cuenta exactamente cinco levantamientos.
- [ ] Finaliza de pie en R5.
- [ ] `eventos_camara_...csv` contiene 19 eventos desde GO hasta R5.
- [ ] Existen `MO1–MO5`, `R1–R5`, `D1–D4` y `S1–S4`.
- [ ] `detecciones_imu_l5_...csv` contiene cinco R y cuatro S.
- [ ] Se guardan el video original y el anotado.
- [ ] `timeline_video_...csv` no está vacío.
- [ ] `metadata_...json` contiene V4.1.0-MAC-M1, ensayo y altura de silla.
- [ ] `resumen_sts_...csv` dice `sppb_total_calculado=False`.
- [ ] La pérdida de muestras IMU y los huecos de cámara son aceptables y quedan
  registrados.

## D. Decisión

- [ ] Si los tres ensayos cumplen, conservar el ZIP original y comenzar la
  recolección sin modificar el código.
- [ ] Si alguno falla, no comenzar la muestra definitiva. Conservar el ensayo
  fallido, documentar el problema y crear una versión nueva después de corregir.

No use los ensayos `PTEST` en el análisis final del paper.
