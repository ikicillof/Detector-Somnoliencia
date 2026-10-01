# Progreso del proyecto

## 2026-10-01 — Port a Raspberry Pi y optimización

Rama: `ccr-a4fc3609-4b8vgx`.

### Situación de partida

El detector ya corría en la Pi, pero con parches a mano y mucho delay:

- `import winsound` rompía en Linux: se había parcheado con un
  `winsound.py` falso dentro del venv.
- mediapipe 1.0.x (la única para Python 3.13) muere en la Pi 3/4 por
  instrucciones AES. Se armó a mano un venv con Python 3.11 (uv) y
  mediapipe 0.10.18 con `--no-deps` (sin jax/jaxlib/scipy). Eso obligó a
  `numpy<2` y `opencv-contrib-python<4.11`.
- OpenCV no estaba en `requirements.txt` (llegaba vía mediapipe).
- Por SSH, `cv2.imshow` abortaba (`could not connect to display`).
- La cámara acumulaba cuadros: la imagen y la alarma llegaban tarde.

### Hecho hoy

**Fase 1 — Alarma multiplataforma**
- Eliminado `winsound`. Nueva clase `Alarma` (`disparar`, `callar`,
  `cerrar`) que elige la salida al arrancar:
  - Raspberry Pi (detectada por `/proc/device-tree/model`): buzzer activo
    con gpiozero + lgpio en `BUZZER_GPIO = 21` (pin físico 40, vía
    transistor NPN; cableado documentado en el código).
  - PC/Windows: tono de 2500 Hz con `sounddevice` + `numpy`.
  - Si falla la librería: advertencia y alarma solo visual/consola.
- Los pitidos corren en un hilo trabajador: nunca bloquean el video.
- Dos niveles: **peligro** (ojos cerrados / cabeza caída: pitido largo
  repetido) y **aviso** (cabeceo brusco: dos pitidos cortos).

**Fase 2 — Modo sin ventana**
- Flag `--sin-ventana`; se activa solo en Linux sin `DISPLAY` ni
  `WAYLAND_DISPLAY`. Sin ventana no se dibuja nada.
- Ctrl+C y SIGTERM cierran limpio (alarma, cámara, MediaPipe).
- Eventos en consola con hora: inicio, calibración, rostro
  detectado/perdido, inicio y fin de cada alerta, cabeceos.
- **Calibración:** si falla, se reintenta sola sin límite (se eliminó la
  "calibración dudosa" y `CALIBRACION_MAX_INTENTOS`). Recalibrar a mano solo
  en modo sin ventana (`c` + Enter); con ventana es solo automática.

**Fase 3 — Rendimiento y delay**
- `CapturaEnHilo`: la cámara se lee en un hilo que guarda solo el último
  cuadro; el bucle siempre procesa el más reciente.
- `abrir_camara`: V4L2 en Linux, DirectShow en Windows, MJPG y
  `CAP_PROP_BUFFERSIZE = 1`.
- `--ancho`/`--alto` (640x480 por defecto en la Pi) y `--saltar N`.
- Cada 5 s: FPS de captura y de procesamiento, ms por cuadro, atraso, y
  EAR/desviación/velocidad (actual, mínimo de la ventana y umbral).
- Aviso de `vcgencmd get_throttled` si no da `0x0`.
- Face Landmarker sin blendshapes ni matrices de transformación.

**Fase 4 — Dependencias e instalación**
- `requirements.txt` (PC) con `opencv-contrib-python` y `sounddevice`.
- `requirements-pi.txt` sin mediapipe (y sin jax/scipy).
- `instalar_pi.sh`: chequeo de espacio, uv, venv 3.11, `TMPDIR=~/tmp`,
  dependencias, mediapipe 0.10.18 `--no-deps`, limpieza, verificación.
  También borra el `winsound.py` falso si quedó.
- `.gitignore`: venv, `__pycache__`, modelos `.task`, `tmp/`.

**Fase 5 — Documentación**
- `INSTALACION_PI.md` para el grupo (acceso remoto, instalación, uso,
  buzzer, problemas conocidos).
- `CLAUDE.md` con las restricciones permanentes.
- Este archivo.

**Cambio posterior:** el buzzer pasó del GPIO18 (pin 12) al **GPIO21
(pin físico 40)**. Buzzer probado en la Pi: funciona (faltaba instalar
`gpiozero`/`lgpio` en el venv con `./instalar_pi.sh`).

**Pitido de calibración:** al terminar bien la calibración suena un pitido
muy corto (`NIVEL_CALIBRADO`, 0,08 s), con menor prioridad que las alertas.
En modo solo visual no imprime nada extra.

**Tic de rostro perdido:** mientras la cámara no ve la cara (después de
`SIN_ROSTRO_ESPERA_SEGUNDOS = 1.0`), suena un "tic" de 0,03 s cada ~1,2 s
(`NIVEL_SIN_ROSTRO`, la prioridad más baja). Se corta apenas vuelve la cara.
Además, ahora un pedido de mayor prioridad corta al instante el patrón de
menor prioridad que esté sonando, para que una alerta nunca espere.

**Umbral de ojos:** a un compañero con ojos achinados lo marcaba como ojos
cerrados con los ojos abiertos. Primero se bajó `EAR_THRESHOLD` de 0.22 a
0.18; después (pedido explícito) el umbral pasó a **calibrarse solo por
persona** (`CalibradorEAR`): durante `EAR_CALIBRACION_SEGUNDOS` se mide la
mediana del EAR con ojos abiertos y el umbral queda en
`EAR_FRACCION_UMBRAL = 0.70` de ese valor, limitado entre 0.12 y 0.26. Si la
mediana da menos de `EAR_ABIERTO_MINIMO = 0.15` (ojos cerrados al calibrar)
se reintenta sola. Mientras tanto se usa `EAR_THRESHOLD = 0.12` de respaldo,
así la alarma de ojos funciona desde el primer segundo sin falsas alarmas
con ojos achinados. La tecla `c` (sin ventana) recalibra también los ojos.
Pendiente: probarlo con el compañero y ajustar `EAR_FRACCION_UMBRAL`.

### Cómo se probó

- Alarma, modo sin ventana, captura en hilo, argumentos y `vcgencmd`:
  con cámara/MediaPipe/GPIO simulados.
- `instalar_pi.sh` corrido completo (dos veces) en una PC x86 con Python
  3.11: instala mediapipe 0.10.18, OpenCV 4.10, numpy 1.26, gpiozero y
  lgpio, **sin jax ni scipy**; el detector arranca con esa mediapipe real.
- **Todavía NO probado en la Pi real** ni con el buzzer físico.

### Pendiente

- [x] Probar en la Pi: `git pull`, `./instalar_pi.sh` y
      `python detector_somnoliencia.py --sin-ventana`.
- [x] Confirmar que el buzzer suena (falta escuchar bien cada patrón).
- [ ] Medir FPS reales en la Pi (línea cada 5 s) y elegir resolución /
      `--saltar` adecuados.
- [ ] Revisar `vcgencmd get_throttled` (fuente y temperatura).
- [ ] Verificar si `lgpio` y `sentencepiece` instalan wheels o compilan en
      la Pi (tiempo de instalación).
- [ ] Recalibrar umbrales con la cámara montada en la Pi si cambia la
      posición respecto de la PC.
- [ ] Cambiar la SD de 8 GB por una de 32 GB.
- [ ] Opcional: que el detector arranque solo al prender la Pi (servicio
      systemd).
- [ ] Opcional: `--guardar-imagen` para ver lo que detecta sin pantalla.
- [ ] Opcional: volver a habilitar la recalibración manual con ventana.
- [ ] Hacer merge de la rama a `main`.
