# CLAUDE.md

Instrucciones para trabajar en este repositorio. Leé también `PROGRESO.md`
(qué está hecho y qué falta) antes de escribir código.

## El proyecto

Detector de somnolencia para conductores, en Python, usando solo una cámara.
Todo está en `detector_somnoliencia.py`:

- **Ojos cerrados:** EAR (Eye Aspect Ratio) sobre los landmarks de MediaPipe
  Face Landmarker.
- **Cabeceos:** pitch de la cabeza con `cv2.solvePnP`; detecta cabeza caída
  sostenida y cabeceo brusco.
- **Alarma:** clase `Alarma` (buzzer por GPIO en la Pi, `sounddevice` +
  `numpy` en la PC, solo visual/consola si nada carga).
- Corre en PC/Windows (desarrollo) y en una Raspberry Pi 3/4 (uso real).

Lo usan y leen estudiantes sin experiencia: el código lleva comentarios muy
detallados, en español, y los mensajes al usuario también son en español.

## Reglas de trabajo

- **No cambiar la lógica de detección** (EAR, solvePnP, umbrales,
  `SIGNO_PITCH`, `CABEZA_CAIDA_GRADOS`, etc.) salvo que se pida
  explícitamente. Los umbrales están calibrados con la cámara real.
- Todos los tiempos de la detección se miden en **segundos con
  `time.time()`**, nunca contando cuadros.
- Toda configuración ajustable va como constante al principio del archivo,
  con un comentario que explique qué hace y en qué rango moverla.
- Trabajar por fases y mostrar un resumen al terminar cada una.

## Restricciones permanentes

- **En la Raspberry Pi se usa Python 3.11 con mediapipe 0.10.18 instalada
  sin dependencias** (`uv pip install --no-deps "mediapipe==0.10.18"`). Las
  dependencias que sí hacen falta están a mano en `requirements-pi.txt`. La
  instalación se hace con `instalar_pi.sh`. Motivo: mediapipe 1.0.x (la
  única para Python 3.13) usa instrucciones AES que la Pi 3/4 no tiene.
- **Nunca agregar `jax`, `jaxlib` ni `scipy`** (a ningún requirements ni
  como import): pesan cientos de MB, llenan `/tmp` (en RAM) y la SD de la Pi,
  y no se usan.
- En la Pi, mediapipe 0.10.18 exige `numpy<2`, y por eso
  `opencv-contrib-python<4.11`. No usar `opencv-python` junto con
  `opencv-contrib-python` (se pisan).
- **Nunca usar `winsound`** ni nada que funcione solo en Windows. La alarma
  pasa siempre por la clase `Alarma`.
- **El programa tiene que funcionar sin pantalla:** con `--sin-ventana` (o
  sin `DISPLAY`/`WAYLAND_DISPLAY` en Linux) no se llama a `cv2.imshow` ni a
  `cv2.waitKey`, todos los eventos importantes se imprimen en consola, y
  Ctrl+C cierra limpio (cámara, buzzer, hilos).
- **La alarma en la Pi es un buzzer activo por GPIO** (gpiozero con backend
  lgpio), en `BUZZER_GPIO = 21` (pin físico 40) a través de un transistor.
  La alarma nunca bloquea el bucle de video ni corta el programa si falla.
- La cámara se lee en un hilo aparte que guarda solo el último cuadro
  (`CapturaEnHilo`); no volver a leer la cámara directamente en el bucle.

## Comandos útiles

```bash
# PC / Windows
pip install -r requirements.txt
python detector_somnoliencia.py

# Raspberry Pi
./instalar_pi.sh
source venv/bin/activate
python detector_somnoliencia.py --sin-ventana [--ancho 320 --alto 240] [--saltar 2]
```

Documentación para el grupo: `README.md` (general) e `INSTALACION_PI.md`
(Raspberry Pi: acceso remoto, instalación y problemas conocidos).
