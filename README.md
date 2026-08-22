# Detector de Somnolencia

Programa en Python que usa la webcam para detectar en tiempo real si el
conductor tiene los ojos cerrados durante varios segundos seguidos, y en ese
caso dispara una alerta visual en pantalla y una alarma sonora.

Funciona midiendo el **EAR (Eye Aspect Ratio)** de ambos ojos a partir de los
puntos faciales que detecta **MediaPipe Face Landmarker**. El código está en
`detector_somnoliencia.py` y tiene comentarios muy detallados pensados para
alguien sin experiencia previa en Python, visión por computadora o IA.

> La primera vez que lo ejecutes, el programa descarga automáticamente el
> modelo de detección facial de MediaPipe (`face_landmarker.task`, ~4 MB) y
> lo guarda en esta misma carpeta. Necesitás conexión a internet solo esa
> primera vez.

## Cómo ejecutarlo desde la consola (cmd)

### Opción recomendada: con entorno virtual

```cmd
cd C:\Users\marie\OneDrive\Documentos\Desarrollo\Detector-Somnoliencia

REM 1. Crear el entorno virtual (una sola vez)
python -m venv venv

REM 2. Activar el entorno virtual (cada vez que abras una consola nueva)
venv\Scripts\activate.bat

REM 3. Instalar las dependencias (una sola vez, o si cambia requirements.txt)
pip install -r requirements.txt

REM 4. Ejecutar el programa
python detector_somnoliencia.py
```

### Opción simple: sin entorno virtual

```cmd
cd C:\Users\marie\OneDrive\Documentos\Desarrollo\Detector-Somnoliencia
pip install -r requirements.txt
python detector_somnoliencia.py
```

## Controles

- Se abre una ventana mostrando la imagen de la cámara con los puntos de los
  ojos marcados, el valor de EAR en vivo y el contador de frames con ojos
  cerrados.
- Presioná **`q`** con la ventana enfocada para cerrar el programa
  correctamente.

## Ajustar la sensibilidad

Todos los parámetros configurables están al principio de
`detector_somnoliencia.py`, en la sección `CONSTANTES DE CONFIGURACIÓN`:

- `EAR_THRESHOLD` (por defecto `0.22`): si el programa dispara la alarma con
  los ojos abiertos, bajalo un poco (por ejemplo `0.18`). Si no detecta
  cuando cerrás los ojos, subilo (por ejemplo `0.25`).
- `DROWSY_TIME_SECONDS` (por defecto `2.0`): cuántos segundos de ojos
  cerrados seguidos se consideran somnolencia.
- `ASSUMED_FPS` (por defecto `15`): a cuántos cuadros por segundo asumimos
  que corre el programa en tu computadora, para convertir los segundos de
  arriba en una cantidad de frames.

## Solución de problemas

- **"ERROR: no se pudo acceder a la webcam"**: revisá que ninguna otra
  aplicación esté usando la cámara (Zoom, Teams, otra pestaña del
  navegador, etc.) y que en **Configuración > Privacidad y seguridad >
  Cámara** de Windows esté permitido el acceso para aplicaciones de
  escritorio.
- **Error al importar `cv2` o `mediapipe`**: significa que las dependencias
  no están instaladas en el entorno de Python activo. Ejecutá de nuevo
  `pip install -r requirements.txt` (asegurándote de tener el entorno
  virtual activado, si estás usando uno).
- **`AttributeError: module 'mediapipe' has no attribute 'solutions'`**: es un
  error de versiones viejas del código. MediaPipe eliminó la API antigua
  (`mp.solutions.face_mesh`) en sus versiones recientes; este proyecto ya usa
  la API nueva (`mediapipe.tasks`), así que asegurate de estar corriendo la
  versión actual de `detector_somnoliencia.py`.
- **Error al descargar el modelo**: revisá tu conexión a internet y volvé a
  ejecutar el programa. También podés descargar el archivo manualmente desde
  la URL que figura en la constante `MODELO_URL` del script y guardarlo en
  esta carpeta con el nombre `face_landmarker.task`.
