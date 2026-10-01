# Detector de Somnolencia

Programa en Python que detecta la somnolencia de un conductor en tiempo real
usando **solo la cámara**. Toda la información sale de la misma malla de
puntos faciales que devuelve **MediaPipe Face Landmarker**: no hay ningún
sensor externo.

Vigila dos cosas al mismo tiempo:

1. **Ojos cerrados**: mide el **EAR (Eye Aspect Ratio)** de ambos ojos. Si
   los ojos quedan cerrados unos 2 segundos seguidos (no un parpadeo), alerta.
2. **Cabeceos**: estima la **pose de la cabeza** (ángulo de *pitch*, la
   inclinación vertical) con `cv2.solvePnP` sobre los mismos landmarks, y
   detecta dos patrones:
   - **Cabeza caída sostenida**: la cabeza queda inclinada hacia abajo más de
     cierto ángulo durante varios segundos.
   - **Cabeceo brusco**: la cabeza cae de golpe y se endereza en menos de un
     segundo (se detecta por la velocidad del movimiento, no por la posición).

Cualquiera de las tres condiciones dispara la misma **alarma sonora** y un
**cartel rojo** en pantalla.

> Apenas arranca, el programa se toma unos segundos para **calibrar** la
> posición neutra de la cabeza: mirá al frente y quedate quieto hasta que
> desaparezca el cartel "Calibrando...".

El código está en un solo archivo, con comentarios muy detallados pensados
para alguien sin experiencia previa en Python, visión por computadora o IA:

- `detector_somnoliencia.py` — programa principal y detección por cámara.

> La primera vez que lo ejecutes, el programa descarga automáticamente el
> modelo de detección facial de MediaPipe (`face_landmarker.task`, ~4 MB) y
> lo guarda en esta misma carpeta. Necesitás conexión a internet solo esa
> primera vez.

## Cómo ejecutarlo desde la consola (cmd)

### Opción recomendada: con entorno virtual

```cmd
cd <carpeta del proyecto>

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
cd <carpeta del proyecto>
pip install -r requirements.txt
python detector_somnoliencia.py
```

## Controles

- Se abre una ventana con la imagen de la cámara (espejada), los puntos de
  los ojos marcados y, abajo a la izquierda, los valores en vivo: EAR y su
  temporizador, **pitch crudo** y **pitch suavizado** por separado,
  desviación respecto del neutro, velocidad angular y el temporizador de
  cabeza caída. Esos números son los que se usan para calibrar los umbrales.
- Al arrancar hay unos segundos de
  **calibración**: mirá al frente y quedate quieto. Si te movés o no mirás
  de frente, la calibración se descarta y se reintenta sola, todas las veces
  que haga falta, sin tocar ninguna tecla. Mientras tanto la detección de
  ojos cerrados sigue funcionando.
- Presioná **`q`** con la ventana enfocada para cerrar el programa
  correctamente.
- En el modo **sin ventana** (`--sin-ventana`), escribí **`c`** + Enter en
  la consola para volver a calibrar la posición neutra de la cabeza (por
  ejemplo si moviste la cámara). Con ventana la calibración es solo
  automática.

## Ajustar la sensibilidad

Todos los parámetros configurables están al principio de
`detector_somnoliencia.py`, en la sección `CONSTANTES DE CONFIGURACIÓN`, cada
uno con un comentario que explica qué representa y en qué rango moverlo.
Toda la lógica temporal se mide en **segundos** (con `time.time()`), nunca en
cantidad de cuadros. Los principales:

**Ojos:**

- `EAR_THRESHOLD` (por defecto `0.22`): si dispara con los ojos abiertos,
  bajalo (p. ej. `0.18`); si no detecta con los ojos cerrados, subilo
  (p. ej. `0.25`).
- `DROWSY_TIME_SECONDS` (por defecto `2.0`): segundos de ojos cerrados
  seguidos que cuentan como somnolencia.

**Cabeceos:**

- `CALIBRACION_SEGUNDOS` (por defecto `3.0`): cuánto dura la calibración
  inicial de la posición neutra de la cabeza.
- `CABEZA_CAIDA_GRADOS` / `CABEZA_CAIDA_SEGUNDOS` (por defecto `15.0` / `1.5`):
  cuántos grados por debajo del neutro y durante cuántos segundos cuenta como
  "cabeza caída sostenida". Mirá el valor `desv` en pantalla para calibrar.
- `CABECEO_VELOCIDAD_GRADOS_POR_SEG` / `CABECEO_AMPLITUD_MINIMA_GRADOS` (por
  defecto `55.0` / `10.0`): qué tan rápida y qué tan amplia tiene que ser una
  caída para contar como "cabeceo brusco". Mirá el valor `vel` en pantalla.
- `EAR_HISTERESIS` / `CABEZA_CAIDA_HISTERESIS_GRADOS` (por defecto `0.02` /
  `3.0`): margen entre el umbral para *entrar* en alerta y el umbral para
  *salir*, para que la alerta no parpadee cuando el valor oscila en el límite.
- `SIGNO_PITCH` (por defecto `None` = automático): el programa verifica solo
  el signo del pitch comparándolo con la geometría de la cara, apenas movés
  la cabeza. Solo tocalo si querés forzarlo (`1.0` o `-1.0`).

**Cámara:**

- `CAMARA_INDICE` (por defecto `0`): si agarra la cámara equivocada, probá
  `1`, `2`, etc.

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
