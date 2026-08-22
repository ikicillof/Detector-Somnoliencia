# Detector de Somnolencia

Programa en Python que detecta la somnolencia de un conductor en tiempo real,
usando **dos señales independientes**:

1. **Ojos cerrados (cámara)**: mide el **EAR (Eye Aspect Ratio)** de ambos ojos
   a partir de los puntos faciales que detecta **MediaPipe Face Landmarker**.
   Si los ojos permanecen cerrados unos 2 segundos seguidos, alerta.
2. **Cabezazos (acelerómetro por Bluetooth)**: detecta movimientos bruscos de
   la cabeza. Si ocurren **2 o más cabezazos en menos de 20 segundos**, alerta.

Ambas disparan la misma alarma sonora y un cartel rojo en pantalla.

El código está repartido en dos archivos, ambos con comentarios muy detallados
pensados para alguien sin experiencia previa en Python, visión por computadora
o IA:

- `detector_somnoliencia.py` — programa principal y detección por cámara.
- `sensor_acelerometro.py` — lectura del acelerómetro y detección de cabezazos.

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
  ojos marcados, el valor de EAR en vivo, el contador de frames con ojos
  cerrados y el estado del acelerómetro.
- Presioná **`q`** con la ventana enfocada para cerrar el programa
  correctamente.
- Presioná **`c`** para simular un cabezazo (solo en modo simulador).

## Acelerómetro: detección de cabezazos

El acelerómetro se coloca en la cabeza (vincha, gorra o auricular) y envía
datos por Bluetooth. El programa detecta movimientos bruscos y alerta cuando
ocurren **2 o más en menos de 20 segundos**.

### Elegir el modo

Se elige con la opción `--modo` al ejecutar, sin necesidad de editar nada:

| Modo | Comando | Requiere |
|---|---|---|
| Simulador (por defecto) | `python detector_somnoliencia.py` | Nada |
| Solo cámara | `python detector_somnoliencia.py --modo desactivado` | Nada |
| Bluetooth Clásico | `python detector_somnoliencia.py --modo clasico --puerto COM5` | `pip install pyserial` |
| Bluetooth BLE | `python detector_somnoliencia.py --modo ble --nombre-ble MiSensor` | `pip install bleak` |

Para ver todas las opciones: `python detector_somnoliencia.py --help`

El valor por defecto es `"simulador"` porque el hardware todavía no está
definido; se puede cambiar en la constante `MODO_SENSOR`.

> **Probar sin hardware:** en modo simulador, apretá **`c`** dos veces con
> menos de 20 segundos de diferencia y debería saltar la alerta. Si dejás
> pasar más de 20 segundos entre una y otra, no salta — así verificás que la
> ventana de tiempo funciona.

Si el sensor falla al conectar, el programa **no se cae**: avisa por consola
y sigue funcionando solo con la cámara.

### Formato de datos que debe enviar el sensor

Cada muestra es **una línea de texto** terminada en salto de línea:

```
t_ms,ax,ay,az
```

- `t_ms`: entero, milisegundos desde que arrancó el microcontrolador
  (en Arduino/ESP32 es directamente `millis()`).
- `ax,ay,az`: decimales, aceleración de cada eje en m/s².

Ejemplo de código para el lado del sensor (Arduino / ESP32):

```cpp
Serial.print(millis());   Serial.print(",");
Serial.print(ax, 2);      Serial.print(",");
Serial.print(ay, 2);      Serial.print(",");
Serial.println(az, 2);
```

> **Por qué el sensor debe mandar su propia marca de tiempo:** el Bluetooth
> introduce un retraso pequeño pero variable (*jitter*). Si midiéramos los 20
> segundos con la hora de llegada a la computadora, ese jitter ensuciaría la
> medición. Usando el tiempo del propio microcontrolador, la distancia entre
> dos cabezazos se mide tal como ocurrió en la cabeza del conductor, sin
> importar cuánto tarden los datos en llegar.
>
> El programa igual acepta líneas de solo `ax,ay,az` (sin marca de tiempo),
> usando la hora de llegada como respaldo, con algo menos de precisión.

### Ajustar la detección de cabezazos

Los parámetros están al principio de `sensor_acelerometro.py`:

- `UMBRAL_MOVIMIENTO_BRUSCO` (por defecto `3.5` m/s²): cuánto tiene que
  apartarse la aceleración de su valor de reposo para contar como cabezazo.
  Si detecta cabezazos que no existen, subilo; si no detecta los reales,
  bajalo. Como referencia, la gravedad es 9.81 m/s².
- `VENTANA_CABEZAZOS_SEG` (por defecto `20.0`): la ventana de tiempo.
- `MIN_CABEZAZOS_PARA_ALERTA` (por defecto `2`): cuántos disparan la alerta.
- `TIEMPO_REFRACTARIO_SEG` (por defecto `0.4`): después de contar un
  cabezazo, cuánto tiempo se ignora el sensor para no contar el mismo
  sacudón muchas veces.

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
- **"AVISO: no se pudo iniciar el acelerometro"**: el programa sigue andando
  solo con la cámara. Revisá que la librería correspondiente esté instalada
  (`pyserial` o `bleak`), que el sensor esté encendido y emparejado, y que
  el `PUERTO_COM` o el nombre BLE configurados sean los correctos.
- **"Acelerometro: SIN SENAL" en pantalla**: la conexión se abrió pero dejaron
  de llegar datos. Suele ser el sensor apagado, sin batería o fuera de rango.
