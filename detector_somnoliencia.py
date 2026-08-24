# ==============================================================================
# DETECTOR DE SOMNOLENCIA EN TIEMPO REAL
# ==============================================================================
#
# ¿QUÉ HACE ESTE PROGRAMA?
# ------------------------
# Detecta si un conductor se está quedando dormido, vigilando DOS SEÑALES
# distintas al mismo tiempo:
#
# 1) LOS OJOS (con la webcam, en este archivo). Usa la cámara para mirar tu
#    cara en tiempo real, detecta tus ojos y calcula qué tan "abiertos" o
#    "cerrados" están usando una fórmula matemática llamada EAR (Eye Aspect
#    Ratio = "Relación de Aspecto del Ojo"). Si tuviste los ojos cerrados
#    durante varios cuadros (frames) seguidos —lo cual indica que
#    probablemente te estás quedando dormido y no simplemente parpadeando—
#    dispara la alarma.
#
# 2) LOS CABEZAZOS (con un acelerómetro por Bluetooth, en el archivo
#    sensor_acelerometro.py). Un acelerómetro colocado en la cabeza mide los
#    movimientos bruscos característicos de alguien que da un cabezazo y se
#    despierta de golpe. Si ocurren dos o más de esos movimientos en menos de
#    20 segundos, también dispara la alarma.
#
# Las dos señales son independientes: cada una puede disparar la alerta por
# su cuenta, y ambas comparten el mismo pitido de alarma y un cartel rojo en
# pantalla. Que sean independientes es importante, porque se complementan: la
# cámara falla si hay poca luz o el conductor usa anteojos oscuros, y el
# acelerómetro no sirve si el conductor se duerme sin mover la cabeza.
#
# ¿QUÉ ES "MEDIAPIPE FACE LANDMARKER"?
# -------------------------------------
# MediaPipe es una librería de Google que, a partir de la imagen de la cámara,
# nos devuelve unos 478 puntos (coordenadas x, y) distribuidos sobre toda la
# cara: ojos, cejas, nariz, boca, contorno de la cara, etc. A esta red de
# puntos se la suele llamar "malla facial". Nosotros solo necesitamos 6 puntos
# de cada ojo para poder calcular el EAR.
#
# Para poder detectar esos puntos, MediaPipe necesita un archivo con una red
# neuronal ya entrenada (un "modelo"), llamado `face_landmarker.task`. Este
# programa lo descarga automáticamente la primera vez que lo corrés (pesa
# unos 4 MB) y lo guarda en la misma carpeta del proyecto, así las próximas
# veces ya no hace falta descargarlo de nuevo.
#
# ¿QUÉ ES EL EAR (Eye Aspect Ratio)?
# ------------------------------------
# Es un número que resume qué tan abierto está un ojo, calculado a partir de
# 6 puntos ubicados en el contorno del ojo:
#
#         p2       p3
#          *       *
#     p1 *           * p4      <- p1 y p4 son las "puntas" (comisuras) del ojo
#          *       *
#         p6       p5
#
# La fórmula es:
#
#     EAR = ( distancia(p2, p6) + distancia(p3, p5) ) / ( 2 * distancia(p1, p4) )
#
# - El NUMERADOR suma dos distancias VERTICALES (qué tan separados están el
#   párpado de arriba y el de abajo, en dos puntos distintos del ojo).
# - El DENOMINADOR es la distancia HORIZONTAL entre las dos puntas del ojo
#   (el "ancho" del ojo), multiplicada por 2 para que la fórmula quede
#   normalizada (es decir, que el resultado no dependa de qué tan cerca o
#   lejos esté tu cara de la cámara).
#
# Cuando el ojo está ABIERTO, la distancia vertical es grande en comparación
# con la horizontal, así que el EAR da un número "alto" (típicamente 0.25-0.35).
# Cuando el ojo se CIERRA, la distancia vertical colapsa casi a cero mientras
# que la distancia horizontal casi no cambia, así que el EAR CAE bruscamente
# (típicamente por debajo de 0.20). Ese es el truco: no medimos "si el ojo
# está cerrado" directamente, medimos una relación geométrica que baja mucho
# cuando el ojo se cierra.
#
# ¿POR QUÉ NO ALCANZA CON UN SOLO CUADRO (FRAME) DE OJOS CERRADOS?
# -------------------------------------------------------------------
# Porque parpadear es normal y sano: un parpadeo dura apenas unos 100-400
# milisegundos. Si disparáramos la alarma apenas viéramos UN frame con EAR
# bajo, la alarma sonaría todo el tiempo por simples parpadeos, lo cual
# sería inútil y molesto. Por eso exigimos que el EAR esté por debajo del
# umbral durante VARIOS FRAMES SEGUIDOS, lo que en la práctica equivale a
# unos segundos de ojos cerrados de forma sostenida: eso sí es una señal
# real de somnolencia (o "microsueño"), no un parpadeo normal.
#
# ==============================================================================

# --- Librerías de la biblioteca estándar de Python (ya vienen instaladas) ---
import argparse  # Para poder elegir opciones al ejecutar desde la consola,
                 # sin tener que editar el archivo cada vez.
import os
import sys
import time
import threading
import urllib.request  # Para descargar el modelo de MediaPipe la primera vez.
import winsound  # Módulo de Windows para reproducir sonidos simples (pitidos).
                 # Solo funciona en Windows, pero como este programa está
                 # pensado para correr desde la consola (cmd) de Windows,
                 # es la opción más simple: no requiere instalar nada extra
                 # ni conseguir archivos de sonido.

# --- Librerías de terceros (hay que instalarlas con pip, ver requirements.txt) ---
import cv2            # OpenCV: nos permite acceder a la webcam, mostrar la
                       # ventana de video y dibujar texto/formas sobre la imagen.
import mediapipe as mp  # MediaPipe: nos da los puntos de la cara (landmarks).
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np    # NumPy: lo usamos para calcular distancias entre puntos
                       # de forma simple y rápida.

# --- Módulos propios de este proyecto ---
# Se ocupan de leer, cada uno, su sensor por Bluetooth y de detectar su señal
# de somnolencia. Están en archivos al lado de este.
import sensor_acelerometro as sensor  # cabezazos (acelerómetro)
import sensor_pulso as pulso          # caídas de frecuencia cardíaca (BPM)


# ==============================================================================
# CONSTANTES DE CONFIGURACIÓN
# ==============================================================================
# Todos los valores que se pueden "ajustar" del programa están acá arriba,
# juntos, para que sea fácil experimentar sin tener que buscar en todo el
# código. Si el programa te resulta muy sensible o muy poco sensible, estos
# son los números que hay que tocar.

# Umbral de EAR: por debajo de este valor consideramos que el ojo está cerrado.
# 0.22 es un valor típico que funciona bien para la mayoría de las caras, pero
# puede variar un poco según la persona, el ángulo de la cámara o la
# iluminación. Si el programa dispara la alarma con los ojos abiertos, bajá
# este número (por ejemplo a 0.18). Si no detecta cuando cerrás los ojos,
# subilo (por ejemplo a 0.25).
EAR_THRESHOLD = 0.22

# Umbral de TIEMPO: cuántos segundos seguidos con los ojos "cerrados" (según
# el EAR) se consideran somnolencia real y no un simple parpadeo.
DROWSY_TIME_SECONDS = 2.0

# FPS (cuadros por segundo) que ASUMIMOS que va a procesar el programa. Esto
# es una aproximación: no todas las computadoras procesan la cámara a la
# misma velocidad. MediaPipe + OpenCV en Python, en una laptop típica, suele
# rondar entre 10 y 20 cuadros por segundo (mucho menos que los 30 FPS
# "nominales" de la cámara), así que 15 es un valor prudente por defecto.
ASSUMED_FPS = 15

# A partir del tiempo (en segundos) y los FPS asumidos, calculamos cuántos
# FRAMES SEGUIDOS con EAR bajo necesitamos para confirmar somnolencia.
# Con los valores de arriba: 2.0 segundos * 15 FPS = 30 frames.
EAR_CONSEC_FRAMES = int(DROWSY_TIME_SECONDS * ASSUMED_FPS)

# Configuración del pitido de alarma: frecuencia del sonido (en Hz, más alto
# = más agudo) y duración de cada pitido (en milisegundos).
ALARM_FREQ_HZ = 2500
ALARM_DURATION_MS = 700

# Dirección de internet de donde se descarga el modelo de detección facial
# de MediaPipe, y nombre con el que se guarda en la carpeta del proyecto.
MODELO_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODELO_NOMBRE_ARCHIVO = "face_landmarker.task"

# --- Configuración del acelerómetro (detección de cabezazos) ---
# De dónde salen los datos del acelerómetro. Las opciones son:
#   "simulador"   -> datos falsos generados por el programa. No necesita
#                    hardware: sirve para probar que todo funciona. En este
#                    modo podés apretar la tecla 'c' para simular un cabezazo.
#   "clasico"     -> Bluetooth Clásico (módulos HC-05 / HC-06 con Arduino).
#                    Requiere instalar pyserial y configurar PUERTO_COM.
#   "ble"         -> Bluetooth Low Energy (ESP32 y sensores modernos).
#                    Requiere instalar bleak y configurar los datos de abajo.
#   "desactivado" -> ignora el acelerómetro por completo y usa solo la cámara.
#
# Este es el valor POR DEFECTO. También se puede elegir al ejecutar el
# programa desde la consola, sin tocar el archivo, así:
#     python detector_somnoliencia.py --modo desactivado
MODO_SENSOR = "simulador"

# Solo se usa si MODO_SENSOR es "clasico". Es el puerto COM que Windows le
# asignó al módulo Bluetooth al emparejarlo (mirar en el Administrador de
# dispositivos, bajo "Puertos (COM y LPT)").
PUERTO_COM = "COM5"

# Solo se usan si MODO_SENSOR es "ble": el nombre con el que el sensor se
# anuncia por Bluetooth, y el UUID de la característica que manda los datos.
# Ambos los define el firmware del sensor.
NOMBRE_DISPOSITIVO_BLE = "AcelerometroCasco"
UUID_CARACTERISTICA_BLE = "0000ffe1-0000-1000-8000-00805f9b34fb"

# Cuántos segundos queda visible en pantalla el cartel de alerta por
# cabezazos. A diferencia de la alerta por ojos cerrados (que se mantiene
# sola mientras los ojos sigan cerrados), un cabezazo es un evento
# instantáneo: si no lo dejáramos fijo un rato, el cartel aparecería y
# desaparecería tan rápido que no llegarías a leerlo.
DURACION_ALERTA_CABEZAZOS_SEG = 5.0

# --- Configuración del sensor de pulso (frecuencia cardíaca / BPM) ---
# De dónde salen los datos de BPM. Las opciones son:
#   "simulador"   -> BPM falso generado por el programa. No necesita
#                    hardware: en este modo podés forzar una caída llamando
#                    a lector_pulso.forzar_caida() (ver sensor_pulso.py).
#   "ble"         -> Bluetooth Low Energy, sensor con Heart Rate Service
#                    estándar (por ejemplo, un Polar H10 o un Wahoo TICKR).
#   "desactivado" -> ignora el sensor de pulso por completo.
MODO_SENSOR_PULSO = "simulador"

# Nombre anunciado por BLE del sensor de pulso. Solo se usa si no se indica
# DIRECCION_BLE_PULSO (ver abajo).
NOMBRE_BLE_PULSO = "Polar H10"

# Dirección MAC del sensor de pulso (ej. "AA:BB:CC:DD:EE:FF"). Es OPCIONAL,
# pero conectar por MAC es más rápido y más confiable para la reconexión
# automática que buscar por nombre. Si es None, se busca por nombre.
DIRECCION_BLE_PULSO = None

# IMPORTANTE: la caída de BPM, por sí sola, NO dispara la alarma (el pulso
# es una señal más débil y más lenta que los ojos cerrados o un cabezazo).
# Solo cuenta como confirmación cuando se combina con OTRA señal (ojos
# cerrados o cabezazos) que esté activa dentro de esta misma ventana de
# tiempo. Este valor es cuánto tiempo (en segundos) después de confirmarse
# una caída de BPM seguimos considerándola "vigente" para poder combinarla
# con una señal de cámara o acelerómetro que aparezca poco después (o que ya
# estuviera activa).
DURACION_VENTANA_COMBINACION_PULSO_SEG = 10.0

# --- Índices de los puntos de MediaPipe que forman cada ojo ---
# MediaPipe numera sus puntos de la cara siempre en el mismo orden. Estos son los
# 6 índices que corresponden al contorno de cada ojo, ya ordenados para que
# coincidan con el dibujo de p1..p6 de la explicación de más arriba:
#   posición 0 -> p1 (comisura izquierda del ojo)
#   posición 1 -> p2 (párpado superior, lado izquierdo)
#   posición 2 -> p3 (párpado superior, lado derecho)
#   posición 3 -> p4 (comisura derecha del ojo)
#   posición 4 -> p5 (párpado inferior, lado derecho)
#   posición 5 -> p6 (párpado inferior, lado izquierdo)
OJO_DERECHO_IDX = [33, 160, 158, 133, 153, 144]   # ojo derecho de la persona
OJO_IZQUIERDO_IDX = [362, 385, 387, 263, 373, 380]  # ojo izquierdo de la persona


# ==============================================================================
# FUNCIONES
# ==============================================================================

def distancia_euclidiana(punto_a, punto_b):
    """Calcula la distancia en línea recta entre dos puntos (x, y).

    Es el simple teorema de Pitágoras: la distancia entre dos puntos es la
    raíz cuadrada de (diferencia en x al cuadrado + diferencia en y al
    cuadrado). NumPy nos hace esta cuenta con una sola función."""
    return np.linalg.norm(np.array(punto_a) - np.array(punto_b))


def obtener_coordenadas_ojo(landmarks, indices, ancho_frame, alto_frame):
    """Convierte los puntos "normalizados" de MediaPipe en coordenadas de píxel.

    MediaPipe no nos da las coordenadas directamente en píxeles: nos las da
    como una fracción entre 0.0 y 1.0 del ancho y del alto de la imagen (por
    ejemplo, x=0.5 significa "la mitad del ancho de la imagen", sin importar
    si la imagen mide 640 o 1920 píxeles de ancho). Para poder dibujar los
    puntos o usarlos en cálculos de píxeles, hay que multiplicarlos por el
    ancho y el alto reales del frame."""
    puntos = []
    for indice in indices:
        punto = landmarks[indice]
        x_pixel = int(punto.x * ancho_frame)
        y_pixel = int(punto.y * alto_frame)
        puntos.append((x_pixel, y_pixel))
    return puntos


def calcular_ear(puntos_ojo):
    """Calcula el EAR (Eye Aspect Ratio) de un ojo a partir de sus 6 puntos.

    'puntos_ojo' debe ser una lista de 6 tuplas (x, y) en el orden
    [p1, p2, p3, p4, p5, p6] descripto en la explicación del inicio del
    archivo. Devuelve un número float: cuanto más bajo, más cerrado está
    el ojo."""
    p1, p2, p3, p4, p5, p6 = puntos_ojo

    # Distancias verticales (entre el párpado de arriba y el de abajo).
    distancia_vertical_1 = distancia_euclidiana(p2, p6)
    distancia_vertical_2 = distancia_euclidiana(p3, p5)

    # Distancia horizontal (el "ancho" del ojo, entre sus dos comisuras).
    distancia_horizontal = distancia_euclidiana(p1, p4)

    # Fórmula del EAR: promedio de las dos distancias verticales, dividido
    # por la distancia horizontal (multiplicada por 2 para normalizar).
    ear = (distancia_vertical_1 + distancia_vertical_2) / (2.0 * distancia_horizontal)
    return ear


def asegurar_modelo_descargado():
    """Se fija si el archivo del modelo de MediaPipe ya existe en la carpeta
    del proyecto. Si no existe (por ejemplo, la primera vez que corrés el
    programa), lo descarga automáticamente de internet y lo guarda ahí.

    Esto evita que tengas que descargar el archivo a mano: el programa lo
    hace solo, una única vez. Devuelve la ruta completa al archivo del
    modelo, ya sea porque ya estaba descargado o porque se acaba de bajar."""
    carpeta_del_script = os.path.dirname(os.path.abspath(__file__))
    ruta_modelo = os.path.join(carpeta_del_script, MODELO_NOMBRE_ARCHIVO)

    if not os.path.exists(ruta_modelo):
        print("Descargando el modelo de detección facial (solo la primera vez, ~4 MB)...")
        try:
            urllib.request.urlretrieve(MODELO_URL, ruta_modelo)
            print("Modelo descargado correctamente.")
        except Exception as error:
            print(f"ERROR: no se pudo descargar el modelo de MediaPipe: {error}")
            print("Verificá tu conexión a internet e intentá ejecutar el programa de nuevo.")
            sys.exit(1)

    return ruta_modelo


def crear_detector_facial():
    """Crea y devuelve el detector de puntos faciales de MediaPipe (Tasks API).

    'num_faces=1' porque solo nos interesa la cara del conductor.
    'running_mode=VIDEO' le dice a MediaPipe que le vamos a ir pasando
    cuadros consecutivos de un video/cámara (en vez de fotos sueltas
    sin relación entre sí), lo que le permite aprovechar información del
    cuadro anterior para trackear la cara de forma más estable.
    'min_face_detection_confidence' / 'min_tracking_confidence' son qué tan
    seguro tiene que estar MediaPipe antes de darnos por válida una cara."""
    ruta_modelo = asegurar_modelo_descargado()
    opciones_base = mp_python.BaseOptions(model_asset_path=ruta_modelo)
    opciones = mp_vision.FaceLandmarkerOptions(
        base_options=opciones_base,
        num_faces=1,
        running_mode=mp_vision.RunningMode.VIDEO,
        min_face_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.FaceLandmarker.create_from_options(opciones)


def procesar_frame(frame, detector_facial, timestamp_ms):
    """Le pasa el frame a MediaPipe y, si encontró una cara, calcula el EAR
    promedio de ambos ojos.

    'timestamp_ms' es un número que tiene que ir SIEMPRE EN AUMENTO entre
    llamada y llamada (MediaPipe, al trabajar en modo VIDEO, exige que cada
    cuadro tenga una marca de tiempo mayor a la del cuadro anterior, para
    saber en qué orden ocurrieron). No hace falta que sea el tiempo real:
    alcanza con un contador que sume de a uno en cada frame.

    Devuelve una tupla (ear_promedio, puntos_ojo_izq, puntos_ojo_der).
    Si NO se detectó ninguna cara en el frame, devuelve (None, None, None)
    para que quien llama a esta función sepa que no hay datos válidos y no
    intente hacer cuentas con "nada"."""
    alto_frame, ancho_frame = frame.shape[:2]

    # MediaPipe espera la imagen en formato RGB, pero OpenCV lee la cámara
    # en formato BGR (los colores en otro orden). Hay que convertir antes
    # de pasarle el frame a MediaPipe, y empaquetarla en un objeto mp.Image.
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imagen_mediapipe = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    resultado = detector_facial.detect_for_video(imagen_mediapipe, timestamp_ms)

    # Si MediaPipe no encontró ninguna cara en este frame (por ejemplo,
    # porque tapaste la cámara), la lista 'face_landmarks' viene vacía.
    if not resultado.face_landmarks:
        return None, None, None

    # Tomamos la primera cara detectada (asumimos que hay un solo conductor
    # mirando a la cámara).
    landmarks = resultado.face_landmarks[0]

    puntos_ojo_der = obtener_coordenadas_ojo(landmarks, OJO_DERECHO_IDX, ancho_frame, alto_frame)
    puntos_ojo_izq = obtener_coordenadas_ojo(landmarks, OJO_IZQUIERDO_IDX, ancho_frame, alto_frame)

    ear_der = calcular_ear(puntos_ojo_der)
    ear_izq = calcular_ear(puntos_ojo_izq)

    # Promediamos ambos ojos: si un solo ojo se detecta mal por un instante
    # (por ejemplo, por un reflejo de luz o un ángulo raro de la cabeza),
    # el promedio con el otro ojo amortigua ese ruido y evita falsas alarmas.
    ear_promedio = (ear_der + ear_izq) / 2.0

    return ear_promedio, puntos_ojo_izq, puntos_ojo_der


def reproducir_alarma():
    """Reproduce un pitido de alarma usando el altavoz de la computadora.

    Esta función se ejecuta DENTRO DE UN HILO SEPARADO (ver 'threading' más
    abajo), nunca directamente en el bucle principal de video. La razón es
    que winsound.Beep() es una función "bloqueante": mientras el pitido está
    sonando, el programa no puede hacer nada más. Si la llamáramos
    directamente en el bucle principal, la ventana de video se congelaría
    (dejaría de actualizarse) durante toda la duración del pitido. Al
    correrla en un hilo aparte, la cámara y el video pueden seguir
    funcionando normalmente mientras el sonido se reproduce de fondo."""
    winsound.Beep(ALARM_FREQ_HZ, ALARM_DURATION_MS)


def dibujar_alerta(frame):
    """Dibuja el aviso visual de alerta (texto rojo) sobre el frame."""
    alto_frame, ancho_frame = frame.shape[:2]
    texto = "ALERTA! SOMNOLENCIA DETECTADA"

    # Dibujamos un rectángulo rojo de fondo para que el texto se lea bien
    # sin importar lo que haya detrás en la imagen de la cámara.
    cv2.rectangle(frame, (0, 0), (ancho_frame, 60), (0, 0, 255), -1)
    cv2.putText(
        frame, texto, (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA
    )


def dibujar_alerta_cabezazos(frame):
    """Dibuja el aviso visual de alerta por cabezazos detectados.

    Se dibuja más abajo que la alerta por ojos cerrados para que, si las dos
    saltan al mismo tiempo, no se pisen y se puedan leer ambas."""
    alto_frame, ancho_frame = frame.shape[:2]
    texto = "ALERTA! CABEZAZOS DETECTADOS"

    cv2.rectangle(frame, (0, 60), (ancho_frame, 115), (0, 0, 255), -1)
    cv2.putText(
        frame, texto, (10, 98),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA
    )


def dibujar_alerta_pulso_combinada(frame):
    """Dibuja el aviso visual de alerta combinada: caída de BPM confirmada
    junto con otra señal (ojos cerrados o cabezazos) ya activa.

    Se dibuja más abajo que las otras dos alertas, en su propia franja, para
    que las tres puedan verse a la vez si llegaran a coincidir."""
    alto_frame, ancho_frame = frame.shape[:2]
    texto = "ALERTA! CAIDA DE PULSO + OTRA SENAL"

    cv2.rectangle(frame, (0, 115), (ancho_frame, 170), (0, 0, 255), -1)
    cv2.putText(
        frame, texto, (10, 153),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA
    )


def disparar_alarma_si_corresponde(hilo_alarma):
    """Lanza el pitido de alarma en un hilo nuevo, si no hay uno sonando ya.

    Devuelve el hilo (nuevo o el que ya estaba) para que quien la llama lo
    siga teniendo a mano. Esta función la usan TANTO la alerta por ojos
    cerrados COMO la alerta por cabezazos: ambas comparten la misma alarma,
    y este guard compartido evita que se pisen entre sí si las dos se
    disparan casi al mismo tiempo."""
    if hilo_alarma is None or not hilo_alarma.is_alive():
        hilo_alarma = threading.Thread(target=reproducir_alarma, daemon=True)
        # 'daemon=True' significa que este hilo no va a impedir que el
        # programa se cierre si vos apretás 'q': Python lo corta junto con
        # todo lo demás al salir.
        hilo_alarma.start()
    return hilo_alarma


def dibujar_info_sensor(frame, detector_cabezazos, sensor_conectado):
    """Muestra en pantalla el estado del acelerómetro.

    Igual que la info de EAR, esto no hace falta para que la alarma
    funcione: sirve para ver en vivo qué está midiendo el sensor y poder
    ajustar el umbral con números reales delante."""
    if not sensor_conectado:
        cv2.putText(frame, "Acelerometro: SIN SENAL", (10, frame.shape[0] - 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        return

    texto = (f"Cabezazos: {detector_cabezazos.cantidad_eventos_recientes()}"
             f"/{sensor.MIN_CABEZAZOS_PARA_ALERTA}"
             f"  (mov: {detector_cabezazos.ultima_desviacion:.1f})")
    cv2.putText(frame, texto, (10, frame.shape[0] - 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)


def dibujar_info_pulso(frame, detector_bpm, sensor_conectado):
    """Muestra en pantalla el estado del sensor de pulso: el último BPM
    válido leído, o un aviso si no hay señal (desconectado o sin buen
    contacto con la piel)."""
    if not sensor_conectado or not detector_bpm.hay_senal_valida():
        cv2.putText(frame, "Pulso: SIN SENAL", (10, frame.shape[0] - 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        return

    texto = f"Pulso: {detector_bpm.ultimo_bpm_valido} BPM"
    cv2.putText(frame, texto, (10, frame.shape[0] - 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)


def dibujar_info_debug(frame, ear_promedio, frames_ojos_cerrados):
    """Dibuja en pantalla el valor actual de EAR y el contador de frames.

    Esto no es necesario para que la alarma funcione: es solo información
    útil para que puedas ver "en vivo" qué está pensando el programa mientras
    lo probás, y así entender por qué dispara (o no dispara) la alarma."""
    texto_ear = f"EAR: {ear_promedio:.3f}"
    texto_contador = f"Frames ojos cerrados: {frames_ojos_cerrados}/{EAR_CONSEC_FRAMES}"

    cv2.putText(frame, texto_ear, (10, frame.shape[0] - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, texto_contador, (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)


def leer_opciones_de_consola():
    """Lee las opciones que se pasan al ejecutar el programa desde la consola.

    Gracias a esto podés cambiar de modo sin editar el archivo. Por ejemplo:

        python detector_somnoliencia.py --modo simulador
        python detector_somnoliencia.py --modo desactivado
        python detector_somnoliencia.py --modo ble --nombre-ble MiSensor

    Si no pasás ninguna opción, se usan los valores por defecto definidos
    arriba en las CONSTANTES DE CONFIGURACIÓN."""
    analizador = argparse.ArgumentParser(
        description="Detector de somnolencia: ojos cerrados (camara) y cabezazos (acelerometro)."
    )
    analizador.add_argument(
        "--modo",
        default=MODO_SENSOR,
        choices=["simulador", "clasico", "ble", "desactivado"],
        help="De donde salen los datos del acelerometro. Por defecto: %(default)s",
    )
    analizador.add_argument(
        "--puerto",
        default=PUERTO_COM,
        help="Puerto COM del modulo Bluetooth Clasico. Por defecto: %(default)s",
    )
    analizador.add_argument(
        "--nombre-ble",
        default=NOMBRE_DISPOSITIVO_BLE,
        help="Nombre del dispositivo BLE. Por defecto: %(default)s",
    )
    analizador.add_argument(
        "--uuid-ble",
        default=UUID_CARACTERISTICA_BLE,
        help="UUID de la caracteristica BLE que envia los datos.",
    )
    analizador.add_argument(
        "--modo-pulso",
        default=MODO_SENSOR_PULSO,
        choices=["simulador", "ble", "desactivado"],
        help="De donde salen los datos de pulso (BPM). Por defecto: %(default)s",
    )
    analizador.add_argument(
        "--nombre-ble-pulso",
        default=NOMBRE_BLE_PULSO,
        help="Nombre del sensor de pulso BLE. Por defecto: %(default)s",
    )
    analizador.add_argument(
        "--direccion-ble-pulso",
        default=DIRECCION_BLE_PULSO,
        help="Direccion MAC del sensor de pulso BLE (opcional, mas confiable que el nombre).",
    )
    return analizador.parse_args()


def main(opciones=None):
    """Función principal: abre la cámara y corre el bucle de detección.

    'opciones' son las elegidas desde la consola. Si no se pasa ninguna
    (por ejemplo, si alguien llama a main() desde otro script), se usan
    los valores por defecto de las constantes."""
    if opciones is None:
        opciones = argparse.Namespace(
            modo=MODO_SENSOR,
            puerto=PUERTO_COM,
            nombre_ble=NOMBRE_DISPOSITIVO_BLE,
            uuid_ble=UUID_CARACTERISTICA_BLE,
            modo_pulso=MODO_SENSOR_PULSO,
            nombre_ble_pulso=NOMBRE_BLE_PULSO,
            direccion_ble_pulso=DIRECCION_BLE_PULSO,
        )

    # --- Preparamos el detector facial de MediaPipe ---
    detector_facial = crear_detector_facial()

    # --- Preparamos el acelerómetro (detección de cabezazos) ---
    # Si algo falla al conectar con el sensor (Bluetooth apagado, puerto COM
    # equivocado, librería sin instalar), NO cortamos el programa: avisamos y
    # seguimos funcionando solo con la cámara. La idea es que un problema con
    # el sensor no te deje sin la detección por ojos, que es la principal.
    lector_sensor = None
    detector_cabezazos = sensor.DetectorCabezazos()

    if opciones.modo != "desactivado":
        try:
            lector_sensor = sensor.crear_lector(
                opciones.modo,
                puerto_com=opciones.puerto,
                nombre_ble=opciones.nombre_ble,
                uuid_ble=opciones.uuid_ble,
            )
            lector_sensor.iniciar()
            print(f"Acelerometro iniciado en modo '{opciones.modo}'.")
        except Exception as error:
            print(f"AVISO: no se pudo iniciar el acelerometro ({error}).")
            print("El programa sigue funcionando solo con la camara.")
            lector_sensor = None

    # --- Preparamos el sensor de pulso (BPM) ---
    # Mismo criterio que con el acelerómetro: si falla la conexión, no
    # cortamos el programa, seguimos con las señales que sí estén disponibles.
    lector_pulso = None
    detector_bpm = pulso.DetectorAnomaliaBPM()

    if opciones.modo_pulso != "desactivado":
        try:
            lector_pulso = pulso.crear_lector(
                opciones.modo_pulso,
                nombre_ble=opciones.nombre_ble_pulso,
                direccion_ble=opciones.direccion_ble_pulso,
            )
            lector_pulso.iniciar()
            print(f"Sensor de pulso iniciado en modo '{opciones.modo_pulso}'.")
        except Exception as error:
            print(f"AVISO: no se pudo iniciar el sensor de pulso ({error}).")
            print("El programa sigue funcionando sin esa señal.")
            lector_pulso = None

    # --- Abrimos la webcam ---
    # El '0' significa "la primera cámara disponible en la computadora".
    captura = cv2.VideoCapture(0)

    if not captura.isOpened():
        # Si no se pudo abrir la cámara (no hay cámara, está siendo usada
        # por otro programa, permisos de Windows bloqueados, etc.), avisamos
        # con un mensaje claro y cortamos el programa en vez de romper con
        # un error críptico más adelante.
        print("ERROR: no se pudo acceder a la webcam.")
        print("Verificá que la cámara esté conectada, que no la esté usando")
        print("otra aplicación, y que Windows tenga permitido el acceso a la")
        print("cámara para aplicaciones de escritorio.")
        return

    # Contador de cuántos frames SEGUIDOS llevamos con el EAR por debajo del
    # umbral. Se reinicia a 0 apenas el EAR vuelve a subir (ojos abiertos),
    # de modo que solo cuenta "rachas" continuas de ojos cerrados, no el
    # total acumulado en todo el programa. Así, parpadear varias veces
    # nunca hace sumar frames de una racha a otra.
    frames_ojos_cerrados = 0

    # Referencia al hilo del pitido de alarma, para no lanzar varios pitidos
    # superpuestos al mismo tiempo (ver comentario más abajo).
    hilo_alarma = None

    # Contador que le vamos a pasar a MediaPipe como "marca de tiempo" de
    # cada frame. Tiene que aumentar siempre (ver comentario en
    # 'procesar_frame'), así que simplemente le sumamos 1 en cada vuelta.
    contador_timestamp_ms = 0

    # Momento (según el reloj de la computadora) en que se disparó la última
    # alerta por cabezazos. Sirve para mantener el cartel visible unos
    # segundos. Vale None mientras no haya habido ninguna alerta.
    momento_alerta_cabezazos = None

    # Momento en que llegó la última muestra del acelerómetro, para poder
    # avisar si el sensor se queda sin señal.
    momento_ultima_muestra = time.monotonic()

    # Lo mismo que arriba, pero para el sensor de pulso.
    momento_ultima_muestra_pulso = time.monotonic()

    # Momento en que se confirmó la última caída sostenida de BPM (o None si
    # no hay ninguna vigente). Se usa solo para la REGLA DE COMBINACIÓN: ver
    # comentario junto a DURACION_VENTANA_COMBINACION_PULSO_SEG.
    momento_alerta_bpm = None

    # Para no imprimir el mismo aviso de alerta combinada en cada vuelta del
    # bucle mientras siga activa: solo avisamos por consola en el instante en
    # que pasa de "no activa" a "activa" (el cartel en pantalla, en cambio,
    # sí se puede seguir dibujando todos los frames sin problema).
    alerta_combinada_bpm_activa = False

    print("Detector de somnolencia iniciado. Presioná 'q' en la ventana de")
    print("video para salir.")
    if opciones.modo == "simulador":
        print("Modo simulador: presioná 'c' para simular un cabezazo.")
    if opciones.modo_pulso == "simulador":
        print("Modo simulador de pulso: presioná 'b' para bajar el pulso")
        print("(y mantenerlo bajo) y 'n' para devolverlo a la normalidad.")

    while True:
        # Leemos un frame (una imagen) de la cámara.
        ret, frame = captura.read()
        if not ret:
            # 'ret' es False si por algún motivo no se pudo leer un frame
            # (por ejemplo, la cámara se desconectó a mitad de la ejecución).
            print("ERROR: se perdió la conexión con la cámara.")
            break

        # Espejamos el frame horizontalmente (efecto "espejo"), para que se
        # sienta más natural mirarse a uno mismo en la pantalla (como en un
        # espejo real, no invertido).
        frame = cv2.flip(frame, 1)

        contador_timestamp_ms += 1
        ear_promedio, puntos_ojo_izq, puntos_ojo_der = procesar_frame(
            frame, detector_facial, contador_timestamp_ms
        )

        # Se recalcula en cada vuelta del bucle: True solo si, EN ESTE FRAME,
        # la racha de ojos cerrados ya llegó al umbral. La usa también la
        # regla de combinación con el sensor de pulso, más abajo.
        somnoliento = False

        if ear_promedio is None:
            # No se detectó ninguna cara en este frame. Reiniciamos el
            # contador para no arrastrar una racha de "ojos cerrados" que en
            # realidad puede deberse a que la cara salió de cuadro, y le
            # avisamos al usuario en pantalla.
            frames_ojos_cerrados = 0
            cv2.putText(frame, "No se detecta rostro", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            # Dibujamos pequeños círculos sobre los puntos de cada ojo, para
            # poder ver visualmente qué está midiendo el programa.
            for punto in puntos_ojo_izq + puntos_ojo_der:
                cv2.circle(frame, punto, 2, (0, 255, 0), -1)

            if ear_promedio < EAR_THRESHOLD:
                # El EAR está por debajo del umbral en este frame: sumamos
                # uno a la racha de "ojos cerrados".
                frames_ojos_cerrados += 1
            else:
                # Los ojos están abiertos en este frame: cortamos la racha.
                frames_ojos_cerrados = 0

            dibujar_info_debug(frame, ear_promedio, frames_ojos_cerrados)

            # ¿La racha de ojos cerrados ya llegó al umbral configurado?
            somnoliento = frames_ojos_cerrados >= EAR_CONSEC_FRAMES

            if somnoliento:
                dibujar_alerta(frame)

                # Lanzamos el pitido solo si no hay ya uno sonando. Esto
                # evita que, si los ojos siguen cerrados durante varios
                # segundos, se disparen decenas de pitidos superpuestos por
                # segundo: en cambio, se va reproduciendo un pitido, y apenas
                # termina, si seguís con los ojos cerrados, arranca el
                # siguiente.
                hilo_alarma = disparar_alarma_si_corresponde(hilo_alarma)

        # ==================================================================
        # PARTE 2: EL ACELERÓMETRO (DETECCIÓN DE CABEZAZOS)
        # ==================================================================
        # Esta parte es independiente de la de la cámara: aunque no se
        # detecte ninguna cara, los cabezazos se siguen midiendo.
        if lector_sensor is not None:
            # Levantamos todas las muestras que hayan llegado por Bluetooth
            # desde la vuelta anterior del bucle. Esta llamada NO espera: si
            # todavía no llegó nada, devuelve una lista vacía y seguimos de
            # largo, así el video nunca se frena esperando al sensor.
            muestras = lector_sensor.leer_muestras()

            if muestras:
                momento_ultima_muestra = time.monotonic()

            for muestra in muestras:
                if detector_cabezazos.procesar_muestra(muestra):
                    # Se juntaron los cabezazos suficientes dentro de la
                    # ventana de tiempo: alerta.
                    print("ALERTA: se detectaron cabezazos / movimientos bruscos.")
                    momento_alerta_cabezazos = time.monotonic()
                    hilo_alarma = disparar_alarma_si_corresponde(hilo_alarma)

            # ¿Hace cuánto que no llega ninguna muestra? Si pasó demasiado
            # tiempo, damos el sensor por desconectado.
            sensor_conectado = (
                (time.monotonic() - momento_ultima_muestra) < sensor.TIMEOUT_SENSOR_SEG
            )
            dibujar_info_sensor(frame, detector_cabezazos, sensor_conectado)

            # Mantenemos el cartel de alerta en pantalla unos segundos
            # después del evento, para que dé tiempo a leerlo.
            if momento_alerta_cabezazos is not None:
                transcurrido = time.monotonic() - momento_alerta_cabezazos
                if transcurrido < DURACION_ALERTA_CABEZAZOS_SEG:
                    dibujar_alerta_cabezazos(frame)
                else:
                    momento_alerta_cabezazos = None

        # ==================================================================
        # PARTE 3: EL SENSOR DE PULSO (CAÍDA DE BPM)
        # ==================================================================
        # También independiente de la cámara y del acelerómetro: se sigue
        # midiendo el pulso aunque no haya rostro en cuadro o no haya habido
        # ningún cabezazo.
        if lector_pulso is not None:
            muestras_pulso = lector_pulso.leer_muestras()

            if muestras_pulso:
                momento_ultima_muestra_pulso = time.monotonic()

            for muestra in muestras_pulso:
                if detector_bpm.procesar_muestra(muestra):
                    # Caída de BPM sostenida confirmada. OJO: esto todavía NO
                    # es una alerta por sí sola (ver comentario en
                    # DURACION_VENTANA_COMBINACION_PULSO_SEG): solo queda
                    # "vigente" por un rato para poder combinarse con la
                    # cámara o el acelerómetro.
                    print("SEÑAL: caída sostenida de BPM (posible somnolencia).")
                    momento_alerta_bpm = time.monotonic()

            pulso_conectado = (
                (time.monotonic() - momento_ultima_muestra_pulso) < pulso.TIMEOUT_SENSOR_SEG
            )
            dibujar_info_pulso(frame, detector_bpm, pulso_conectado)

            # --- Regla de combinación ---
            # La caída de BPM, sola, no dispara la alarma. Solo cuenta cuando
            # todavía está "vigente" (dentro de su ventana de combinación) Y,
            # al mismo tiempo, hay otra señal activa: ojos cerrados AHORA
            # MISMO, o un cabezazo cuyo cartel siga en pantalla. Como esta
            # comprobación se repite en cada vuelta del bucle, funciona sin
            # importar cuál de las dos señales haya aparecido primero.
            pulso_vigente = (
                momento_alerta_bpm is not None
                and (time.monotonic() - momento_alerta_bpm) < DURACION_VENTANA_COMBINACION_PULSO_SEG
            )
            cabezazos_activo = momento_alerta_cabezazos is not None

            if pulso_vigente and (somnoliento or cabezazos_activo):
                if not alerta_combinada_bpm_activa:
                    print("ALERTA: caída de BPM confirmada junto con otra señal de somnolencia.")
                alerta_combinada_bpm_activa = True
                dibujar_alerta_pulso_combinada(frame)
                hilo_alarma = disparar_alarma_si_corresponde(hilo_alarma)
            else:
                alerta_combinada_bpm_activa = False

        # Mostramos el frame resultante en una ventana.
        cv2.imshow("Detector de Somnolencia", frame)

        # Esperamos 1 milisegundo a que se presione una tecla. El '& 0xFF' es
        # una forma estándar de comparar la tecla en distintos sistemas
        # operativos.
        tecla = cv2.waitKey(1) & 0xFF

        if tecla == ord('q'):
            print("Saliendo del programa...")
            break

        # En modo simulador, la tecla 'c' genera un cabezazo falso. Sirve
        # para probar la alerta sin tener el sensor real: apretala dos veces
        # con menos de 20 segundos de diferencia y debería saltar la alarma.
        if tecla == ord('c') and isinstance(lector_sensor, sensor.LectorSimulado):
            lector_sensor.forzar_cabezazo()
            print("Cabezazo simulado.")

        # En modo simulador de pulso, 'b' baja el BPM simulado y lo mantiene
        # bajo (para probar la caída sostenida y la alerta combinada sin
        # tener el sensor real), y 'n' lo devuelve a la normalidad.
        if tecla == ord('b') and isinstance(lector_pulso, pulso.LectorSimulado):
            lector_pulso.forzar_caida_manual()
            print("Pulso bajo simulado (sostenido). Presioná 'n' para volver a la normalidad.")

        if tecla == ord('n') and isinstance(lector_pulso, pulso.LectorSimulado):
            lector_pulso.restaurar_pulso_normal()
            print("Pulso simulado vuelto a la normalidad.")

    # --- Liberamos los recursos antes de terminar ---
    # Muy importante: si no liberamos la cámara, puede quedar "ocupada" y
    # otras aplicaciones (o el propio programa, si lo volvés a correr) no
    # van a poder usarla hasta reiniciar la computadora.
    captura.release()
    cv2.destroyAllWindows()
    detector_facial.close()

    # También cerramos la conexión con el acelerómetro y con el sensor de
    # pulso, para liberar el Bluetooth y que quede disponible para la
    # próxima ejecución.
    if lector_sensor is not None:
        lector_sensor.detener()
    if lector_pulso is not None:
        lector_pulso.detener()


# Este bloque hace que 'main()' se ejecute solo cuando corrés este archivo
# directamente (por ejemplo, con "python detector_somnoliencia.py"), y no si
# alguna vez este archivo se importa desde otro script de Python.
if __name__ == "__main__":
    main(leer_opciones_de_consola())
