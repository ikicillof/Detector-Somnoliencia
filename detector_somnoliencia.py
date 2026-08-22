# ==============================================================================
# DETECTOR DE SOMNOLENCIA EN TIEMPO REAL
# ==============================================================================
#
# ¿QUÉ HACE ESTE PROGRAMA?
# ------------------------
# Usa la webcam para mirar tu cara en tiempo real. Detecta tus ojos y calcula
# qué tan "abiertos" o "cerrados" están usando una fórmula matemática llamada
# EAR (Eye Aspect Ratio = "Relación de Aspecto del Ojo"). Si detecta que
# tuviste los ojos cerrados durante varios cuadros (frames) seguidos —lo cual
# indica que probablemente te estás quedando dormido y no simplemente
# parpadeando— dispara una alarma sonora y muestra un aviso en pantalla.
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


def main():
    """Función principal: abre la cámara y corre el bucle de detección."""

    # --- Preparamos el detector facial de MediaPipe ---
    detector_facial = crear_detector_facial()

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

    print("Detector de somnolencia iniciado. Presioná 'q' en la ventana de")
    print("video para salir.")

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

                # Lanzamos el pitido en un hilo nuevo SOLO si no hay ya un
                # pitido sonando en este momento (hilo_alarma is None, o el
                # hilo anterior ya terminó de sonar). Esto evita que, si los
                # ojos siguen cerrados durante varios segundos, se disparen
                # decenas de pitidos superpuestos por segundo: en cambio,
                # se va reproduciendo un pitido, y apenas termina, si seguís
                # con los ojos cerrados, arranca el siguiente.
                if hilo_alarma is None or not hilo_alarma.is_alive():
                    hilo_alarma = threading.Thread(target=reproducir_alarma, daemon=True)
                    # 'daemon=True' significa que este hilo no va a impedir
                    # que el programa se cierre si vos apretás 'q': Python
                    # lo corta junto con todo lo demás al salir.
                    hilo_alarma.start()

        # Mostramos el frame resultante en una ventana.
        cv2.imshow("Detector de Somnolencia", frame)

        # Esperamos 1 milisegundo a que se presione una tecla. Si la tecla
        # presionada es 'q', salimos del bucle. El '& 0xFF' es una forma
        # estándar de comparar la tecla en distintos sistemas operativos.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Saliendo del programa...")
            break

    # --- Liberamos los recursos antes de terminar ---
    # Muy importante: si no liberamos la cámara, puede quedar "ocupada" y
    # otras aplicaciones (o el propio programa, si lo volvés a correr) no
    # van a poder usarla hasta reiniciar la computadora.
    captura.release()
    cv2.destroyAllWindows()
    detector_facial.close()


# Este bloque hace que 'main()' se ejecute solo cuando corrés este archivo
# directamente (por ejemplo, con "python detector_somnoliencia.py"), y no si
# alguna vez este archivo se importa desde otro script de Python.
if __name__ == "__main__":
    main()
