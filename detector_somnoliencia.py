# ==============================================================================
# DETECTOR DE SOMNOLENCIA EN TIEMPO REAL
# ==============================================================================
#
# ¿QUÉ HACE ESTE PROGRAMA?
# ------------------------
# Detecta si un conductor se está quedando dormido mirándolo con la webcam.
# Toda la información sale de la MISMA malla de puntos faciales que devuelve
# MediaPipe: no hay ningún sensor externo. Vigila DOS cosas al mismo tiempo:
#
# 1) LOS OJOS. Calcula qué tan "abiertos" o "cerrados" están con una fórmula
#    geométrica llamada EAR (Eye Aspect Ratio). Si los ojos quedan cerrados
#    varios segundos seguidos —no un simple parpadeo— dispara la alarma.
#
# 2) LOS CABECEOS. A partir de los mismos puntos de la cara estima la POSE de
#    la cabeza (hacia dónde está inclinada) y saca el ángulo de PITCH, que es
#    la inclinación vertical (cabeza mirando al frente vs. cabeza caída hacia
#    el pecho). Con ese ángulo detecta dos patrones distintos:
#      - Cabeza caída sostenida: la cabeza queda inclinada hacia abajo más de
#        cierto ángulo durante varios segundos.
#      - Cabeceo brusco: la cabeza cae de golpe y se endereza en menos de un
#        segundo (el clásico "cabezazo" de quien pega una cabeceada y se
#        despierta). Se detecta por la VELOCIDAD del movimiento, no por la
#        posición final.
#
# Cualquiera de las tres condiciones (ojos cerrados, cabeza caída, cabeceo
# brusco) dispara la alarma: pitidos y un cartel rojo en pantalla. En la PC
# suena por los parlantes; en la Raspberry Pi, por un buzzer en un pin GPIO.
# Hay dos niveles con patrones de pitidos distintos (ver PATRONES_ALARMA):
# ojos cerrados / cabeza caída = peligro; cabeceo brusco = aviso.
#
# ¿QUÉ ES "MEDIAPIPE FACE LANDMARKER"?
# -------------------------------------
# MediaPipe es una librería de Google que, a partir de la imagen de la cámara,
# nos devuelve unos 478 puntos (coordenadas x, y) distribuidos sobre toda la
# cara: ojos, cejas, nariz, boca, contorno de la cara, etc. A esta red de
# puntos se la suele llamar "malla facial". Para el EAR usamos 6 puntos de
# cada ojo; para la pose de la cabeza usamos 6 puntos más (nariz, mentón,
# comisuras de ojos y boca).
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
# ¿QUÉ ES EL PITCH Y CÓMO SE ESTIMA?
# ------------------------------------
# La "pose de la cabeza" son los tres ángulos que describen hacia dónde está
# orientada: yaw (girar a los costados, como decir "no"), pitch (asentir,
# como decir "sí": cabeza al frente vs. mirando al piso) y roll (ladear la
# cabeza hacia un hombro). Para la somnolencia solo nos interesa el PITCH.
#
# Se estima con cv2.solvePnP: le damos 6 puntos de la cara EN LA IMAGEN (2D)
# y las mismas 6 posiciones EN UN MODELO 3D GENÉRICO de cabeza humana, y la
# función calcula qué rotación y traslación de ese modelo 3D explica lo que
# vemos en la imagen. De esa rotación sacamos el pitch, en grados, tomándolo
# como la inclinación del eje "arriba de la cara". La intención es que mirar
# hacia abajo dé pitch NEGATIVO; como el signo de solvePnP puede salir
# invertido según la cámara, el programa lo verifica y corrige solo,
# comparándolo con una medida geométrica 2D independiente, apenas el
# conductor mueve un poco la cabeza.
#
# IMPORTANTE: la imagen se muestra espejada (efecto espejo, más cómodo para
# mirarse), pero el cálculo de la pose se hace sobre el frame SIN espejar,
# porque una imagen reflejada rompería la geometría de solvePnP.
#
# El valor ABSOLUTO del pitch no nos sirve directamente, porque depende de
# cómo esté puesta la cámara y de la postura natural de cada persona. Por eso,
# al arrancar, el programa se toma unos segundos para CALIBRAR: promedia el
# pitch mientras el conductor mira al frente y guarda eso como "posición
# neutra". De ahí en más todo se mide como DESVIACIÓN respecto de ese neutro.
# La calibración se valida (si el conductor se movió o no miraba al frente,
# se reintenta sola, sin tocar ninguna tecla). En el modo sin ventana además
# se puede rehacer a mano escribiendo 'c' + Enter en la consola.
#
# ¿POR QUÉ TODO SE MIDE EN SEGUNDOS Y NO EN "CUADROS"?
# -----------------------------------------------------
# Parpadear es normal y sano: un parpadeo dura apenas 100-400 milisegundos.
# Si disparáramos la alarma apenas viéramos un instante con los ojos abajo
# del umbral, sonaría todo el tiempo por simples parpadeos. Por eso exigimos
# que la condición se mantenga durante VARIOS SEGUNDOS seguidos.
#
# Ese "varios segundos" se mide con el reloj (time.time()), NUNCA contando
# cuántos cuadros seguidos pasaron. La cantidad de cuadros por segundo cambia
# según la cámara, la luz y la computadora (una Raspberry no procesa al mismo
# ritmo que una laptop), así que un umbral atado a "cuadros" quedaría mal
# calibrado apenas cambia algo del entorno. El reloj, en cambio, mide siempre
# lo mismo.
#
# ==============================================================================

# --- Librerías de la biblioteca estándar de Python (ya vienen instaladas) ---
import os
import sys
import time
import argparse   # Para leer opciones de la línea de comandos (--sin-ventana).
import queue      # Cola segura entre hilos (comandos escritos en la consola).
import signal     # Para cerrar ordenadamente si el sistema pide terminar.
import subprocess # Para consultar 'vcgencmd' en la Raspberry Pi.
import platform   # Para saber en qué sistema operativo estamos (Windows,
                  # Linux...) y elegir cómo hacer sonar la alarma.
import threading
import urllib.request  # Para descargar el modelo de MediaPipe la primera vez.
from collections import deque, namedtuple  # 'deque': cola doble para guardar
                               # los últimos valores de pitch (suavizado y
                               # velocidad). 'namedtuple': para devolver la
                               # pose de la cabeza (pitch, yaw, proxy) junta.

# --- Librerías de terceros (hay que instalarlas con pip, ver requirements.txt) ---
import cv2            # OpenCV: nos permite acceder a la webcam, mostrar la
                       # ventana de video, dibujar sobre la imagen y estimar
                       # la pose de la cabeza (cv2.solvePnP).
import mediapipe as mp  # MediaPipe: nos da los puntos de la cara (landmarks).
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np    # NumPy: distancias entre puntos, y el ajuste de recta
                       # con el que medimos la velocidad del cabeceo.


# ==============================================================================
# CONSTANTES DE CONFIGURACIÓN
# ==============================================================================
# Todos los valores que se pueden "ajustar" del programa están acá arriba,
# juntos, para que sea fácil experimentar sin tener que buscar en todo el
# código. Si el programa te resulta muy sensible o muy poco sensible, estos
# son los números que hay que tocar. Para calibrarlos: corré el programa y
# mirá en pantalla el EAR y el pitch en vivo, con la cabeza y los ojos en
# distintas posiciones, y ajustá cada número según lo que veas.

# --- Cámara ---
# Índice de la webcam. 0 es "la primera cámara disponible". Si tenés varias
# cámaras y agarra la equivocada, probá con 1, 2, etc.
CAMARA_INDICE = 0

# --- Ojos cerrados (EAR) ---
# Umbral de EAR: por debajo de este valor consideramos que el ojo está cerrado.
# 0.22 es un valor típico, pero varía según la persona, el ángulo de la cámara
# y la iluminación. Si dispara la alarma con los ojos abiertos, bajalo (p. ej.
# 0.18). Si no detecta cuando cerrás los ojos, subilo (p. ej. 0.25).
EAR_THRESHOLD = 0.22

# Cuántos SEGUNDOS seguidos con los ojos por debajo del umbral se consideran
# somnolencia real y no un simple parpadeo. Rango razonable: 1.0 a 3.0 s.
DROWSY_TIME_SECONDS = 2.0

# Histéresis del EAR. Una vez que el ojo cuenta como "cerrado" (EAR por
# debajo de EAR_THRESHOLD), no vuelve a contar como "abierto" hasta que el
# EAR sube por encima de EAR_THRESHOLD + este margen. Sirve para que el
# contador de tiempo no se reinicie por oscilaciones del EAR justo en el
# límite (si no, con los ojos entornados la alerta nunca llegaría a los 2 s).
# Rango razonable: 0.01 a 0.04.
EAR_HISTERESIS = 0.02

# --- Cabeceos (pose de la cabeza / pitch) ---
# Duración de la calibración inicial, en segundos. Apenas arranca el programa,
# durante este tiempo se promedia el pitch para fijar la "posición neutra"
# del conductor mirando al frente. Todo lo que viene después se mide como
# desviación respecto de ese neutro. Rango razonable: 2.0 a 5.0 s (menos es
# poco promedio; más hace esperar de gancho al arrancar).
CALIBRACION_SEGUNDOS = 3.0

# Ventana de la media móvil que SUAVIZA el pitch antes de evaluar los
# umbrales, en segundos. Los landmarks tienen un poco de ruido cuadro a
# cuadro; promediar los últimos ~0.12 s lo saca sin agregar un retraso
# perceptible. Cuanto más grande, más estable pero más lento para reaccionar
# a un cabeceo rápido. Rango razonable: 0.08 a 0.20 s.
SUAVIZADO_PITCH_SEGUNDOS = 0.12

# Cuánto hacia atrás se mira, en segundos, para medir la VELOCIDAD angular
# del pitch (grados por segundo). La velocidad es (pitch_suavizado_ahora -
# pitch_suavizado_hace_este_tiempo) dividido ese tiempo. Más chico = más
# sensible a movimientos rápidos pero más ruidoso. Rango razonable:
# 0.10 a 0.25 s.
VENTANA_VELOCIDAD_SEGUNDOS = 0.18

# --- Patrón 1: cabeza caída sostenida ---
# Cuántos GRADOS por debajo del neutro tiene que estar el pitch para
# considerar que la cabeza "está caída". Mirá el número "desv" en pantalla
# con la cabeza derecha (cerca de 0) y con la cabeza caída hacia el pecho
# (bien negativo) para elegir un valor en el medio. Rango típico: 12 a 25.
#
# Calibrado con la cámara real (valores de "desv" medidos tras recalibrar):
#   mirando al frente ......................  ~0
#   mirando el volante .....................  ~-5.5
#   cabeza caída como si me durmiera .......  ~-17.5
# 11.0 queda entre "mirar el volante" y "cabeza de sueño", más cerca del
# primero para no perder detecciones reales; el requisito de que dure
# CABEZA_CAIDA_SEGUNDOS descarta igual los vistazos rápidos al volante.
CABEZA_CAIDA_GRADOS = 11.0

# Cuántos SEGUNDOS seguidos tiene que mantenerse esa caída para disparar la
# alerta. Una caída más corta puede ser mirar el tablero, la palanca o los
# espejos. Rango típico: 1.0 a 3.0 s.
CABEZA_CAIDA_SEGUNDOS = 1.5

# Histéresis de la cabeza caída: margen (en grados) entre el umbral para
# ENTRAR en estado de "cabeza caída" (-CABEZA_CAIDA_GRADOS) y el umbral para
# SALIR de ese estado (-(CABEZA_CAIDA_GRADOS - este margen)). Evita que la
# alerta parpadee cuando el pitch queda oscilando cerca del límite. Rango
# razonable: 2 a 6 grados.
CABEZA_CAIDA_HISTERESIS_GRADOS = 3.0

# --- Patrón 2: cabeceo brusco (la "cabeceada" de sueño) ---
# Velocidad mínima de caída, en GRADOS POR SEGUNDO, para empezar a contar un
# cabeceo brusco. Es una caída rápida hacia abajo, no una posición. Un
# movimiento tranquilo de la cabeza (mirar un espejo) ronda los 20-40 °/s;
# una cabeceada de sueño suele estar entre 60 y 200 °/s. Mirá el valor "vel"
# en pantalla moviendo la cabeza para calibrarlo. Rango típico: 45 a 120.
CABECEO_VELOCIDAD_GRADOS_POR_SEG = 55.0

# Cuántos grados como mínimo tiene que abarcar esa caída rápida, para no
# contar temblores chiquitos que casualmente tuvieron velocidad alta.
# Rango típico: 8 a 20.
CABECEO_AMPLITUD_MINIMA_GRADOS = 10.0

# Ventana máxima, en segundos, dentro de la cual la cabeza tiene que volver a
# subir para que el evento cuente como "cabeceo brusco" (caída + recuperación
# rápida). Si tarda más que esto en volver, ya no es un cabeceo: es una
# cabeza caída sostenida, y la agarra el patrón 1. Rango típico: 0.5 a 1.2 s.
CABECEO_VENTANA_RECUPERACION_SEG = 1.0

# Qué fracción de lo que bajó la cabeza tiene que volver a subir para dar el
# cabeceo por "recuperado". 0.5 = tiene que remontar al menos la mitad de la
# caída. Rango razonable: 0.4 a 0.7.
CABECEO_FRACCION_RECUPERACION = 0.5

# Cuánto hacia atrás se mira, en segundos, para encontrar el pitch "de antes
# del cabeceo" (el punto más alto reciente), que es contra el que se mide la
# amplitud de la caída. Tiene que ser un poco más largo que una cabeceada
# típica. Rango razonable: 0.4 a 0.8 s.
VENTANA_PICO_CABECEO_SEG = 0.5

# Cuántos segundos queda en pantalla el cartel de alerta por cabeceo brusco.
# Como es un evento instantáneo, sin esto el cartel aparecería y desaparecería
# tan rápido que no se llegaría a leer. Rango razonable: 2.0 a 5.0 s.
DURACION_ALERTA_CABECEO_SEG = 3.0

# --- Validación de la calibración inicial ---
# Si el pitch varió más que esto (grados, pico a pico) durante los segundos
# de calibración, quiere decir que el conductor se movió: se descarta esa
# calibración y se reintenta. Rango razonable: 4 a 12 grados.
CALIBRACION_ESTABILIDAD_MAX_GRADOS = 8.0

# Si el yaw promedio (giro a los costados) durante la calibración supera esto
# en valor absoluto, el conductor no está mirando de frente a la cámara: se
# descarta y se reintenta. Rango razonable: 12 a 25 grados.
CALIBRACION_YAW_MAX_GRADOS = 18.0

# Si la calibración falla, se REINTENTA SOLA, sin límite y sin tocar ninguna
# tecla, hasta que salga bien: nunca se acepta una calibración mala. Mientras
# tanto la detección de ojos cerrados funciona igual; solo la de cabeceos
# queda en pausa. Cada esta cantidad de intentos fallidos seguidos se imprime
# un recordatorio de que los cabeceos todavía no se están vigilando.
CALIBRACION_AVISO_CADA_INTENTOS = 3

# --- Autoverificación del signo del pitch ---
# El signo del pitch de solvePnP no es 100% predecible de antemano. El
# programa lo verifica solo comparándolo con un proxy geométrico 2D
# independiente (la distancia vertical nariz-ojos, que baja al mirar hacia
# abajo). Estos dos valores controlan esa verificación:
#
# Ventana de tiempo, en segundos, sobre la que se compara el pitch con el
# proxy. Rango razonable: 3 a 6 s.
VENTANA_VERIF_SIGNO_SEG = 4.0
# Cuánto se tiene que haber movido la cabeza (grados de pitch, pico a pico)
# dentro de esa ventana para que la comparación tenga sentido. Con menos
# movimiento no hay señal suficiente y no se toca nada. Rango razonable: 5 a 12.
VERIF_SIGNO_MOV_MINIMO_GRADOS = 6.0

# Signo del pitch. None = automático: el programa lo corrige solo apenas
# movés un poco la cabeza. Si lo fijás en 1.0 o -1.0 se desactiva la
# autodetección. Referencia: mirar HACIA ABAJO tiene que hacer que "desv" en
# pantalla se vuelva NEGATIVO.
#
# Fijado en 1.0 tras probar con la cámara real: con este signo, bajar la
# cabeza da "desv" negativo (mirando el volante ~-5.5, cabeza de sueño
# ~-17.5). La autodetección (comparación con el proxy geométrico) NO
# funcionó en la prueba real, por eso queda forzado a mano.
SIGNO_PITCH = 1.0

# --- Alarma sonora ---
# Frecuencia del pitido (en Hz, más alto = más agudo) y duración de cada
# pitido (en milisegundos). La frecuencia solo se usa en la PC (por los
# parlantes); el buzzer activo de la Raspberry Pi tiene un tono fijo.
ALARM_FREQ_HZ = 2500
ALARM_DURATION_MS = 700

# Pin GPIO (numeración BCM, NO el número de pin físico) donde está conectada
# la base del transistor que maneja el buzzer activo en la Raspberry Pi.
# Conexión actual:
#   pin físico 12 (GPIO18) -> resistencia de 1 kΩ -> base del transistor NPN
#   pin físico 2 (5 V)     -> + del buzzer;  - del buzzer -> colector
#   pin físico 14 (GND)    -> emisor
# El GPIO en alto satura el transistor y el buzzer suena.
BUZZER_GPIO = 18

# Patrones de pitidos de cada nivel de alerta, como lista de
# (segundos_sonando, segundos_en_silencio). Así se distinguen de oído sin
# mirar la pantalla:
#   NIVEL_AVISO   -> cabeceo brusco (evento instantáneo): dos pitidos cortos.
#   NIVEL_PELIGRO -> ojos cerrados o cabeza caída sostenida: un pitido largo
#                    que se repite mientras dure la condición.
NIVEL_AVISO = 1
NIVEL_PELIGRO = 2
PATRONES_ALARMA = {
    NIVEL_AVISO: [(0.12, 0.08), (0.12, 0.30)],
    NIVEL_PELIGRO: [(ALARM_DURATION_MS / 1000.0, 0.10)],
}

# --- Modelo de MediaPipe ---
# Dirección de internet de donde se descarga el modelo de detección facial, y
# nombre con el que se guarda en la carpeta del proyecto.
MODELO_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODELO_NOMBRE_ARCHIVO = "face_landmarker.task"

# --- Índices de los puntos de MediaPipe que forman cada ojo ---
# MediaPipe numera sus puntos de la cara siempre en el mismo orden. Estos son
# los 6 índices del contorno de cada ojo, ordenados para que coincidan con el
# dibujo de p1..p6 de la explicación de más arriba:
#   posición 0 -> p1 (comisura izquierda del ojo)
#   posición 1 -> p2 (párpado superior, lado izquierdo)
#   posición 2 -> p3 (párpado superior, lado derecho)
#   posición 3 -> p4 (comisura derecha del ojo)
#   posición 4 -> p5 (párpado inferior, lado derecho)
#   posición 5 -> p6 (párpado inferior, lado izquierdo)
OJO_DERECHO_IDX = [33, 160, 158, 133, 153, 144]   # ojo derecho de la persona
OJO_IZQUIERDO_IDX = [362, 385, 387, 263, 373, 380]  # ojo izquierdo de la persona

# --- Modelo 3D genérico de cara, para estimar la pose de la cabeza ---
# Son 6 puntos de una cara humana "promedio", en milímetros aproximados, con
# la nariz en el origen. Es el modelo clásico que se usa con cv2.solvePnP.
# El orden de estos puntos tiene que coincidir con MODELO_POSE_IDX de abajo.
MODELO_POSE_3D = np.array([
    (0.0,     0.0,     0.0),      # punta de la nariz
    (0.0,  -330.0,   -65.0),      # mentón
    (-225.0, 170.0,  -135.0),     # comisura externa del ojo del lado izq. de la imagen
    (225.0,  170.0,  -135.0),     # comisura externa del ojo del lado der. de la imagen
    (-150.0,-150.0,  -125.0),     # comisura de la boca del lado izq. de la imagen
    (150.0, -150.0,  -125.0),     # comisura de la boca del lado der. de la imagen
], dtype=np.float64)

# Índices de MediaPipe que se corresponden, uno a uno y en el mismo orden,
# con los puntos de MODELO_POSE_3D.
MODELO_POSE_IDX = [1, 152, 33, 263, 61, 291]

# Resultado de estimar la pose de la cabeza en un frame:
#   pitch : inclinación vertical, en grados. ~0 de frente, NEGATIVO al mirar
#           hacia abajo. Es el valor CRUDO (sin suavizar) y sin el signo
#           automático aplicado (eso lo hace el DetectorCabeceos).
#   yaw   : giro a los costados, en grados. Solo se usa para validar la
#           calibración (mirar de frente = yaw chico).
#   proxy : medida geométrica 2D auxiliar (distancia vertical nariz-ojos
#           normalizada). Baja al mirar hacia abajo. Sirve para verificar
#           que el signo del pitch sea el correcto.
PoseCabeza = namedtuple("PoseCabeza", "pitch yaw proxy")


# ==============================================================================
# FUNCIONES DE GEOMETRÍA (EAR Y POSE DE LA CABEZA)
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


def estimar_pose(landmarks, ancho_frame, alto_frame):
    """Estima la pose de la cabeza a partir de los landmarks de un frame.

    IMPORTANTE: 'landmarks' tienen que venir del frame SIN espejar. La imagen
    se espeja solo para mostrarla; si le pasáramos landmarks espejados, la
    geometría de solvePnP quedaría reflejada (un espejo no es una rotación) y
    el resultado sería inconsistente.

    Usa cv2.solvePnP para encontrar la rotación del modelo 3D genérico
    (MODELO_POSE_3D) que explica lo que se ve en la imagen. De las columnas de
    la matriz de rotación (que son los ejes del rostro vistos desde la cámara)
    saca el pitch y el yaw:

      - pitch: la inclinación del eje "arriba de la cara". La intención es que
        de frente dé ~0 y mirar hacia abajo dé NEGATIVO. Este método
        (dirección de un eje + atan2) no tiene la ambigüedad de ±180° que
        tenía descomponer en ángulos de Euler con RQDecomp3x3, pero el signo
        de solvePnP igual puede salir invertido según cómo esté puesta la
        cámara: por eso el DetectorCabeceos lo verifica y corrige solo,
        comparándolo con 'proxy' (ver _verificar_signo).
      - yaw: giro a los costados. Solo se usa para validar la calibración.

    Devuelve un PoseCabeza(pitch, yaw, proxy) en grados (pitch/yaw), o None si
    solvePnP no pudo resolver la pose en este frame."""
    puntos_2d = np.array(
        [(landmarks[idx].x * ancho_frame, landmarks[idx].y * alto_frame)
         for idx in MODELO_POSE_IDX],
        dtype=np.float64,
    )

    # Matriz de cámara aproximada: no calibramos la cámara real, alcanza con
    # una estimación razonable (distancia focal ~ ancho del frame, centro
    # óptico en el medio de la imagen). Sin coeficientes de distorsión.
    focal = float(ancho_frame)
    centro = (ancho_frame / 2.0, alto_frame / 2.0)
    matriz_camara = np.array([
        [focal, 0.0,   centro[0]],
        [0.0,   focal, centro[1]],
        [0.0,   0.0,   1.0],
    ], dtype=np.float64)
    sin_distorsion = np.zeros((4, 1), dtype=np.float64)

    ok, vector_rotacion, _ = cv2.solvePnP(
        MODELO_POSE_3D, puntos_2d, matriz_camara, sin_distorsion,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None

    matriz_rotacion, _ = cv2.Rodrigues(vector_rotacion)
    # Las columnas de la matriz de rotación son los ejes del rostro expresados
    # en coordenadas de cámara (X derecha, Y abajo, Z hacia la escena).
    eje_arriba = matriz_rotacion[:, 1]   # "arriba de la cara"
    eje_frente = matriz_rotacion[:, 2]   # "hacia adelante de la cara"

    # Pitch: elevación del eje "arriba". De frente el eje apunta hacia arriba
    # en la imagen (Y de cámara negativa) -> atan2(0, 1) = 0. Al mirar hacia
    # abajo, ese eje se inclina hacia la cámara (Z negativa) -> el ángulo
    # crece, y le ponemos el signo menos para que "abajo" quede NEGATIVO.
    pitch = -float(np.degrees(np.arctan2(-eje_arriba[2], -eje_arriba[1])))

    # Yaw: hacia dónde apunta el frente de la cara. De frente a la cámara el
    # eje apunta hacia el visor (Z negativa) -> atan2(0, 1) = 0.
    yaw = float(np.degrees(np.arctan2(eje_frente[0], -eje_frente[2])))

    # Proxy geométrico 2D, independiente de solvePnP: distancia vertical entre
    # la nariz y la línea de los ojos, normalizada por la distancia entre
    # ojos. Al mirar hacia abajo la nariz "sube" hacia los ojos en la imagen,
    # así que el proxy BAJA. Se usa para verificar el signo del pitch.
    nariz = landmarks[1]
    ojo_a = landmarks[33]
    ojo_b = landmarks[263]
    ojos_medio_y = (ojo_a.y + ojo_b.y) / 2.0
    interocular = abs(ojo_b.x - ojo_a.x)
    proxy = (nariz.y - ojos_medio_y) / interocular if interocular > 1e-6 else 0.0

    return PoseCabeza(pitch=pitch, yaw=yaw, proxy=proxy)


# ==============================================================================
# DETECTOR DE CABECEOS
# ==============================================================================

class DetectorCabeceos:
    """Recibe la pose de la cabeza cuadro a cuadro y avisa cuando hay una
    señal de somnolencia por movimiento de cabeza.

    Se usa así: se le pasa el PoseCabeza de cada frame con
    'procesar(pose, ahora)' y devuelve una tupla (cabeza_caida, cabeceo):

      - cabeza_caida: True MIENTRAS la cabeza siga caída más de lo permitido
        durante el tiempo requerido (patrón 1).
      - cabeceo:      True SOLO en el instante en que se confirma un cabeceo
        brusco (patrón 2).

    Antes de detectar nada se toma CALIBRACION_SEGUNDOS para promediar el
    pitch neutro (conductor mirando al frente). Esa calibración se valida (si
    el conductor se movió mucho o no miraba al frente, se descarta y se
    reintenta sola, todas las veces que haga falta). Mientras calibra,
    'calibrando' vale True y siempre devuelve (False, False). Se puede rehacer
    en cualquier momento con recalibrar().

    Además verifica solo el signo del pitch: si el de solvePnP resulta
    invertido para este equipo, lo corrige apenas el conductor mueve un poco
    la cabeza (ver _verificar_signo)."""

    def __init__(self):
        # Momento en que llegó la primera muestra (arranque de la calibración).
        self.tiempo_arranque = None

        # --- Calibración ---
        # Muestras juntadas durante la calibración (pitch YA con el signo
        # efectivo aplicado, yaw y proxy), y los promedios finales (None hasta
        # que la calibración termina y queda validada).
        self._muestras_calibracion = []
        self._yaws_calibracion = []
        self._proxys_calibracion = []
        self.pitch_neutro = None
        self.proxy_neutro = None
        self.calibrando = True
        self._intentos_calibracion = 0
        self.aviso_calibracion = ""       # motivo del reintento, para el HUD

        # --- Signo del pitch ---
        # Signo autodetectado (se usa solo si SIGNO_PITCH es None). Arranca en
        # +1 y se puede invertir una vez en _verificar_signo.
        self._signo_auto = 1.0
        self._signo_verificado = False
        # Historial (tiempo, desviacion_pitch, proxy - proxy_neutro) para la
        # verificación del signo.
        self._historial_signo = deque()

        # Historial reciente de (tiempo, pitch_crudo_con_signo), para la media
        # móvil que suaviza el pitch.
        self._historial = deque()

        # Historial de (tiempo, pitch_YA_suavizado), para medir la velocidad
        # angular y encontrar el pico previo a un cabeceo.
        self._historial_suave = deque()

        # --- Patrón 1: cabeza caída sostenida ---
        self._tiempo_inicio_caida = None
        self._cabeza_caida_estado = False   # con histéresis (ver _evaluar_cabeza_caida)

        # --- Patrón 2: cabeceo brusco ---
        self._en_cabeceo = False
        self._cabeceo_tiempo_inicio = None
        self._cabeceo_pitch_inicio = None
        self._cabeceo_pitch_minimo = None

        # Valores expuestos solo para mostrar en pantalla y calibrar.
        self.pitch_crudo = 0.0        # pitch del frame, con signo, sin suavizar
        self.pitch_suavizado = 0.0
        self.desviacion = 0.0
        self.ultima_velocidad = 0.0
        self.segundos_caida = 0.0

    def _signo_efectivo(self):
        """+1 o -1: el SIGNO_PITCH manual si está fijado, o el autodetectado."""
        if SIGNO_PITCH is not None:
            return float(SIGNO_PITCH)
        return self._signo_auto

    def recalibrar(self):
        """Descarta la calibración actual y arranca una nueva. La detección
        queda en pausa hasta que la nueva calibración termine y valide.
        El signo del pitch ya aprendido NO se toca (es propiedad del equipo,
        no de la postura)."""
        self.tiempo_arranque = None
        self._muestras_calibracion = []
        self._yaws_calibracion = []
        self._proxys_calibracion = []
        self.pitch_neutro = None
        self.proxy_neutro = None
        self.calibrando = True
        self._intentos_calibracion = 0
        self.aviso_calibracion = ""
        self._historial_signo.clear()
        self._historial.clear()
        self._historial_suave.clear()
        self._tiempo_inicio_caida = None
        self._cabeza_caida_estado = False
        self._en_cabeceo = False
        self._cabeceo_tiempo_inicio = None
        self._cabeceo_pitch_inicio = None
        self._cabeceo_pitch_minimo = None
        self.desviacion = 0.0
        self.ultima_velocidad = 0.0
        self.segundos_caida = 0.0

    def procesar(self, pose, ahora):
        """Procesa el PoseCabeza de un frame, tomado en el instante 'ahora'
        (segundos, de time.time()). Devuelve (cabeza_caida, cabeceo)."""
        if self.tiempo_arranque is None:
            self.tiempo_arranque = ahora

        # El pitch entra ya con el signo efectivo aplicado; de acá para
        # adelante todo (neutro, desviación, velocidad) es consistente.
        pitch = pose.pitch * self._signo_efectivo()
        self.pitch_crudo = pitch

        # --- Suavizado: media móvil corta sobre el pitch crudo ---
        self._historial.append((ahora, pitch))
        while (len(self._historial) > 2
               and self._historial[0][0] < ahora - SUAVIZADO_PITCH_SEGUNDOS):
            self._historial.popleft()
        self.pitch_suavizado = sum(p for _, p in self._historial) / len(self._historial)

        # --- Historial del pitch suavizado, para medir la velocidad y para
        # encontrar el "pico" de pitch justo antes de un cabeceo ---
        self._historial_suave.append((ahora, self.pitch_suavizado))
        ventana_larga = max(VENTANA_VELOCIDAD_SEGUNDOS, VENTANA_PICO_CABECEO_SEG)
        while (len(self._historial_suave) > 2
               and self._historial_suave[0][0] < ahora - ventana_larga):
            self._historial_suave.popleft()

        # --- Calibración de la posición neutra (con validación) ---
        if self.pitch_neutro is None:
            self._muestras_calibracion.append(pitch)
            self._yaws_calibracion.append(pose.yaw)
            self._proxys_calibracion.append(pose.proxy)
            listo = (ahora - self.tiempo_arranque) >= CALIBRACION_SEGUNDOS
            if listo and len(self._muestras_calibracion) >= 5:
                self._cerrar_calibracion(ahora)
            return False, False

        self.desviacion = self.pitch_suavizado - self.pitch_neutro
        self.ultima_velocidad = self._velocidad_angular(ahora)
        self._verificar_signo(ahora, pose.proxy)

        # El orden importa poco, pero evaluamos primero el cabeceo brusco:
        # si NO se recupera a tiempo, deja de ser cabeceo y el patrón 1 se
        # encarga de la cabeza caída.
        cabeceo = self._evaluar_cabeceo_brusco(ahora)
        cabeza_caida = self._evaluar_cabeza_caida(ahora)
        return cabeza_caida, cabeceo

    def _cerrar_calibracion(self, ahora):
        """Se llama cuando ya pasó el tiempo de calibración y hay suficientes
        muestras. Valida que el conductor estuviera quieto y mirando de
        frente; si no, descarta y reintenta (o acepta con aviso tras agotar
        los intentos)."""
        pitchs = self._muestras_calibracion
        spread = max(pitchs) - min(pitchs)
        yaw_medio = sum(self._yaws_calibracion) / len(self._yaws_calibracion)

        if spread > CALIBRACION_ESTABILIDAD_MAX_GRADOS:
            motivo = "te moviste demasiado"
        elif abs(yaw_medio) > CALIBRACION_YAW_MAX_GRADOS:
            motivo = "no estas mirando de frente a la camara"
        else:
            motivo = None

        def fijar_neutro():
            self.pitch_neutro = sum(pitchs) / len(pitchs)
            self.proxy_neutro = (
                sum(self._proxys_calibracion) / len(self._proxys_calibracion)
            )
            self.calibrando = False

        if motivo is None:
            fijar_neutro()
            self.aviso_calibracion = ""
            print(f"Calibracion OK. Pitch neutro = {self.pitch_neutro:+.1f} grados.")
            return

        # Falló: se reintenta sola. Vaciamos las muestras y volvemos a
        # arrancar el reloj; no hace falta tocar ninguna tecla.
        self._intentos_calibracion += 1
        self.aviso_calibracion = (f"Calibracion: {motivo}. Reintentando "
                                  f"(intento {self._intentos_calibracion + 1})...")
        print(f"Calibracion fallida ({motivo}). Reintentando automaticamente "
              f"(intento {self._intentos_calibracion + 1}): mira al frente y "
              f"quedate quieto.")
        if self._intentos_calibracion % CALIBRACION_AVISO_CADA_INTENTOS == 0:
            print("AVISO: la deteccion de cabeceos sigue en pausa hasta que la "
                  "calibracion salga bien (la de ojos cerrados funciona igual).")
        self._muestras_calibracion = []
        self._yaws_calibracion = []
        self._proxys_calibracion = []
        self.tiempo_arranque = ahora

    def _verificar_signo(self, ahora, proxy):
        """Comprueba una sola vez que el signo del pitch de solvePnP sea el
        correcto, comparándolo con el proxy geométrico 2D. Ambos deberían
        bajar juntos al mirar hacia abajo; si van al revés, invierte el signo.

        No hace nada si el signo está fijado a mano (SIGNO_PITCH) o si todavía
        no hubo suficiente movimiento de cabeza para poder juzgar."""
        if SIGNO_PITCH is not None or self._signo_verificado:
            return
        if self.proxy_neutro is None:
            return

        self._historial_signo.append((ahora, self.desviacion, proxy - self.proxy_neutro))
        while (len(self._historial_signo) > 2
               and self._historial_signo[0][0] < ahora - VENTANA_VERIF_SIGNO_SEG):
            self._historial_signo.popleft()
        if len(self._historial_signo) < 8:
            return

        desv_pitch = [d for _, d, _ in self._historial_signo]
        desv_proxy = [q for _, _, q in self._historial_signo]
        if (max(desv_pitch) - min(desv_pitch)) < VERIF_SIGNO_MOV_MINIMO_GRADOS:
            return  # la cabeza casi no se movió: no hay con qué juzgar todavía

        correlacion = float(np.corrcoef(desv_pitch, desv_proxy)[0, 1])
        if np.isnan(correlacion):
            return

        if correlacion <= -0.4:
            # Pitch y proxy se mueven al revés -> el pitch está invertido.
            self._signo_auto = -self._signo_auto
            # El neutro se calculó con el signo viejo: al invertir el signo,
            # el neutro nuevo es el opuesto. Los historiales quedaron con el
            # signo viejo, así que se limpian.
            self.pitch_neutro = -self.pitch_neutro
            self._historial.clear()
            self._historial_suave.clear()
            self._tiempo_inicio_caida = None
            self._cabeza_caida_estado = False
            self._en_cabeceo = False
            self.desviacion = 0.0
            self._signo_verificado = True
            print("AVISO: el pitch venia invertido para esta camara; corregido "
                  "automaticamente.")
        elif correlacion >= 0.4:
            # Signo correcto confirmado: no volvemos a chequear.
            self._signo_verificado = True

    def _velocidad_angular(self, ahora):
        """Velocidad de cambio del pitch, en grados por segundo: cuánto cambió
        el pitch suavizado en los últimos ~VENTANA_VELOCIDAD_SEGUNDOS,
        dividido ese tiempo. Negativa = la cabeza está bajando."""
        if len(self._historial_suave) < 2:
            return 0.0

        objetivo = ahora - VENTANA_VELOCIDAD_SEGUNDOS
        # Muestra más nueva que sea al menos tan vieja como 'objetivo'.
        referencia = self._historial_suave[0]
        for muestra in self._historial_suave:
            if muestra[0] <= objetivo:
                referencia = muestra
            else:
                break

        t_ref, p_ref = referencia
        t_nuevo, p_nuevo = self._historial_suave[-1]
        dt = t_nuevo - t_ref
        if dt < 1e-3:
            return 0.0
        return (p_nuevo - p_ref) / dt

    def _evaluar_cabeza_caida(self, ahora):
        """Patrón 1: la cabeza quedó inclinada hacia abajo más de
        CABEZA_CAIDA_GRADOS durante más de CABEZA_CAIDA_SEGUNDOS.

        Con HISTÉRESIS: para ENTRAR en estado de "cabeza caída" el pitch tiene
        que bajar de -CABEZA_CAIDA_GRADOS; para SALIR tiene que volver por
        encima de -(CABEZA_CAIDA_GRADOS - CABEZA_CAIDA_HISTERESIS_GRADOS). Así
        el temporizador no se reinicia por oscilaciones chicas cerca del
        límite y la alerta no parpadea."""
        umbral_entrar = -CABEZA_CAIDA_GRADOS
        umbral_salir = -(CABEZA_CAIDA_GRADOS - CABEZA_CAIDA_HISTERESIS_GRADOS)

        if not self._cabeza_caida_estado:
            if self.desviacion <= umbral_entrar:
                self._cabeza_caida_estado = True
        else:
            if self.desviacion > umbral_salir:
                self._cabeza_caida_estado = False

        if self._cabeza_caida_estado:
            if self._tiempo_inicio_caida is None:
                self._tiempo_inicio_caida = ahora
            self.segundos_caida = ahora - self._tiempo_inicio_caida
            return self.segundos_caida >= CABEZA_CAIDA_SEGUNDOS

        self._tiempo_inicio_caida = None
        self.segundos_caida = 0.0
        return False

    def _evaluar_cabeceo_brusco(self, ahora):
        """Patrón 2: caída rápida de la cabeza seguida de una recuperación
        (vuelve a subir) en menos de CABECEO_VENTANA_RECUPERACION_SEG."""
        if not self._en_cabeceo:
            # ¿Arranca una caída rápida?
            if self.ultima_velocidad <= -CABECEO_VELOCIDAD_GRADOS_POR_SEG:
                self._en_cabeceo = True
                self._cabeceo_tiempo_inicio = ahora
                # El punto de partida NO es el pitch de ahora (que, por el
                # suavizado y por la ventana con que medimos la velocidad, ya
                # bajó bastante): es el pitch más alto de los últimos
                # instantes, o sea la posición de la cabeza justo antes de
                # empezar a caer.
                self._cabeceo_pitch_inicio = max(
                    p for _, p in self._historial_suave
                )
                self._cabeceo_pitch_minimo = self.pitch_suavizado
            return False

        # Estamos en medio de un posible cabeceo: seguimos el punto más bajo.
        self._cabeceo_pitch_minimo = min(self._cabeceo_pitch_minimo,
                                         self.pitch_suavizado)
        amplitud = self._cabeceo_pitch_inicio - self._cabeceo_pitch_minimo
        transcurrido = ahora - self._cabeceo_tiempo_inicio

        if transcurrido > CABECEO_VENTANA_RECUPERACION_SEG:
            # No volvió a subir a tiempo: no es un cabeceo "brusco". Si la
            # cabeza sigue abajo, lo toma el patrón 1.
            self._en_cabeceo = False
            return False

        # ¿Ya remontó al menos una fracción de lo que había bajado?
        subida_desde_minimo = self.pitch_suavizado - self._cabeceo_pitch_minimo
        recuperado = subida_desde_minimo >= CABECEO_FRACCION_RECUPERACION * amplitud

        if (amplitud >= CABECEO_AMPLITUD_MINIMA_GRADOS
                and recuperado
                and subida_desde_minimo >= CABECEO_AMPLITUD_MINIMA_GRADOS * 0.5):
            self._en_cabeceo = False
            return True

        return False


# ==============================================================================
# MODELO DE MEDIAPIPE Y PROCESAMIENTO DE FRAMES
# ==============================================================================

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
        # Salidas extra que NO usamos: apagarlas ahorra trabajo en cada
        # cuadro (sobre todo los "blendshapes", que corren otra red
        # neuronal más). Solo necesitamos los puntos de la cara.
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return mp_vision.FaceLandmarker.create_from_options(opciones)


def procesar_frame(frame, detector_facial, timestamp_ms):
    """Le pasa el frame a MediaPipe y, si encontró una cara, calcula el EAR
    promedio de ambos ojos y la pose de la cabeza.

    IMPORTANTE: 'frame' tiene que ser el frame CRUDO de la cámara, sin
    espejar. El espejado se hace después, solo sobre la copia que se muestra
    en pantalla, para no romper la geometría de la pose (ver estimar_pose).

    'timestamp_ms' es un número que tiene que ir SIEMPRE EN AUMENTO entre
    llamada y llamada (MediaPipe, al trabajar en modo VIDEO, exige que cada
    cuadro tenga una marca de tiempo mayor a la del cuadro anterior, para
    saber en qué orden ocurrieron). No hace falta que sea el tiempo real:
    alcanza con un contador que sume de a uno en cada frame. OJO: esto es
    solo para que MediaPipe ordene los cuadros; NADA de la lógica de
    detección se mide con este contador (esa se mide con time.time()).

    Devuelve la tupla (ear_promedio, pose, puntos_ojo_izq, puntos_ojo_der),
    con 'pose' un PoseCabeza (o None). Los puntos de los ojos son en píxeles
    del frame CRUDO. Si NO se detectó ninguna cara, devuelve
    (None, None, None, None). 'pose' puede ser None aunque haya cara, si
    solvePnP no pudo resolver la pose en ese frame."""
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
        return None, None, None, None

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

    # Pose de la cabeza (puede fallar y devolver None en algún frame suelto).
    try:
        pose = estimar_pose(landmarks, ancho_frame, alto_frame)
    except cv2.error:
        pose = None

    return ear_promedio, pose, puntos_ojo_izq, puntos_ojo_der


# ==============================================================================
# ALARMA (MULTIPLATAFORMA)
# ==============================================================================
#
# El resto del programa solo conoce la clase 'Alarma' y sus tres métodos:
#
#   alarma.disparar(nivel)  -> pide que suene el patrón de ese nivel.
#   alarma.callar()         -> corta lo que esté sonando.
#   alarma.cerrar()         -> apaga todo al salir del programa.
#
# Por dentro, al arrancar, elige el "backend" (la forma concreta de hacer
# ruido) según la máquina:
#
#   - Raspberry Pi -> buzzer activo por GPIO, a través de un transistor, con
#                     la librería gpiozero (backend lgpio).
#   - Windows / PC -> tono por los parlantes, generado con numpy y reproducido
#                     con sounddevice.
#   - Si la librería que hace falta no carga (no está instalada, no hay
#     permisos sobre el GPIO, no hay placa de sonido...) se muestra una
#     advertencia y la alarma queda solo en pantalla/consola. El programa
#     NUNCA se corta por culpa de la alarma: detectar somnolencia es más
#     importante que el sonido.
#
# Los pitidos se reproducen en un HILO aparte (un "trabajador" que espera
# pedidos). Así 'disparar()' vuelve al instante y el bucle de video nunca se
# congela mientras suena la alarma.

def es_raspberry_pi():
    """True si el programa está corriendo en una Raspberry Pi.

    Linux expone el modelo de la placa en /proc/device-tree/model (por
    ejemplo "Raspberry Pi 4 Model B Rev 1.4"). En Windows o en una PC común
    ese archivo no existe, y devolvemos False."""
    if platform.system() != "Linux":
        return False
    try:
        with open("/proc/device-tree/model", "r", errors="ignore") as archivo:
            return "raspberry pi" in archivo.read().lower()
    except OSError:
        return False


class _SalidaBuzzer:
    """Buzzer activo conectado a un pin GPIO de la Raspberry Pi (vía
    transistor). Un buzzer ACTIVO suena solo con recibir tensión, así que
    alcanza con prenderlo y apagarlo."""

    nombre = "buzzer GPIO"

    def __init__(self, pin):
        # gpiozero puede usar varias librerías de bajo nivel para manejar los
        # pines; en Raspberry Pi OS actual la que funciona es lgpio. La
        # elegimos explícitamente, salvo que el usuario haya fijado otra.
        os.environ.setdefault("GPIOZERO_PIN_FACTORY", "lgpio")
        from gpiozero import Buzzer  # import acá: solo existe en la Pi
        self._buzzer = Buzzer(pin)
        self.nombre = f"buzzer GPIO{pin}"

    def encender(self):
        self._buzzer.on()

    def apagar(self):
        self._buzzer.off()

    def cerrar(self):
        self._buzzer.off()
        self._buzzer.close()


class _SalidaParlante:
    """Tono senoidal por los parlantes de la computadora, con sounddevice."""

    nombre = "parlantes (sounddevice)"
    _MUESTRAS_POR_SEG = 44100

    def __init__(self):
        import sounddevice  # import acá: si falta, se usa el modo sin sonido
        self._sd = sounddevice
        # Verificamos que exista un dispositivo de salida; si no, esto lanza
        # una excepción y caemos al modo sin sonido.
        self._sd.query_devices(kind="output")
        self._tonos = {}  # duración -> arreglo con el tono ya generado

    def _tono(self, segundos):
        """Genera (una sola vez por duración) la onda del pitido."""
        if segundos not in self._tonos:
            t = np.arange(int(self._MUESTRAS_POR_SEG * segundos)) / self._MUESTRAS_POR_SEG
            onda = 0.5 * np.sin(2 * np.pi * ALARM_FREQ_HZ * t)
            # Rampa de 5 ms al principio y al final para que no haga "clic".
            rampa = min(len(onda) // 2, int(0.005 * self._MUESTRAS_POR_SEG))
            if rampa > 0:
                envolvente = np.linspace(0.0, 1.0, rampa)
                onda[:rampa] *= envolvente
                onda[-rampa:] *= envolvente[::-1]
            self._tonos[segundos] = onda.astype(np.float32)
        return self._tonos[segundos]

    def encender(self, segundos):
        # sd.play() NO bloquea: arranca el sonido y vuelve enseguida.
        self._sd.play(self._tono(segundos), self._MUESTRAS_POR_SEG)

    def apagar(self):
        self._sd.stop()

    def cerrar(self):
        self._sd.stop()


class Alarma:
    """Alarma con la misma interfaz en la Raspberry Pi y en la PC. Ver la
    explicación al principio de esta sección."""

    def __init__(self):
        self._salida = self._elegir_salida()
        self.solo_visual = self._salida is None

        # Comunicación con el hilo trabajador: el nivel pedido (0 = nada) y
        # un "Condition" para despertarlo cuando llega un pedido.
        self._condicion = threading.Condition()
        self._nivel_pedido = 0
        self._cerrando = False
        # Se activa para interrumpir un patrón a mitad de camino.
        self._cortar = threading.Event()
        self._ultimo_aviso_consola = 0.0

        self._hilo = None
        if not self.solo_visual:
            self._hilo = threading.Thread(target=self._trabajador, daemon=True)
            self._hilo.start()

    def _elegir_salida(self):
        """Prueba la salida que corresponde a esta máquina. Si falla, avisa y
        devuelve None (alarma solo visual/consola)."""
        if es_raspberry_pi():
            try:
                salida = _SalidaBuzzer(BUZZER_GPIO)
            except Exception as error:
                print("ADVERTENCIA: no se pudo usar el buzzer por GPIO "
                      f"({type(error).__name__}: {error}).")
                print("  Revisá que estén instalados gpiozero y lgpio, y que "
                      "tu usuario esté en el grupo 'gpio'.")
                print("  El programa sigue, pero la alarma va a ser SOLO "
                      "VISUAL/CONSOLA.")
                return None
        else:
            try:
                salida = _SalidaParlante()
            except Exception as error:
                print("ADVERTENCIA: no se pudo usar el sonido "
                      f"({type(error).__name__}: {error}).")
                print("  Revisá que esté instalado sounddevice "
                      "(pip install sounddevice) y que haya parlantes.")
                print("  El programa sigue, pero la alarma va a ser SOLO "
                      "VISUAL/CONSOLA.")
                return None
        print(f"Alarma: usando {salida.nombre}.")
        return salida

    def disparar(self, nivel):
        """Pide que suene el patrón de 'nivel' (NIVEL_AVISO o NIVEL_PELIGRO).
        Vuelve al instante. Si ya está sonando un patrón, el pedido queda
        anotado y se reproduce cuando ese termina; varios pedidos seguidos se
        juntan en uno solo (gana el nivel más alto). Así, mientras una
        condición se mantiene, el patrón se repite sin superponerse."""
        if self.solo_visual:
            # Sin sonido: avisamos por consola, como mucho una vez por segundo
            # para no inundarla (el bucle llama a esto en cada frame).
            ahora = time.time()
            if ahora - self._ultimo_aviso_consola >= 1.0:
                self._ultimo_aviso_consola = ahora
                print("\a*** ALARMA ***", flush=True)  # '\a' = campana de la terminal
            return
        with self._condicion:
            self._nivel_pedido = max(self._nivel_pedido, nivel)
            self._condicion.notify()

    def callar(self):
        """Corta el patrón que esté sonando y descarta pedidos pendientes."""
        if self.solo_visual:
            return
        with self._condicion:
            self._nivel_pedido = 0
        self._cortar.set()

    def cerrar(self):
        """Apaga la alarma y termina el hilo. Llamar siempre al salir: en la
        Pi, si el programa se corta con el buzzer prendido, puede quedar
        sonando."""
        if self.solo_visual:
            return
        with self._condicion:
            self._cerrando = True
            self._nivel_pedido = 0
            self._condicion.notify()
        self._cortar.set()
        if self._hilo is not None:
            self._hilo.join(timeout=2.0)
        try:
            self._salida.cerrar()
        except Exception:
            pass

    def _trabajador(self):
        """Corre en su propio hilo: espera pedidos y reproduce los patrones."""
        while True:
            with self._condicion:
                while self._nivel_pedido == 0 and not self._cerrando:
                    self._condicion.wait()
                if self._cerrando:
                    return
                nivel = self._nivel_pedido
                self._nivel_pedido = 0
                self._cortar.clear()
            try:
                self._reproducir(PATRONES_ALARMA[nivel])
            except Exception as error:
                # Un error de audio/GPIO a mitad de camino no debe matar el
                # hilo ni el programa: avisamos y seguimos.
                print(f"ADVERTENCIA: fallo al hacer sonar la alarma: {error}")
            finally:
                try:
                    self._salida.apagar()
                except Exception:
                    pass

    def _reproducir(self, patron):
        """Reproduce un patrón (lista de (sonando, silencio) en segundos).
        Usa Event.wait() en lugar de time.sleep() para poder cortarlo al
        instante con callar() o cerrar()."""
        for sonando, silencio in patron:
            if isinstance(self._salida, _SalidaParlante):
                self._salida.encender(sonando)
            else:
                self._salida.encender()
            if self._cortar.wait(sonando):
                return
            self._salida.apagar()
            if self._cortar.wait(silencio):
                return


# ==============================================================================
# DIBUJO SOBRE LA IMAGEN
# ==============================================================================

def dibujar_alerta_ojos(frame):
    """Cartel rojo de alerta por ojos cerrados (franja de arriba de todo)."""
    ancho_frame = frame.shape[1]
    cv2.rectangle(frame, (0, 0), (ancho_frame, 55), (0, 0, 255), -1)
    cv2.putText(frame, "ALERTA! OJOS CERRADOS", (10, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)


def dibujar_alerta_cabeza_caida(frame):
    """Cartel rojo de alerta por cabeza caída sostenida (segunda franja).

    Se dibuja más abajo que la de ojos para que, si las dos saltan al mismo
    tiempo, no se pisen y se puedan leer ambas."""
    ancho_frame = frame.shape[1]
    cv2.rectangle(frame, (0, 55), (ancho_frame, 105), (0, 0, 255), -1)
    cv2.putText(frame, "ALERTA! CABEZA CAIDA", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)


def dibujar_alerta_cabeceo(frame):
    """Cartel rojo de alerta por cabeceo brusco (tercera franja)."""
    ancho_frame = frame.shape[1]
    cv2.rectangle(frame, (0, 105), (ancho_frame, 155), (0, 0, 255), -1)
    cv2.putText(frame, "ALERTA! CABECEO BRUSCO", (10, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)


def dibujar_hud(frame, ear_promedio, segundos_ojos_cerrados, detector_cabeceos):
    """Muestra en pantalla, abajo a la izquierda, los valores en vivo (EAR,
    pitch crudo y suavizado, desviación, velocidad, temporizadores). No hace
    falta para que la alarma funcione: es para poder CALIBRAR los umbrales
    mirando la imagen. Mostrar el pitch crudo y el suavizado por separado
    permite ver si el ruido viene del cálculo (salta el crudo, no el
    suavizado) o de algún otro lado."""
    alto_frame = frame.shape[0]
    verde = (0, 255, 0)
    amarillo = (0, 255, 255)
    fuente = cv2.FONT_HERSHEY_SIMPLEX

    # Línea 1: EAR y cuánto tiempo llevan los ojos cerrados.
    if ear_promedio is not None:
        texto_ear = (f"EAR: {ear_promedio:.3f} (umbral {EAR_THRESHOLD:.2f})   "
                     f"ojos cerrados: {segundos_ojos_cerrados:.1f}s"
                     f" / {DROWSY_TIME_SECONDS:.1f}s")
    else:
        texto_ear = "EAR: --   (sin rostro)"
    cv2.putText(frame, texto_ear, (10, alto_frame - 66),
                fuente, 0.6, verde, 2, cv2.LINE_AA)

    # Línea 2 y 3: estado de la pose de la cabeza.
    if detector_cabeceos.calibrando:
        mensaje = (detector_cabeceos.aviso_calibracion
                   or "Calibrando pose de la cabeza... MIRA AL FRENTE Y QUEDATE QUIETO")
        cv2.putText(frame, mensaje, (10, alto_frame - 40),
                    fuente, 0.6, amarillo, 2, cv2.LINE_AA)
    elif detector_cabeceos.pitch_neutro is not None:
        texto_pose = (f"pitch crudo: {detector_cabeceos.pitch_crudo:+.1f}   "
                      f"suav: {detector_cabeceos.pitch_suavizado:+.1f}   "
                      f"desv: {detector_cabeceos.desviacion:+.1f}")
        cv2.putText(frame, texto_pose, (10, alto_frame - 40),
                    fuente, 0.6, verde, 2, cv2.LINE_AA)
        texto_pose2 = (f"vel: {detector_cabeceos.ultima_velocidad:+.0f}/s   "
                       f"caida: {detector_cabeceos.segundos_caida:.1f}s"
                       f" / {CABEZA_CAIDA_SEGUNDOS:.1f}s")
        cv2.putText(frame, texto_pose2, (10, alto_frame - 16),
                    fuente, 0.6, verde, 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "pitch: -- (sin pose)", (10, alto_frame - 40),
                    fuente, 0.6, amarillo, 2, cv2.LINE_AA)


# ==============================================================================
# CAPTURA DE LA CÁMARA EN UN HILO APARTE
# ==============================================================================
#
# ¿POR QUÉ UN HILO APARTE?
# La cámara entrega cuadros a un ritmo fijo (por ejemplo 30 por segundo). Si
# el programa procesa más lento que eso (le pasa a la Raspberry Pi, donde
# MediaPipe tarda bastante por cuadro), los cuadros que no llegó a leer se
# van acumulando en una cola dentro del driver de la cámara. Cada vez que el
# programa pide "el próximo cuadro" recibe uno VIEJO, de hace varios
# segundos: eso es el "delay" que se veía en la Pi. Para un detector de
# somnolencia eso es grave: la alarma sonaría tarde.
#
# La solución: un hilo dedicado que lee la cámara sin parar, lo más rápido
# que ella entregue, y guarda SOLO el último cuadro (los anteriores se
# descartan). El bucle de procesamiento, cuando termina con un cuadro, toma
# siempre el más reciente. Así nunca se procesa una imagen vieja.

def abrir_camara(indice, ancho, alto):
    """Abre la cámara con la configuración que menos delay genera.

    - Backend según la plataforma: V4L2 en Linux (el nativo; es el que
      respeta bien formato y tamaño de buffer) y DirectShow en Windows (que
      acepta pedir MJPG). Si con ese backend no abre, se prueba el automático.
    - Formato MJPG: la cámara manda cada cuadro ya comprimido en JPEG. Por
      USB pasan muchos menos datos que en crudo (YUYV), así que la cámara
      puede dar más cuadros por segundo a la misma resolución.
    - Buffer de 1 cuadro: que el driver no guarde cuadros viejos.
    - Resolución: si 'ancho'/'alto' son None se deja la que trae la cámara.

    Devuelve el cv2.VideoCapture abierto, o None si no se pudo abrir."""
    sistema = platform.system()
    if sistema == "Linux":
        backend = cv2.CAP_V4L2
    elif sistema == "Windows":
        backend = cv2.CAP_DSHOW
    else:
        backend = cv2.CAP_ANY

    captura = cv2.VideoCapture(indice, backend)
    if not captura.isOpened() and backend != cv2.CAP_ANY:
        captura.release()
        captura = cv2.VideoCapture(indice)
    if not captura.isOpened():
        captura.release()
        return None

    # Orden importante: primero el formato, después la resolución (algunos
    # drivers solo ofrecen ciertas resoluciones en MJPG).
    captura.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    if ancho and alto:
        captura.set(cv2.CAP_PROP_FRAME_WIDTH, ancho)
        captura.set(cv2.CAP_PROP_FRAME_HEIGHT, alto)
    captura.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # no todos los backends lo aceptan

    # Mostramos lo que la cámara REALMENTE aceptó (puede ignorar el pedido).
    fourcc = int(captura.get(cv2.CAP_PROP_FOURCC))
    formato = "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4))
    log(f"Cámara {indice}: {int(captura.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
        f"{int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT))}, formato "
        f"{formato.strip() or '?'}, {captura.get(cv2.CAP_PROP_FPS):.0f} FPS "
        f"nominales, backend {captura.getBackendName()}.")
    return captura


class CapturaEnHilo:
    """Lee la cámara en un hilo propio y se queda solo con el último cuadro.

    Uso:
        camara = CapturaEnHilo(captura)
        numero, frame, momento = camara.esperar_cuadro_nuevo(numero_anterior)
        ...
        camara.detener()

    Cada cuadro tiene un 'numero' que va en aumento, para saber si es nuevo,
    y el 'momento' (time.time()) en que se capturó."""

    # Si la cámara no entrega ningún cuadro durante este tiempo, se considera
    # perdida (desconectada).
    SEGUNDOS_SIN_CUADROS_PERDIDA = 5.0

    def __init__(self, captura):
        self._captura = captura
        self._condicion = threading.Condition()
        self._frame = None
        self._numero = 0
        self._momento = 0.0
        self._activa = True
        self.perdida = False
        self.cuadros_leidos = 0   # contador total, para calcular FPS de captura
        self._hilo = threading.Thread(target=self._leer_sin_parar, daemon=True)
        self._hilo.start()

    def _leer_sin_parar(self):
        ultimo_ok = time.time()
        while self._activa:
            ok, frame = self._captura.read()
            ahora = time.time()
            if not ok:
                if ahora - ultimo_ok > self.SEGUNDOS_SIN_CUADROS_PERDIDA:
                    with self._condicion:
                        self.perdida = True
                        self._condicion.notify_all()
                    return
                time.sleep(0.01)
                continue
            ultimo_ok = ahora
            with self._condicion:
                # Pisamos el cuadro anterior: si nadie lo procesó, se pierde.
                # Es justamente lo que queremos.
                self._frame = frame
                self._numero += 1
                self._momento = ahora
                self.cuadros_leidos += 1
                self._condicion.notify_all()

    def esperar_cuadro_nuevo(self, numero_anterior, numero_minimo=None,
                             timeout=1.0):
        """Espera hasta que haya un cuadro con número > 'numero_anterior'
        (o >= 'numero_minimo', si se indica) y devuelve (numero, frame,
        momento). Devuelve (None, None, None) si pasó 'timeout' sin cuadros
        nuevos o si la cámara se perdió (mirar el atributo 'perdida')."""
        if numero_minimo is None:
            numero_minimo = numero_anterior + 1
        with self._condicion:
            hay_nuevo = self._condicion.wait_for(
                lambda: self._numero >= numero_minimo or self.perdida
                or not self._activa,
                timeout=timeout)
            if not hay_nuevo or self.perdida or not self._activa:
                return None, None, None
            return self._numero, self._frame, self._momento

    def detener(self):
        """Frena el hilo y libera la cámara."""
        self._activa = False
        with self._condicion:
            self._condicion.notify_all()
        self._hilo.join(timeout=2.0)
        self._captura.release()


def revisar_alimentacion_pi():
    """En la Raspberry Pi, pregunta al firmware (vcgencmd get_throttled) si
    hubo bajo voltaje o exceso de temperatura. Las dos cosas hacen que la Pi
    baje su velocidad a propósito ("throttling") y el detector vaya lento.
    Si el comando no existe, no hace nada."""
    if not es_raspberry_pi():
        return
    try:
        salida = subprocess.run(["vcgencmd", "get_throttled"],
                                capture_output=True, text=True, timeout=3)
    except (OSError, subprocess.SubprocessError):
        return
    texto = salida.stdout.strip()          # por ejemplo "throttled=0x50005"
    if "=" not in texto:
        return
    valor_texto = texto.split("=", 1)[1]
    try:
        valor = int(valor_texto, 16)
    except ValueError:
        return
    if valor == 0:
        log("Alimentación y temperatura de la Pi: OK (throttled=0x0).")
        return

    # Significado de cada bit (documentación oficial de Raspberry Pi):
    # los bits 0-3 son lo que pasa AHORA; los 16-19, lo que pasó desde que
    # se prendió la Pi.
    problemas = {
        0: "bajo voltaje AHORA (fuente insuficiente o cable malo)",
        1: "frecuencia de la CPU limitada AHORA",
        2: "la Pi está frenada (throttled) AHORA",
        3: "límite de temperatura AHORA",
        16: "hubo bajo voltaje desde que se prendió",
        17: "hubo frecuencia limitada desde que se prendió",
        18: "estuvo frenada desde que se prendió",
        19: "llegó al límite de temperatura desde que se prendió",
    }
    log(f"ADVERTENCIA: vcgencmd get_throttled = {valor_texto} (debería ser 0x0).")
    for bit, descripcion in problemas.items():
        if valor & (1 << bit):
            print(f"  - {descripcion}")
    print("  Usá la fuente oficial (5 V / 3 A en la Pi 4, 2,5 A en la Pi 3) y "
          "un disipador o ventilador: si no, el detector va a ir más lento.")


# ==============================================================================
# BUCLE PRINCIPAL
# ==============================================================================

def log(mensaje):
    """Imprime un evento en la consola con la hora adelante. En modo sin
    ventana la consola es la única forma de ver qué está pasando."""
    print(f"[{time.strftime('%H:%M:%S')}] {mensaje}", flush=True)


def leer_argumentos():
    """Lee las opciones de la línea de comandos. Ejemplo:
        python detector_somnoliencia.py --sin-ventana"""
    parser = argparse.ArgumentParser(
        description="Detector de somnolencia por cámara (ojos y cabeceos).")
    parser.add_argument(
        "--sin-ventana", action="store_true",
        help="no abre la ventana de video (para usar por SSH o sin monitor). "
             "Salís con Ctrl+C.")
    parser.add_argument(
        "--ancho", type=int, default=None,
        help="ancho de la imagen pedida a la cámara, en píxeles "
             "(por defecto 640 en la Raspberry Pi; en la PC, el de la cámara).")
    parser.add_argument(
        "--alto", type=int, default=None,
        help="alto de la imagen pedida a la cámara, en píxeles "
             "(por defecto 480 en la Raspberry Pi; en la PC, el de la cámara).")
    parser.add_argument(
        "--saltar", type=int, default=1, metavar="N",
        help="procesar 1 de cada N cuadros de la cámara (por defecto 1 = "
             "todos). Sirve si la máquina no da abasto. Los tiempos de "
             "detección siguen medidos en segundos, no en cuadros.")
    argumentos = parser.parse_args()
    if argumentos.saltar < 1:
        parser.error("--saltar tiene que ser 1 o más.")
    if (argumentos.ancho is None) != (argumentos.alto is None):
        parser.error("--ancho y --alto se usan juntos.")
    if argumentos.ancho is None and es_raspberry_pi():
        argumentos.ancho, argumentos.alto = 640, 480
    return argumentos


def hay_pantalla():
    """False si estamos en Linux sin entorno gráfico (por ejemplo, conectados
    por SSH a la Raspberry Pi). Ahí cv2.imshow aborta el programa con
    'qt.qpa.xcb: could not connect to display', así que hay que evitarlo.
    En Windows y macOS siempre hay pantalla."""
    if platform.system() != "Linux":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def escuchar_teclado(comandos):
    """Corre en un hilo aparte en modo sin ventana: lee líneas de la consola
    y las deja en la cola 'comandos'. Así se puede recalibrar escribiendo 'c'
    + Enter (y salir con 'q' + Enter, además de Ctrl+C), aunque no haya
    ventana donde apretar teclas."""
    while True:
        try:
            linea = sys.stdin.readline()
        except (OSError, ValueError):
            return
        if not linea:  # se cerró la entrada (por ejemplo, corre como servicio)
            return
        comandos.put(linea.strip().lower())


def main():
    """Función principal: abre la cámara y corre el bucle de detección."""
    argumentos = leer_argumentos()

    # --- ¿Con o sin ventana? ---
    mostrar_ventana = not argumentos.sin_ventana
    if mostrar_ventana and not hay_pantalla():
        mostrar_ventana = False
        log("No hay pantalla (no existe DISPLAY ni WAYLAND_DISPLAY): se activa "
            "automáticamente el modo --sin-ventana.")

    # En Linux, si alguien manda la señal de terminar (por ejemplo, 'kill' o
    # al detener un servicio), la tratamos igual que Ctrl+C para cerrar
    # ordenadamente y no dejar el buzzer sonando.
    if platform.system() == "Linux":
        def _al_terminar(_senal, _marco):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, _al_terminar)

    # En la Pi, avisar de entrada si la fuente o la temperatura la frenan.
    revisar_alimentacion_pi()

    # --- Preparamos el detector facial de MediaPipe ---
    detector_facial = crear_detector_facial()

    # --- Detector de cabeceos (pose de la cabeza) ---
    detector_cabeceos = DetectorCabeceos()

    # --- Abrimos la webcam ---
    captura = abrir_camara(CAMARA_INDICE, argumentos.ancho, argumentos.alto)

    if captura is None:
        # Si no se pudo abrir la cámara (no hay cámara, está siendo usada
        # por otro programa, permisos de Windows bloqueados, etc.), avisamos
        # con un mensaje claro y cortamos el programa en vez de romper con
        # un error críptico más adelante.
        print(f"ERROR: no se pudo acceder a la webcam (indice {CAMARA_INDICE}).")
        print("Verificá que la cámara esté conectada, que no la esté usando")
        print("otra aplicación, y que el sistema tenga permitido el acceso a")
        print("la cámara. Si tenés varias cámaras, probá otro valor en la")
        print("constante CAMARA_INDICE.")
        detector_facial.close()
        return

    # Momento en que empezó la racha actual de "ojos cerrados" (None si los
    # ojos están abiertos). La duración de la racha se calcula como
    # 'ahora - este_momento', SIEMPRE con el reloj, nunca contando frames.
    tiempo_ojos_cerrados_inicio = None
    segundos_ojos_cerrados = 0.0

    # Estado "ojos cerrados" con histéresis: se entra con EAR < EAR_THRESHOLD
    # y se sale recién con EAR > EAR_THRESHOLD + EAR_HISTERESIS, así el
    # contador de tiempo no se reinicia por oscilaciones del EAR en el límite.
    ojos_cerrados_estado = False

    # Alarma (buzzer en la Pi, parlantes en la PC, o solo visual si ninguno
    # funciona). Nunca bloquea el bucle: los pitidos suenan en otro hilo.
    alarma = Alarma()

    # Momento del último cabeceo brusco confirmado (None si no hubo, o si ya
    # pasó su tiempo en pantalla). El cartel se mantiene visible unos
    # segundos porque el cabeceo es un evento instantáneo.
    momento_alerta_cabeceo = None

    # Estado anterior de cada alerta, para imprimir en consola solo cuando
    # EMPIEZA o TERMINA (y no una línea por cada frame).
    alerta_ojos_antes = False
    alerta_cabeza_antes = False
    hay_rostro_antes = None
    calibrando_antes = True

    # Contador que le pasamos a MediaPipe como "marca de tiempo" de cada
    # frame. Solo sirve para que MediaPipe ordene los cuadros; no se usa para
    # ninguna medición de la lógica de detección.
    contador_timestamp_ms = 0

    # Desde acá la cámara se lee en su propio hilo (ver CapturaEnHilo): el
    # bucle siempre toma el cuadro más reciente y nunca uno atrasado.
    camara = CapturaEnHilo(captura)
    numero_procesado = 0   # número del último cuadro que procesamos

    # Estadísticas de rendimiento, que se imprimen cada 5 segundos.
    SEGUNDOS_ENTRE_ESTADISTICAS = 5.0
    momento_estadisticas = time.time()
    leidos_antes = 0
    procesados = 0
    tiempo_procesando = 0.0

    # En modo sin ventana, los comandos se escriben en la consola.
    comandos = queue.Queue()
    if not mostrar_ventana and sys.stdin is not None and sys.stdin.isatty():
        threading.Thread(target=escuchar_teclado, args=(comandos,),
                         daemon=True).start()

    log("Detector de somnolencia iniciado"
        + (" (modo sin ventana)." if not mostrar_ventana else "."))
    log(f"Calibrando la pose de la cabeza durante {CALIBRACION_SEGUNDOS:.0f} "
        f"segundos: mirá al frente y quedate quieto.")
    if mostrar_ventana:
        print("Presioná 'q' para salir.")
    else:
        print("Escribí 'c' + Enter para volver a calibrar la pose. "
              "Ctrl+C (o 'q' + Enter) para salir.")

    try:
        while True:
            # Tomamos el cuadro MÁS RECIENTE de la cámara (esperando si
            # todavía no llegó uno nuevo). Con --saltar N se espera a que la
            # cámara haya entregado N cuadros desde el último procesado.
            # Este 'frame_crudo' NO se espeja: toda la detección (MediaPipe,
            # EAR, pose) trabaja sobre él.
            #
            # 'ahora' es el momento en que la cámara capturó el cuadro, en
            # segundos (time.time()). TODA la lógica temporal usa esto.
            numero, frame_crudo, ahora = camara.esperar_cuadro_nuevo(
                numero_procesado,
                numero_minimo=numero_procesado + argumentos.saltar)
            if numero is None:
                if camara.perdida:
                    # La cámara dejó de entregar cuadros (por ejemplo, se
                    # desconectó a mitad de la ejecución).
                    log("ERROR: se perdió la conexión con la cámara.")
                    break
                continue  # todavía no llegó un cuadro nuevo; seguimos esperando
            numero_procesado = numero
            inicio_proceso = time.time()

            contador_timestamp_ms += 1
            ear_promedio, pose, puntos_ojo_izq, puntos_ojo_der = procesar_frame(
                frame_crudo, detector_facial, contador_timestamp_ms
            )

            # A partir de acá dibujamos sobre 'lienzo': la copia ESPEJADA que
            # se muestra en pantalla (efecto espejo, más natural para
            # mirarse). Es solo visual; la detección ya se hizo sobre el
            # frame crudo. Sin ventana no hay nada que dibujar: 'lienzo'
            # queda en None y nos ahorramos ese trabajo.
            lienzo = cv2.flip(frame_crudo, 1) if mostrar_ventana else None

            hay_rostro = ear_promedio is not None
            if hay_rostro != hay_rostro_antes:
                log("Rostro detectado." if hay_rostro else "No se detecta rostro.")
                hay_rostro_antes = hay_rostro

            # ==============================================================
            # SEÑAL 1: OJOS CERRADOS (EAR)
            # ==============================================================
            alerta_ojos = False
            if ear_promedio is None:
                # No se detectó ninguna cara. Cortamos la racha de ojos
                # cerrados para no arrastrar una que en realidad es "la cara
                # salió de cuadro", y avisamos en pantalla.
                tiempo_ojos_cerrados_inicio = None
                segundos_ojos_cerrados = 0.0
                ojos_cerrados_estado = False
                if lienzo is not None:
                    cv2.putText(lienzo, "No se detecta rostro", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
                                cv2.LINE_AA)
            else:
                # Dibujamos los puntos de cada ojo (con la x espejada, porque
                # los puntos vienen en coordenadas del frame crudo y el lienzo
                # está espejado).
                if lienzo is not None:
                    ancho_lienzo = lienzo.shape[1]
                    for (x, y) in puntos_ojo_izq + puntos_ojo_der:
                        cv2.circle(lienzo, (ancho_lienzo - 1 - x, y), 2,
                                   (0, 255, 0), -1)

                # Histéresis: entrar en "ojos cerrados" con EAR < umbral,
                # salir recién cuando el EAR supera umbral + EAR_HISTERESIS.
                if not ojos_cerrados_estado and ear_promedio < EAR_THRESHOLD:
                    ojos_cerrados_estado = True
                elif ojos_cerrados_estado and ear_promedio > EAR_THRESHOLD + EAR_HISTERESIS:
                    ojos_cerrados_estado = False

                if ojos_cerrados_estado:
                    # Si es el comienzo de la racha anotamos el momento; si ya
                    # venía, medimos cuánto lleva (con el reloj, no con
                    # frames).
                    if tiempo_ojos_cerrados_inicio is None:
                        tiempo_ojos_cerrados_inicio = ahora
                    segundos_ojos_cerrados = ahora - tiempo_ojos_cerrados_inicio
                else:
                    tiempo_ojos_cerrados_inicio = None
                    segundos_ojos_cerrados = 0.0

                if segundos_ojos_cerrados >= DROWSY_TIME_SECONDS:
                    alerta_ojos = True
                    if lienzo is not None:
                        dibujar_alerta_ojos(lienzo)
                    alarma.disparar(NIVEL_PELIGRO)

            if alerta_ojos != alerta_ojos_antes:
                log("ALERTA: OJOS CERRADOS." if alerta_ojos
                    else "Fin de alerta: ojos abiertos.")
                alerta_ojos_antes = alerta_ojos

            # ==============================================================
            # SEÑAL 2: CABECEOS (POSE DE LA CABEZA)
            # ==============================================================
            # Independiente de los ojos: se calcula siempre que haya una pose
            # válida, aunque el EAR de ese frame haya fallado.
            alerta_cabeza = False
            if pose is not None:
                cabeza_caida, cabeceo_brusco = detector_cabeceos.procesar(pose, ahora)

                if cabeza_caida:
                    alerta_cabeza = True
                    if lienzo is not None:
                        dibujar_alerta_cabeza_caida(lienzo)
                    alarma.disparar(NIVEL_PELIGRO)

                if cabeceo_brusco:
                    log("ALERTA: cabeceo brusco detectado.")
                    momento_alerta_cabeceo = ahora
                    alarma.disparar(NIVEL_AVISO)

            if alerta_cabeza != alerta_cabeza_antes:
                log("ALERTA: CABEZA CAIDA." if alerta_cabeza
                    else "Fin de alerta: cabeza levantada.")
                alerta_cabeza_antes = alerta_cabeza

            # La calibración imprime su propio resultado; acá solo marcamos
            # cuándo se termina, con hora, para seguirlo en la consola.
            if calibrando_antes and not detector_cabeceos.calibrando:
                log("Calibración terminada: detección de cabeceos activa.")
            calibrando_antes = detector_cabeceos.calibrando

            # Mantenemos el cartel de cabeceo unos segundos después del evento.
            if momento_alerta_cabeceo is not None:
                if (ahora - momento_alerta_cabeceo) < DURACION_ALERTA_CABECEO_SEG:
                    if lienzo is not None:
                        dibujar_alerta_cabeceo(lienzo)
                else:
                    momento_alerta_cabeceo = None

            # --- Rendimiento: FPS de captura y de procesamiento ---
            procesados += 1
            tiempo_procesando += time.time() - inicio_proceso
            transcurrido = time.time() - momento_estadisticas
            if transcurrido >= SEGUNDOS_ENTRE_ESTADISTICAS:
                leidos = camara.cuadros_leidos
                fps_captura = (leidos - leidos_antes) / transcurrido
                fps_proceso = procesados / transcurrido
                ms_por_cuadro = 1000.0 * tiempo_procesando / procesados
                atraso_ms = 1000.0 * (time.time() - ahora)
                log(f"FPS captura: {fps_captura:.1f} | FPS procesamiento: "
                    f"{fps_proceso:.1f} | {ms_por_cuadro:.0f} ms por cuadro | "
                    f"atraso del último cuadro: {atraso_ms:.0f} ms")
                momento_estadisticas = time.time()
                leidos_antes = leidos
                procesados = 0
                tiempo_procesando = 0.0

            # --- Teclas / comandos ---
            tecla = None
            if mostrar_ventana:
                # Info en vivo para calibrar (EAR y pitch en pantalla).
                dibujar_hud(lienzo, ear_promedio, segundos_ojos_cerrados,
                            detector_cabeceos)
                # Mostramos el lienzo (imagen espejada) en una ventana.
                cv2.imshow("Detector de Somnolencia", lienzo)
                # Esperamos 1 milisegundo a que se presione una tecla. El
                # '& 0xFF' es una forma estándar de comparar la tecla en
                # distintos sistemas operativos.
                codigo = cv2.waitKey(1) & 0xFF
                if codigo != 0xFF:
                    tecla = chr(codigo)
            else:
                try:
                    tecla = comandos.get_nowait()
                except queue.Empty:
                    pass

            if tecla == "q":
                log("Saliendo del programa...")
                break
            # Recalibrar a mano solo en modo sin ventana ('c' + Enter). Con
            # ventana, por ahora, la calibración es solo automática.
            if tecla == "c" and not mostrar_ventana:
                detector_cabeceos.recalibrar()
                calibrando_antes = True
                log("Recalibrando la pose de la cabeza: mirá al frente y "
                    "quedate quieto.")

    except KeyboardInterrupt:
        # Ctrl+C: no es un error, es la forma normal de salir sin ventana.
        print()
        log("Ctrl+C: saliendo del programa...")

    finally:
        # --- Liberamos los recursos antes de terminar ---
        # Muy importante: si no liberamos la cámara, puede quedar "ocupada"
        # y otras aplicaciones (o el propio programa, si lo volvés a correr)
        # no van a poder usarla. Esto se hace SIEMPRE, se salga como se
        # salga (q, Ctrl+C o un error).
        alarma.cerrar()  # en la Pi, deja el buzzer apagado sí o sí
        camara.detener()  # frena el hilo de captura y libera la cámara
        if mostrar_ventana:
            cv2.destroyAllWindows()
        detector_facial.close()
        log("Recursos liberados. Chau.")


# Este bloque hace que 'main()' se ejecute solo cuando corrés este archivo
# directamente (por ejemplo, con "python detector_somnoliencia.py"), y no si
# alguna vez este archivo se importa desde otro script de Python.
if __name__ == "__main__":
    main()
