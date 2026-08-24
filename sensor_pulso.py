# ==============================================================================
# SENSOR DE PULSO: LECTURA DE FRECUENCIA CARDÍACA (BPM) POR BLUETOOTH (BLE)
# ==============================================================================
#
# ¿QUÉ HACE ESTE ARCHIVO?
# ------------------------
# El detector principal (detector_somnoliencia.py) mira los OJOS con la cámara
# y los CABEZAZOS con el acelerómetro (sensor_acelerometro.py). Este archivo
# se ocupa de la tercera señal: las PULSACIONES (frecuencia cardíaca, en
# latidos por minuto o "BPM").
#
# La idea fisiológica es la siguiente: cuando una persona se relaja y empieza
# a quedarse dormida, el sistema nervioso pasa a predominar en modo
# "parasimpático" y la frecuencia cardíaca suele BAJAR de forma sostenida
# respecto de su valor habitual mientras se está manejando (que normalmente
# está algo elevado por la actividad de conducir, con foco y tensión). Una
# caída sostenida de varios latidos por minuto, mantenida durante bastantes
# segundos (no un segundo suelto, que puede ser ruido del sensor), es una
# señal de posible somnolencia. Por eso este archivo NO dispara la alarma por
# sí solo: solo devuelve una señal booleana que la lógica central combina con
# la cámara y el acelerómetro (ver más abajo, "CÓMO SE INTEGRA CON EL RESTO").
#
# ¿QUÉ SENSOR DE PULSO CONVIENE USAR?
# --------------------------------------
# Recomendación: una BANDA PECTORAL que hable el protocolo BLE ESTÁNDAR
# "Heart Rate Service" (definido por el Bluetooth SIG, UUID 0x180D). Es el
# mismo protocolo que usan apps de fitness/ciclismo, no es algo propietario:
# cualquier sensor que lo implemente funciona con este script sin cambiarle
# una línea. Dos opciones concretas, ambas usadas y documentadas:
#
#   - Polar H10 (recomendado): banda pectoral, mide por ECG (mucho más
#     precisa que un sensor óptico de muñeca), pila de botón con meses de
#     autonomía, BLE muy estable y ampliamente probado en proyectos con
#     Raspberry Pi. Es el "estándar de oro" de referencia en la industria.
#   - Wahoo TICKR: alternativa más económica, mismo protocolo estándar
#     (0x180D / 0x2A37), también banda pectoral, también confiable.
#
# Por qué NO conviene una pulsera/smartband genérica de muñeca: la mayoría
# (Xiaomi Mi Band y clones incluidos) usan protocolos BLE PROPIETARIOS, no el
# Heart Rate Service estándar, así que no se pueden leer con un cliente BLE
# genérico como este sin ingeniería inversa del protocolo de cada marca. Una
# banda pectoral con GATT estándar es la opción robusta y mantenible.
#
# ¿QUÉ ES EL "GATT HEART RATE SERVICE"?
# ----------------------------------------
# BLE organiza los datos que expone un dispositivo en "servicios" y, dentro
# de cada servicio, "características". El Heart Rate Service (UUID
# 0000180d-...) tiene una característica llamada Heart Rate Measurement
# (UUID 00002a37-...) a la que nos podemos "suscribir": el sensor nos avisa
# solo, cada vez que tiene un nuevo latido calculado (normalmente 1 vez por
# segundo), sin que tengamos que estar preguntando todo el tiempo. El
# contenido de cada aviso es un paquete de bytes con un formato fijo definido
# por el estándar (ver 'parsear_heart_rate_measurement' más abajo).
#
# ¿QUÉ LIBRERÍA CONVIENE PARA LEER BLE EN LA RASPBERRY?
# ----------------------------------------------------------
# 'bleak' (la misma que ya usa sensor_acelerometro.py en su modo "ble").
# Es la elección correcta acá porque:
#   - Es la que ya eligió este proyecto para BLE: mantener una sola librería
#     de Bluetooth simplifica la instalación y el mantenimiento.
#   - En Raspberry Pi OS (Linux), bleak habla con el Bluetooth del sistema a
#     través de BlueZ (el stack de Bluetooth estándar de Linux) vía D-Bus,
#     que es la forma soportada y recomendada de usar BLE en Linux.
#   - 'bluepy', la otra opción típica en Raspberry, es específica de Linux,
#     tiene menos mantenimiento activo y una API de más bajo nivel (hay que
#     lidiar directamente con hilos y con el binario bluepy-helper). No suma
#     nada frente a bleak para este caso de uso, así que no se usa acá.
#
# ¿CÓMO MANEJAMOS RUIDO Y LECTURAS INVÁLIDAS?
# -----------------------------------------------
# Un valor de BPM fuera del rango fisiológico humano razonable (por ejemplo,
# 3 o 400) no es una pulsación real: es ruido del sensor, de la conexión BLE,
# o un instante de mal contacto con la piel. Esas lecturas se descartan: no
# entran en el promedio ni cortan una racha de caída que ya esté en curso
# (ver 'DetectorAnomaliaBPM.procesar_muestra').
#
# ¿CÓMO SE INTEGRA CON EL RESTO? (LEER ANTES DE TOCAR detector_somnoliencia.py)
# ----------------------------------------------------------------------------------
# Este archivo sigue EXACTAMENTE el mismo patrón de integración que ya usa
# sensor_acelerometro.py, para no inventar un mecanismo nuevo:
#
#     lector = crear_lector("ble", nombre_ble="Polar H10 12345678")
#     lector.iniciar()
#     ...
#     muestras = lector.leer_muestras()   # no bloquea, se llama en el bucle principal
#     for muestra in muestras:
#         if detector_bpm.procesar_muestra(muestra):
#             ...  # posible señal de somnolencia por caída de BPM
#     ...
#     lector.detener()
#
# Es decir: NO se usa cola/MQTT/sockets separados del resto del programa. La
# lectura BLE corre en su propio hilo en segundo plano (para no bloquear la
# cámara ni el acelerómetro) y dejar las muestras en una 'queue.Queue'
# interna; el bucle principal las retira cuando quiere, sin esperar.
# Esto es deliberado: es el mismo mecanismo que 'LectorBLE' de
# sensor_acelerometro.py, así detector_somnoliencia.py puede tratar a los
# tres sensores de la misma forma.
#
# IMPORTANTE: este archivo, a propósito, NO llama a ninguna función de
# alarma/parlante/motor vibrador. Devuelve una señal booleana
# (True = "hay una caída de BPM sostenida ahora mismo") para que sea la
# lógica central (detector_somnoliencia.py) la que decida qué hacer con esa
# señal, tal como ya hace con 'DetectorCabezazos'. La propuesta concreta de
# integración (qué líneas agregar a detector_somnoliencia.py) se entrega por
# separado, para revisar antes de tocar el código de disparo de alarma.
#
# ==============================================================================

# --- Librerías de la biblioteca estándar de Python ---
import csv
import logging
import os
import queue
import threading
import time
from collections import deque

# NOTA: 'bleak' NO se importa acá arriba a propósito, sino adentro de
# LectorBLE, igual que hace sensor_acelerometro.py. Así este módulo se puede
# importar y usar en modo simulador aunque bleak no esté instalado.


# ==============================================================================
# CONSTANTES DE CONFIGURACIÓN
# ==============================================================================

# --- UUIDs estándar del "Heart Rate Service" (definidos por el Bluetooth SIG,
# no por ningún fabricante en particular: cualquier sensor de pulso que
# cumpla el estándar los expone con estos mismos valores). ---
UUID_SERVICIO_HEART_RATE = "0000180d-0000-1000-8000-00805f9b34fb"
UUID_CARACTERISTICA_HEART_RATE = "00002a37-0000-1000-8000-00805f9b34fb"

# Rango de BPM que consideramos fisiológicamente posible en un ser humano
# despierto o adormilado. Fuera de este rango, el dato es ruido del sensor
# (mal contacto, glitch de la conexión BLE) y se descarta.
BPM_MINIMO_VALIDO = 30
BPM_MAXIMO_VALIDO = 220

# Cuántos BPM por debajo del promedio habitual del conductor consideramos una
# "caída". Es un valor RELATIVO (respecto del propio promedio de cada
# persona), no un número de BPM fijo, porque el pulso de reposo varía mucho
# de una persona a otra.
CAIDA_BPM_UMBRAL = 15

# Cuántos segundos tiene que mantenerse la caída para considerarla una señal
# real (y no un instante de ruido o una respiración profunda pasajera).
TIEMPO_MINIMO_CAIDA_SEG = 8.0

# Qué tan rápido se actualiza el promedio lento que estima el "BPM habitual"
# del conductor en este viaje. Cuanto más chico, más lento se adapta: con
# sensores que mandan ~1 muestra por segundo, 0.02 tarda más o menos un
# minuto en asentarse, lo cual está bien porque el pulso de reposo cambia
# despacio (no queremos que el promedio "seatore" cada 2 latidos).
FACTOR_PROMEDIO_LENTO_BPM = 0.02

# Si pasa esta cantidad de segundos sin recibir ninguna muestra VÁLIDA,
# consideramos que el sensor se desconectó o perdió contacto con la piel.
# Es más alto que el de acelerómetro porque un sensor de pulso BLE típico
# manda datos mucho menos seguido (≈1 vez por segundo, no 50 veces).
TIMEOUT_SENSOR_SEG = 5.0

# --- Reconexión automática BLE ---
# Cuántos segundos esperamos antes de reintentar conectar, si no se encontró
# el sensor o si la conexión se cortó.
REINTENTO_CONEXION_SEG = 5.0

# Cuánto tiempo (segundos) escaneamos buscando el sensor antes de darlo por
# no encontrado en ese intento (y pasar a reintentar).
ESCANEO_TIMEOUT_SEG = 10.0

# --- Log de lecturas a archivo, con marca de tiempo ---
# Cada lectura (válida o descartada por ruido) se guarda en este archivo CSV
# para poder correlacionarla después con los registros de cámara y
# acelerómetro. El archivo se guarda al lado de este script.
NOMBRE_ARCHIVO_LOG = "log_pulso.csv"


# ==============================================================================
# LOG DE LECTURAS A ARCHIVO (PARA ANÁLISIS POSTERIOR)
# ==============================================================================

def _crear_logger_bpm(ruta_archivo):
    """Crea (o reutiliza) un logger que escribe cada lectura de BPM a un
    archivo, con marca de tiempo, en formato fácil de abrir con una planilla
    de cálculo (CSV: timestamp,bpm,valida).

    Usamos el módulo 'logging' de la biblioteca estándar en vez de abrir y
    cerrar el archivo a mano en cada lectura: 'logging' ya se encarga de que
    escribir desde un hilo en segundo plano (como el de BLE) sea seguro, y de
    ir agregando líneas al archivo sin pisar lo que ya estaba guardado."""
    logger = logging.getLogger("sensor_pulso.lecturas")
    logger.setLevel(logging.INFO)

    # Si esta función se llama más de una vez (por ejemplo, si se crea más de
    # un LectorBLE), evitamos agregar el mismo archivo de log dos veces: si
    # no, cada lectura se escribiría duplicada.
    if not logger.handlers:
        manejador = logging.FileHandler(ruta_archivo, encoding="utf-8")
        formato = logging.Formatter("%(asctime)s.%(msecs)03d,%(message)s",
                                     datefmt="%Y-%m-%d %H:%M:%S")
        manejador.setFormatter(formato)
        logger.addHandler(manejador)
        logger.propagate = False  # no repetir estas líneas en la consola

    return logger


def _registrar_lectura(logger, bpm, valida):
    """Agrega una línea al log: <hora>,<bpm>,<True/False según si el dato
    pasó el filtro de rango fisiológico>. Guardamos también las lecturas
    inválidas (marcadas como tales) porque saber CUÁNTO ruido hubo también es
    información útil al analizar el viaje después."""
    logger.info(f"{bpm},{valida}")


# ==============================================================================
# REPRESENTACIÓN DE UNA MUESTRA
# ==============================================================================

class MuestraPulso:
    """Una única lectura de frecuencia cardíaca, ya interpretada.

    'tiempo_seg' es la hora (reloj monotónico de la computadora) en que
    LLEGÓ la muestra. A diferencia del acelerómetro, acá no usamos una marca
    de tiempo propia del sensor: el estándar BLE Heart Rate Measurement no
    incluye una, y como los cambios de BPM son lentos (segundos, no
    milisegundos), el jitter de Bluetooth no afecta la medición de forma
    apreciable.

    'valida' es False cuando el valor de BPM está fuera del rango
    fisiológico posible (ver BPM_MINIMO_VALIDO / BPM_MAXIMO_VALIDO): en ese
    caso el dato se guarda igual (para el log) pero no participa de la
    detección de anomalías."""

    def __init__(self, tiempo_seg, bpm, valida=True):
        self.tiempo_seg = tiempo_seg
        self.bpm = bpm
        self.valida = valida


def parsear_heart_rate_measurement(datos):
    """Interpreta los bytes crudos que manda la característica estándar
    'Heart Rate Measurement' (UUID 0x2A37) y devuelve el valor de BPM.

    El formato está fijado por el estándar del Bluetooth SIG, no depende del
    fabricante del sensor:

      byte 0        : "flags" (banderas). El bit 0 nos dice si el valor de
                       BPM viene en 1 byte (0) o en 2 bytes (1). Los sensores
                       de pulso humano casi siempre usan 1 byte (BPM < 256),
                       pero soportamos ambos casos por las dudas.
      byte 1 (o 1-2): el valor de BPM en sí.
      bytes restantes: energía gastada / intervalos RR, si el sensor los
                       manda. No los necesitamos para este proyecto, así que
                       los ignoramos: no hace falta leerlos para sacar el BPM,
                       que siempre viene primero.

    Devuelve el BPM (entero) o None si el paquete viene incompleto/corrupto."""
    if not datos or len(datos) < 2:
        return None

    flags = datos[0]
    formato_16_bits = bool(flags & 0x01)

    if formato_16_bits:
        if len(datos) < 3:
            return None
        bpm = int.from_bytes(datos[1:3], byteorder="little")
    else:
        bpm = datos[1]

    return bpm


# ==============================================================================
# DETECTOR DE ANOMALÍAS DE BPM (POSIBLE SEÑAL DE SOMNOLENCIA)
# ==============================================================================

class DetectorAnomaliaBPM:
    """Recibe muestras de BPM y marca cuándo hay una caída sostenida respecto
    del promedio habitual del conductor en este viaje.

    Se usa exactamente igual que 'DetectorCabezazos' de sensor_acelerometro.py:
    se le van pasando las muestras una por una con 'procesar_muestra', y esa
    función devuelve True en el instante exacto en que se confirma la caída
    sostenida. Por diseño, ESTA CLASE NO DISPARA NINGUNA ALARMA: solo informa
    la señal, para que sea detector_somnoliencia.py quien decida combinarla
    con la cámara y el acelerómetro."""

    def __init__(self,
                 caida_bpm=CAIDA_BPM_UMBRAL,
                 tiempo_minimo_caida_seg=TIEMPO_MINIMO_CAIDA_SEG,
                 factor_promedio=FACTOR_PROMEDIO_LENTO_BPM,
                 bpm_minimo_valido=BPM_MINIMO_VALIDO,
                 bpm_maximo_valido=BPM_MAXIMO_VALIDO):
        self.caida_bpm = caida_bpm
        self.tiempo_minimo_caida_seg = tiempo_minimo_caida_seg
        self.factor_promedio = factor_promedio
        self.bpm_minimo_valido = bpm_minimo_valido
        self.bpm_maximo_valido = bpm_maximo_valido

        # Promedio lento que estima el "BPM habitual" del conductor en este
        # viaje. Empieza en None: se inicializa con la primera muestra válida.
        self.promedio_lento = None

        # Último BPM válido recibido, y desde cuándo estamos (si estamos) en
        # medio de una posible racha de caída sostenida.
        self.ultimo_bpm_valido = None
        self.tiempo_inicio_caida = None

        # Cuántas muestras inválidas (ruido / fuera de rango) llevamos
        # seguidas. Sirve para que la lógica central pueda distinguir "el
        # sensor no tiene buen contacto" de "hay una señal real de
        # somnolencia": ver 'hay_senal_valida'.
        self.muestras_invalidas_seguidas = 0

    def procesar_muestra(self, muestra):
        """Procesa una muestra de BPM.

        Devuelve True SOLO en el instante en que se confirma una caída
        sostenida (es decir, cuando el BPM estuvo por debajo del promedio en
        más de 'caida_bpm' latidos durante al menos 'tiempo_minimo_caida_seg'
        segundos seguidos). En cualquier otro caso devuelve False."""
        if not muestra.valida or not (self.bpm_minimo_valido <= muestra.bpm <= self.bpm_maximo_valido):
            # Dato de ruido: no lo usamos para nada del cálculo. Importante:
            # tampoco cortamos una racha de caída que ya estuviera en curso,
            # porque un instante de mal contacto no significa que el pulso
            # haya vuelto a subir.
            self.muestras_invalidas_seguidas += 1
            return False

        self.muestras_invalidas_seguidas = 0
        self.ultimo_bpm_valido = muestra.bpm

        if self.promedio_lento is None:
            # Primera muestra válida: la tomamos como punto de partida del
            # promedio y no evaluamos ninguna caída todavía (no hay contra
            # qué comparar).
            self.promedio_lento = muestra.bpm
            return False

        caida_actual = self.promedio_lento - muestra.bpm

        if caida_actual >= self.caida_bpm:
            # Estamos por debajo del promedio en más de lo permitido.
            if self.tiempo_inicio_caida is None:
                self.tiempo_inicio_caida = muestra.tiempo_seg

            duracion = muestra.tiempo_seg - self.tiempo_inicio_caida

            # A propósito NO actualizamos 'promedio_lento' mientras estamos
            # dentro de una posible caída: si lo hiciéramos, el promedio
            # empezaría a "perseguir" al BPM bajo y la caída terminaría
            # cancelándose sola antes de llegar al tiempo mínimo, sin importar
            # cuánto durase.
            if duracion >= self.tiempo_minimo_caida_seg:
                # Caída confirmada. Reiniciamos para poder detectar una
                # próxima caída más adelante en el viaje.
                self.tiempo_inicio_caida = None
                return True
        else:
            # El BPM volvió a estar dentro de lo normal: cortamos cualquier
            # racha de caída en curso y seguimos actualizando el promedio con
            # este valor, mezclando un poquito del valor nuevo (factor chico
            # = adaptación lenta, ver comentario de FACTOR_PROMEDIO_LENTO_BPM).
            self.tiempo_inicio_caida = None
            self.promedio_lento = (
                self.factor_promedio * muestra.bpm
                + (1.0 - self.factor_promedio) * self.promedio_lento
            )

        return False

    def hay_senal_valida(self, maximo_muestras_invalidas_seguidas=5):
        """Indica si el sensor está dando datos con sentido ahora mismo (no
        necesariamente indica somnolencia: solo indica que el dato de BPM es
        confiable). Sirve para mostrar en pantalla, por ejemplo, "sin buen
        contacto" en vez de un BPM que en realidad es ruido."""
        return self.muestras_invalidas_seguidas < maximo_muestras_invalidas_seguidas


# ==============================================================================
# LECTORES DE SENSOR (LA "CAPA INTERCAMBIABLE")
# ==============================================================================
#
# Mismo contrato que en sensor_acelerometro.py, para que detector_somnoliencia.py
# pueda tratar los tres sensores de la misma forma:
#
#     lector.iniciar()
#     muestras = lector.leer_muestras()   # se llama seguido, no bloquea
#     lector.detener()


class LectorBase:
    """Clase base de la que heredan todos los lectores de pulso."""

    def iniciar(self):
        """Abre la conexión con el sensor."""
        raise NotImplementedError

    def leer_muestras(self):
        """Devuelve la lista de muestras (MuestraPulso) nuevas que llegaron
        desde la última vez que se llamó a esta función.

        MUY IMPORTANTE: igual que en sensor_acelerometro.py, esta función
        nunca se queda esperando. Si todavía no llegó nada, devuelve una
        lista vacía y listo, para no congelar el bucle principal (cámara +
        acelerómetro) esperando al Bluetooth del sensor de pulso."""
        raise NotImplementedError

    def detener(self):
        """Cierra la conexión con el sensor."""
        raise NotImplementedError


class LectorSimulado(LectorBase):
    """Simulador: genera BPM falso, sin ningún hardware.

    Sirve para probar toda la lógica de detección (y el resto del programa)
    antes de tener el sensor real conectado. Genera un pulso que ronda los
    75 BPM con variación normal, y ofrece DOS formas de forzar una caída
    artificial para probar el detector:

      - 'forzar_caida(cantidad_muestras)': una caída CRONOMETRADA, que se
        recupera sola después de esa cantidad de muestras. Útil para scripts
        de prueba automáticos.
      - 'forzar_caida_manual()' / 'restaurar_pulso_normal()': una caída
        MANUAL, indefinida, que se mantiene baja hasta que vos mismo la
        cancelás. Pensada para atarla a dos teclas (una para bajar el pulso,
        otra para devolverlo a la normalidad), igual que la tecla 'c' ya
        simula un cabezazo en sensor_acelerometro.py."""

    def __init__(self, bpm_base=75.0, frecuencia_hz=1.0):
        self.bpm_base = bpm_base
        self.frecuencia_hz = frecuencia_hz
        self.tiempo_inicio = None
        self.tiempo_ultima_muestra = None
        self.caida_pendiente = False
        self.muestras_de_caida_restantes = 0
        # Caída "manual": a diferencia de 'muestras_de_caida_restantes' (que
        # se cuenta sola y se agota), esta queda prendida hasta que se llame
        # a 'restaurar_pulso_normal()', sin importar cuánto tiempo pase.
        self._caida_manual_activa = False
        # Semilla fija para que el comportamiento sea reproducible al probar.
        import random
        self._random = random.Random(7)

    def iniciar(self):
        self.tiempo_inicio = time.monotonic()
        self.tiempo_ultima_muestra = self.tiempo_inicio

    def forzar_caida(self, cantidad_muestras=12):
        """Simula una caída sostenida de BPM a partir de la próxima muestra,
        que se recupera sola después de 'cantidad_muestras' lecturas (para
        probar el detector sin intervención manual)."""
        self.caida_pendiente = True
        self.muestras_de_caida_restantes = cantidad_muestras

    def forzar_caida_manual(self):
        """Baja el BPM simulado ya mismo, y lo deja bajo indefinidamente
        hasta que se llame a 'restaurar_pulso_normal()'. Pensado para
        controlarse a mano, por ejemplo apretando una tecla."""
        self._caida_manual_activa = True

    def restaurar_pulso_normal(self):
        """Cancela la caída manual (si había una) y vuelve a generar BPM
        dentro de lo normal."""
        self._caida_manual_activa = False

    def leer_muestras(self):
        ahora = time.monotonic()
        muestras = []

        periodo = 1.0 / self.frecuencia_hz
        while (ahora - self.tiempo_ultima_muestra) >= periodo:
            self.tiempo_ultima_muestra += periodo
            tiempo_muestra = self.tiempo_ultima_muestra - self.tiempo_inicio

            if self.caida_pendiente:
                self.caida_pendiente = False

            if self.muestras_de_caida_restantes > 0:
                self.muestras_de_caida_restantes -= 1

            # El BPM sale bajo si CUALQUIERA de las dos formas de caída está
            # activa ahora mismo: la manual (indefinida) o la cronometrada
            # (todavía le quedan muestras por descontar).
            en_caida = self._caida_manual_activa or self.muestras_de_caida_restantes > 0

            if en_caida:
                bpm = self.bpm_base - 20 + self._random.uniform(-2.0, 2.0)
            else:
                bpm = self.bpm_base + self._random.uniform(-3.0, 3.0)

            muestras.append(MuestraPulso(tiempo_muestra, round(bpm)))

        return muestras

    def detener(self):
        pass


class LectorBLE(LectorBase):
    """Lee la frecuencia cardíaca por BLE desde un sensor que implementa el
    Heart Rate Service estándar (UUID 0x180D), como un Polar H10 o un Wahoo
    TICKR.

    A diferencia de 'LectorBLE' en sensor_acelerometro.py, este lector SÍ
    reintenta la conexión solo, en un bucle, si el sensor no se encuentra o
    si la conexión se corta (se pidió explícitamente reconexión automática,
    algo importante acá porque el sensor puede perder contacto con la piel
    o quedar momentáneamente fuera de rango durante el viaje)."""

    def __init__(self, nombre_dispositivo=None, direccion_ble=None,
                 registrar_en_archivo=True, ruta_log=None):
        if not nombre_dispositivo and not direccion_ble:
            raise ValueError(
                "Hay que indicar 'nombre_dispositivo' o 'direccion_ble' "
                "para saber a qué sensor conectarse."
            )

        self.nombre_dispositivo = nombre_dispositivo
        # Conectar por dirección MAC (si se conoce) es más rápido y más
        # confiable para la RECONEXIÓN que buscar por nombre: no depende de
        # que el nombre anunciado por BLE sea siempre exactamente igual, y
        # evita tener que re-escanear todos los dispositivos cercanos cada
        # vez. Se recomienda usarla cuando se conoce de antemano el sensor
        # (por ejemplo, leyendo la MAC impresa en la banda pectoral o con
        # 'bluetoothctl scan on' una vez).
        self.direccion_ble = direccion_ble

        self.cola = queue.Queue()
        self.hilo = None
        self.debe_seguir = False
        self._conectado = False

        if registrar_en_archivo:
            carpeta_del_script = os.path.dirname(os.path.abspath(__file__))
            ruta_log = ruta_log or os.path.join(carpeta_del_script, NOMBRE_ARCHIVO_LOG)
            self._logger = _crear_logger_bpm(ruta_log)
        else:
            self._logger = None

    def iniciar(self):
        self.debe_seguir = True
        self.hilo = threading.Thread(target=self._bucle_asyncio, daemon=True)
        self.hilo.start()

    def esta_conectado(self):
        """True mientras haya una conexión BLE activa con el sensor ahora
        mismo (no solo mientras el hilo esté vivo: el hilo sigue vivo incluso
        mientras se está reintentando la conexión)."""
        return self._conectado

    def _al_recibir_datos(self, _caracteristica, datos):
        """Callback que llama 'bleak' cada vez que el sensor manda una nueva
        medición de frecuencia cardíaca (típicamente, una vez por segundo).

        Corre dentro del hilo/bucle de asyncio de este lector, no en el hilo
        principal del programa: por eso deja la muestra en 'self.cola' en vez
        de devolverla directamente, siguiendo el mismo patrón que
        LectorBLE.al_recibir_datos en sensor_acelerometro.py."""
        bpm = parsear_heart_rate_measurement(bytes(datos))
        if bpm is None:
            return

        valida = BPM_MINIMO_VALIDO <= bpm <= BPM_MAXIMO_VALIDO
        muestra = MuestraPulso(time.monotonic(), bpm, valida=valida)
        self.cola.put(muestra)

        if self._logger is not None:
            _registrar_lectura(self._logger, bpm, valida)

    def _bucle_asyncio(self):
        """Igual que en sensor_acelerometro.py: bleak trabaja de forma
        asincrónica, así que encapsulamos todo eso en su propio hilo para no
        complicar el resto del programa (que no usa asyncio).

        A diferencia del acelerómetro, acá el bucle NO termina si el sensor
        no aparece o si se desconecta: reintenta solo cada
        REINTENTO_CONEXION_SEG segundos, hasta que se llame a 'detener()'."""
        import asyncio

        try:
            from bleak import BleakClient, BleakScanner
        except ImportError:
            print("ERROR: el sensor de pulso necesita la libreria 'bleak'.")
            print("Instalala con:  pip install bleak")
            return

        async def buscar_dispositivo():
            """Busca el sensor por dirección MAC (si la tenemos: es más
            rápido y confiable) o, si no, por nombre anunciado."""
            if self.direccion_ble:
                return await BleakScanner.find_device_by_address(
                    self.direccion_ble, timeout=ESCANEO_TIMEOUT_SEG
                )
            return await BleakScanner.find_device_by_name(
                self.nombre_dispositivo, timeout=ESCANEO_TIMEOUT_SEG
            )

        async def principal():
            while self.debe_seguir:
                try:
                    dispositivo = await buscar_dispositivo()

                    if dispositivo is None:
                        objetivo = self.direccion_ble or self.nombre_dispositivo
                        print(f"AVISO: no se encontro el sensor de pulso "
                              f"'{objetivo}'. Reintentando en "
                              f"{REINTENTO_CONEXION_SEG:.0f}s...")
                        await asyncio.sleep(REINTENTO_CONEXION_SEG)
                        continue

                    # Evento que se activa cuando bleak detecta que el sensor
                    # se desconectó (se apagó, salió de rango, etc.), para
                    # poder salir del bucle de "estoy conectado" y pasar a
                    # reintentar.
                    desconectado = asyncio.Event()

                    def al_desconectarse(_cliente):
                        desconectado.set()

                    async with BleakClient(
                        dispositivo, disconnected_callback=al_desconectarse
                    ) as cliente:
                        print(f"Sensor de pulso conectado "
                              f"({self.nombre_dispositivo or self.direccion_ble}).")
                        self._conectado = True

                        await cliente.start_notify(
                            UUID_CARACTERISTICA_HEART_RATE, self._al_recibir_datos
                        )

                        # Nos quedamos "de guardia" mientras la conexión siga
                        # viva. No hacemos nada activamente acá: los datos
                        # llegan solos por el callback 'al_recibir_datos'.
                        while self.debe_seguir and not desconectado.is_set():
                            await asyncio.sleep(0.2)

                    self._conectado = False

                    if self.debe_seguir:
                        print("AVISO: se perdio la conexion BLE con el sensor "
                              f"de pulso. Reintentando en "
                              f"{REINTENTO_CONEXION_SEG:.0f}s...")
                        await asyncio.sleep(REINTENTO_CONEXION_SEG)

                except Exception as error:
                    # Cualquier error de conexión (timeout, sensor ocupado con
                    # otro dispositivo, error de BlueZ, etc.) no debe tirar
                    # abajo el programa: lo avisamos y reintentamos.
                    self._conectado = False
                    print(f"ERROR en la conexion BLE del sensor de pulso: {error}")
                    await asyncio.sleep(REINTENTO_CONEXION_SEG)

        asyncio.run(principal())

    def leer_muestras(self):
        muestras = []
        while True:
            try:
                muestras.append(self.cola.get_nowait())
            except queue.Empty:
                break
        return muestras

    def detener(self):
        self.debe_seguir = False


# ==============================================================================
# FUNCIÓN DE AYUDA PARA ELEGIR EL LECTOR
# ==============================================================================

def crear_lector(modo, nombre_ble="Polar H10", direccion_ble=None):
    """Devuelve el lector que corresponda según el modo elegido.

    Modos disponibles:
      "simulador" : BPM falso, no necesita hardware. Ideal para probar.
      "ble"       : Bluetooth Low Energy, sensor con Heart Rate Service
                    estándar (Polar H10, Wahoo TICKR, etc.).

    'direccion_ble' (la MAC del sensor, ej. "AA:BB:CC:DD:EE:FF") es opcional
    pero recomendada para uso en la Raspberry: conecta más rápido y
    reconecta de forma más confiable que buscar por nombre. Si no se indica,
    se busca por 'nombre_ble'."""
    if modo == "simulador":
        return LectorSimulado()
    if modo == "ble":
        return LectorBLE(nombre_dispositivo=nombre_ble, direccion_ble=direccion_ble)

    raise ValueError(
        f"Modo de sensor de pulso desconocido: '{modo}'. "
        f"Las opciones válidas son 'simulador' o 'ble'."
    )


# ==============================================================================
# PRUEBA INDEPENDIENTE (opcional, solo para verificar el sensor por separado)
# ==============================================================================
# Este bloque NO forma parte del contrato de integración (ese es
# 'LectorBase'/'crear_lector', igual que en sensor_acelerometro.py). Es
# simplemente una forma rápida de probar, desde la Raspberry, que el sensor
# de pulso conecta y manda datos, ANTES de integrarlo al programa principal.
#
# Uso:
#     python sensor_pulso.py --modo simulador
#     python sensor_pulso.py --modo ble --nombre-ble "Polar H10 12345678"
#     python sensor_pulso.py --modo ble --direccion-ble AA:BB:CC:DD:EE:FF
if __name__ == "__main__":
    import argparse

    analizador = argparse.ArgumentParser(
        description="Prueba independiente del lector de pulso (BPM) por BLE."
    )
    analizador.add_argument("--modo", default="simulador", choices=["simulador", "ble"])
    analizador.add_argument("--nombre-ble", default="Polar H10")
    analizador.add_argument("--direccion-ble", default=None)
    opciones = analizador.parse_args()

    lector = crear_lector(
        opciones.modo,
        nombre_ble=opciones.nombre_ble,
        direccion_ble=opciones.direccion_ble,
    )
    detector = DetectorAnomaliaBPM()

    lector.iniciar()
    print("Leyendo pulso. Presioná Ctrl+C para salir.")
    if opciones.modo == "simulador":
        print("Modo simulador: la caída de BPM se puede forzar editando este bloque,")
        print("o usando LectorSimulado.forzar_caida() desde otro script de prueba.")

    try:
        while True:
            for muestra in lector.leer_muestras():
                estado = "valida" if muestra.valida else "RUIDO (descartada)"
                print(f"BPM: {muestra.bpm:3d}  ({estado})")

                if detector.procesar_muestra(muestra):
                    print(">>> Posible señal de somnolencia: caída sostenida de BPM.")

            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nSaliendo...")
    finally:
        lector.detener()
