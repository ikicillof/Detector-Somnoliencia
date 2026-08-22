# ==============================================================================
# SENSOR ACELERÓMETRO: DETECCIÓN DE CABEZAZOS Y MOVIMIENTOS BRUSCOS
# ==============================================================================
#
# ¿QUÉ HACE ESTE ARCHIVO?
# ------------------------
# El detector principal (detector_somnoliencia.py) mira los OJOS con la cámara.
# Este archivo se ocupa de la otra señal de somnolencia: los "cabezazos", esos
# movimientos bruscos de la cabeza que hace alguien cuando se está quedando
# dormido y se despierta de golpe.
#
# Para eso usamos un ACELERÓMETRO: un sensor chiquito que se coloca en la
# cabeza (o en una vincha/gorra/auricular) y que mide la aceleración a la que
# está sometido, en los tres ejes del espacio (X, Y, Z). Cuando la cabeza cae
# de golpe y se recupera, ese sensor registra un "pico" de aceleración mucho
# más grande que el de un movimiento normal.
#
# La regla que queremos implementar es: si ocurren DOS O MÁS cabezazos en
# menos de 20 segundos, disparar una alerta.
#
# ¿CÓMO SE DETECTA UN CABEZAZO A PARTIR DE NÚMEROS?
# ---------------------------------------------------
# El sensor nos manda todo el tiempo tres números (ax, ay, az). El primer paso
# es reducir esos tres números a UNO SOLO que represente "cuánta aceleración
# hay en total, sin importar la dirección". Eso se hace con la MAGNITUD:
#
#     magnitud = raiz_cuadrada( ax^2 + ay^2 + az^2 )
#
# El problema es que, incluso con el sensor totalmente quieto arriba de una
# mesa, esa magnitud NO da cero: da alrededor de 9.81, porque el acelerómetro
# siempre mide la gravedad de la Tierra. Entonces lo que nos interesa no es la
# magnitud en sí, sino CUÁNTO SE APARTA de su valor "de reposo".
#
# Para saber cuál es el valor de reposo no lo fijamos en 9.81 a mano (cada
# sensor está calibrado un poco distinto), sino que lo vamos estimando solo,
# con un promedio que se actualiza muy lentamente. Como se mueve lento, la
# gravedad (que es constante) queda capturada en ese promedio, mientras que un
# golpe seco —que dura milésimas de segundo— no llega a afectarlo y aparece
# como una desviación grande. A esa técnica se la llama "filtro paso-alto":
# dejamos pasar los cambios rápidos y descartamos los lentos.
#
#     desviacion = valor_absoluto( magnitud - promedio_lento )
#
# Si esa desviación supera un umbral, decimos que hubo un movimiento brusco.
#
# ¿POR QUÉ HAY UN "TIEMPO REFRACTARIO"?
# ---------------------------------------
# Un solo cabezazo no es un único pico limpio: es un sacudón que dura unas
# décimas de segundo y que, mirado muestra por muestra, supera el umbral
# muchas veces seguidas. Si contáramos cada una de esas muestras como un
# cabezazo distinto, un solo movimiento contaría como veinte y la alerta
# saltaría siempre.
#
# Para evitarlo, después de contar un cabezazo ignoramos todo lo que pase
# durante un ratito corto (el "tiempo refractario", por ejemplo 0.4 segundos).
# Así, un sacudón cuenta como UN evento, que es lo que queremos.
#
# ¿CÓMO MANEJAMOS EL DELAY DEL BLUETOOTH?
# -----------------------------------------
# Los datos llegan por Bluetooth, que introduce un retraso pequeño pero real,
# y —peor aún— un retraso que VARÍA de muestra a muestra (a eso se lo llama
# "jitter"). Si midiéramos los 20 segundos usando la hora a la que los datos
# LLEGAN a la computadora, ese jitter nos ensuciaría la medición.
#
# La solución es que el sensor incluya SU PROPIA marca de tiempo en cada
# muestra (el tiempo medido por el microcontrolador, que no sufre el retraso
# del Bluetooth). Nosotros hacemos todas las cuentas de tiempo con esa marca.
# Así, aunque los datos lleguen tarde, la distancia entre dos cabezazos se
# mide exactamente como ocurrió en la cabeza del conductor.
#
# Si el sensor no manda su propia marca de tiempo, igual funciona: usamos la
# hora de llegada como respaldo. Un retraso constante no molesta (corre todo
# por igual); lo que degrada un poco la precisión es el jitter.
#
# ==============================================================================
#
# FORMATO DE DATOS QUE DEBE ENVIAR EL SENSOR
# --------------------------------------------
# Como todavía no está definido el hardware, acá va el formato que propone
# este programa. Es simple de generar desde un Arduino o un ESP32 y fácil de
# leer. Cada muestra es UNA LÍNEA DE TEXTO terminada en salto de línea:
#
#     t_ms,ax,ay,az\n
#
# donde:
#   t_ms : entero. Milisegundos desde que arrancó el microcontrolador
#          (en Arduino/ESP32 es directamente la función millis()).
#   ax,ay,az : números decimales. Aceleración en cada eje, en m/s^2.
#
# Ejemplo de tres muestras reales:
#
#     12500,0.12,-0.34,9.79
#     12520,0.15,-0.30,9.81
#     12540,4.90,-2.10,13.44      <- acá hay un movimiento brusco
#
# Código de ejemplo para el lado del sensor (Arduino / ESP32):
#
#     Serial.print(millis());   Serial.print(",");
#     Serial.print(ax, 2);      Serial.print(",");
#     Serial.print(ay, 2);      Serial.print(",");
#     Serial.println(az, 2);
#
# Este programa también acepta líneas de solo tres valores ("ax,ay,az"), sin
# la marca de tiempo. En ese caso usa la hora de llegada, con la pérdida de
# precisión que se explicó más arriba.
#
# ==============================================================================

# --- Librerías de la biblioteca estándar de Python ---
import math
import queue
import threading
import time
from collections import deque

# NOTA: las librerías de Bluetooth (pyserial y bleak) NO se importan acá
# arriba a propósito. Se importan adentro de cada clase, solo cuando
# realmente se usan. De esa forma, el programa funciona con el simulador
# aunque no tengas instalada ninguna librería de Bluetooth.


# ==============================================================================
# CONSTANTES DE CONFIGURACIÓN
# ==============================================================================

# Cuánto tiene que apartarse la aceleración de su valor de reposo para que la
# consideremos un movimiento brusco, en m/s^2. Como referencia: la gravedad
# es 9.81 m/s^2, así que 3.5 equivale a un sacudón considerable, bastante más
# fuerte que mover la cabeza para mirar por el espejo retrovisor.
# Si detecta cabezazos que no existen, SUBÍ este número. Si no detecta los
# cabezazos reales, BAJALO.
UMBRAL_MOVIMIENTO_BRUSCO = 3.5

# Ventana de tiempo dentro de la cual contamos los cabezazos, en segundos.
VENTANA_CABEZAZOS_SEG = 20.0

# Cuántos cabezazos dentro de esa ventana disparan la alerta.
MIN_CABEZAZOS_PARA_ALERTA = 2

# Tiempo refractario: después de contar un cabezazo, cuántos segundos
# ignoramos el sensor para no contar el mismo sacudón muchas veces.
TIEMPO_REFRACTARIO_SEG = 0.4

# Qué tan rápido se actualiza el promedio lento que estima la gravedad.
# Es un número entre 0 y 1: cuanto MÁS CHICO, más lento se adapta el promedio
# y mejor distingue un golpe seco. 0.01 significa que hacen falta unas cien
# muestras para que el promedio se acomode, o sea un par de segundos si el
# sensor manda ~50 muestras por segundo.
FACTOR_PROMEDIO_LENTO = 0.01

# Si pasa esta cantidad de segundos sin recibir ni una muestra, consideramos
# que el sensor se desconectó (se apagó, se salió de rango, se quedó sin
# batería) y lo avisamos en pantalla.
TIMEOUT_SENSOR_SEG = 3.0


# ==============================================================================
# REPRESENTACIÓN DE UNA MUESTRA
# ==============================================================================

class Muestra:
    """Una única lectura del acelerómetro, ya interpretada.

    'tiempo_seg' es el momento en que ocurrió la lectura, expresado en
    segundos. Puede venir del propio sensor (lo ideal, porque no sufre el
    retraso del Bluetooth) o, si el sensor no lo manda, de la hora de
    llegada a la computadora."""

    def __init__(self, tiempo_seg, ax, ay, az):
        self.tiempo_seg = tiempo_seg
        self.ax = ax
        self.ay = ay
        self.az = az

    def magnitud(self):
        """Devuelve la magnitud total de la aceleración, combinando los tres
        ejes en un solo número (ver explicación del inicio del archivo)."""
        return math.sqrt(self.ax ** 2 + self.ay ** 2 + self.az ** 2)


def parsear_linea(linea, tiempo_de_llegada_seg):
    """Convierte una línea de texto recibida del sensor en un objeto Muestra.

    Acepta los dos formatos descriptos arriba:
      - "t_ms,ax,ay,az"  (con marca de tiempo del sensor: lo recomendado)
      - "ax,ay,az"       (sin marca de tiempo: usa la hora de llegada)

    Si la línea está incompleta o tiene basura (algo bastante común al
    arrancar una conexión Bluetooth, donde suele llegar media línea suelta),
    devuelve None en vez de romper el programa."""
    partes = linea.strip().split(",")

    try:
        if len(partes) == 4:
            # El sensor mandó su propia marca de tiempo, en milisegundos.
            # La pasamos a segundos dividiendo por mil.
            tiempo_seg = float(partes[0]) / 1000.0
            ax, ay, az = float(partes[1]), float(partes[2]), float(partes[3])
            return Muestra(tiempo_seg, ax, ay, az)

        if len(partes) == 3:
            # Sin marca de tiempo: usamos la hora a la que llegó el dato.
            ax, ay, az = float(partes[0]), float(partes[1]), float(partes[2])
            return Muestra(tiempo_de_llegada_seg, ax, ay, az)

    except ValueError:
        # Alguno de los pedazos no era un número válido: línea corrupta.
        return None

    return None


# ==============================================================================
# DETECTOR DE CABEZAZOS
# ==============================================================================

class DetectorCabezazos:
    """Recibe muestras del acelerómetro y avisa cuando hubo demasiados
    cabezazos en poco tiempo.

    La forma de usarlo es simple: se le pasan las muestras una por una con
    'procesar_muestra', y esa función devuelve True en el momento exacto en
    que se alcanza la condición de alerta (dos o más cabezazos dentro de la
    ventana de 20 segundos)."""

    def __init__(self,
                 umbral=UMBRAL_MOVIMIENTO_BRUSCO,
                 ventana_seg=VENTANA_CABEZAZOS_SEG,
                 min_eventos=MIN_CABEZAZOS_PARA_ALERTA,
                 refractario_seg=TIEMPO_REFRACTARIO_SEG,
                 factor_promedio=FACTOR_PROMEDIO_LENTO):
        self.umbral = umbral
        self.ventana_seg = ventana_seg
        self.min_eventos = min_eventos
        self.refractario_seg = refractario_seg
        self.factor_promedio = factor_promedio

        # Promedio lento que estima el valor de reposo (la gravedad). Empieza
        # en None porque todavía no vimos ninguna muestra: se inicializa con
        # la primera que llegue.
        self.promedio_lento = None

        # Momento del último cabezazo contado, para aplicar el tiempo
        # refractario.
        self.tiempo_ultimo_evento = None

        # Lista de los momentos en que ocurrieron los cabezazos recientes.
        # Usamos un 'deque' (una lista optimizada para agregar y sacar de los
        # extremos) porque constantemente agregamos eventos nuevos al final y
        # descartamos los viejos del principio.
        self.eventos = deque()

        # Última desviación calculada. Sirve solo para mostrarla en pantalla
        # y poder ajustar el umbral viendo números reales.
        self.ultima_desviacion = 0.0

    def procesar_muestra(self, muestra):
        """Procesa una muestra del acelerómetro.

        Devuelve True SOLO en el instante en que se dispara la alerta (es
        decir, cuando se junta la cantidad de cabezazos requerida dentro de
        la ventana). En cualquier otro caso devuelve False."""
        magnitud = muestra.magnitud()

        # --- Primer paso: estimar el valor de reposo (la gravedad) ---
        if self.promedio_lento is None:
            # Es la primera muestra: no tenemos con qué comparar todavía, así
            # que la tomamos como valor de reposo inicial y no detectamos nada.
            self.promedio_lento = magnitud
            self.ultima_desviacion = 0.0
            return False

        # Cuánto se aparta esta muestra del valor de reposo.
        desviacion = abs(magnitud - self.promedio_lento)
        self.ultima_desviacion = desviacion

        # Actualizamos el promedio lento, mezclando un poquito de la muestra
        # nueva con lo que ya teníamos. Al ser 'factor_promedio' muy chico,
        # el promedio se mueve despacio y no se deja arrastrar por los golpes.
        self.promedio_lento = (
            self.factor_promedio * magnitud
            + (1.0 - self.factor_promedio) * self.promedio_lento
        )

        # --- Segundo paso: decidir si esto es un cabezazo ---
        if desviacion < self.umbral:
            # Movimiento normal, no pasa nada.
            return False

        # Estamos por encima del umbral. Antes de contarlo, verificamos el
        # tiempo refractario: si hace muy poquito que contamos un cabezazo,
        # esto es la continuación del mismo sacudón, no uno nuevo.
        if self.tiempo_ultimo_evento is not None:
            if (muestra.tiempo_seg - self.tiempo_ultimo_evento) < self.refractario_seg:
                return False

        # --- Tercer paso: contarlo y ver si hay que alertar ---
        self.tiempo_ultimo_evento = muestra.tiempo_seg
        self.eventos.append(muestra.tiempo_seg)

        # Descartamos los eventos que ya quedaron fuera de la ventana de
        # tiempo: si un cabezazo ocurrió hace más de 20 segundos, ya no
        # cuenta para la alerta.
        limite = muestra.tiempo_seg - self.ventana_seg
        while self.eventos and self.eventos[0] < limite:
            self.eventos.popleft()

        if len(self.eventos) >= self.min_eventos:
            # ¡Alerta! Vaciamos la lista de eventos para no volver a disparar
            # la alarma en la muestra siguiente por los mismos cabezazos: el
            # conteo arranca de cero después de cada alerta.
            self.eventos.clear()
            return True

        return False

    def cantidad_eventos_recientes(self):
        """Cuántos cabezazos hay actualmente acumulados en la ventana."""
        return len(self.eventos)


# ==============================================================================
# LECTORES DE SENSOR (LA "CAPA INTERCAMBIABLE")
# ==============================================================================
#
# Todos los lectores de acá abajo se usan exactamente igual desde afuera:
#
#     lector.iniciar()
#     muestras = lector.leer_muestras()   # se llama seguido, no bloquea
#     lector.detener()
#
# Gracias a eso, el programa principal no necesita saber si los datos vienen
# de un simulador, de Bluetooth Clásico o de BLE: cambiar de uno a otro es
# cambiar una sola línea. Esto es útil porque el hardware todavía no está
# definido: podés desarrollar y probar todo con el simulador, y el día que
# tengas el sensor real solo cambiás qué lector usás.


class LectorBase:
    """Clase base de la que heredan todos los lectores. Define qué funciones
    tiene que tener cualquier lector para que el programa principal lo pueda
    usar sin cambios."""

    def iniciar(self):
        """Abre la conexión con el sensor."""
        raise NotImplementedError

    def leer_muestras(self):
        """Devuelve la lista de muestras nuevas que llegaron desde la última
        vez que se llamó a esta función.

        MUY IMPORTANTE: esta función nunca se queda esperando. Si todavía no
        llegó nada, devuelve una lista vacía y listo. Eso es fundamental
        porque se la llama desde el bucle de video: si se quedara esperando
        datos del Bluetooth, la imagen de la cámara se congelaría."""
        raise NotImplementedError

    def detener(self):
        """Cierra la conexión con el sensor."""
        raise NotImplementedError


class LectorSimulado(LectorBase):
    """Simulador: genera datos falsos de acelerómetro, sin ningún hardware.

    Sirve para dos cosas: probar que toda la lógica de detección funciona
    antes de tener el sensor, y poder mostrar el sistema andando aunque el
    hardware no esté disponible.

    Genera ruido suave todo el tiempo (como una cabeza quieta con pequeños
    movimientos normales) y, cada tanto, un 'cabezazo' artificial. Podés
    forzar un cabezazo cuando quieras llamando a 'forzar_cabezazo()'."""

    def __init__(self, frecuencia_hz=50.0):
        self.frecuencia_hz = frecuencia_hz
        self.tiempo_inicio = None
        self.tiempo_ultima_muestra = None
        self.cabezazo_pendiente = False
        self.muestras_de_cabezazo_restantes = 0
        # Usamos una semilla fija para que el ruido sea siempre parecido y el
        # comportamiento sea reproducible al probar.
        import random
        self._random = random.Random(42)

    def iniciar(self):
        self.tiempo_inicio = time.monotonic()
        self.tiempo_ultima_muestra = self.tiempo_inicio

    def forzar_cabezazo(self):
        """Simula un cabezazo en este instante (para probar a mano)."""
        self.cabezazo_pendiente = True

    def leer_muestras(self):
        ahora = time.monotonic()
        muestras = []

        # Generamos tantas muestras como correspondan al tiempo transcurrido,
        # para imitar un sensor que manda datos a un ritmo constante.
        periodo = 1.0 / self.frecuencia_hz
        while (ahora - self.tiempo_ultima_muestra) >= periodo:
            self.tiempo_ultima_muestra += periodo
            tiempo_muestra = self.tiempo_ultima_muestra - self.tiempo_inicio

            if self.cabezazo_pendiente:
                # Arranca un cabezazo: dura unas pocas muestras.
                self.cabezazo_pendiente = False
                self.muestras_de_cabezazo_restantes = 5

            if self.muestras_de_cabezazo_restantes > 0:
                # Estamos en medio de un cabezazo simulado: metemos una
                # aceleración bien grande para que supere el umbral.
                self.muestras_de_cabezazo_restantes -= 1
                ax = self._random.uniform(4.0, 7.0)
                ay = self._random.uniform(-6.0, -3.0)
                az = 9.81 + self._random.uniform(3.0, 6.0)
            else:
                # Reposo: la gravedad más un ruidito chico.
                ax = self._random.uniform(-0.3, 0.3)
                ay = self._random.uniform(-0.3, 0.3)
                az = 9.81 + self._random.uniform(-0.3, 0.3)

            muestras.append(Muestra(tiempo_muestra, ax, ay, az))

        return muestras

    def detener(self):
        pass


class LectorBluetoothClasico(LectorBase):
    """Lee el acelerómetro por Bluetooth Clásico (SPP), típico de módulos
    HC-05 / HC-06 conectados a un Arduino.

    Con este tipo de Bluetooth, Windows crea un PUERTO COM virtual cuando
    emparejás el dispositivo. Desde el punto de vista del programa, entonces,
    leer el Bluetooth es igual que leer un cable serie: por eso alcanza con
    la librería 'pyserial'.

    Para saber qué puerto COM te tocó: emparejá el módulo desde
    Configuración > Dispositivos > Bluetooth, y después miralo en el
    Administrador de dispositivos, bajo 'Puertos (COM y LPT)'."""

    def __init__(self, puerto="COM5", baudios=115200):
        self.puerto = puerto
        self.baudios = baudios
        self.conexion = None
        # La lectura del puerto se hace en un hilo aparte, y las muestras se
        # van dejando en esta cola para que el programa principal las levante
        # cuando pueda. Así el bucle de video nunca espera al Bluetooth.
        self.cola = queue.Queue()
        self.hilo = None
        self.debe_seguir = False

    def iniciar(self):
        import serial  # pyserial. Se importa acá para que el resto del
                       # programa funcione aunque no esté instalada.

        self.conexion = serial.Serial(self.puerto, self.baudios, timeout=1)
        self.debe_seguir = True
        self.hilo = threading.Thread(target=self._bucle_de_lectura, daemon=True)
        self.hilo.start()

    def _bucle_de_lectura(self):
        """Corre en un hilo aparte: lee líneas del puerto sin parar y las va
        dejando en la cola."""
        while self.debe_seguir:
            try:
                linea_bytes = self.conexion.readline()
                if not linea_bytes:
                    continue
                # Los datos llegan como bytes; hay que pasarlos a texto.
                # 'errors="ignore"' descarta cualquier byte corrupto en vez
                # de romper, algo habitual en conexiones inalámbricas.
                linea = linea_bytes.decode("utf-8", errors="ignore")
                muestra = parsear_linea(linea, time.monotonic())
                if muestra is not None:
                    self.cola.put(muestra)
            except Exception:
                # Si el sensor se desconecta a mitad de camino, cortamos el
                # hilo en silencio. El programa principal se va a dar cuenta
                # porque dejan de llegar muestras (ver TIMEOUT_SENSOR_SEG).
                break

    def leer_muestras(self):
        muestras = []
        # Vaciamos la cola de golpe, sin esperar: agarramos todo lo que haya
        # llegado y salimos enseguida.
        while True:
            try:
                muestras.append(self.cola.get_nowait())
            except queue.Empty:
                break
        return muestras

    def detener(self):
        self.debe_seguir = False
        if self.conexion is not None:
            try:
                self.conexion.close()
            except Exception:
                pass


class LectorBLE(LectorBase):
    """Lee el acelerómetro por BLE (Bluetooth Low Energy), típico de un ESP32
    o de sensores más modernos.

    BLE funciona distinto al Bluetooth Clásico: no hay puerto COM. En su
    lugar, el dispositivo publica "características" identificadas por un
    código largo llamado UUID, y nosotros nos suscribimos a la que manda los
    datos para que nos avise cada vez que hay algo nuevo.

    Necesita la librería 'bleak', y que sepas el nombre (o la dirección) del
    dispositivo y el UUID de la característica: esos dos datos los define el
    firmware del sensor."""

    def __init__(self, nombre_dispositivo, uuid_caracteristica):
        self.nombre_dispositivo = nombre_dispositivo
        self.uuid_caracteristica = uuid_caracteristica
        self.cola = queue.Queue()
        self.hilo = None
        self.debe_seguir = False
        # Buffer donde vamos pegando los pedazos de texto que llegan, porque
        # BLE puede partir una línea en varios envíos.
        self._buffer_texto = ""

    def iniciar(self):
        self.debe_seguir = True
        self.hilo = threading.Thread(target=self._bucle_asyncio, daemon=True)
        self.hilo.start()

    def _bucle_asyncio(self):
        """La librería bleak trabaja de forma asincrónica, un estilo de
        programación distinto al del resto de este proyecto. Para no
        complicar el programa principal, encapsulamos todo eso acá adentro,
        en su propio hilo."""
        import asyncio

        try:
            from bleak import BleakClient, BleakScanner
        except ImportError:
            # La librería no está instalada. Avisamos con un mensaje claro en
            # vez de dejar que el hilo muera con un error incomprensible.
            print("ERROR: el modo BLE necesita la libreria 'bleak'.")
            print("Instalala con:  pip install bleak")
            return

        async def principal():
            dispositivo = await BleakScanner.find_device_by_name(
                self.nombre_dispositivo, timeout=10.0
            )
            if dispositivo is None:
                print(f"ERROR: no se encontró el dispositivo BLE "
                      f"'{self.nombre_dispositivo}'.")
                return

            def al_recibir_datos(_caracteristica, datos):
                # Esta función la llama bleak cada vez que el sensor manda
                # algo. Vamos acumulando el texto y, cada vez que aparece un
                # salto de línea, procesamos la línea completa.
                self._buffer_texto += datos.decode("utf-8", errors="ignore")
                while "\n" in self._buffer_texto:
                    linea, self._buffer_texto = self._buffer_texto.split("\n", 1)
                    muestra = parsear_linea(linea, time.monotonic())
                    if muestra is not None:
                        self.cola.put(muestra)

            async with BleakClient(dispositivo) as cliente:
                await cliente.start_notify(self.uuid_caracteristica, al_recibir_datos)
                while self.debe_seguir:
                    await asyncio.sleep(0.1)

        try:
            asyncio.run(principal())
        except Exception as error:
            print(f"ERROR en la conexión BLE: {error}")

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

def crear_lector(modo, puerto_com="COM5", nombre_ble="AcelerometroCasco",
                 uuid_ble="0000ffe1-0000-1000-8000-00805f9b34fb"):
    """Devuelve el lector que corresponda según el modo elegido.

    Modos disponibles:
      "simulador" : datos falsos, no necesita hardware. Ideal para probar.
      "clasico"   : Bluetooth Clásico (HC-05/HC-06) a través de un puerto COM.
      "ble"       : Bluetooth Low Energy (ESP32 y similares).

    El UUID que figura por defecto es el que usan los módulos BLE más comunes
    y baratos (los HM-10 y clones); si tu sensor usa otro, hay que cambiarlo."""
    if modo == "simulador":
        return LectorSimulado()
    if modo == "clasico":
        return LectorBluetoothClasico(puerto=puerto_com)
    if modo == "ble":
        return LectorBLE(nombre_dispositivo=nombre_ble, uuid_caracteristica=uuid_ble)

    raise ValueError(
        f"Modo de sensor desconocido: '{modo}'. "
        f"Las opciones válidas son 'simulador', 'clasico' o 'ble'."
    )
