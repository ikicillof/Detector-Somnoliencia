# Detector de somnolencia en la Raspberry Pi

Guía para instalar, correr y mantener el detector en la Raspberry Pi del
proyecto. Está pensada para cualquiera del grupo, aunque no hayas tocado
nunca una Pi ni Linux.

- **Placa:** Raspberry Pi 3 o 4, con Raspberry Pi OS Trixie de **64 bits**.
- **Usuario:** `detector-somnoliencia`
- **IP fija (Tailscale):** `100.66.6.49`
- **Alarma:** buzzer activo en el **GPIO21** (pin físico 40), a través de un
  transistor (ver [Conexión del buzzer](#5-conexión-del-buzzer)).

---

## 1. Conectarse a la Pi

La Pi no necesita monitor ni teclado: se maneja a distancia. Hay tres formas,
según desde dónde te conectes.

### Desde tu computadora: Tailscale + SSH

Tailscale arma una red privada entre tus dispositivos y la Pi, así podés
llegar a ella desde cualquier lado (no hace falta estar en la misma WiFi).

1. Instalá Tailscale desde <https://tailscale.com/download> e iniciá sesión
   con la cuenta del grupo (pedile acceso a quien la administra).
2. Abrí una terminal (en Windows: `cmd` o PowerShell) y escribí:

   ```
   ssh detector-somnoliencia@100.66.6.49
   ```

3. La primera vez te pregunta si confiás en la máquina: escribí `yes`.
   Después te pide la contraseña del usuario de la Pi.

### Desde el celular: Tailscale + Termius

1. Instalá **Tailscale** (Play Store / App Store) e iniciá sesión con la
   cuenta del grupo. Dejalo conectado.
2. Instalá **Termius** y creá un *Host* nuevo con:
   - Address: `100.66.6.49`
   - Username: `detector-somnoliencia`
   - Password: la contraseña de la Pi
3. Tocá el host para conectarte. Te queda una terminal igual que la de la
   computadora.

### Desde una PC sin permisos de administrador (por ejemplo, las notebooks de la escuela)

Ahí no se puede instalar Tailscale. Usá **Raspberry Pi Connect**, que
funciona desde el navegador:

1. Entrá a <https://connect.raspberrypi.com> e iniciá sesión con la cuenta
   Raspberry Pi ID del grupo.
2. Elegí la Pi y después:
   - **Remote shell:** una terminal, igual que por SSH.
   - **Screen sharing:** ves el escritorio de la Pi (sirve para correr el
     detector *con ventana*, ver más abajo).

---

## 2. Instalación (una sola vez)

Conectado a la Pi por cualquiera de los métodos de arriba:

```bash
# 1. Bajar el proyecto (si todavía no está en la Pi)
git clone https://github.com/ikicillof/Detector-Somnoliencia.git
cd Detector-Somnoliencia

# 2. Instalar todo
./instalar_pi.sh
```

El script hace todo solo y va mostrando cada paso:

1. Avisa si queda menos de 1 GB libre en la tarjeta SD.
2. Instala `uv` (un instalador de Python mucho más liviano que `pip`).
3. Crea el entorno virtual `venv` con **Python 3.11**.
4. Usa `~/tmp` como carpeta temporal (no `/tmp`, que está en la RAM).
5. Instala las dependencias de `requirements-pi.txt`.
6. Instala **mediapipe 0.10.18 sin sus dependencias** (para no bajar
   `jax`/`scipy`, que pesan cientos de MB y no se usan).
7. Borra la carpeta temporal y muestra las versiones instaladas.

Tarda varios minutos. Si termina con `Instalación terminada`, está listo.

> El script se puede volver a correr sin problema (por ejemplo, después de
> actualizar el proyecto): reutiliza lo que ya está instalado.

---

## 3. Correr el detector

```bash
cd ~/Detector-Somnoliencia
source venv/bin/activate
python detector_somnoliencia.py --sin-ventana
```

- `source venv/bin/activate` "entra" al entorno virtual. Hay que hacerlo
  **cada vez** que abrís una terminal nueva. Te das cuenta de que está
  activo porque el renglón empieza con `(venv)`.
- Al arrancar se **calibra solo**: mirá al frente, con los ojos abiertos
  normalmente, y quedate quieto unos segundos. Se calibra la posición de la
  cabeza y también el umbral de ojos cerrados de esa persona (así funciona
  bien con ojos achinados). Si te movés, reintenta solo hasta que salga bien. Cuando
  termina, el buzzer hace un pitido muy cortito.
- En la consola van apareciendo los eventos con la hora: calibración,
  rostro detectado/perdido y las alertas (`ALERTA: OJOS CERRADOS`,
  `ALERTA: CABEZA CAIDA`, `ALERTA: cabeceo brusco detectado`).
- Cada 5 segundos imprime los FPS y los valores en vivo (EAR, desviación y
  velocidad de la cabeza, con su mínimo y su umbral), para ajustar la
  sensibilidad.
- **Para recalibrar:** escribí `c` y Enter.
- **Para salir:** `Ctrl+C` (o `q` y Enter). Libera la cámara y apaga el
  buzzer.

### Opciones

| Opción | Qué hace |
|---|---|
| `--sin-ventana` | No abre ventana de video. Se activa sola si no hay pantalla (por SSH). |
| `--ancho 320 --alto 240` | Resolución pedida a la cámara. Por defecto 640x480 en la Pi. Más chica = más rápido. |
| `--saltar 2` | Procesa 1 de cada 2 cuadros. Usar solo si la Pi no da abasto. |

### Verlo con ventana (para ajustar)

Para ver la imagen con los puntos de los ojos y los valores en pantalla,
corré el detector **sin** `--sin-ventana` desde una sesión que tenga
escritorio:

- con un monitor conectado a la Pi (abriendo una terminal en el escritorio), o
- con **Screen sharing** de Raspberry Pi Connect.

Desde SSH también podés mandar la ventana al monitor de la Pi (si hay una
sesión de escritorio abierta) con `DISPLAY=:0 python detector_somnoliencia.py`.
Ojo: con ventana la Pi trabaja más, así que los FPS van a ser menores que
en el uso real.

---

## 4. Actualizar la Pi cuando hay cambios en GitHub

```bash
cd ~/Detector-Somnoliencia
git pull
./instalar_pi.sh      # solo si cambiaron requirements-pi.txt o instalar_pi.sh
```

---

## 5. Conexión del buzzer

Buzzer **activo** (suena solo con recibir tensión) manejado por un
transistor NPN, porque el GPIO no da corriente suficiente:

| Pin de la Pi | Va a |
|---|---|
| Pin 2 (5 V) | + del buzzer |
| — | − del buzzer → colector del transistor |
| Pin 40 (GPIO21) | resistencia de 1 kΩ → base del transistor |
| Pin 14 (GND) | emisor del transistor |

Si cambiás de pin, actualizá la constante `BUZZER_GPIO` al principio de
`detector_somnoliencia.py` (usa la numeración **GPIO/BCM**, no el número de
pin físico).

Patrones de la alarma:

- **Peligro** (ojos cerrados o cabeza caída): pitido largo que se repite
  mientras dure.
- **Aviso** (cabeceo brusco): dos pitidos cortos.
- **Calibración lista:** un pitido muy cortito cuando termina de calibrar
  (ya podés dejar de mirar fijo al frente). No es una alerta.
- **Rostro perdido:** un "tic" cortísimo cada ~1 segundo mientras la cámara
  no ve la cara (empieza tras 1 segundo sin verla y se corta apenas la
  vuelve a ver). Si suena todo el tiempo, revisá la posición de la cámara
  o la luz.

---

## 6. Problemas conocidos y solución

### `FATAL ERROR: This binary was compiled with aes enabled...`

Es **mediapipe 1.0.x**, que viene compilada con instrucciones AES de ARMv8
que la Pi 3/4 no tiene. Pasa si se instaló mediapipe con el Python del
sistema (3.13) o con `pip install mediapipe` a secas.

**Solución:** usar el entorno que arma `instalar_pi.sh` (Python 3.11 +
mediapipe 0.10.18). Si el `venv` quedó mal, borralo y reinstalá:

```bash
rm -rf venv
./instalar_pi.sh
```

### `No space left on device` durante la instalación (`/tmp` lleno)

En la Pi, `/tmp` está en la RAM (~450 MB) y se llena al descomprimir
paquetes grandes como `jax`/`jaxlib`.

**Solución:** `instalar_pi.sh` ya usa `~/tmp` como carpeta temporal y no
instala `jax`/`scipy`. Si instalás algo a mano, hacé antes
`mkdir -p ~/tmp && export TMPDIR=~/tmp`, y **nunca** instales mediapipe sin
`--no-deps`.

### La tarjeta SD está llena

Una SD de 8 GB queda casi llena solo con el sistema.

- Ver cuánto queda: `df -h /`
- Liberar espacio:
  ```bash
  sudo apt clean
  sudo apt autoremove
  rm -rf ~/.cache/pip ~/.cache/uv
  ```
- **Solución de fondo:** pasar a una tarjeta de **32 GB**.

### `qt.qpa.xcb: could not connect to display` (por SSH)

Por SSH no hay pantalla donde abrir la ventana de video.

**Solución:** correr con `--sin-ventana` (el programa ya lo activa solo si
no encuentra pantalla). Para ver la imagen, ver
[Verlo con ventana](#verlo-con-ventana-para-ajustar).

### Mucho delay (la imagen o la alarma llegan tarde)

Pasaba porque la Pi procesa más lento de lo que la cámara entrega, y los
cuadros se acumulaban. Ya está resuelto: la cámara se lee en un hilo aparte
y siempre se procesa el cuadro más reciente.

Si igual va lento, mirá la línea de FPS que sale cada 5 segundos y probá:

1. Bajar la resolución: `--ancho 320 --alto 240`.
2. Procesar menos cuadros: `--saltar 2`.
3. Revisar la advertencia de `vcgencmd get_throttled` al arrancar: si no da
   `0x0`, la fuente no alcanza o la Pi está muy caliente, y se frena sola.
   Usá la fuente oficial y un disipador/ventilador.

### El buzzer no suena

- Al arrancar el programa tiene que decir `Alarma: usando buzzer GPIO21.`
  Si dice `ADVERTENCIA: no se pudo usar el buzzer por GPIO`, el programa
  sigue pero con alarma solo en la consola. Revisá:
  - que la instalación haya terminado bien (`gpiozero` y `lgpio`);
  - que tu usuario esté en el grupo `gpio`: `groups` (si no está:
    `sudo usermod -aG gpio $USER` y volvé a entrar).
- Revisá el cableado (sección 5) y que el buzzer sea **activo**.

### `ERROR: no se pudo acceder a la webcam`

- Revisá que la cámara esté conectada: `ls /dev/video*`.
- Que no haya otra copia del detector corriendo: `pkill -f detector_somnoliencia`.
- Si hay varias cámaras, cambiá `CAMARA_INDICE` en el código.
