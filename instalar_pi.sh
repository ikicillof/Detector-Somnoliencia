#!/usr/bin/env bash
# ==============================================================================
# Instalación del detector de somnolencia en la Raspberry Pi (3 o 4).
#
# Uso (desde la carpeta del proyecto):
#     ./instalar_pi.sh
#
# Qué hace y por qué (más detalle en INSTALACION_PI.md):
#   - Crea un entorno virtual con Python 3.11 usando 'uv' (Raspberry Pi OS
#     Trixie trae Python 3.13, y la única mediapipe para 3.13 no anda en la
#     Pi 3/4: le faltan las instrucciones AES).
#   - Instala requirements-pi.txt y después mediapipe==0.10.18 SIN sus
#     dependencias, para no bajar jax/jaxlib/scipy (cientos de MB que no se
#     usan y que llenan /tmp y la SD).
#   - Usa $HOME/tmp como carpeta temporal, porque /tmp está en RAM (~450 MB).
#
# Se puede correr más de una vez: si el entorno ya existe con Python 3.11,
# lo reutiliza y solo instala lo que falte.
# ==============================================================================

set -euo pipefail

VERSION_MEDIAPIPE="0.10.18"
VERSION_PYTHON="3.11"
CARPETA_TMP="$HOME/tmp"

paso() { echo; echo "==> $*"; }
ok()   { echo "    OK: $*"; }
aviso(){ echo "    ADVERTENCIA: $*"; }
error(){ echo; echo "ERROR: $*" >&2; exit 1; }

# Trabajamos siempre desde la carpeta del proyecto (donde está este script),
# aunque lo llamen desde otro lado.
cd "$(dirname "$(readlink -f "$0")")"
[ -f requirements-pi.txt ] || error "no encuentro requirements-pi.txt en $(pwd)."

# Pase lo que pase (incluso si algo falla a mitad de camino), al salir se
# borra la carpeta temporal para no dejar la SD llena de basura.
limpiar() { rm -rf "$CARPETA_TMP"; }
trap limpiar EXIT

echo "Instalación del detector de somnolencia para Raspberry Pi"
echo "Carpeta del proyecto: $(pwd)"

# ------------------------------------------------------------------------------
paso "Revisando el espacio libre en la tarjeta SD"
# ------------------------------------------------------------------------------
LIBRE_KB=$(df -Pk / | awk 'NR==2 {print $4}')
LIBRE_MB=$((LIBRE_KB / 1024))
if [ "$LIBRE_MB" -lt 1024 ]; then
    aviso "quedan solo ${LIBRE_MB} MB libres en / (se recomienda al menos 1 GB)."
    echo "    La instalación puede fallar a mitad de camino por falta de espacio."
    echo "    Para liberar lugar: sudo apt clean ; sudo apt autoremove ;"
    echo "    borrar descargas viejas ; rm -rf ~/.cache/pip ~/.cache/uv"
    echo "    A largo plazo: usar una tarjeta de 32 GB."
    if [ -t 0 ]; then
        read -r -p "    ¿Seguir igual? [s/N] " respuesta
        case "$respuesta" in
            s|S|si|SI|sí|Sí) ;;
            *) echo "Instalación cancelada."; exit 1 ;;
        esac
    fi
else
    ok "${LIBRE_MB} MB libres."
fi

ARQUITECTURA=$(uname -m)
if [ "$ARQUITECTURA" != "aarch64" ]; then
    aviso "la arquitectura es '$ARQUITECTURA', no 'aarch64'."
    echo "    mediapipe solo tiene versión para Raspberry Pi OS de 64 bits."
fi

# ------------------------------------------------------------------------------
paso "1/7 Instalando uv (si hace falta)"
# ------------------------------------------------------------------------------
if [ -f "$HOME/.local/bin/env" ]; then
    # shellcheck disable=SC1091
    source "$HOME/.local/bin/env"
fi
if command -v uv >/dev/null 2>&1; then
    ok "uv ya estaba instalado ($(uv --version))."
else
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # shellcheck disable=SC1091
    source "$HOME/.local/bin/env"
    command -v uv >/dev/null 2>&1 || error "no se pudo instalar uv."
    ok "uv instalado ($(uv --version))."
fi

# ------------------------------------------------------------------------------
paso "2/7 Creando el entorno virtual 'venv' con Python $VERSION_PYTHON"
# ------------------------------------------------------------------------------
PYTHON_VENV="venv/bin/python"
if [ -x "$PYTHON_VENV" ] && \
   "$PYTHON_VENV" -c "import sys; sys.exit(sys.version_info[:2] != (3, 11))" 2>/dev/null; then
    ok "ya existe un venv con Python $VERSION_PYTHON; se reutiliza."
else
    if [ -e venv ]; then
        aviso "hay un venv viejo que no es Python $VERSION_PYTHON: se borra y se crea de nuevo."
        rm -rf venv
    fi
    # Sin --system-site-packages: el venv no ve los paquetes del sistema
    # (que son para Python 3.13 y no sirven acá).
    uv venv --python "$VERSION_PYTHON" venv
    ok "venv creado ($("$PYTHON_VENV" --version))."
fi

# Restos de la instalación vieja: un 'winsound.py' falso que se había creado
# a mano dentro del venv. El programa ya no usa winsound.
WINSOUND_FALSO="venv/lib/python${VERSION_PYTHON}/site-packages/winsound.py"
if [ -f "$WINSOUND_FALSO" ]; then
    rm -f "$WINSOUND_FALSO"
    ok "borrado el winsound.py falso que había quedado en el venv."
fi

# ------------------------------------------------------------------------------
paso "3/7 Usando $CARPETA_TMP como carpeta temporal (no /tmp, que está en RAM)"
# ------------------------------------------------------------------------------
mkdir -p "$CARPETA_TMP"
export TMPDIR="$CARPETA_TMP"
ok "TMPDIR=$TMPDIR"

# ------------------------------------------------------------------------------
paso "4/7 Instalando las dependencias de requirements-pi.txt"
# ------------------------------------------------------------------------------
echo "    (puede tardar varios minutos)"
uv pip install --python "$PYTHON_VENV" --no-cache -r requirements-pi.txt
ok "dependencias instaladas."

# ------------------------------------------------------------------------------
paso "5/7 Instalando mediapipe==$VERSION_MEDIAPIPE SIN dependencias (sin jax/scipy)"
# ------------------------------------------------------------------------------
uv pip install --python "$PYTHON_VENV" --no-cache --no-deps "mediapipe==$VERSION_MEDIAPIPE"
ok "mediapipe instalada."

# ------------------------------------------------------------------------------
paso "6/7 Borrando la carpeta temporal $CARPETA_TMP"
# ------------------------------------------------------------------------------
limpiar
ok "listo."

# ------------------------------------------------------------------------------
paso "7/7 Verificando la instalación"
# ------------------------------------------------------------------------------
"$PYTHON_VENV" -c "import cv2, mediapipe" \
    || error "no se pudo importar cv2 o mediapipe. Revisá los mensajes de arriba."
"$PYTHON_VENV" - <<'PYTHON'
import sys, cv2, mediapipe, numpy
print(f"    Python    {sys.version.split()[0]}")
print(f"    OpenCV    {cv2.__version__}")
print(f"    mediapipe {mediapipe.__version__}")
print(f"    numpy     {numpy.__version__}")
try:
    import gpiozero, lgpio  # noqa: F401
    from importlib.metadata import version
    print(f"    gpiozero  {version('gpiozero')} (con lgpio {version('lgpio')})")
except Exception as error:
    print(f"    ADVERTENCIA: gpiozero/lgpio no cargan ({error}); "
          "la alarma va a ser solo visual.")
PYTHON

echo
echo "Instalación terminada. Para correr el detector:"
echo "    source venv/bin/activate"
echo "    python detector_somnoliencia.py --sin-ventana"
