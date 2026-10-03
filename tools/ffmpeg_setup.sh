#!/usr/bin/env bash
# Menyiapkan ffmpeg di sandbox yang tidak punya akses apt dan jaringannya
# dibatasi allowlist (hanya pypi.org + github.com yang bisa dijangkau).
#
# imageio-ffmpeg membawa binary ffmpeg statis (build 7.0.2) lewat PyPI,
# lengkap dengan libx264, aac, libmp3lame, zoompan, xfade, dan drawtext.
#
# Pakai:  source tools/ffmpeg_setup.sh   ->  variabel $FFMPEG siap dipakai
set -euo pipefail

VENV="${VENV:-/tmp/vid}"

if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
fi

if ! "$VENV/bin/python" -c "import imageio_ffmpeg" >/dev/null 2>&1; then
  "$VENV/bin/pip" install --quiet imageio-ffmpeg
fi

FFMPEG="$("$VENV/bin/python" -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')"
export FFMPEG

mkdir -p "$HOME/.local/bin"
ln -sf "$FFMPEG" "$HOME/.local/bin/ffmpeg"

echo "FFMPEG=$FFMPEG"
"$FFMPEG" -version | head -1
