# Archer — Stickman Psychology Video Factory

Pipeline produksi video faceless bergaya **animasi stickman** dengan voice-over pria (Bahasa Indonesia), tanpa teks/subtitle on-screen.

Niche: **Psikologi Gelap / Dark Psychology** — niche kompetisi rendah dengan CPM tinggi dan potensi viral besar di Shorts.

## Struktur

```
assets/short/      gambar scene + voice-over video short (9:16)
assets/long/       gambar scene + voice-over video long (16:9)
content/           naskah (JSON) + metadata judul/deskripsi/tag
build/             build_video.py (ffmpeg: ken burns, crossfade, mixing audio)
output/            hasil render .mp4 + thumbnail
```

## Render

```bash
pip install --break-system-packages imageio-ffmpeg
python3 build/build_video.py short
python3 build/build_video.py long
```

Setiap scene: gambar → ken burns (zoom in/out bergantian) → transisi crossfade/smoothleft 0.5–0.6 s, voice-over ditempatkan presisi di scene-nya, plus bed ambient halus (-26 dB) dan loudness normalisasi ke -14 LUFS (standar YouTube).

## Rilis

Video short dipublikasikan di [GitHub Releases](../../releases).
