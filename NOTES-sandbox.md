# Catatan lingkungan sandbox (3 Oktober 2026)

Ringkasan temuan teknis saat menyiapkan pipeline video. Disimpan supaya tidak perlu
diuji ulang di sesi berikutnya.

## 1. Jaringan dibatasi allowlist

Hanya beberapa host yang bisa dijangkau dari dalam sandbox:

| Host | Status |
|---|---|
| `github.com` | ✅ 200 |
| `api.github.com` | ✅ 200 |
| `codeload.github.com` | ✅ 301 (artinya `git clone` jalan) |
| `pypi.org`, `files.pythonhosted.org` | ✅ 200 |
| `raw.githubusercontent.com` | ❌ TLS ditolak (curl exit 35) |
| `youtube.com`, `www.youtube.com` | ❌ TLS ditolak |
| `pexels.com`, `pixabay.com`, `mixkit.co` | ❌ TLS ditolak |
| `archive.org`, `upload.wikimedia.org` | ❌ TLS ditolak |
| `commondatastorage.googleapis.com`, `google.com` | ❌ TLS ditolak |

**Konsekuensi:** `yt-dlp` / `curl` **tidak bisa** mengunduh video dari YouTube, TikTok,
Instagram, maupun situs stock footage. Satu-satunya cara memasukkan video asli ke sandbox:

1. **`git clone` repo publik GitHub** yang memang menyimpan file video, atau
2. **file di-upload/di-attach langsung oleh pengguna** ke dalam repo ini.

## 2. ffmpeg

Tidak ada `apt` (bukan root) dan paket `ffmpeg` tidak tersedia. Solusinya memakai binary
statis dari PyPI:

```bash
source tools/ffmpeg_setup.sh     # -> $FFMPEG, ffmpeg 7.0.2
```

Encoder & filter yang tersedia dan sudah diverifikasi: `libx264`, `aac`, `libmp3lame`,
`zoompan`, `xfade`, `drawtext`, `subtitles`/`ass` (libass), `gblur`, `overlay`, `concat`.
Catatan: `ffprobe` **tidak** ikut dalam paket ini — `tools/make_short.py` membaca durasi
dari keluaran stderr `ffmpeg -i`.

## 3. Font

Tidak ada fontconfig; sistem hanya punya DejaVu. Roboto Black/Bold diambil dari paket PyPI
`font-roboto` (Apache-2.0) dan disimpan di
`projects/timnas-final-2026/assets/fonts/`.

## 4. Footage asli yang terbukti bisa diunduh

`git clone --depth 1 https://github.com/intel-iot-devkit/sample-videos.git`
— 17 file MP4, lisensi **CC BY 4.0**, jadi legal untuk diremix dan dimonetisasi.

Namun isinya klip uji computer-vision (960x540, 59.94 fps): buah/sayur di konveyor,
lorong toko, deteksi orang, mur-baut. Secara teknis "video asli", tapi bukan konten viral
dan kualitasnya di bawah standar Shorts.
