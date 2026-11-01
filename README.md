# Rahasia di Balik Tembok Besar China — animasi 3D vertikal

Video pendek (**2:47, 1080×1920, 90 fps**) bergaya *history short* Indonesia: narasi bahasa
Indonesia, dunia 3D low-poly yang **dirender penuh** (bukan slideshow gambar), kamera
sinematik yang bergerak, karakter berjalan, kavaleri, asap api beacon, kabut, siang →
senja → malam, caption per-kata, dan skor musik prosedural.

Referensi gaya: video TikTok/Shorts berbahasa Indonesia tentang Tembok Besar China
(render 3D bergaya *game-mobile*, narasi dokumenter, caption kuning). Semua aset di sini
dibuat dari nol secara prosedural — tidak ada footage atau gambar pihak ketiga.

## Hasil

| Berkas | Keterangan |
| --- | --- |
| `video/rahasia_tembok_besar_china_1080x1920.mp4` | video final **2:47.24 · 15.052 frame** — 1080×1920, **90 fps**, H.264 High 3,05 Mbps + AAC 192 kbps 48 kHz, 68,1 MB, caption terbakar, −14,4 LUFS |
| `video/poster.jpg` | poster/thumbnail 1080×1920 |
| `video/captions.subs.ass` | caption per-kata (libass) untuk versi lain |
| `ref/qc90.jpg` | QC frame terpilih dari master 90 fps |

## Cara kerja pipeline

```
src/story.json     naskah 10 adegan (narasi + kicker)
        │
        ▼
assets/audio/vo_XX.mp3  voice-over per adegan (TTS, suara pria, id-ID)
        │
        ▼
src/film.py        timeline (durasi nyata VO) · skor prosedural · caption ASS per-kata · mix + ducking
        │
        ▼
src/shots.py       18 shot 3D: kamera, aktor, palette (golden/day/dusk/night/blue/macro)
        │
        ▼
src/world.py       dunia prosedural: ridge terrain, tembok bergerigi + menara, hutan,
                   prajurit, kuda, kafilah, asap, api, kabut, reruntuhan
        │
        ▼
src/engine3d.py    renderer software: proyeksi perspektif, backface cull, cull sub-piksel,
                   near-clip, painter's algorithm, Lambert + hemisphere, fog ketinggian,
                   emissive, transparansi, langit prosedural + bintang
        │
        ▼
src/shadow.py      bayangan matahari (shadow map orthografis) dipanggang sekali per shot
        │
        ▼
src/gfx.py         finishing OpenCV: resolve supersample INTER_AREA, bloom 2 skala +
                   halation, LUT filmic (rolloff highlight, mid-grey tetap), clarity,
                   micro-sharpen, vignette, dither + grain
        │
        ▼
src/render.py      pipe raw RGB → ffmpeg (libx264, `-tune animation`) + burn caption (libass)
        │
        ▼
src/make.py        render paralel per rentang frame → concat → mux AAC + loudnorm
```

### Modul yang dipakai (semua versi terbaru)

| Modul | Versi | Peran |
| --- | --- | --- |
| `numpy` | 2.4.6 | seluruh matematika vertex, normal, pencahayaan, fog, mix audio |
| `opencv-python-headless` | 5.0.0.93 | resolve supersample, bloom, blur, LUT, vignette, dither (SIMD) |
| `pillow` | 13.0.0 | rasterisasi poligon (C-speed), sky compositing, ukur lebar caption |
| `scipy` | 1.17.1 | `cKDTree` untuk oklusi kontak antar-face |
| `fonttools` | 4.68.0 | verifikasi font Anton |
| `imageio-ffmpeg` | 0.7.0 | FFmpeg 7.0.2 portabel (libass + libx264) |

Tidak ada GPU, WebGL, Blender, atau pustaka 3D pihak ketiga: renderer 3D-nya ditulis
sendiri dan berjalan **~3.1 frame/detik per proses** di 1080×1920 dengan supersampling
1.6×. `make.py` membagi frame ke semua core, jadi master penuh (15.052 frame) selesai
**±45 menit** di 2 core (render paralel 39,9 menit + burn caption 5 menit + mux).

## Menjalankan ulang

```bash
python3 -m venv .venv && .venv/bin/pip install numpy opencv-python-headless pillow scipy imageio-ffmpeg fonttools
cd src
../.venv/bin/python make.py                  # master penuh: paralel + concat + mux (~30 menit di 2 core)
../.venv/bin/python make.py --test 20        # bukti 20 detik
../.venv/bin/python render.py                # satu proses saja (lebih lambat)
../.venv/bin/python render.py --test 10      # 10 detik pertama
../.venv/bin/python render.py --audio-only   # hanya timeline + musik + caption
../.venv/bin/python render.py --f0 3000 --f1 4500 --out ../out/part.mp4 --no-final  # satu rentang frame
../.venv/bin/python test_shots.py            # contact sheet look-dev 18 shot
```

Flag tuning: `--ss` (supersampling, default 1.5), `--crf` (default 17), `--preset`,
`--ao` (oklusi kontak, default 0 = mati), `--fps` (default 60).

## Struktur film (10 adegan = 18 shot)

| Adegan | Shot | Visual |
| --- | --- | --- |
| 1 Pembuka | `hero` | dolly rendah sepanjang tembok di golden hour, burung, kabut |
| 2 Bukan satu tembok | `many` | orbit udara atas diorama tiga generasi tembok + menara beacon |
| 3 Mitos dari bulan | `orbit`, `camo` | aerial ekstrem berkabut → tembok menyatu dengan batuan |
| 4 Tentara & petani | `workers`, `chisel` | pekerja menggotong batu di atas tembok, pahat batu close-up |
| 5 Yang tidak pulang | `memorial` | crane naik di senja berkabut, gundukan makam + dupa menyala |
| 6 Perekat ketan | `rice`, `cutaway` | makro bata + butir ketan → potongan melintang struktur tembok |
| 7 Kota memanjang | `gate`, `barracks` | gerbang pajak dilewati kafilah, barak & barisan prajurit |
| 8 Asap dan api | `beacon_day`, `beacon_night` | rantai asap siang → api beacon malam berbintang |
| 9 Pernah ditembus | `mongol`, `betray` | kavaleri Mongol memacu kuda, gerbang dibuka malam hari |
| 10 Yang hilang | `ruin`, `villager`, `final` | tembok runtuh ditelan hutan, warga menumpuk bata, hero shot malam |

## Catatan produksi

* **90 fps**: timeline dihitung per frame, animasi berbasis waktu, jadi kamera dan aktor
  tetap sinkron di semua fps. `make.py` memotong tepat di batas frame sehingga concat tidak
  bergeser satu frame pun.
* **Gerak mulus (anti-patah-patah)**: semua indeks path yang dulu dibulatkan `int()`
  sekarang fraksional (`sample_path`/`path_frame_f`), sehingga dolly kamera di atas tembok
  bergerak kontinu, bukan melompat tiap 4,4 unit. Api beacon tidak lagi me-reseed RNG 8×/
  detik (`fire_glow` sekarang berosilasi kontinu), dan tiap frame memakai *shutter trail*
  0,35 dengan frame sebelumnya — gerak cepat terbaca sebagai blur, bukan deretan still.
* **Bayangan matahari asli**: `shadow.py` memanggang peta bayangan orthografis sekali per
  shot (matahari konstan dalam satu shot) lalu mengalikannya **hanya pada komponen cahaya
  langsung** — jadi area terbayang tetap dapat cahaya langit biru, bukan sekadar menggelap.
* **Specular + rim light**: respons permukaan tipis di shader per-face supaya batu dan atap
  tidak terlihat seperti karton.
* **Anti-aliasing nyata**: frame dirender 1.5× (1620×2880) lalu di-resolve dengan
  `INTER_AREA`, bukan di-blur setelahnya. Face yang lebih kecil dari satu piksel dibuang
  sebelum rasterisasi.
* Durasi shot mengikuti durasi voice-over asli, jadi setiap potongan jatuh di beat narasi.
* Dissolve 0.55 s antar shot; transisi siang→malam memakai dissolve yang sama.
* Caption: kata per kata (bukan per cue 3 kata) — kata terbaru kuning `#FFD93B` dengan
  pop-scale, kata sebelumnya putih; sisa kata di-placeholder transparan agar baris tidak
  bergeser; ukuran font menyusut otomatis (diukur dengan Pillow) supaya selalu masuk frame.
* Musik: progresi Am–F–C–G, drone + taiko + angin + shimmer, di-duck −72 % saat narasi.
* Semua angka (kamera, palette, kepadatan hutan, grade) dapat diubah di `src/shots.py`,
  `src/world.py`, dan `src/gfx.py`.
