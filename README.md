# Rahasia di Balik Tembok Besar China — animasi 3D vertikal

Video pendek (2:47, 1080×1920, 30 fps) bergaya *history short* Indonesia: narasi bahasa
Indonesia, dunia 3D low-poly yang **dirender penuh** (bukan slideshow gambar), kamera
sinematik yang bergerak, karakter berjalan, kavaleri, asap api beacon, kabut, siang →
senja → malam, caption per-kata, dan skor musik prosedural.

Referensi gaya: video TikTok/Shorts berbahasa Indonesia tentang Tembok Besar China
(render 3D bergaya *game-mobile*, narasi dokumenter, caption kuning). Semua aset di sini
dibuat dari nol secara prosedural — tidak ada footage atau gambar pihak ketiga.

## Hasil

| Berkas | Keterangan |
| --- | --- |
| `video/rahasia_tembok_besar_china_1080x1920.mp4` | video final 2:47 — 1080×1920, 30 fps, H.264 ~1.8 Mbps + AAC 192 kbps 48 kHz, caption terbakar, −14 LUFS |
| `video/poster.jpg` | poster/thumbnail 1080×1920 |
| `video/captions.subs.ass` | caption per-kata (libass) untuk versi lain |
| `ref/contact.jpg`, `ref/final_sheet.jpg` | analisis video referensi + contact sheet hasil akhir |

## Cara kerja pipeline

```
src/story.json     naskah 10 adegan (narasi + kicker)
        │
        ▼
assets/audio/vo_XX.mp3  voice-over per adegan (TTS, suara pria, id-ID)
        │
        ▼
src/film.py        timeline (durasi nyata VO) · skor prosedural · caption ASS · mix + ducking
        │
        ▼
src/shots.py       18 shot 3D: kamera, aktor, palette (golden/day/dusk/night/blue/macro)
        │
        ▼
src/world.py       dunia prosedural: ridge terrain, tembok bergerigi + menara, hutan,
                   prajurit, kuda, kafilah, asap, api, kabut, reruntuhan
        │
        ▼
src/engine3d.py    renderer software: proyeksi perspektif, backface cull, near-clip,
                   painter's algorithm, Lambert + hemisphere, fog ketinggian, emissive,
                   transparansi, lalu post: bloom, vignette, grade, grain
        │
        ▼
src/render.py      pipe raw RGB → ffmpeg (libx264) + burn caption (libass) + mux AAC
```

### Modul yang dipakai

* **NumPy** — seluruh matematika vertex, normal, pencahayaan, fog, dan mix audio.
* **Pillow** — rasterisasi poligon (C-speed), sky gradient, dan compositing.
* **FFmpeg 7 + libass + libx264** — encoding, burn subtitle, loudness normalisation
  (`loudnorm -14 LUFS`).
* **fontTools** — verifikasi font (Anton untuk tipografi caption ala Shorts).
* **TTS** + **imageio-ffmpeg** — voice-over Indonesia dan binary ffmpeg yang portabel.

Tidak ada GPU, WebGL, Blender, atau pustaka 3D pihak ketiga: renderer 3D-nya ditulis
sendiri (~420 baris) dan berjalan ~3 frame/detik di 1080×1920 pada 2 core.

## Menjalankan ulang

```bash
python3 -m venv .venv && .venv/bin/pip install numpy pillow imageio-ffmpeg fonttools
cd src
../.venv/bin/python render.py                # render penuh (~30-40 menit di 2 core)
../.venv/bin/python render.py --test 10      # 10 detik pertama saja
../.venv/bin/python render.py --audio-only   # hanya timeline + musik + caption
../.venv/bin/python test_shots.py            # contact sheet look-dev 18 shot
../.venv/bin/python render.py --from 47 --to 68 --out ../out/partA.mp4 --no-final  # re-render 1 rentang
```

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

* Durasi shot mengikuti durasi voice-over asli, jadi setiap potongan jatuh di beat narasi.
* Dissolve 0.55 s antar shot; transisi siang→malam memakai dissolve yang sama.
* Caption: 3 kata per cue, Anton, kuning `#FFD93B`, outline hitam 7 px, pop-in halus.
* Musik: progresi Am–F–C–G, drone + taiko + angin + shimmer, di-duck −72 % saat narasi.
* Semua angka (kamera, palette, kepadatan hutan) dapat diubah di `src/shots.py` / `src/world.py`.
