# raw/ — tempat menaruh video sumber

Letakkan file video asli (mp4 / mov / webm) di folder ini, lalu jalankan:

```bash
source tools/ffmpeg_setup.sh
python3 tools/make_short.py --src raw/NAMA_FILE.mp4 \
  --segments 8.2-17,31.5-44 --mode cover \
  --vo audio/vo.mp3 --captions script/captions.json \
  --out output/short.mp4
```

Sandbox ini tidak bisa menjangkau YouTube/TikTok/situs stock footage
(lihat `NOTES-sandbox.md`), jadi file sumber harus masuk lewat folder ini
atau lewat `git clone` repo GitHub publik.
