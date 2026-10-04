# Menjaga repo ini aman dari flag/suspend GitHub

Masalahnya bukan konten, tapi **ukuran biner**. GitHub memperingatkan file >50 MB,
menolak file >100 MB, dan repo yang membengkak (ratusan MB sampai GB) berisiko kena
pembatasan abuse/bandwidth. Langkah yang sudah diterapkan di sini:

1. **Riwayat dipadatkan.** Semua commit lama yang memuat render MP4 (±150 MB blob)
   dibuang dengan membuat commit tunggal (orphan) lalu force-push. Blob lama jadi
   tidak terjangkau dan akan dibersihkan GitHub otomatis.
2. **Release/tag lama dihapus**, karena tag menahan commit lama tetap "hidup"
   sehingga blob besarnya tidak pernah bisa di-GC.
3. **Aset gambar disimpan sebagai JPEG q92**, bukan PNG: 130 MB -> 16,2 MB
   (kualitas untuk render video 1080p praktis identik).
4. **Render mentah tidak di-commit** (`output/*.mp4` masuk .gitignore). Hanya versi
   final yang sudah dikompresi (<25 MB) yang ditambahkan manual dengan `git add -f`.
5. **Audio VO tetap MP3** (kecil), dan folder kerja `build/tmp_*` diabaikan.
6. Jika suatu saat butuh menyimpan banyak video: pakai **GitHub Releases** sebagai
   tempat file besar (bukan di dalam git), atau aktifkan **Git LFS**.

Cek cepat kesehatan repo:

```bash
git count-objects -vH | grep size-pack     # ukuran lokal
gh api repos/<user>/<repo> --jq '.size'    # ukuran remote (KB)
find . -size +40M -not -path './.git/*'    # file besar yang belum di-ignore
```
