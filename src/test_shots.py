import sys, time, numpy as np
from PIL import Image
import shots as S
from engine3d import Renderer, post

S.W_IMG, S.H_IMG = 432, 768
W, H = 432, 768
tiles = []
for cls in S.SHOT_CLASSES:
    t0 = time.time()
    try:
        sh = cls(12.0)
        meshes, cam, lk = sh.frame(6.2)
        r = Renderer(W, H)
        bg = r.sky_image(cam, lk.light, top=lk.sky_top, bottom=lk.sky_bot,
                         sun_glow=lk.sky_glow, sun_size=lk.sky_size, stars=lk.stars)
        img = r.render(meshes, cam, lk.light, background=bg)
        img = post(img, seed=1, **lk.post)
        dt = time.time()-t0
        nf = sum(len(m.f) for m in meshes)
        print(f"{cls.name:14s} faces={nf:6d} {dt*1000:6.0f}ms")
    except Exception as e:
        import traceback; traceback.print_exc()
        img = Image.new('RGB', (W,H), (30,30,30))
    tiles.append(img)

cols, rows = 6, 3
sheet = Image.new('RGB', (W*cols//2, H*rows//2))
for i, im in enumerate(tiles):
    im2 = im.resize((W//2, H//2))
    sheet.paste(im2, ((i%cols)*W//2, (i//cols)*H//2))
sheet.save('/home/user/Archer/ref/sheet.jpg', quality=88)
print("saved")
