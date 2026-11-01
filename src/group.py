import sys, numpy as np
from PIL import Image
import shots as S
from engine3d import Renderer, post

names = sys.argv[1].split(',')
T = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
Wp, Hp = 540, 960
S.W_IMG, S.H_IMG = Wp, Hp
by = {c.name: c for c in S.SHOT_CLASSES}
tiles = []
for n in names:
    sh = by[n](12.0)
    meshes, cam, lk = sh.frame(T * sh.dur)
    r = Renderer(Wp, Hp)
    bg = r.sky_image(cam, lk.light, top=lk.sky_top, bottom=lk.sky_bot,
                     sun_glow=lk.sky_glow, sun_size=lk.sky_size, stars=lk.stars)
    img = r.render(meshes, cam, lk.light, background=bg)
    img = post(img, seed=3, **lk.post)
    tgt = cam.eye + cam.R[2]
    print(f"{n:13s} eye=({cam.eye[0]:7.1f},{cam.eye[1]:6.1f},{cam.eye[2]:7.1f}) terrain@eye=? faces={len(meshes)}")
    tiles.append(img)
cols = min(2, len(tiles)); rows = (len(tiles)+cols-1)//cols
sheet = Image.new('RGB', (Wp*cols, Hp*rows))
for i, im in enumerate(tiles):
    sheet.paste(im, ((i%cols)*Wp, (i//cols)*Hp))
sheet.save('/home/user/Archer/ref/group.jpg', quality=92)
