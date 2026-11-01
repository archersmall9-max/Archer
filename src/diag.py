import numpy as np
from PIL import Image
import shots as S
import world as W
from engine3d import Renderer, icosphere, Mesh, post, pad_faces

W.W_IMG, W.H_IMG = 400, 400
S.W_IMG, S.H_IMG = 400, 400
Wp, Hp = 400, 400
tiles = []
for cls in S.SHOT_CLASSES:
    sh = cls(12.0)
    meshes, cam, lk = sh.frame(6.2)
    V = np.concatenate([m.v for m in meshes])
    lo, hi = V.min(0), V.max(0)
    ctr = (lo + hi) / 2
    span = float((hi - lo).max())
    # markers
    mk = [icosphere(cam.eye, span*0.02, np.array([1,0.1,0.1]), glow=1.0)]
    f = cam.R[2]
    for d in (10, 25, 45, 70, 100):
        mk.append(icosphere(cam.eye + f*d, span*0.012, np.array([0.1,1,1]), glow=1.0))
    r = Renderer(Wp, Hp)
    eye = np.array([ctr[0], ctr[1] + span*1.05, ctr[2] + 0.001])
    from engine3d import Camera
    top = Camera(eye, ctr, fov=60, w=Wp, h=Hp, near=1.0)
    lk.light.fog_density = 0.0
    img = r.render(meshes + mk, top, lk.light, background=Image.new('RGB',(Wp,Hp),(20,20,26)))
    img = post(img, bloom=0.1, vignette=0.0, grain=0.0, saturation=1.0, contrast=1.0)
    print(f"{cls.name:14s} ctr=({ctr[0]:6.0f},{ctr[1]:6.0f},{ctr[2]:6.0f}) span={span:6.0f} eye=({cam.eye[0]:6.0f},{cam.eye[1]:6.0f},{cam.eye[2]:6.0f})")
    tiles.append(img)

cols, rows = 6, 3
sheet = Image.new('RGB', (Wp*cols, Hp*rows))
for i, im in enumerate(tiles):
    sheet.paste(im, ((i%cols)*Wp, (i//cols)*Hp))
sheet.save('/home/user/Archer/ref/map.jpg', quality=85)
