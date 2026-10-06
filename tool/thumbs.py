"""Item thumbnails in the style of the game's own shoe icons: the pair standing on
grey mannequin legs, seen from the front three-quarters, framed tight, on
transparency (the menu draws the rarity background). A small z-buffered software
rasteriser: textured, smooth-shaded, supersampled."""
import numpy as np
from PIL import Image

LEG_COLOUR = np.array([0.74, 0.74, 0.75])   # the game icons' mannequin: ~175 lit edge, ~150 facing, ~130 shaded edge
LEG_SHADE = (0.69, 0.26)       # ambient, diffuse
LEG_INSIDE = 0.05              # the legs start this far below the top of the collar (the foot is culled in game)
BODY_MESH = '/cas_rsp_body_complex_dmpreset_cas_main_bundlereftable'
STANCE = 0.085         # icon only: each shoe's centre this far from the middle, like the icon pose
CAMERA = (0.48, 0.41, 0.40)   # eye offset from the pair: 50 deg to the side, 33 deg up, 0.75 m


def body_legs(g, top=0.65):
    """The game's own body mesh below `top` (metres): welded positions, smooth normals and
    triangles, in the same character space as the shoes. None if this build lacks it."""
    import fb
    import meshset
    from shoe_mesh import MESHSET_RES
    _, bundles, toc_chunks = g.toc('Win32/items.toc')
    toc = {c.guid: c for c in toc_chunks}
    b = next((x for x in bundles if x.name.endswith(BODY_MESH)), None)
    if b is None:
        return None
    files, _ = fb.read_bundle_region(b.region)
    ebx, res, _, _ = g.manifest(files)
    for a, f in zip(res, files[1 + len(ebx):]):
        if a.res_type != MESHSET_RES:
            continue
        lod = meshset.MeshSet(g.payload(f)).lods[0]
        c = toc.get(lod.chunk)
        if c is None:
            return None
        data = fb.decode_cas(g.read(fb.FileInfo(c.patch, c.install_chunk, c.archive, c.offset, c.size)), g.root)
        v = meshset.decode_section(lod, lod.sections[0], data)
        break
    else:
        return None
    pos, inv = np.unique(np.round(v['pos'][:, :3], 5), axis=0, return_inverse=True)   # weld the uv seams
    tri = inv.reshape(-1)[v['triangles']]
    tri = tri[(pos[tri][:, :, 1].max(1) < top) & (tri[:, 0] != tri[:, 1]) & (tri[:, 1] != tri[:, 2]) & (tri[:, 0] != tri[:, 2])]
    nrm = np.zeros_like(pos)
    face = np.cross(pos[tri[:, 1]] - pos[tri[:, 0]], pos[tri[:, 2]] - pos[tri[:, 0]])
    for k in range(3):
        np.add.at(nrm, tri[:, k], face)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    return pos, nrm, tri


def _look_at(eye, target, up=(0, 1, 0)):
    f = np.asarray(target, float) - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return r, u, f


def _leg(centre, bottom, top, r0, r1, segments=40, rings=12):
    """A tapered cylinder (one shin)."""
    pos, nrm = [], []
    for j in range(rings + 1):
        t = j / rings
        y = bottom + (top - bottom) * t
        r = r0 + (r1 - r0) * t
        for i in range(segments):
            a = 2 * np.pi * i / segments
            d = np.array([np.sin(a), 0, np.cos(a)])
            pos.append([centre[0] + r * d[0], y, centre[1] + r * d[2]])
            nrm.append(d)
    tri = []
    for j in range(rings):
        for i in range(segments):
            a = j * segments + i
            b = j * segments + (i + 1) % segments
            c, d = a + segments, b + segments
            tri += [[a, b, c], [b, d, c]]
    return np.array(pos), np.array(nrm), np.array(tri)


def render(shoe, texture, size=768, supersample=2, view_side=1, legs=None):
    """Renders shoe (positions/normals/uv/triangles) with its colour texture, standing on
    legs (body_legs(); plain tapered shins without them)."""
    big = size * supersample
    tex = np.asarray(texture.convert('RGB'), dtype=np.float32) / 255.0
    positions = shoe.positions.copy()
    shift = {}
    for side in (1, -1):                       # the icon pose stands with the feet closer together
        m = positions[:, 0] * side > 0
        shift[side] = side * STANCE - positions[m, 0].mean()
        positions[m, 0] += shift[side]
    th, tw = tex.shape[:2]

    # legs: from inside each collar up past the top of the frame, moved with their shoe
    meshes = [(positions, shoe.normals, shoe.uv, shoe.triangles, None)]
    for side in (1, -1):
        p = positions[positions[:, 0] * side > 0]
        top = p[:, 1].max()
        if legs is not None:
            lp, ln, lt = legs
            corners = lp[lt]
            keep = lt[(corners[:, :, 0] * side > 0).all(1) & (corners[:, :, 1].min(1) > top - LEG_INSIDE)]
            used, local = np.unique(keep, return_inverse=True)
            q = lp[used].copy()
            q[:, 0] += shift[side]
            meshes.append((q, ln[used], None, local.reshape(-1, 3), LEG_COLOUR))
        else:
            collar = p[p[:, 1] > top - 0.015][:, [0, 2]].mean(0)
            lp, ln, lt = _leg(collar, top - 0.04, top + 0.45, 0.036, 0.048)
            meshes.append((lp, ln, None, lt, LEG_COLOUR))

    # camera: in front of the pair, off to one side and above, like the game's icons
    feet = positions
    centre = (feet.min(0) + feet.max(0)) / 2
    target = centre + [0, -0.01, 0.02]
    eye = target + np.array([CAMERA[0] * view_side, CAMERA[1], CAMERA[2]])
    right, up, fwd = _look_at(eye, target)
    light = -fwd * 0.55 + up * 0.75 + right * (0.35 * view_side)
    light /= np.linalg.norm(light)
    leg_light = -fwd * 0.41 + up * 0.15 - right * 0.9   # the game's mannequin is lit from the picture's left
    leg_light /= np.linalg.norm(leg_light)

    def project(p):
        rel = p - eye
        z = rel @ fwd
        return np.stack([(rel @ right) / z, (rel @ up) / z], 1), z

    # frame on the shoes alone: fill the width, sit the soles near the bottom
    s2d, _ = project(feet)
    lo, hi = s2d.min(0), s2d.max(0)
    scale = big * 0.96 / max(hi[0] - lo[0], (hi[1] - lo[1]) / 0.80)
    offset_x = (big - (hi[0] - lo[0]) * scale) / 2 - lo[0] * scale
    offset_y = big * 0.90 + lo[1] * scale      # screen y grows downwards

    colour = np.zeros((big, big, 3), np.float32)
    alpha = np.zeros((big, big), np.float32)
    depth = np.full((big, big), np.inf, np.float32)
    for pos, nrm, uv, tri, flat in meshes:
        s, z = project(pos)
        sx = s[:, 0] * scale + offset_x
        sy = offset_y - s[:, 1] * scale
        if flat is None:
            shade_v = np.clip(nrm @ light, 0, 1) * 0.62 + 0.38 + np.clip(nrm @ up, 0, 1) * 0.05
        else:                                  # the mannequin's legs: matte grey, lit from the left
            shade_v = np.clip(nrm @ leg_light, 0, 1) * LEG_SHADE[1] + LEG_SHADE[0]
        x0, y0, x1, y1, x2, y2 = (sx[tri[:, 0]], sy[tri[:, 0]], sx[tri[:, 1]], sy[tri[:, 1]], sx[tri[:, 2]], sy[tri[:, 2]])
        area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        visible = area < 0                     # counter-clockwise in world -> clockwise on screen (y down)
        visible &= (np.minimum(np.minimum(z[tri[:, 0]], z[tri[:, 1]]), z[tri[:, 2]]) > 0.05)   # in front of the eye
        for k in np.flatnonzero(visible):
            a, b, c = tri[k]
            xs = (sx[a], sx[b], sx[c])
            ys = (sy[a], sy[b], sy[c])
            bx0, bx1 = max(int(np.floor(min(xs))), 0), min(int(np.ceil(max(xs))), big - 1)
            by0, by1 = max(int(np.floor(min(ys))), 0), min(int(np.ceil(max(ys))), big - 1)
            if bx0 > bx1 or by0 > by1:
                continue
            gx, gy = np.meshgrid(np.arange(bx0, bx1 + 1) + 0.5, np.arange(by0, by1 + 1) + 0.5)
            d = area[k]
            w0 = ((xs[1] - gx) * (ys[2] - gy) - (xs[2] - gx) * (ys[1] - gy)) / d
            w1 = ((xs[2] - gx) * (ys[0] - gy) - (xs[0] - gx) * (ys[2] - gy)) / d
            w2 = 1 - w0 - w1
            inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
            if not inside.any():
                continue
            iz = w0 / z[a] + w1 / z[b] + w2 / z[c]      # perspective-correct weights
            zz = 1 / iz
            region = depth[by0:by1 + 1, bx0:bx1 + 1]
            take = inside & (zz < region)
            if not take.any():
                continue
            pw0, pw1, pw2 = (w0 / z[a]) * zz, (w1 / z[b]) * zz, (w2 / z[c]) * zz
            shade = pw0 * shade_v[a] + pw1 * shade_v[b] + pw2 * shade_v[c]
            if flat is None:
                u = (pw0 * uv[a, 0] + pw1 * uv[b, 0] + pw2 * uv[c, 0]) % 1
                v = (pw0 * uv[a, 1] + pw1 * uv[b, 1] + pw2 * uv[c, 1]) % 1
                rgb = tex[np.minimum((v * th).astype(int), th - 1), np.minimum((u * tw).astype(int), tw - 1)]
            else:
                rgb = np.broadcast_to(flat, take.shape + (3,))
            region[take] = zz[take]
            colour[by0:by1 + 1, bx0:bx1 + 1][take] = (rgb * shade[..., None])[take]
            alpha[by0:by1 + 1, bx0:bx1 + 1][take] = 1
    img = np.dstack([np.clip(colour, 0, 1) * 255, alpha * 255]).astype(np.uint8)
    return Image.fromarray(img, 'RGBA').resize((size, size), Image.LANCZOS)
