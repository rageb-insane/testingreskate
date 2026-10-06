"""Gives each foot's outsole its own artwork. Models usually mirror one shoe for the other, so
both soles sample the same texels; the real shoe may differ per
foot (the Dior Jordan 1: "DIOR" under the left sole, the Air Dior Wings under the right).
The texture's UV islands are repacked so each chosen foot's sole island gets a copy of its
own, painted with that foot's image."""
import numpy as np
from PIL import Image

PAD = 6 / 1024          # gap around every island, in UV units (6 texels at the game's 1024)


def islands(triangles, n_vertices):
    """Island (connected component) index per vertex; triangles sharing a vertex connect."""
    parent = np.arange(n_vertices)

    def find(i):
        root = i
        while parent[root] != root:
            root = parent[root]
        while parent[i] != root:
            parent[i], i = root, parent[i]
        return root
    for a, b, c in triangles:
        ra, rb, rc = find(a), find(b), find(c)
        parent[rb] = ra
        parent[find(rc)] = ra
    return np.array([find(i) for i in range(n_vertices)])


def sole_island(shoe, side, comp):
    """The island holding most of one foot's ground-facing triangles."""
    p, t = shoe.positions, shoe.triangles
    fn = np.cross(p[t[:, 1]] - p[t[:, 0]], p[t[:, 2]] - p[t[:, 0]])
    fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
    centre = p[t].mean(1)
    ground = (fn[:, 1] < -0.8) & (centre[:, 1] < p[:, 1].min() + 0.01) & (np.sign(centre[:, 0]) == side)
    ids, counts = np.unique(comp[t[ground, 0]], return_counts=True)
    return ids[np.argmax(counts)]


def sole_artwork(shoe, texture, side):
    """That foot's sole region of the texture, as an image (what own_soles replaces)."""
    comp = islands(shoe.triangles, len(shoe.positions))
    vid = np.flatnonzero(comp == sole_island(shoe, side, comp))
    lo, hi = shoe.uv[vid].min(0), shoe.uv[vid].max(0)
    w, h = texture.size
    return texture.convert('RGB').crop((round(lo[0] * w), round(lo[1] * h), round(hi[0] * w), round(hi[1] * h)))


def _skyline(sizes, scale, order):
    """Bottom-left skyline packing of (w, h) boxes in the given order. None if they do not fit."""
    sky = [(0.0, 0.0, 1.0)]               # (x, top, width) segments
    at = {}
    for i in order:
        w, h = sizes[i][0] * scale + 2 * PAD, sizes[i][1] * scale + 2 * PAD
        best = None
        for k, (x, _, _) in enumerate(sky):
            if x + w > 1 + 1e-9:
                break
            top, span, j = 0.0, 0.0, k
            while span < w - 1e-12 and j < len(sky):
                top, span, j = max(top, sky[j][1]), span + sky[j][2], j + 1
            if span >= w - 1e-9 and top + h <= 1 + 1e-9 and (best is None or (top, x) < best[:2]):
                best = (top, x)
        if best is None:
            return None
        top, x = best
        at[i] = (x + PAD, top + PAD)
        new, end = [], x + w
        for sx, st, sw in sky:             # raise the skyline under the new box
            s_end = sx + sw
            if s_end <= x or sx >= end:
                new.append((sx, st, sw))
                continue
            if sx < x:
                new.append((sx, st, x - sx))
            if s_end > end:
                new.append((end, st, s_end - end))
        new.append((x, top + h, w))
        new.sort()
        merged = []
        for seg in new:
            if merged and abs(merged[-1][1] - seg[1]) < 1e-12 and abs(merged[-1][0] + merged[-1][2] - seg[0]) < 1e-12:
                merged[-1] = (merged[-1][0], seg[1], merged[-1][2] + seg[2])
            else:
                merged.append(seg)
        sky = merged
    return at


def _pack(sizes, scale):
    """The first of a few orderings whose skyline packing fits."""
    n = range(len(sizes))
    for key in (lambda i: -sizes[i][1], lambda i: -max(sizes[i]), lambda i: -sizes[i][0] * sizes[i][1],
                lambda i: -sizes[i][0]):
        at = _skyline(sizes, scale, sorted(n, key=key))
        if at:
            return at
    return None


def own_soles(shoe, texture, artworks, size=4096):
    """Repacks the shoe's UVs so each foot in `artworks` ({+1: image for the x>0 foot, -1: for
    x<0}) gets its own copy of its sole island, painted with that image (which covers the sole's
    region of the original texture, at any resolution). Returns (new texture, scale the rest
    of the texture kept)."""
    comp = islands(shoe.triangles, len(shoe.positions))
    soles = {sole_island(shoe, side, comp): side for side in artworks}
    uv = shoe.uv
    # boxes: one per distinct UV footprint (both feet share theirs), plus the chosen sole's copy
    keys, members = {}, []
    for c in np.unique(comp):
        idx = np.flatnonzero(comp == c)
        lo, hi = uv[idx].min(0), uv[idx].max(0)
        key = ('own', soles[c]) if c in soles else tuple(np.round(np.r_[lo, hi], 4))
        if key not in keys:
            keys[key] = len(members)
            members.append((lo, hi, []))
        members[keys[key]][2].append(idx)
    sizes = [tuple(hi - lo) for lo, hi, _ in members]
    lo_s, hi_s = 0.0, 1.0
    for _ in range(30):                   # largest scale at which everything still fits
        mid = (lo_s + hi_s) / 2
        lo_s, hi_s = (mid, hi_s) if _pack(sizes, mid) else (lo_s, mid)
    scale = lo_s
    place = _pack(sizes, scale)

    src = texture.convert('RGB').resize((size, size), Image.LANCZOS)
    out = Image.new('RGB', (size, size), (0, 0, 0))
    filled = Image.new('L', (size, size), 0)
    new_uv = uv.copy()
    for i, (lo, hi, idx_lists) in enumerate(members):
        dx, dy = place[i]
        for idx in idx_lists:
            new_uv[idx] = (uv[idx] - lo) * scale + (dx, dy)
        # copy the island's box with its padding (so filtering at the edges finds its colours)
        pad = PAD / scale
        box = [int(np.floor((lo[0] - pad) * size)), int(np.floor((lo[1] - pad) * size)),
               int(np.ceil((hi[0] + pad) * size)), int(np.ceil((hi[1] + pad) * size))]
        w = max(1, round((box[2] - box[0]) * scale))
        h = max(1, round((box[3] - box[1]) * scale))
        own = [side for side in artworks if keys.get(('own', side)) == i]
        if own:
            inner = artworks[own[0]].convert('RGB').resize((max(1, round((hi[0] - lo[0]) * size)), max(1, round((hi[1] - lo[1]) * size))), Image.LANCZOS)
            patch = src.crop(box)
            patch.paste(inner, (int(round(lo[0] * size)) - box[0], int(round(lo[1] * size)) - box[1]))
        else:
            patch = src.crop(box)
        patch = patch.resize((w, h), Image.LANCZOS)
        at = (int(round((dx - PAD) * size)), int(round((dy - PAD) * size)))
        out.paste(patch, at)
        filled.paste(255, (at[0], at[1], at[0] + w, at[1] + h))
    out = _dilate(out, filled)
    shoe.uv = new_uv
    return out, scale


def _dilate(image, mask, steps=48):
    """Fills the unused texels from their neighbours, so mips do not bleed black into edges."""
    a = np.asarray(image, np.float32)
    m = np.asarray(mask) > 0
    small = 4                              # work at a quarter size: fast, and only gaps need it
    h, w = m.shape
    ac = a.reshape(h // small, small, w // small, small, 3).mean((1, 3))
    mc = m.reshape(h // small, small, w // small, small).any((1, 3))
    for _ in range(steps):
        if mc.all():
            break
        acc = np.zeros_like(ac)
        cnt = np.zeros(mc.shape, np.float32)
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            acc += np.roll(np.roll(ac * mc[..., None], dy, 0), dx, 1)
            cnt += np.roll(np.roll(mc, dy, 0), dx, 1)
        grow = (~mc) & (cnt > 0)
        ac[grow] = acc[grow] / cnt[grow, None]
        mc = mc | grow
    ac[~mc] = a[m].mean(0)
    fill = np.repeat(np.repeat(ac, small, 0), small, 1)
    a = np.where(m[..., None], a, fill)
    return Image.fromarray(a.clip(0, 255).astype(np.uint8))


def coverage(uv, triangles, size):
    """The texels a mesh's UV triangles cover in a size x size texture, plus a one-texel rim
    (bilinear filtering reads it)."""
    import cv2
    mask = np.zeros((size, size), np.uint8)
    pts = np.round(np.clip(uv, 0, 1) * size * 16).astype(np.int32)[triangles]   # 4 bits of subpixel
    for t in pts:
        cv2.fillConvexPoly(mask, t, 255, lineType=cv2.LINE_8, shift=4)
    return cv2.dilate(mask, np.ones((3, 3), np.uint8)) > 0


def pad(image, mask):
    """The texels outside `mask` filled from the painted ones nearby (push-pull over a pyramid),
    so filtering and smaller mips at the edges of UV islands do not pull in whatever the
    unused background held (a light seam across a dark panel)."""
    import cv2
    mode = image.mode
    a = np.asarray(image, np.float32)
    flat = a.ndim == 2
    if flat:
        a = a[..., None]
    m = mask.astype(np.float32)
    levels = [(a * m[..., None], m)]
    while min(levels[-1][1].shape) > 1:
        c, w = levels[-1]
        half = (max(1, c.shape[1] // 2), max(1, c.shape[0] // 2))
        c2 = cv2.resize(c, half, interpolation=cv2.INTER_AREA).reshape(half[1], half[0], a.shape[2])
        levels.append((c2, cv2.resize(w, half, interpolation=cv2.INTER_AREA).reshape(half[1], half[0])))
    c, w = levels[-1]
    fill = c / max(float(w.max()), 1e-6)
    for c, w in reversed(levels[:-1]):
        up = cv2.resize(fill, (c.shape[1], c.shape[0]), interpolation=cv2.INTER_LINEAR).reshape(c.shape)
        known = (w > 1e-6)[..., None]
        fill = np.where(known, c / np.maximum(w, 1e-6)[..., None], up)
    out = np.where(mask[..., None], a, fill).clip(0, 255).astype(np.uint8)
    return Image.fromarray(out[..., 0] if flat else out, mode)
