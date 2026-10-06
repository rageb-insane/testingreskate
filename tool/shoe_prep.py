"""Turns what a downloaded shoe model contains into the pair shoe_mesh.build expects: the left
shoe at x>0 and the right at x<0, both standing on y=0, toes toward +z.

Models arrive posed for display: a backdrop or floor around them, one shoe lying on its side,
the pair in any arrangement, sometimes two copies of the same foot. So:
  * props go: any part far bigger than the shoes (backdrops, floors, turntables);
  * the rest splits into objects: connected pieces whose boxes overlap belong together
    (upper, sole and laces), separate objects are separate shoes;
  * each shoe stands up on its own: the sole is the big flat face at one extreme with nothing
    like it on the opposite side (the collar is open), the heel is the end the collar rises at;
  * each shoe says which foot it is from its sole: the arch dips in on the inside (medial) edge.
    Two of the same foot, or a single shoe, get a mirrored copy for the missing foot."""
import numpy as np

import atlas


def _faces(p, tri):
    a, b, c = p[tri[:, 0]], p[tri[:, 1]], p[tri[:, 2]]
    cross = np.cross(b - a, c - a)
    area = np.linalg.norm(cross, axis=1) / 2
    normal = cross / np.maximum(2 * area, 1e-12)[:, None]
    return (a + b + c) / 3, normal, area


class _Piece:
    """One connected piece of one primitive."""
    def __init__(self, prim, vids):
        self.prim, self.vids = prim, vids
        p = prim.positions[vids]
        self.lo, self.hi = p.min(0), p.max(0)


def _objects(prims):
    pieces = []
    for prim in prims:
        comp = atlas.islands(prim.indices, len(prim.positions))
        for k in np.unique(comp[np.unique(prim.indices)]):
            pieces.append(_Piece(prim, np.flatnonzero(comp == k)))
    parent = list(range(len(pieces)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    vol = [np.prod(np.maximum(q.hi - q.lo, 1e-4)) for q in pieces]
    for i in range(len(pieces)):
        for j in range(i + 1, len(pieces)):
            inter = np.minimum(pieces[i].hi, pieces[j].hi) - np.maximum(pieces[i].lo, pieces[j].lo)
            overlap = np.prod(np.where(inter >= 0, np.maximum(inter, 1e-4), 0))   # flat pieces count as 0.1 mm thick
            if overlap > 0.2 * min(vol[i], vol[j]):
                parent[find(i)] = find(j)
    groups = {}
    for i, q in enumerate(pieces):
        groups.setdefault(find(i), []).append(q)
    return sorted(groups.values(), key=lambda g: -sum(len(q.vids) for q in g))


def _merge(group):
    """One object's vertices, normals, uv and triangles, in one array set: each primitive's
    vertices in their original order (a shoe that is one whole primitive comes out unchanged)."""
    by_prim = {}
    for q in group:
        by_prim.setdefault(id(q.prim), (q.prim, []))[1].append(q.vids)
    pos, nrm, uv, tri, base = [], [], [], [], 0
    for prim, vid_lists in by_prim.values():
        vids = np.sort(np.concatenate(vid_lists))
        remap = np.full(len(prim.positions), -1)
        remap[vids] = np.arange(len(vids)) + base
        t = remap[prim.indices]
        tri.append(t[(t >= 0).all(1)])
        pos.append(prim.positions[vids])
        nrm.append(prim.normals[vids] if prim.normals is not None else np.zeros((len(vids), 3)))
        uv.append(prim.uv[vids])
        base += len(vids)
    pos, nrm, uv, tri = np.concatenate(pos), np.concatenate(nrm), np.concatenate(uv), np.concatenate(tri)
    if not np.abs(nrm).any(1).all():                   # a part without normals: from its faces
        centre, fn, area = _faces(pos, tri)
        acc = np.zeros_like(pos)
        for k in range(3):
            np.add.at(acc, tri[:, k], fn * area[:, None])
        missing = ~np.abs(nrm).any(1)
        nrm[missing] = acc[missing] / np.maximum(np.linalg.norm(acc[missing], axis=1, keepdims=True), 1e-12)
    return pos, nrm, uv, tri


def stand_up(pos, tri):
    """Rotation (rows: new x, y, z) that stands a shoe on its sole with the toe toward +z."""
    centre = pos.mean(0)
    fc, fn, area = _faces(pos, tri)

    def flat_at_extreme(d):
        proj = pos @ d
        reach = proj.max() - 0.12 * (proj.max() - proj.min())
        return (np.abs(fn @ d) > 0.9) & (fc @ d > reach)
    # every direction on the sphere: the sole is the most flat area at one extreme, with little
    # like it at the opposite extreme (the collar is open); side panels have a twin opposite
    k = np.arange(800) + 0.5
    polar, turn = np.arccos(1 - 2 * k / 800), np.pi * (1 + 5 ** 0.5) * k
    dirs = np.c_[np.cos(turn) * np.sin(polar), np.sin(turn) * np.sin(polar), np.cos(polar)]
    scores = [area[flat_at_extreme(d)].sum() - area[flat_at_extreme(-d)].sum() for d in dirs]
    down = dirs[int(np.argmax(scores))]
    sel = flat_at_extreme(down)                          # refine: the sole faces' own mean normal
    n = fn[sel] * np.sign(fn[sel] @ down)[:, None]
    down = (n * area[sel, None]).sum(0)
    down /= np.linalg.norm(down)
    up = -down
    axis = np.eye(3)[np.argmax(np.abs(up))] * np.sign(up[np.argmax(np.abs(up))])
    if up @ axis > np.cos(np.radians(3)):              # already standing on one of the model's axes:
        up = axis                                       # keep it (toe spring tilts the sole's own normal)
    flat = (pos - centre) - np.outer((pos - centre) @ up, up)   # the footprint, seen from above
    w, vec = np.linalg.eigh(np.cov(flat.T))
    length = vec[:, np.argmax(w)]
    length -= up * (length @ up)
    length /= np.linalg.norm(length)
    height = pos @ up
    top = height > height.max() - 0.15 * (height.max() - height.min())
    along = (pos - centre) @ length
    toe = -length if along[top].mean() > 0 else length  # the collar rises at the heel
    x = np.cross(up, toe)
    return np.array([x / np.linalg.norm(x), up, toe])


def which_foot(pos):
    """+1 for a left shoe, -1 for a right one, of a shoe standing up with its toe toward +z
    (x>0 is the wearer's left): the sole's outline dips in at the arch on the inner side."""
    y = pos[:, 1]
    sole = pos[y < y.min() + 0.15 * (y.max() - y.min())]
    z0, z1 = sole[:, 2].min(), sole[:, 2].max()

    def edges(f0, f1):
        sel = sole[(sole[:, 2] >= z0 + f0 * (z1 - z0)) & (sole[:, 2] <= z0 + f1 * (z1 - z0))]
        return sel[:, 0].min(), sel[:, 0].max()
    heel, ball, arch = edges(0.12, 0.25), edges(0.62, 0.75), edges(0.38, 0.52)
    dip_minus = arch[0] - (heel[0] + ball[0]) / 2        # how far the -x edge comes in at the arch
    dip_plus = (heel[1] + ball[1]) / 2 - arch[1]         # how far the +x edge comes in
    # the inner (medial) side dips more; for the left foot the inner side faces -x
    return (1 if dip_minus > dip_plus else -1), abs(dip_minus - dip_plus)


def _mirrored(shoe):
    pos, nrm, uv, tri = shoe
    return pos * [-1, 1, 1], nrm * [-1, 1, 1], uv, tri[:, ::-1]


def _signed_volume(pos, tri):
    return float(np.einsum('ij,ij->i', pos[tri[:, 0]], np.cross(pos[tri[:, 1]], pos[tri[:, 2]])).sum() / 6)


def _outward_share(pos, tri):
    """Area share of a shoe's faces that face away from its middle (minus those facing in): a shoe
    with pieces wound inside-out scores well under its twin; it does not depend on placement."""
    a, b, c = pos[tri[:, 0]], pos[tri[:, 1]], pos[tri[:, 2]]
    n = np.cross(b - a, c - a)
    area = np.linalg.norm(n, axis=1) / 2
    return float((np.sign((n * ((a + b + c) / 3 - pos.mean(0))).sum(1)) * area).sum() / max(area.sum(), 1e-12))


TWIN_GAP = 0.1    # outward shares further apart than this: one twin has inside-out pieces


def _match_twin_winding(placed, log, reach=0.004):
    """A pair's two shoes are mirror twins, but a model can have pieces of one shoe wound the
    wrong way round (its triangles and normals facing into the shoe: the Jordan 4 pack's left
    outsole, heel and wing parts). The game draws only a triangle's front, so those pieces show
    their inside: a sole that floats, a heel without its print. Only when the twins' outward
    shares differ by more than TWIN_GAP (the Jordan 4: 0.14 against 0.36; a sound pair like the
    Jordan 1 pack: 0.30 against 0.26, where laces crossing the other way and thin double-sided
    parts would only fool the comparison), the twin facing out more is taken as right, and every
    triangle of the other that faces opposite to its mirrored twin (within `reach` metres) is
    turned round, its normals with it. Nothing moves. Changes `placed` in place."""
    share = {s: _outward_share(placed[s][0], placed[s][3]) for s in placed}
    good, fix = sorted(placed, key=lambda s: -share[s])
    if share[good] - share[fix] <= TWIN_GAP:
        return
    gp, gn, _, gt = placed[good]
    pos, nrm, uv, tri = placed[fix]

    def faces(p, t):
        a, b, c = p[t[:, 0]], p[t[:, 1]], p[t[:, 2]]
        n = np.cross(b - a, c - a)
        return (a + b + c) / 3 - p.mean(0), n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    gc, gfn = faces(gp, gt)
    gc, gfn = gc * [-1, 1, 1], gfn * [-1, 1, 1]                   # the good twin mirrored onto this foot
    fc, ffn = faces(pos, tri)
    cell = lambda c: np.floor(c / reach).astype(np.int64)
    buckets = {}
    for i, k in enumerate(map(tuple, cell(gc))):
        buckets.setdefault(k, []).append(i)
    agree = np.zeros(len(tri))
    offsets = [(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)]
    for i, k in enumerate(map(tuple, cell(fc))):
        cand = [j for o in offsets for j in buckets.get((k[0] + o[0], k[1] + o[1], k[2] + o[2]), ())]
        if not cand:
            continue
        cand = np.array(cand)
        d = np.linalg.norm(gc[cand] - fc[i], axis=1)
        if d.min() <= reach:
            agree[i] = gfn[cand[d.argmin()]] @ ffn[i]
    flip = agree < -0.3
    if flip.sum() < 0.005 * len(tri):
        return
    # a vertex shared by turned and kept triangles is split, so each keeps its own normal (the
    # copies sit at the same spot; nothing moves)
    used_flip = np.zeros(len(pos), bool); used_flip[tri[flip].ravel()] = True
    used_keep = np.zeros(len(pos), bool); used_keep[tri[~flip].ravel()] = True
    both = np.flatnonzero(used_flip & used_keep)
    remap = np.arange(len(pos))
    remap[both] = len(pos) + np.arange(len(both))
    pos = np.concatenate([pos, pos[both]])
    nrm = np.concatenate([nrm, nrm[both]])
    uv = np.concatenate([uv, uv[both]])
    tri = tri.copy()
    tri[flip] = remap[tri[flip]][:, ::-1]
    nrm[np.unique(tri[flip])] *= -1
    placed[fix] = (pos, nrm, uv, tri)
    log(f'  {"left" if fix > 0 else "right"} shoe: {int(flip.sum())} triangles faced into the shoe (opposite to the '
        f'{"left" if good > 0 else "right"} shoe\'s) and were turned round; nothing moved')


def prepare(prims, log=print):
    """(primitive-like objects for shoe_mesh.build, material index the shoes use)."""
    usable = [p for p in prims if p.uv is not None and len(p.indices)]
    if not usable:
        raise SystemExit('The model has no textured (UV-mapped) mesh.')
    biggest = max(usable, key=lambda p: len(p.positions))
    size = np.linalg.norm(np.ptp(biggest.positions, 0))
    props = [p for p in usable if np.linalg.norm(np.ptp(p.positions, 0)) > 2.5 * size]
    for p in props:
        log(f'  left out "{p.name or "a part"}": far bigger than the shoe (a backdrop or floor)')
    kept = [p for p in usable if p not in props]
    objects = _objects(kept)
    counts = [sum(len(q.vids) for q in g) for g in objects]
    shoes = [objects[0]] + ([objects[1]] if len(objects) > 1 and counts[1] > 0.5 * counts[0] else [])
    dropped = sum(counts[len(shoes):])
    if dropped:
        log(f'  left out {len(objects) - len(shoes)} small loose piece(s) ({dropped} vertices) away from the shoes')
    materials = {}
    for g in shoes:
        for q in g:
            materials[q.prim.material] = materials.get(q.prim.material, 0) + len(q.vids)
    material = max(materials, key=materials.get)
    if len([m for m in materials if m is not None]) > 1:
        log(f'  note: the shoes use {len(materials)} materials; only the main one\'s texture is used')

    placed = {}
    for g in shoes:
        pos, nrm, uv, tri = _merge(g)
        rot = stand_up(pos, tri)
        pos = (pos - pos.mean(0)) @ rot.T
        nrm = nrm @ rot.T
        if np.linalg.det(rot) < 0:
            tri = tri[:, ::-1]
        side, sureness = which_foot(pos)
        if side in placed:                               # two of the same foot
            log(f'  both shoes are {"left" if side > 0 else "right"} shoes: mirrored one for the other foot')
            side = -side
            pos, nrm, uv, tri = _mirrored((pos, nrm, uv, tri))
        placed[side] = (pos, nrm, uv, tri)
        log(f'  {"left" if side > 0 else "right"} shoe: stood up, toe forward '
            f'(arch on the inside by {sureness * 1000:.1f} mm)')
    if len(placed) == 1:
        side = next(iter(placed))
        placed[-side] = _mirrored(placed[side])
        log(f'  one shoe in the model: mirrored it for the {"left" if side < 0 else "right"} foot')

    if len(placed) == 2:
        _match_twin_winding(placed, log)

    out = []
    for side, (pos, nrm, uv, tri) in placed.items():
        pos = pos.copy()
        pos[:, 1] -= pos[:, 1].min()
        pos[:, 0] += side * 0.15 - (pos[:, 0].min() + pos[:, 0].max()) / 2
        out.append(type(biggest)(name=f'{"left" if side > 0 else "right"} shoe', node=None, positions=pos,
                                 normals=nrm, uv=uv, indices=tri, material=material))
    return out, material
