"""Reduces a dense model (made for 3D printing or close-up renders) to what the game can hold: a
shoe's mesh is indexed with 16 bits, at most 65535 vertices for both feet.

Grid clustering with quadric error placement (Lindstrom, 2000): space is cut into cubes, every
vertex in a cube merges into one, and that one sits where it best keeps the planes of all the
triangles around it (so edges, creases and flat faces stay where they were instead of rounding
off, as plain averaging would). Triangles whose corners fall into fewer than three cubes vanish.
Vertices only merge within the same colour (UV), so painted parts keep their borders. The cube
size is searched to land just under the vertex budget."""
import numpy as np

SMALL_PART = 2000     # parts of one colour smaller than this are left as modelled


def _weld(pos, uv, tri, nrm):
    key = np.round(np.concatenate([pos, uv * 1e3], 1), 6)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    n = np.zeros((len(first), 3))
    np.add.at(n, inv, nrm)                               # welded corners keep the mean of their normals
    return pos[first], uv[first], inv[tri], n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def _cluster(pos, uv, tri, cell, nrm):
    """One pass at cube size `cell`: (positions, uv, triangles, normals). A merged vertex's normal
    is the mean of the original ones it replaces: the model's own smooth shading, so detail too
    small to keep (studs, stitching) leaves a calm surface instead of dark facets."""
    origin = pos.min(0)
    c = np.floor((pos - origin) / cell).astype(np.int64)
    tile = np.round(uv * 4096).astype(np.int64)
    key = np.concatenate([c, tile], 1)
    _, cid, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    n = len(cid)
    # face quadrics, area-weighted, gathered into each corner's cube
    a, b, d = pos[tri[:, 0]], pos[tri[:, 1]], pos[tri[:, 2]]
    face = np.cross(b - a, d - a)
    area = np.linalg.norm(face, axis=1)
    ok = area > 1e-15
    unit = np.zeros_like(face)
    unit[ok] = face[ok] / area[ok, None]
    plane = np.concatenate([unit, -(unit * a).sum(1, keepdims=True)], 1)       # n.x + w = 0
    K = plane[:, :, None] * plane[:, None, :] * (area / 2)[:, None, None]
    Q = np.zeros((n, 4, 4))
    for k in range(3):
        np.add.at(Q, inv[tri[:, k]], K)
    mean = np.zeros((n, 3))
    np.add.at(mean, inv, pos)
    count = np.bincount(inv, minlength=n)[:, None]
    mean /= count
    A, rhs = Q[:, :3, :3], -Q[:, :3, 3]
    lam = 1e-3 * np.maximum(np.trace(A, axis1=1, axis2=2), 1e-12)[:, None, None]
    x = np.linalg.solve(A + lam * np.eye(3), (rhs + lam[:, :, 0] * mean)[:, :, None])[:, :, 0]
    # keep each vertex near its cube (a near-flat quadric can send it far)
    far = np.linalg.norm(x - mean, axis=1) > cell
    x[far] = mean[far]
    new_uv = np.zeros((n, uv.shape[1]))
    np.add.at(new_uv, inv, uv)
    new_uv /= count
    vn = np.zeros((n, 3))
    np.add.at(vn, inv, nrm)
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
    t = inv[tri]
    keep = (t[:, 0] != t[:, 1]) & (t[:, 1] != t[:, 2]) & (t[:, 0] != t[:, 2])
    t = t[keep]
    _, first = np.unique(np.sort(t, 1), axis=0, return_index=True)          # one of each triangle
    t = t[np.sort(first)]
    used = np.unique(t)
    remap = np.full(n, -1)
    remap[used] = np.arange(len(used))
    return x[used], new_uv[used], remap[t], vn[used]


def normals(pos, tri):
    """Area-weighted vertex normals."""
    a, b, c = pos[tri[:, 0]], pos[tri[:, 1]], pos[tri[:, 2]]
    fn = np.cross(b - a, c - a)
    out = np.zeros_like(pos)
    for k in range(3):
        np.add.at(out, tri[:, k], fn)
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


def reduce(pos, uv, tri, max_vertices, log=print, label='mesh', nrm=None):
    """(positions, normals, uv, triangles) with at most max_vertices vertices; unchanged if it fits.
    nrm: the model's own vertex normals (else computed from its faces)."""
    pos, uv, tri = np.asarray(pos, float), np.asarray(uv, float), np.asarray(tri)
    pos, uv, tri, nrm = _weld(pos, uv, tri, normals(pos, tri) if nrm is None else np.asarray(nrm, float))
    if len(pos) <= max_vertices:
        return pos, nrm, uv, tri
    try:
        import fast_simplification
    except ImportError:
        fast_simplification = None
    if fast_simplification is not None:
        # layers just under other parts sink out of reach first (a shoe is about 300 mm long)
        mm = float(np.ptp(pos, 0).max()) / 300
        pos = sink_covered(pos, nrm, tri, 1.5 * mm, 1.5 * mm, log, label)
        return _quadric(fast_simplification, pos, uv, tri, nrm, max_vertices, log, label)
    size = float(np.linalg.norm(np.ptp(pos, 0)))
    lo, hi = size * 1e-4, size * 0.05
    best = None
    for _ in range(18):                                  # largest detail that still fits the budget
        mid = np.sqrt(lo * hi)
        p, u, t, vn = _cluster(pos, uv, tri, mid, nrm)
        if len(p) <= max_vertices:
            best, hi = (p, u, t, vn, mid), mid
        else:
            lo = mid
    p, u, t, vn, cell = best
    log(f'  {label}: {len(pos)} vertices -> {len(p)} ({len(t)} triangles), detail kept down to '
        f'{cell / size * 100:.2f}% of its size')
    # where a merged normal disagrees with its faces (a thin part folded onto itself), use the faces'
    fn = normals(p, t)
    bad = (vn * fn).sum(1) < 0.2
    vn[bad] = fn[bad]
    return p, fill_normals(vn, p, t), u, t


def _small_pieces(pos, tri, size, most_vertices=64, most_size=0.015):
    """Per vertex: the id of the tiny separate piece it belongs to (studs, rivets: closed shapes
    too small for edge collapse to touch), or -1."""
    import atlas
    comp = atlas.islands(tri, len(pos))
    ids, inv, counts = np.unique(comp, return_inverse=True, return_counts=True)
    lo = np.full((len(ids), 3), np.inf); hi = np.full((len(ids), 3), -np.inf)
    np.minimum.at(lo, inv, pos); np.maximum.at(hi, inv, pos)
    diag = np.linalg.norm(hi - lo, axis=1)
    small = (counts <= most_vertices) & (diag < most_size * size)
    out = np.where(small[inv], inv, -1)
    return out, diag[inv]


def _shrink_pieces(pos, uv, tri, nrm, piece, diag, most=6, tol=0.06):
    """Each tiny piece rebuilt from the few of its own points that span it most (a stud keeps its
    tip and base corners, a rod its ends): the two farthest apart, the farthest from that line,
    the farthest from that plane, then whichever lies farthest outside the shape so far, while
    one lies more than `tol` of the piece's size outside it (at most `most` points). Their hull
    is the new piece, every face pointing out. Other vertices and triangles stay as they were."""
    small_v = piece >= 0
    small_t = small_v[tri].all(1)
    if not small_t.any():
        return pos, uv, tri, nrm, np.zeros(len(pos), bool)
    keep_v = ~small_v
    hulls = []
    order = np.argsort(piece, kind='stable')
    pieces = piece[order]
    starts = np.flatnonzero(np.r_[True, pieces[1:] != pieces[:-1]])
    ends = np.r_[starts[1:], len(order)]
    for s0, e0 in zip(starts, ends):
        if pieces[s0] < 0:
            continue
        ids = order[s0:e0]
        chosen = _piece_points(pos[ids], most, tol * diag[ids[0]])
        if len(chosen) < 4:
            keep_v[ids] = True                           # too flat to rebuild: leave it
            continue
        sel = ids[chosen]
        keep_v[sel] = True
        hulls.append(sel[_hull(pos[sel])])
    t = np.concatenate([tri[~small_t]] + hulls) if hulls else tri[~small_t]
    used = np.unique(t)
    remap = np.full(len(pos), -1)
    remap[used] = np.arange(len(used))
    t = remap[t]
    p = pos[used]
    vn = nrm[used].copy()
    rebuilt = np.zeros(len(pos), bool)
    if hulls:
        rebuilt[np.concatenate(hulls).ravel()] = True
    fn = normals(p, t)                                   # a rebuilt piece shades by its own faces
    vn[rebuilt[used]] = fn[rebuilt[used]]
    return p, uv[used], t, vn, rebuilt[used]


def _max_triangle(pts2):
    """Indices of the three 2D points spanning the largest triangle."""
    n = len(pts2)
    best, area = (0, 1, 2), -1.0
    for a in range(n):
        for b in range(a + 1, n):
            d = pts2[b] - pts2[a]
            cr = np.abs(d[0] * (pts2[:, 1] - pts2[a, 1]) - d[1] * (pts2[:, 0] - pts2[a, 0]))
            c = int(cr.argmax())
            if cr[c] > area:
                best, area = (a, b, c), cr[c]
    return list(best)


def _piece_points(pts, most, tol):
    """The points a tiny piece is rebuilt from, chosen by its shape so identical pieces (a stud
    field, a row of stitches) come out identical: a flat piece with a tip (a pyramid stud) keeps
    its tip and the three base points spanning the largest triangle; a long one (a rod) the
    largest triangle at each end; anything else its spanning points."""
    c = pts.mean(0)
    w, v = np.linalg.eigh(np.cov((pts - c).T))          # extents, smallest first
    ext = np.ptp((pts - c) @ v, 0)
    loc = (pts - c) @ v
    if ext[2] > 2.0 * ext[1]:                            # long: a rod
        along = loc[:, 2]
        ends = [np.flatnonzero(along <= along.min() + 0.15 * ext[2]), np.flatnonzero(along >= along.max() - 0.15 * ext[2])]
        chosen = []
        for e in ends:
            if len(e) < 3:
                return _span_points(pts, most, tol)
            chosen += list(e[_max_triangle(loc[e][:, :2])])
        return chosen
    if ext[0] < 0.75 * ext[1]:                           # flat with a tip: a stud
        h = loc[:, 0]
        # the base is the side with the wide footprint, the tip the lone point on the other
        tip_up = (h > h.mean()).sum() < (h < h.mean()).sum()
        tip = int(h.argmax() if tip_up else h.argmin())
        base_side = h <= h.min() + 0.35 * ext[0] if tip_up else h >= h.max() - 0.35 * ext[0]
        base = np.flatnonzero(base_side)
        if len(base) < 3:
            return _span_points(pts, most, tol)
        return [tip] + list(base[_max_triangle(loc[base][:, 1:])])
    return _span_points(pts, most, tol)


def _span_points(pts, most, tol):
    """Indices of the few points of a small cloud that span it (see _shrink_pieces)."""
    d = np.linalg.norm(pts[:, None] - pts[None], axis=2)
    i, j = np.unravel_index(d.argmax(), d.shape)
    chosen = [int(i), int(j)]
    line = pts[j] - pts[i]
    off = pts - pts[i]
    dist = np.linalg.norm(np.cross(off, line), axis=1) / max(np.linalg.norm(line), 1e-12)
    chosen.append(int(dist.argmax()))
    nr = np.cross(pts[chosen[1]] - pts[chosen[0]], pts[chosen[2]] - pts[chosen[0]])
    h = np.abs((pts - pts[chosen[0]]) @ nr) / max(np.linalg.norm(nr), 1e-12)
    if h.max() <= tol:
        return chosen
    chosen.append(int(h.argmax()))
    while len(chosen) < most:
        faces = _hull(pts[chosen])
        a = pts[chosen][faces[:, 0]]
        fn = np.cross(pts[chosen][faces[:, 1]] - a, pts[chosen][faces[:, 2]] - a)
        fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
        outside = ((pts[:, None] - a[None]) * fn[None]).sum(2).max(1)
        outside[chosen] = -np.inf
        k = int(outside.argmax())
        if outside[k] <= tol:
            break
        chosen.append(k)
    return chosen


def _hull(points):
    """Triangles (indices into points) of the convex hull of a handful of points, facing out."""
    rng = np.random.default_rng(0)
    q = points + rng.normal(0, 1e-9 * max(float(np.ptp(points)), 1e-12), points.shape)   # no exact coplanar ties
    c = q.mean(0)
    n = len(q)
    faces = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                nr = np.cross(q[j] - q[i], q[k] - q[i])
                side = (q - q[i]) @ nr
                side[[i, j, k]] = 0
                if (side <= 0).all() or (side >= 0).all():
                    faces.append((i, j, k) if (c - q[i]) @ nr < 0 else (i, k, j))
    return np.array(faces, np.int64).reshape(-1, 3)


def _quadric(fs, pos, uv, tri, nrm, max_vertices, log, label):
    """Edge-collapse simplification by quadric error (fast_simplification, Garland-Heckbert): the
    edges whose removal changes the shape least go first, so flat areas thin out and detail keeps
    its vertices. Parts not joined by edges (each colour after welding) never merge. The collapse
    history maps every original vertex to the one it became, which carries its colour (UV) and
    the mean of the model's normals over."""
    n0 = len(pos)
    size = float(np.linalg.norm(np.ptp(pos, 0)))
    piece, diag = _small_pieces(pos, tri, size)
    studs = None
    if (piece >= 0).any():
        before = int((piece >= 0).sum())
        pos, uv, tri, nrm, rebuilt = _shrink_pieces(pos, uv, tri, nrm, piece, diag)
        log(f'  {label}: {len(np.unique(piece[piece >= 0]))} tiny separate pieces (studs) rebuilt on their own: '
            f'{before} -> {int(rebuilt.sum())} vertices')
        # the rebuilt pieces are final: they sit out the edge collapse, which would flatten them
    else:
        rebuilt = np.zeros(len(pos), bool)
    # a model painted in flat colours (a palette: few distinct UVs): its small parts (one colour,
    # under SMALL_PART points: tongue linings, labels) stay as modelled; cheap, and reduced they are
    # the thin layers whose facets cut through the part over them
    tile = np.unique(np.round(uv * 4096).astype(np.int64), axis=0, return_inverse=True)[1].ravel()
    part_size = np.bincount(tile)
    if len(part_size) <= 256:
        whole = (part_size[tile] < SMALL_PART) & ~rebuilt
        if whole.any():
            log(f'  {label}: {int((part_size < SMALL_PART).sum())} small parts kept as modelled ({int(whole.sum())} vertices)')
            rebuilt = rebuilt | whole
    if rebuilt.any():                                   # set aside: they sit out the edge collapse
        stud_t = rebuilt[tri].all(1)
        sv = np.flatnonzero(rebuilt)
        sr = np.full(len(pos), -1); sr[sv] = np.arange(len(sv))
        studs = (pos[sv], uv[sv], sr[tri[stud_t]], nrm[sv])
        rest_v = np.flatnonzero(~rebuilt)
        rr = np.full(len(pos), -1); rr[rest_v] = np.arange(len(rest_v))
        pos, uv, tri, nrm = pos[rest_v], uv[rest_v], rr[tri[~stud_t]], nrm[rest_v]
        max_vertices -= len(sv)
    target = max_vertices * 2
    for _ in range(8):
        p, t, collapses = fs.simplify(pos, tri.astype(np.int32), target_count=int(target), return_collapses=True)
        used = np.unique(t)
        if len(used) <= max_vertices:
            break
        target *= 0.97 * max_vertices / len(used)
    # the replay gives each original vertex the index it ended up as (same triangles, same order);
    # positions come from simplify itself (the replay can place a point far off)
    _, t_replay, mapping = fs.replay_simplification(pos, tri.astype(np.int32), collapses)
    assert np.array_equal(t_replay, t), 'collapse replay does not match the simplified mesh'
    remap = np.full(len(p), -1)
    remap[used] = np.arange(len(used))
    t = remap[t]
    m = remap[np.asarray(mapping)]
    n = len(used)
    p = p[used]
    keep = m >= 0
    new_uv = np.zeros((n, uv.shape[1]))
    new_uv[m[keep]] = uv[keep]                        # one piece, one colour: any of them will do
    near = keep.copy()
    near[keep] = np.linalg.norm(pos[keep] - p[m[keep]], axis=1) < 0.02 * size
    vn = np.zeros((n, 3))
    np.add.at(vn, m[near], nrm[near])                 # the model's normals of the points merged here
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
    fn = normals(p, t)
    bad = (vn * fn).sum(1) < 0.2
    vn[bad] = fn[bad]
    if studs is not None:
        sp, su, st, sn = studs
        t = np.concatenate([t, st + len(p)])
        p, vn, new_uv = np.concatenate([p, sp]), np.concatenate([vn, sn]), np.concatenate([new_uv, su])
    log(f'  {label}: {n0} vertices -> {len(p)} ({len(t)} triangles), by quadric edge collapse')
    return p, fill_normals(vn, p, t), new_uv, t


def fill_normals(vn, pos, tri):
    """Any vertex left without a normal (only collapsed triangles around it) takes its faces', else up."""
    vn = np.asarray(vn, float).copy()
    weak = ~np.isfinite(vn).all(1) | (np.linalg.norm(np.nan_to_num(vn), axis=1) < 1e-6)
    if weak.any():
        fn = normals(pos, tri)
        vn[weak] = fn[weak]
        still = np.linalg.norm(vn, axis=1) < 1e-6
        vn[still] = (0.0, 1.0, 0.0)
    return vn


def unstable_triangles(pos, tri, nrm):
    """Triangles the game would draw wrong once positions are stored as 16-bit floats (its shoe
    vertex format): ones that collapse, or whose facing turns against their normals. On a model
    reduced from millions of points these are sub-millimetre slivers, invisible but turned over."""
    q = pos.astype(np.float16).astype(np.float64)
    a, b, c = q[tri[:, 0]], q[tri[:, 1]], q[tri[:, 2]]
    fn = np.cross(b - a, c - a)
    facing = (fn * (nrm[tri[:, 0]] + nrm[tri[:, 1]] + nrm[tri[:, 2]])).sum(1)
    return (np.linalg.norm(fn, axis=1) < 1e-14) | (facing <= 0)


def sink_covered(pos, nrm, tri, delta, push, log=print, label='mesh'):
    """Moves every point that lies just under another piece of the model (a surface of a different
    piece within `delta` in front of it, along its normal: a trim under a heel cover, a lining
    under a sole) `push` further in. Hidden there anyway, it then keeps clear of the surface over
    it when both are reduced (a reduced surface moves about a millimetre, enough for a layer
    close under it to show through). Uncovered points stay; the push tapers off at the edge of a
    cover. Returns the new positions."""
    import atlas
    piece = atlas.islands(tri, len(pos))
    n = nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    cell = delta / 2
    key3 = np.floor(pos / cell).astype(np.int64)
    base = key3.min(0)
    dims = key3.max(0) - base + 3
    def flat(k):
        k = k - base + 1
        return (k[:, 0] * dims[1] + k[:, 1]) * dims[2] + k[:, 2]
    keys = flat(key3)
    order = np.argsort(keys, kind='stable')
    sk = keys[order]
    covered = np.zeros(len(pos), bool)
    for t in (0.25, 0.5, 0.75, 1.0):
        probe = pos + n * (t * delta)
        pk = flat(np.floor(probe / cell).astype(np.int64))
        lo = np.searchsorted(sk, pk, 'left'); hi = np.searchsorted(sk, pk, 'right')
        cnt = hi - lo
        has = np.flatnonzero((cnt > 0) & ~covered)
        if not len(has):
            continue
        rep = np.repeat(has, cnt[has])
        offs = np.arange(len(rep)) - np.repeat(np.cumsum(cnt[has]) - cnt[has], cnt[has])
        q = order[np.repeat(lo[has], cnt[has]) + offs]
        rel = pos[q] - pos[rep]
        along = (rel * n[rep]).sum(1)
        side = np.linalg.norm(rel - along[:, None] * n[rep], axis=1)
        hit = (piece[q] != piece[rep]) & (along > 0) & (along < delta) & (side < 0.5 * delta) & ((n[q] * n[rep]).sum(1) > 0.3)
        covered[np.unique(rep[hit])] = True
    w = covered.astype(float)
    # taper: a point's push is the share of covered points around it (two rings), so a layer dips in
    # gradually where it slides under its cover
    edges = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    for _ in range(2):
        acc = np.zeros(len(pos)); cntv = np.zeros(len(pos))
        np.add.at(acc, edges[:, 0], w[edges[:, 1]]); np.add.at(cntv, edges[:, 0], 1)
        w = np.where(covered, np.maximum(w, acc / np.maximum(cntv, 1)), acc / np.maximum(cntv, 1) * 0.5)
    small, _ = _small_pieces(pos, tri, float(np.linalg.norm(np.ptp(pos, 0))))
    w[small >= 0] = 0                                   # studs and rivets sit on top: they stay
    covered &= small < 0
    log(f'  {label}: {int(covered.sum())} points lie just under another part; sunk up to {push:.3g} out of reach')
    return pos - n * (push * w)[:, None]
