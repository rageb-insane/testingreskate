"""Turns a pair of shoe meshes (glTF) into skate. shoe geometry: aligned to the
game's own sneaker, skinned by copying the nearest game-shoe vertex's bone
weights, with tangent frames from the model's normals and UVs."""
import numpy as np

import fb
import meshset

BASE_MESH_BUNDLE = ('win32/characters/maincharacters/generic/cas/clothing/unlicensed/generic/footwear/shoe/sneaker/'
                    'vertclassic/2025/gen_sneaker_vertclassic_cas_main_bundlereftable')
BASE_MESH_NAME = ('characters/maincharacters/generic/cas/clothing/unlicensed/generic/footwear/shoe/sneaker/'
                  'vertclassic/2025/gen_sneaker_vertclassic_mesh')
MESHSET_RES = 0x49B156D4
BODY_BUNDLE = ('win32/characters/maincharacters/generic/cas/body/bodyblends/rsp/'
               'cas_rsp_body_complex_dmpreset_cas_main_bundlereftable')
# the game's Nike SB Dunk High: a real high-top's proportions, with the pants fit the game ships for it
DUNK_HIGH_BUNDLE = ('win32/characters/maincharacters/generic/cas/clothing/licensed/nike/footwear/shoe/sneaker/'
                    'sbdunkhigh/2026/nike_sneaker_sbdunkhigh_cas_main_bundlereftable')
HIGH_TOP = 0.15        # a shoe this tall (metres, on the game's foot) is fitted to the Dunk High
# culling regions of make_tuck_mod (djb2-xor of the names): pants triangles under a high-top's collar
# rim, which the high-tops built here cull, and the rest of the pants, which nothing culls
TUCK_REGION = 0xF4CB24E3      # "Socioculture_HighTopTuck"
PANTS_REGION = 0x225E6277     # "Socioculture_Pants"
TUCK_BELOW_RIM = 0.012        # the tucked part ends this far under the collar rim
# a high-top placed as modelled (its collar is not the Dunk's) also culls a region of its own: the
# pants wholly under its own rim that TUCK_REGION leaves (named "Socioculture_Tuck_<shoe>")
OWN_TUCK_PREFIX = 'Socioculture_Tuck_'


class Reference:
    """A game shoe, highest detail: positions and 8-bone skinning per vertex (by default the
    sneaker custom shoes are added to; DUNK_HIGH_BUNDLE with mesh=None for the Dunk High)."""

    def __init__(self, g, bundle=BASE_MESH_BUNDLE, mesh=BASE_MESH_NAME):
        _, bundles, toc_chunks = g.toc('Win32/items.toc')
        toc = {c.guid: c for c in toc_chunks}
        b = [x for x in bundles if x.name == bundle][0]
        files, _ = fb.read_bundle_region(b.region)
        ebx, res, ch, meta = g.manifest(files)
        i = [k for k, a in enumerate(res) if a.res_type == MESHSET_RES and (mesh is None or a.name == mesh)][0]
        self.meshset_bytes = g.payload(files[1 + len(ebx) + i])
        self.meshset = meshset.MeshSet(self.meshset_bytes)
        lod = self.meshset.lods[0]
        c = toc[lod.chunk]
        chunk = fb.decode_cas(g.read(fb.FileInfo(c.patch, c.install_chunk, c.archive, c.offset, c.size)), g.root)
        v = meshset.decode_section(lod, lod.sections[0], chunk)
        self.positions = v['pos'][:, :3]
        self.bones = np.concatenate([v['bone_idx0'], v['bone_idx1']], 1)
        self.weights = np.concatenate([v['bone_w0'], v['bone_w1']], 1)


def _heading(p):
    xz = p[:, [0, 2]] - p[:, [0, 2]].mean(0)
    w, vec = np.linalg.eigh(np.cov(xz.T))
    d = vec[:, np.argmax(w)]
    return d if d[1] >= 0 else -d


def _frame(p):
    """Heel-to-toe axis, its length and the footprint centre."""
    d = _heading(p)
    t = p[:, [0, 2]] @ d
    centre_along = (t.min() + t.max()) / 2
    side = np.array([d[1], -d[0]])
    s = p[:, [0, 2]] @ side
    centre = d * centre_along + side * (s.min() + s.max()) / 2
    return d, t.max() - t.min(), centre


def align(shoe_positions, shoe_normals, reference_positions):
    """Rotate/scale/translate one shoe onto the reference foot (same heading, length, floor, centre)."""
    d_src, len_src, c_src = _frame(shoe_positions)
    d_dst, len_dst, c_dst = _frame(reference_positions)
    angle = np.arctan2(d_dst[0], d_dst[1]) - np.arctan2(d_src[0], d_src[1])
    ca, sa = np.cos(angle), np.sin(angle)
    rot = np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])
    scale = len_dst / len_src
    p = shoe_positions.copy()
    p[:, [0, 2]] -= c_src
    p = (p @ rot.T) * scale
    p[:, 1] += reference_positions[:, 1].min() - p[:, 1].min()
    p[:, [0, 2]] += c_dst
    n = shoe_normals @ rot.T
    return p, n, dict(angle_deg=float(np.degrees(angle)), scale=float(scale))


def level(pos, normals, heel=(0.0, 0.25), ball=(0.55, 0.80), most_deg=8.0, step_deg=0.05):
    """Tips an aligned shoe heel-down or toe-down (about its side-to-side axis, nothing reshaped)
    until its heel and the ball of its foot both rest on the floor, as the game's shoes do: a
    model whose sole rocks (it sat on its forefoot where it was modelled) otherwise touches the
    game's flat foot at the ball only and its heel floats. heel and ball are stretches of the
    length from the heel end. Returns (positions, normals, degrees turned)."""
    head, length, centre = _frame(pos)
    side = np.array([head[1], 0.0, -head[0]])
    pivot = np.array([centre[0], pos[:, 1].min(), centre[1]])

    def turn(p, deg):
        c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
        return p * c + np.cross(side, p) * s + np.outer(p @ side, side) * (1 - c)
    along = (pos[:, [0, 2]] - centre) @ head + length / 2
    in_heel = (along >= heel[0] * length) & (along < heel[1] * length)
    in_ball = (along >= ball[0] * length) & (along < ball[1] * length)
    best = (np.inf, 0.0)
    for deg in np.arange(-most_deg, most_deg + step_deg / 2, step_deg):
        y = turn(pos - pivot, deg)[:, 1]
        gap = abs(y[in_heel].min() - y[in_ball].min())
        if gap < best[0] - 1e-9:
            best = (gap, float(deg))
    deg = best[1]
    out = turn(pos - pivot, deg) + pivot
    out[:, 1] += pivot[1] - out[:, 1].min()
    _, _, moved_centre = _frame(out)
    out[:, [0, 2]] += centre - moved_centre
    return out, turn(normals, deg), deg


def lean_like(pos, normals, target, heights=(0.07, 0.15), rear=0.4, most_deg=6.0, step_deg=0.1):
    """Tips a placed shoe sideways (about its heel-to-toe axis, nothing reshaped) until the back
    of its upper stands over the foot where `target` (the game shoe it is placed on, built
    around the game's ankle) has its own: a model whose upper leans in over its sole shows the
    leg off-centre in the collar and the heel counter slanting (seen from behind). Measured as
    the middle of the rear `rear` of the upper, side to side, at `heights` (metres). The sole
    stays on the floor and in place. Returns (positions, normals, degrees toward the outside)."""
    head, length, centre = _frame(target)
    side = np.array([head[1], 0.0, -head[0]])
    axis = np.array([head[0], 0.0, head[1]])
    rows = np.arange(heights[0], heights[1] + 1e-6, 0.02)

    def middle(p):
        t = (p[:, [0, 2]] - centre) @ head + length / 2
        s = p[:, [0, 2]] @ side[[0, 2]]
        out = []
        for y in rows:
            m = (np.abs(p[:, 1] - y) < 0.006) & (t < rear * length)
            out.append((s[m].max() + s[m].min()) / 2 if m.sum() > 5 else np.nan)
        return np.array(out)
    want = middle(target)
    pivot = np.array([centre[0], pos[:, 1].min(), centre[1]])

    def turn(p, deg):
        c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
        return p * c + np.cross(axis, p) * s + np.outer(p @ axis, axis) * (1 - c)

    def placed(deg):                   # about the floor under the footprint's centre: the sole stays put
        q = turn(pos - pivot, deg) + pivot
        q[:, 1] += pivot[1] - q[:, 1].min()
        return q
    best = (np.inf, 0.0)
    for deg in np.arange(-most_deg, most_deg + step_deg / 2, step_deg):
        off = abs(np.nanmean(middle(placed(deg)) - want))
        if off < best[0] - 1e-9:
            best = (off, float(deg))
    deg = best[1]
    out = placed(deg)
    up = out[:, 1] > np.percentile(out[:, 1], 90)
    outward = np.sign((np.abs(out[up, 0]).mean() - np.abs(pos[up, 0]).mean()) or 1.0)
    return out, turn(normals, deg), float(outward * abs(deg))


def fit_inside(pos, reference, start=0.07, ramp=0.015, margin=0.002, height_step=0.005, angle_step=10):
    """Keeps the upper inside the game sneaker's outline around the ankle, so pants
    shaped for that sneaker are not poked through. Works in cylindrical coordinates
    about the ankle axis: above `start` metres, any point farther out than the game
    shoe's surface at that height and angle (minus `margin`) is pulled in, blending
    in over `ramp` metres. Returns (positions, number of points moved)."""
    top = reference[:, 1].max()
    axis = reference[reference[:, 1] > top - 0.02][:, [0, 2]].mean(0)

    def cylinder(p):
        v = p[:, [0, 2]] - axis
        return np.hypot(v[:, 0], v[:, 1]), np.arctan2(v[:, 0], v[:, 1]), v

    rb, ab, _ = cylinder(reference)
    hb = np.floor(reference[:, 1] / height_step).astype(int)
    sb = np.floor((np.degrees(ab) % 360) / angle_step).astype(int)
    n_sectors = 360 // angle_step
    limit = {}
    for h, s, r in zip(hb, sb, rb):
        if r > limit.get((h, s), 0):
            limit[(h, s)] = r
    out = pos.copy()
    r, a, v = cylinder(pos)
    moved = 0
    for i in np.flatnonzero(pos[:, 1] > start):
        h = int(np.floor(pos[i, 1] / height_step))
        s = int((np.degrees(a[i]) % 360) // angle_step)
        # the widest the game shoe gets in this and the neighbouring cells; walk down
        # if the Dior is taller than the sneaker there
        bound = 0
        for hh in range(h, h - 6, -1):
            for ds in (-1, 0, 1):
                bound = max(bound, limit.get((hh, (s + ds) % n_sectors), 0))
            if bound:
                break
        if not bound:
            continue
        allowed = bound - margin
        if r[i] > allowed:
            w = min(1.0, (pos[i, 1] - start) / ramp)
            new_r = r[i] + (allowed - r[i]) * w
            out[i, [0, 2]] = axis + v[i] * (new_r / r[i])
            moved += 1
    return out, moved


def fit_inside_smooth(pos, reference, start=0.07, ramp=0.015, margin=0.002, height_step=0.005, angle_step=10):
    """fit_inside for bulky collars and tall tongues (a Yeezy 2), where pulling points in one
    by one squashes the outer wall onto the outline while the lining stays put, and the
    collar crumples. On a grid of heights x angles about the ankle axis, every point of a cell
    is drawn toward the axis by the same fraction (walls keep their thickness order, the
    lining never crosses the outer wall, nothing folds), by a smooth field: enough to bring the cell's outermost point onto the game shoe's surface
    minus `margin`. Above the sneaker's top (a taller collar or tongue) the highest outline
    below carries on up. Starts at `start` metres, blending in over `ramp`.
    Returns (positions, number of points moved)."""
    top = reference[:, 1].max()
    axis = reference[reference[:, 1] > top - 0.02][:, [0, 2]].mean(0)
    n_s = 360 // angle_step
    n_h = int(np.ceil((max(top, pos[:, 1].max()) + 0.01) / height_step))
    outline = _polar_max(reference, axis, n_h, n_s, height_step, angle_step)
    filled = _fill_sideways(outline)
    for k in range(1, n_h):                       # taller than the sneaker: its top outline carries on up
        filled[k] = np.where(filled[k] > 0, filled[k], filled[k - 1])
    r, h, a, v = _polar(pos, axis, n_h, n_s, height_step, angle_step)
    outer = np.zeros((n_h, n_s))
    np.maximum.at(outer, (h.astype(int), a.astype(int) % n_s), r)
    allowed = filled - margin
    shrink = np.where((outer > allowed) & (allowed > 0), 1 - allowed / np.maximum(outer, 1e-9), 0)
    for _ in range(2):                            # widen each peak first, so smoothing does not shave it
        shrink = np.maximum.reduce([shrink, np.roll(shrink, 1, 1), np.roll(shrink, -1, 1),
                                    np.vstack([shrink[:1], shrink[:-1]]), np.vstack([shrink[1:], shrink[-1:]])])
    shrink = _smooth(shrink, 4)
    f = _sample(shrink, h, a) * np.clip((pos[:, 1] - start) / ramp, 0, 1)
    moved = f > 1e-4
    out = pos.copy()
    out[:, 0] = axis[0] + v[:, 0] * (1 - f)
    out[:, 2] = axis[1] + v[:, 1] * (1 - f)
    return out, int(moved.sum())


def match_outline(pos, target, normals=None, start=0.06, ramp=0.02, height_step=0.005, angle_step=10, least=0.7, most=1.4):
    """Shapes a high-top's ankle like the game's Dunk High (`target`, one foot): every height x
    angle ring of the upper is scaled about the ankle axis so its outer wall follows the
    Dunk's, flared parts drawn in and narrow ones let out, by a smooth field (each ring as a
    whole, so walls keep their order). Above the Dunk's top its top outline carries on up.
    Starts at `start` metres, blending in over `ramp`. Returns (positions, number moved)."""
    top = target[:, 1].max()
    axis = target[target[:, 1] > top - 0.02][:, [0, 2]].mean(0)
    n_s = 360 // angle_step
    n_h = int(np.ceil((max(top, pos[:, 1].max()) + 0.01) / height_step))
    want = _fill_sideways(_polar_max(target, axis, n_h, n_s, height_step, angle_step))
    for k in range(1, n_h):
        want[k] = np.where(want[k] > 0, want[k], want[k - 1])
    r, h, a, v = _polar(pos, axis, n_h, n_s, height_step, angle_step)
    # a ring is measured by its outward-facing surface (the outer wall): rows that hold only the
    # lining (it folds over the rim into the shoe, facing in) would read as a narrow wall, be let
    # out and come through the outer one; they take the wall's measure from around them instead
    wall = np.ones(len(pos), bool)
    if normals is not None:
        wall = (normals[:, 0] * v[:, 0] + normals[:, 2] * v[:, 1]) > 0.2 * np.maximum(r, 1e-9)
    outer = np.zeros((n_h, n_s))
    np.maximum.at(outer, (h[wall].astype(int), a[wall].astype(int) % n_s), r[wall])
    have = _fill_sideways(outer)
    f = np.where((have > 0) & (want > 0), np.clip(want / np.maximum(have, 1e-9), least, most), 1.0)
    f = 1 + _smooth(f - 1, 3)
    scale = 1 + (_sample(f, h, a) - 1) * np.clip((pos[:, 1] - start) / ramp, 0, 1)
    out = pos.copy()
    out[:, 0] = axis[0] + v[:, 0] * scale
    out[:, 2] = axis[1] + v[:, 1] * scale
    return out, int((np.abs(scale - 1) > 1e-3).sum())


def _polar(p, axis, n_h, n_s, height_step, angle_step):
    """Radius, height cell (float), angle cell (float) and offset from the ankle axis."""
    v = p[:, [0, 2]] - axis
    r = np.hypot(v[:, 0], v[:, 1])
    a = (np.degrees(np.arctan2(v[:, 0], v[:, 1])) % 360) / angle_step
    return r, np.clip(p[:, 1] / height_step, 0, n_h - 1), a, v


def _polar_max(p, axis, n_h, n_s, height_step, angle_step):
    r, h, a, _ = _polar(p, axis, n_h, n_s, height_step, angle_step)
    grid = np.zeros((n_h, n_s))
    np.maximum.at(grid, (h.astype(int), a.astype(int) % n_s), r)
    return grid


def _fill_sideways(grid):
    """Cells where a surface has nothing at some angle and height borrow from neighbours."""
    filled = grid.copy()
    for _ in range(4):
        grow = np.maximum.reduce([filled, np.roll(filled, 1, 1), np.roll(filled, -1, 1),
                                  np.vstack([filled[:1], filled[:-1]]), np.vstack([filled[1:], filled[-1:]])])
        filled = np.where(filled > 0, filled, grow)
    return filled


def _sample(grid, h, a):
    """Bilinear sample of a height x angle grid at cell coordinates (angle wraps)."""
    n_h, n_s = grid.shape
    h0 = np.clip(np.floor(h - 0.5).astype(int), 0, n_h - 1)
    h1 = np.clip(h0 + 1, 0, n_h - 1)
    th = np.clip(h - 0.5 - h0, 0, 1)
    a0 = np.floor(a - 0.5).astype(int) % n_s
    a1 = (a0 + 1) % n_s
    ta = (a - 0.5) % 1
    return ((grid[h0, a0] * (1 - ta) + grid[h0, a1] * ta) * (1 - th) +
            (grid[h1, a0] * (1 - ta) + grid[h1, a1] * ta) * th)


LEGWEAR = ('_pants_', '_overalls_', '_leggings', '_costumebottom_', 'cas_rsp_body')   # not full-body costumes


def lower_leg_garments(g, top):
    """Rest-pose vertices near the ankles (below `top` metres) of the game's legwear: every
    pair of pants, overalls, leggings and costume bottoms, and the body itself. LOD0."""
    _, bundles, toc_chunks = g.toc('Win32/items.toc')
    toc = {c.guid: c for c in toc_chunks}
    out = []
    for b in bundles:
        name = b.name.split('/')[-1].replace('_complex_dmpreset_cas_main_bundlereftable', '')
        if not b.name.endswith('_complex_dmpreset_cas_main_bundlereftable') or not any(k in name for k in LEGWEAR):
            continue
        files, _ = fb.read_bundle_region(b.region)
        ebx, res, _, _ = g.manifest(files)
        for i, a in enumerate(res):
            if a.res_type != MESHSET_RES:
                continue
            ms = meshset.MeshSet(g.payload(files[1 + len(ebx) + i]))
            lod = ms.lods[0]
            c = toc.get(lod.chunk)
            if c is None:
                continue
            chunk = fb.decode_cas(g.read(fb.FileInfo(c.patch, c.install_chunk, c.archive, c.offset, c.size)), g.root)
            p = np.concatenate([meshset.decode_section(lod, s, chunk)['pos'][:, :3] for s in lod.sections])
            low = p[(p[:, 1] < top) & (np.abs(np.abs(p[:, 0]) - 0.17) < 0.12)]
            if len(low):
                out.append((name, low))
    return out


def _smooth(grid, passes):
    """Box blur over height (clamped) and angle (wrapping)."""
    for _ in range(passes):
        up = np.vstack([grid[:1], grid[:-1]])
        down = np.vstack([grid[1:], grid[-1:]])
        grid = (grid * 2 + up + down + np.roll(grid, 1, 1) + np.roll(grid, -1, 1)) / 6
    return grid


def fit_around_garments(pos, reference, garments, start=0.09, ramp=0.02, clearance=0.006, margin=0.002,
                        most=0.045, front_most=0.010, height_step=0.005, angle_step=10, thin_padding=False):
    """Widens the ankle collar so tight legwear passes inside it, as it does into the game
    sneaker's padded collar (the game hides nothing there: skinny jeans end 11 cm up, inside
    the shoe). On a grid of heights x angles about the ankle axis, every garment the game
    sneaker swallows (its surface inside the sneaker's outline) sets a floor `clearance`
    outside itself. Each cell's points all move out together (walls keep their thickness, the
    lining never crosses the outer wall) by a smooth field: enough to bring the innermost
    surface to the floor, but at most `most`, and only as far as the outer wall has room
    before the sneaker's outline minus `margin`, so loose pants still drape over. At the front
    (tongue) the push is capped at `front_most`, easing to `most` at the back: a widened tongue
    and quarter make the shoe look wide from the front (the user compared it with photos of the
    real pair), while the heel can bulge unseen; only loose, wide-hem pants poke through more.
    thin_padding (for collars fit_inside_smooth has drawn in to the outline, which leaves the
    outer wall no room): the lining moves the full amount, the outer wall only as far as it
    has room, the points between by their place across the wall.
    Returns (positions, number of points moved)."""
    top = reference[:, 1].max()
    axis = reference[reference[:, 1] > top - 0.02][:, [0, 2]].mean(0)
    n_s = 360 // angle_step
    n_h = int(np.ceil((max(top, pos[:, 1].max()) + 0.01) / height_step))
    outline = _polar_max(reference, axis, n_h, n_s, height_step, angle_step)
    filled = _fill_sideways(outline)
    side = np.sign(reference[:, 0].mean())
    floor = np.zeros((n_h, n_s))
    for _, gp in garments:
        gp = gp[np.sign(gp[:, 0]) == side]
        if len(gp):
            gr = _polar_max(gp, axis, n_h, n_s, height_step, angle_step)
            tucked = (gr > 0) & (gr < filled - margin)
            floor = np.where(tucked, np.maximum(floor, gr + clearance), floor)
    floor[:int(start / height_step)] = 0
    r, h, a, v = _polar(pos, axis, n_h, n_s, height_step, angle_step)
    cell = (h.astype(int), a.astype(int) % n_s)
    inner = np.full((n_h, n_s), np.inf)
    np.minimum.at(inner, cell, r)
    outer = np.zeros((n_h, n_s))
    np.maximum.at(outer, cell, r)
    toward_back = (1 - np.cos(np.radians((np.arange(n_s) + 0.5) * angle_step))) / 2   # 0 at the toe side, 1 at the heel
    most = front_most + (most - front_most) * toward_back
    need = np.where((floor > 0) & np.isfinite(inner), np.clip(floor - inner, 0, most), 0)
    # every point of a cell moves by the same amount (the lining stays behind the outer
    # wall), so the cap is the room the outer wall has before the sneaker's outline
    room = _smooth(np.clip(filled - margin - outer, 0, most), 2)
    # widen each need to its neighbours first, so smoothing does not shave the peaks
    for _ in range(3):
        need = np.maximum.reduce([need, np.roll(need, 1, 1), np.roll(need, -1, 1),
                                  np.vstack([need[:1], need[:-1]]), np.vstack([need[1:], need[-1:]])])
    push_out = _sample(_smooth(np.minimum(need, room), 6), h, a)
    if not thin_padding:
        d = push_out * np.clip((pos[:, 1] - (start - ramp)) / ramp, 0, 1)
        return _move_out(pos, axis, r, v, d)
    # the outer wall moves only as far as it has room; the lining may move further (the padding
    # gets thinner), each point in between by its place across the wall, so nothing crosses
    push_in = _sample(_smooth(need, 6), h, a)
    seen = np.isfinite(inner)
    inner_s = _sample(_smooth(_fill_sideways(np.where(seen, inner, 0)), 2), h, a)
    outer_s = _sample(_smooth(_fill_sideways(outer), 2), h, a)
    wall = np.maximum(outer_s - inner_s, 1e-3)
    push_in = np.minimum(np.maximum(push_in, push_out), push_out + np.maximum(wall - 0.002, 0))
    t = np.clip((r - inner_s) / wall, 0, 1)
    d = (push_in * (1 - t) + push_out * t) * np.clip((pos[:, 1] - (start - ramp)) / ramp, 0, 1)
    return _move_out(pos, axis, r, v, d)


def _move_out(pos, axis, r, v, d):
    """Moves every point `d` farther from the ankle axis. Returns (positions, number moved)."""
    new_r = r + d
    moved = (d > 1e-5) & (r > 1e-6)
    out = pos.copy()
    scale = np.where(moved, new_r / np.maximum(r, 1e-9), 1.0)
    out[:, 0] = axis[0] + v[:, 0] * scale
    out[:, 2] = axis[1] + v[:, 1] * scale
    return out, int(moved.sum())


def legwear_showing(positions, reference, garments, start=0.09, margin=0.002, height_step=0.005, angle_step=10):
    """Clipping check: of the (garment, height, angle) cells where legwear tucks inside the
    game sneaker's outline, those where it lies outside this shoe's outer wall (it would show
    through). The game's own sneaker scores 0. Returns (showing, total)."""
    showing = total = 0
    for side in (1, -1):
        foot = reference[np.sign(reference[:, 0]) == side]
        shoe = positions[np.sign(positions[:, 0]) == side]
        top = foot[:, 1].max()
        axis = foot[foot[:, 1] > top - 0.02][:, [0, 2]].mean(0)

        def outer(p):
            v = p[:, [0, 2]] - axis
            keys = zip((p[:, 1] // height_step).astype(int),
                       ((np.degrees(np.arctan2(v[:, 0], v[:, 1])) % 360) // angle_step).astype(int))
            out = {}
            for k, r in zip(keys, np.hypot(v[:, 0], v[:, 1])):
                out[k] = max(out.get(k, 0), r)
            return out
        outline, wall = outer(foot), outer(shoe)
        for _, gp in garments:
            for k, r in outer(gp[np.sign(gp[:, 0]) == side]).items():
                if k[0] * height_step < start or k not in wall or k not in outline or r >= outline[k] - margin:
                    continue
                total += 1
                showing += r > wall[k]
    return showing, total


def nearest(src, dst, chunk=2048):
    """Index of the nearest dst point for every src point (brute force, chunked)."""
    out = np.empty(len(src), dtype=np.int64)
    dd = (dst * dst).sum(1)
    for i in range(0, len(src), chunk):
        s = src[i:i + chunk]
        dist = dd[None, :] - 2 * s @ dst.T
        out[i:i + chunk] = np.argmin(dist, 1)
    return out


def tangents(positions, normals, uv, triangles):
    """Per-vertex tangent (orthogonalised to the normal) and bitangent from UV derivatives."""
    v0, v1, v2 = (positions[triangles[:, k]] for k in range(3))
    w0, w1, w2 = (uv[triangles[:, k]] for k in range(3))
    e1, e2 = v1 - v0, v2 - v0
    d1, d2 = w1 - w0, w2 - w0
    r = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]
    r = np.where(np.abs(r) < 1e-12, 1e-12, r)
    sdir = (e1 * d2[:, 1:2] - e2 * d1[:, 1:2]) / r[:, None]
    tdir = (e2 * d1[:, 0:1] - e1 * d2[:, 0:1]) / r[:, None]
    tan = np.zeros_like(positions)
    bit = np.zeros_like(positions)
    for k in range(3):
        np.add.at(tan, triangles[:, k], sdir)
        np.add.at(bit, triangles[:, k], tdir)
    n = normals
    t = tan - (tan * n).sum(1, keepdims=True) * n
    bad = np.linalg.norm(t, axis=1) < 1e-9
    if bad.any():   # no UV gradient: any perpendicular will do
        helper = np.where(np.abs(n[bad, 0:1]) < 0.9, [[1, 0, 0]], [[0, 1, 0]])
        t[bad] = np.cross(n[bad], helper)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    b = np.cross(n, t)
    sign = np.where((b * bit).sum(1) < 0, -1.0, 1.0)
    return t, b * sign[:, None]


class Shoe:
    """Both feet merged into one vertex/triangle set, ready to encode. high_top: fitted to the
    Dunk High, so its item also culls TUCK_REGION (pants tucked under its collar)."""

    def __init__(self, positions, normals, uv, triangles, bones, weights, high_top=False):
        self.positions, self.normals, self.uv, self.triangles = positions, normals, uv, triangles
        self.bones, self.weights = bones, weights
        self.high_top = high_top
        self.tangents, self.bitangents = tangents(positions, normals, uv, triangles)


TUCK_AT_MOST = 0.05    # a collar the classic tuck-in would move more of than this is left as modelled


def blend_skin(b1, w1, b2, w2, t):
    """Per vertex (1 - t) of one skinning plus t of another: the 8 strongest influences of both,
    weights summing to 255 as the game's do."""
    bones, weights = b1.copy(), w1.copy()
    for i in np.flatnonzero(t > 0):
        acc = {}
        for b, w, f in ((b1[i], w1[i], 1 - t[i]), (b2[i], w2[i], t[i])):
            for bb, ww in zip(b, w):
                if ww:
                    acc[int(bb)] = acc.get(int(bb), 0.0) + float(ww) * f
        top = sorted(acc.items(), key=lambda kv: -kv[1])[:bones.shape[1]]
        total = sum(w for _, w in top)
        q = np.array([w / total * 255 for _, w in top])
        r = np.floor(q).astype(int)
        r[np.argsort(-(q - r))[:255 - r.sum()]] += 1             # round so they still add up to 255
        bones[i] = 0
        weights[i] = 0
        bones[i, :len(top)] = [b for b, _ in top]
        weights[i, :len(top)] = r
    return bones, weights


LEG_SKIN_RAMP = 0.01   # under the Dunk High's collar rim, the shoe's skinning turns into the leg's over this


def skin_above_rim(positions, bones, weights, shape, leg, side):
    """A high-top placed as modelled is skinned by copying the nearest Dunk High vertex, but where
    its collar or tongue rises above the Dunk's (the back of an Air Jordan 1's padded collar) the
    nearest Dunk vertex is the Dunk's rim below, which follows the foot more than the leg: when
    the ankle bends, the pants (skinned like the body) push through it. Above the Dunk's rim at
    that angle, the skinning becomes the body's own there (nearest body vertex), blending in over
    LEG_SKIN_RAMP under it. Only how those points follow the ankle changes, not their shape.
    Returns (bones, weights, points changed)."""
    axis, rim = collar_rim(shape, side)
    v = positions[:, [0, 2]] - axis
    sector = ((np.degrees(np.arctan2(v[:, 0], v[:, 1])) % 360) // (360 // len(rim))).astype(int)
    t = np.clip((positions[:, 1] - (rim[sector] - LEG_SKIN_RAMP)) / LEG_SKIN_RAMP, 0, 1)
    if not (t > 0).any():
        return bones, weights, 0
    body = (leg.positions[:, 0] * side > 0) & (leg.positions[:, 1] < 0.4)
    sel = np.flatnonzero(t > 0)
    near = np.flatnonzero(body)[nearest(positions[sel], leg.positions[body])]
    b2 = np.zeros_like(bones)
    w2 = np.zeros_like(weights)
    b2[sel], w2[sel] = leg.bones[near], leg.weights[near]
    bones, weights = blend_skin(bones, weights, b2, w2, t)
    return bones, weights, len(sel)


def skin_collar_to_leg(positions, bones, weights, rims, leg, depth=0.03, ramp=0.02):
    """A shoe's collar follows the leg like the pants over it do: points within `depth` under the
    shoe's own collar rim (rims: {side: (ankle axis, rim height per angle sector)}) take the body's
    skinning there (nearest body point), blending in over `ramp`. Pants are skinned like the body,
    so a collar skinned like the foot slides under them when the ankle bends (the jeans' hem rides
    down over the heel). Only how those points follow the bones changes, not their shape.
    Returns (bones, weights, points changed)."""
    t = np.zeros(len(positions))
    for side, (axis, rim) in rims.items():
        on = positions[:, 0] * side > 0
        v = positions[on][:, [0, 2]] - axis
        sector = ((np.degrees(np.arctan2(v[:, 0], v[:, 1])) % 360) // (360 / len(rim))).astype(int)
        t[on] = np.clip((positions[on, 1] - (rim[sector] - depth)) / ramp, 0, 1)
    sel = np.flatnonzero(t > 0)
    if not len(sel):
        return bones, weights, 0
    b2, w2 = np.zeros_like(bones), np.zeros_like(weights)
    for side in (1, -1):
        part = sel[positions[sel, 0] * side > 0]
        if not len(part):
            continue
        body = np.flatnonzero((leg.positions[:, 0] * side > 0) & (leg.positions[:, 1] < 0.4))
        near = body[nearest(positions[part], leg.positions[body])]
        b2[part], w2[part] = leg.bones[near], leg.weights[near]
    bones, weights = blend_skin(bones, weights, b2, w2, t)
    return bones, weights, len(sel)


def build(prims, reference, garments=(), log=print, collar='classic', shape=None, as_modelled=False, leg=None,
          adjust=True, base_shoe=None, base_name=None):
    """glTF primitives (a pair, or one shoe to mirror) -> Shoe on the game's feet. A high-top
    (HIGH_TOP or taller) with `shape` (the Dunk High Reference) is placed, shaped at the ankle
    and skinned like the game's Dunk High, so it wears pants the way that shoe does. Anything
    else: its ankle fitted between the game sneaker's outline and the legwear in `garments`.
    as_modelled: nothing reshaped, the shoe only turned, scaled to the game's foot and set level
    on the floor (heel and ball down), then skinned like the game shoe it is placed on (the
    Dunk High for a high-top when `shape` is given, else the sneaker); with `leg` (the body
    Reference, BODY_BUNDLE) a high-top's points above the Dunk's collar follow the leg instead.
    adjust=False (with as_modelled): only turned, scaled and set on the floor, no tipping or
    leaning, skinned by the nearest game-shoe vertex alone (a first look in game).
    base_shoe (with as_modelled): the game shoe (Reference) to place on and skin from instead,
    whatever the height (named base_name in the log); still a high-top for tucking if tall."""
    ref = reference.positions
    feet = {+1: ref[:, 0] > 0, -1: ref[:, 0] < 0}
    by_side = {}
    for p in prims:                                  # by the middle of its box: shoe_prep puts it at x = +-0.15
        side = 1 if p.positions[:, 0].min() + p.positions[:, 0].max() > 0 else -1
        by_side.setdefault(side, []).append(p)
    if len(by_side) == 1:   # a single shoe: mirror it for the other foot
        side = next(iter(by_side))
        mirrored = []
        for p in by_side[side]:
            q = type(p)(**p.__dict__)
            q.positions = p.positions * [-1, 1, 1]
            q.normals = p.normals * [-1, 1, 1]
            q.indices = p.indices[:, ::-1]
            mirrored.append(q)
        by_side[-side] = mirrored
        log('  one shoe in the model: mirrored it for the other foot')
    parts, high_top = [], False
    for side, plist in by_side.items():
        pos = np.concatenate([p.positions for p in plist])
        nrm = np.concatenate([p.normals for p in plist])
        uv = np.concatenate([p.uv for p in plist])
        tri, base = [], 0
        for p in plist:
            tri.append(p.indices + base)
            base += len(p.positions)
        tri = np.concatenate(tri)
        foot = ref[feet[side]]
        name = f'{"right" if side < 0 else "left"} foot (x{"<" if side < 0 else ">"}0)'
        if as_modelled:
            on, src = feet[side], reference
            if base_shoe is not None:
                on, src = base_shoe.positions[:, 0] * side > 0, base_shoe
                if align(pos, nrm, base_shoe.positions[on])[0][:, 1].max() >= HIGH_TOP:
                    high_top = True
            elif shape is not None:
                on_dunk = shape.positions[:, 0] * side > 0
                if align(pos, nrm, shape.positions[on_dunk])[0][:, 1].max() >= HIGH_TOP:
                    on, src, high_top = on_dunk, shape, True
            target = src.positions[on]
            src_name = base_name if src is base_shoe else ('Dunk High' if src is shape else 'sneaker')
            apos, anrm, info = align(pos, nrm, target)
            if not adjust:
                log(f"  {name}: as modelled, on the game's {src_name}: only turned "
                    f'{info["angle_deg"]:+.1f} deg and scaled x{info["scale"]:.3f} to the foot, set on the floor')
                idx = np.flatnonzero(on)[nearest(apos, target)]
                parts.append((apos, anrm, uv, tri, src.bones[idx], src.weights[idx]))
                continue
            apos, anrm, tipped = level(apos, anrm)
            apos, anrm, leaned = lean_like(apos, anrm, target)
            apos, anrm, retipped = level(apos, anrm)
            log(f"  {name}: as modelled, on the game's {src_name}: turned "
                f'{info["angle_deg"]:+.1f} deg, scaled x{info["scale"]:.3f}, tipped {tipped + retipped:+.2f} deg so heel '
                f'and ball rest on the floor, leaned {leaned:+.1f} deg outward so the upper stands over the ankle, '
                'nothing reshaped')
            idx = np.flatnonzero(on)[nearest(apos, target)]
            bones, weights = src.bones[idx], src.weights[idx]
            if src is shape and leg is not None:
                bones, weights, moved = skin_above_rim(apos, bones, weights, shape, leg, side)
                if moved:
                    log(f"  {name}: {moved} collar and tongue points above the Dunk High's collar follow the leg "
                        "(skinned like the body there, as pants are)")
            parts.append((apos, anrm, uv, tri, bones, weights))
            continue
        if shape is not None:
            on_dunk = shape.positions[:, 0] * side > 0
            dunk = shape.positions[on_dunk]
            dpos, dnrm, dinfo = align(pos, nrm, dunk)
            if dpos[:, 1].max() >= HIGH_TOP:
                dpos, reshaped = match_outline(dpos, dunk, dnrm)
                log(f"  {name}: a high-top, shaped like the game's Dunk High: turned {dinfo['angle_deg']:+.1f} deg, "
                    f'scaled x{dinfo["scale"]:.3f}, {reshaped} ankle points reshaped')
                idx = np.flatnonzero(on_dunk)[nearest(dpos, dunk)]
                parts.append((dpos, dnrm, uv, tri, shape.bones[idx], shape.weights[idx]))
                high_top = True
                continue
        pos, nrm, info = align(pos, nrm, foot)
        smooth = collar == 'smooth'                # opt-in: the user preferred the classic fit on the Dior
        tucked, moved = (fit_inside_smooth if smooth else fit_inside)(pos, foot)
        if not smooth and moved > TUCK_AT_MOST * len(pos):
            log(f'  {"right" if side < 0 else "left"} foot: collar left as modelled (tucking {moved} points into '
                "the game sneaker's outline would crumple the tongue and collar)")
            moved = 0
        else:
            pos = tucked
        pos, widened = fit_around_garments(pos, foot, garments, thin_padding=smooth)
        log(f'  {name}: '
            f'turned {info["angle_deg"]:+.1f} deg, scaled x{info["scale"]:.3f}, '
            f'{moved} ankle points pulled inside the game shoe\'s outline, '
            f'{widened} pushed out to clear tight legwear')
        idx = np.flatnonzero(feet[side])[nearest(pos, foot)]
        parts.append((pos, nrm, uv, tri, reference.bones[idx], reference.weights[idx]))
    pos = np.concatenate([p[0] for p in parts])
    nrm = np.concatenate([p[1] for p in parts])
    uv = np.concatenate([p[2] for p in parts])
    offset = np.cumsum([0] + [len(p[0]) for p in parts])
    tri = np.concatenate([p[3] + offset[i] for i, p in enumerate(parts)])
    bones = np.concatenate([p[4] for p in parts])
    weights = np.concatenate([p[5] for p in parts])
    if len(pos) > 65535:
        raise SystemExit(f'{len(pos)} vertices: the game indexes a mesh with 16 bits (max 65535). Reduce the model first.')
    return Shoe(pos, nrm / np.linalg.norm(nrm, axis=1, keepdims=True), uv, tri, bones, weights, high_top)


def collar_rim(shape, side, angle_step=10):
    """The Dunk High's collar rim on one foot: (ankle axis xz, rim height per angle sector),
    angle measured about the axis from the toe (+z) as in _polar."""
    foot = shape.positions[shape.positions[:, 0] * side > 0]
    top = foot[:, 1].max()
    axis = foot[foot[:, 1] > top - 0.02][:, [0, 2]].mean(0)
    v = foot[:, [0, 2]] - axis
    sector = ((np.degrees(np.arctan2(v[:, 0], v[:, 1])) % 360) // angle_step).astype(int)
    rim = np.zeros(360 // angle_step)
    np.maximum.at(rim, sector, foot[:, 1])
    rim = np.minimum.reduce([rim, np.roll(rim, 1), np.roll(rim, -1)])     # the lower of each neighbour
    return axis, rim


def own_collar_rims(positions, shape, angle_step=10):
    """A shoe's own collar rim per foot ({side: (ankle axis xz, rim height per angle sector)}),
    about the Dunk High's ankle axis and in its sectors, as collar_rim: make_tuck_mod hides the
    pants wholly under it for a high-top placed as modelled, whose collar is not the Dunk's."""
    out = {}
    for side in (1, -1):
        axis, _ = collar_rim(shape, side, angle_step)
        foot = positions[positions[:, 0] * side > 0]
        v = foot[:, [0, 2]] - axis
        sector = ((np.degrees(np.arctan2(v[:, 0], v[:, 1])) % 360) // angle_step).astype(int)
        rim = np.zeros(360 // angle_step)
        np.maximum.at(rim, sector, foot[:, 1])
        out[side] = (axis, np.minimum.reduce([rim, np.roll(rim, 1), np.roll(rim, -1)]))
    return out
