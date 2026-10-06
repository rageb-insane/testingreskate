"""Puts a logo on the chest of a black crop top: a ReSkate mod that adds it to skate. as a new top.

    python make_top_mod.py logo.png --name "Wonderland Crop Top" [--size 0.42] [--install]

The logo is printed the way the game prints its own chest logos: the crop top's graphic slot
(param_graphic0_*), which lays a 2:1 colour+opacity texture over the shirt through the mesh's
second UV set (tile 0 = the front panel), centred on the body's centre line. A transparent PNG
works best; an image without transparency has its background colour (taken from the corners)
keyed out.

The new top is a copy of the game's black Vans crop crewneck colorway (the one colorway of this
shirt with a chest print): its appearance preset with the print swapped for the logo, renumbered,
in that colorway's bundle, which no other installed mod may ship (ReSkate would merge it and drop
the texture's metadata); plus the item, thumbnails, item-list entry and bundle-reference row in
the shared bundle. The thumbnail is the game's own render of that shirt with its print painted
out and the logo drawn through the same projection.
"""
import argparse
import hashlib
import json
import os
import random
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import uuid
import zipfile

import cv2
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fb  # noqa: E402
import bundleref  # noqa: E402
import meshset  # noqa: E402
from make_deck_mod import (EBXTOOL, DEFAULT_GAME, SHARED_BUNDLE, BUNDLE_REF_TABLE, PATCH_DIRECTORY, TEXTURE_RES,  # noqa: E402
                           ITEM_COLLECTION, TEMPLATE, djb, guid_bytes, new_guid, ebxtool, ebx_info, edit_ebx, bc7,
                           texture_header, CasWriter, slugify)
from make_shoe_mod import bundle_ref, bundles_of_other_mods, earlier_builds, ebx_file_guid  # noqa: E402
from shoe_mesh import MESHSET_RES  # noqa: E402

BASE = 'vans_shirt_cropcrewneck_00001'
COLORWAY_BUNDLE = ('win32/characters/maincharacters/generic/cas/clothing/licensed/vans/apparel/top/shirt/cropcrewneck/'
                   f'2023/colorways/{BASE}_ap_cas_main_bundlereftable')
BASE_ITEM = 'items/cust_tops/own_topshirt_vans_cropcrewneck_00001'
BASE_THUMB = 'thumbnail/tool/own_topshirt_vans_cropcrewneck_00001_lrg'
MESH_BUNDLE = '/gen_shirt_cropcrewneck_complex_dmpreset_cas_main_bundlereftable'
LOGO_W, LOGO_H = 1024, 512     # 2:1 like the game's chest prints (param_graphic0_aspect 2)
INK_WIDTH, INK_HEIGHT = 0.94, 0.80   # most of the texture the logo may fill
DEFAULT_SIZE = 0.42            # texture width in the shirt's print UVs (the Vans print is 0.373)
MIN_FACING = 0.75              # every part of the print must face the front this much (normal z)
THUMB, THUMB_LARGE = 256, 768


def read(path):
    with open(path, 'rb') as f:
        return f.read()


# ------------------------------------------------------------------ the shirt
def shirt_front(g):
    """The crop top's front panel in print-UV space (uv1 tile 0): per texel of a 1024x1024 grid,
    the 3D point and normal under it (NaN off the panel), and the u of the body's centre line."""
    _, bundles, toc_chunks = g.toc('Win32/items.toc')
    toc = {c.guid: c for c in toc_chunks}
    b = next(x for x in bundles if x.name.endswith(MESH_BUNDLE))
    files, _ = fb.read_bundle_region(b.region)
    ebx, res, chunks, _ = g.manifest(files)
    for a, f in zip(res, files[1 + len(ebx):]):
        if a.res_type != MESHSET_RES:
            continue
        lod = meshset.MeshSet(g.payload(f)).lods[0]
        c = toc[lod.chunk]
        data = fb.decode_cas(g.read(fb.FileInfo(c.patch, c.install_chunk, c.archive, c.offset, c.size)), g.root)
        v = meshset.decode_section(lod, lod.sections[0], data)
        break
    else:
        raise SystemExit('The crop top mesh was not found in this skate. build.')
    p, t, uv = v['pos'][:, :3], v['triangles'], v['uv1']
    n = np.zeros_like(p)
    fn = np.cross(p[t[:, 1]] - p[t[:, 0]], p[t[:, 2]] - p[t[:, 0]])
    for k in range(3):
        np.add.at(n, t[:, k], fn)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    N = 1024
    P = np.full((N, N, 3), np.nan)
    NN = np.full((N, N, 3), np.nan)
    for tri in t:
        if (uv[tri, 0] >= 1).any():
            continue
        xs, ys = uv[tri, 0] * N, uv[tri, 1] * N
        x0, x1 = max(int(xs.min()), 0), min(int(xs.max()) + 1, N - 1)
        y0, y1 = max(int(ys.min()), 0), min(int(ys.max()) + 1, N - 1)
        d = (xs[1] - xs[0]) * (ys[2] - ys[0]) - (xs[2] - xs[0]) * (ys[1] - ys[0])
        if abs(d) < 1e-12:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        w1 = ((gx - xs[0]) * (ys[2] - ys[0]) - (xs[2] - xs[0]) * (gy - ys[0])) / d
        w2 = ((xs[1] - xs[0]) * (gy - ys[0]) - (gx - xs[0]) * (ys[1] - ys[0])) / d
        w0 = 1 - w1 - w2
        inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
        for out, src in ((P, p), (NN, n)):
            val = w0[..., None] * src[tri[0]] + w1[..., None] * src[tri[1]] + w2[..., None] * src[tri[2]]
            out[y0:y1 + 1, x0:x1 + 1][inside] = val[inside]
    centre = (uv[:, 0] < 1) & (np.abs(p[:, 0]) < 0.005) & (p[:, 2] > 0.05)
    return P, NN, float(np.median(uv[centre, 0]))


def print_coverage(front, alpha, offh, offv, scale, aspect):
    """Where the print's ink lands: the share of it off the front panel, and the least it faces
    the front (normal z) where it is on it."""
    P, NN, _ = front
    N = P.shape[0]
    ys, xs = np.nonzero(alpha > 0.25)
    u = ((xs + 0.5) / alpha.shape[1] - 0.5) * scale + 0.5 + offh
    v = ((ys + 0.5) / alpha.shape[0] - 0.5) * scale / aspect + 0.5 - offv
    iu, iv = np.clip((u * N).astype(int), 0, N - 1), np.clip((v * N).astype(int), 0, N - 1)
    pos, nrm = P[iv, iu], NN[iv, iu]
    on = ~np.isnan(pos[:, 0]) & (u >= 0) & (u < 1) & (v >= 0) & (v < 1)
    return 1 - on.mean(), float(nrm[on, 2].min()) if on.any() else -1.0


# ------------------------------------------------------------------ the logo
def load_logo(path):
    """The logo centred on a transparent 1024x512 texture, as large as fits."""
    im = Image.open(path)
    im.load()
    a = np.asarray(im.convert('RGBA')).copy()
    if a[..., 3].min() > 250:                     # no transparency: key out the background
        corners = np.concatenate([a[:4, :4], a[:4, -4:], a[-4:, :4], a[-4:, -4:]]).reshape(-1, 4)[:, :3]
        bg = np.median(corners.astype(np.float32), 0)
        dist = np.abs(a[..., :3].astype(np.float32) - bg).max(2)
        a[..., 3] = (np.clip((dist - 12) / 40, 0, 1) * 255).astype(np.uint8)
    im = Image.fromarray(a, 'RGBA')
    box = im.getchannel('A').point(lambda x: 255 if x > 8 else 0).getbbox()
    if not box:
        raise SystemExit('The logo image is empty (fully transparent).')
    im = im.crop(box)
    w = LOGO_W * INK_WIDTH
    h = w * im.height / im.width
    if h > LOGO_H * INK_HEIGHT:
        h = LOGO_H * INK_HEIGHT
        w = h * im.width / im.height
    w, h = max(1, round(w)), max(1, round(h))
    canvas = Image.new('RGBA', (LOGO_W, LOGO_H), (0, 0, 0, 0))
    canvas.alpha_composite(im.resize((w, h), Image.LANCZOS), ((LOGO_W - w) // 2, (LOGO_H - h) // 2))
    return canvas


def _bleed(rgba):
    """Transparent texels take the colour of the nearest ink, so filtering never pulls in black."""
    a = rgba[..., 3:]
    pm = rgba[..., :3] * a
    rgb = np.where(a > 1 / 255, pm / np.maximum(a, 1e-6), 0)
    have = (a > 1 / 255).astype(np.float32)[..., 0]
    for sigma in (1, 2, 4, 8, 16, 32, 64):
        if have.min() > 0:
            break
        bw = cv2.GaussianBlur(have, (0, 0), sigma)
        br = cv2.GaussianBlur(np.ascontiguousarray(rgb * have[..., None]), (0, 0), sigma)
        fill = (have == 0) & (bw > 1e-3)
        rgb[fill] = br[fill] / bw[fill][:, None]
        have = np.maximum(have, fill.astype(np.float32))
    if have.min() == 0:
        rgb[have == 0] = rgb[have > 0].mean(0) if (have > 0).any() else 0
    return np.dstack([rgb, a])


def logo_mips(canvas, work):
    """BC7 mips, largest first: averaged with premultiplied alpha, colour bled into the clear."""
    a = np.asarray(canvas, np.float32) / 255
    pm = np.dstack([a[..., :3] * a[..., 3:], a[..., 3:]])
    mips, level = [], 0
    while True:
        h, w = pm.shape[:2]
        straight = np.dstack([pm[..., :3] / np.maximum(pm[..., 3:], 1e-6), pm[..., 3:]])
        img = Image.fromarray((np.clip(_bleed(straight), 0, 1) * 255 + 0.5).astype(np.uint8), 'RGBA')
        mips.append(bc7(img, work, f'logo_mip{level}'))
        if w == 1 and h == 1:
            return mips
        h2, w2 = max(1, h // 2), max(1, w // 2)
        pm = pm.reshape(h2, h // h2, w2, w // w2, 4).mean((1, 3))
        level += 1


def logo_header(template, name, chunk_guid, mips, offsets):
    """The logo's texture resource, from the base print's header: size, mips, the whole texture
    in the bundle (first mip 0), its chunk and its own name hash."""
    header = bytearray(template)
    sizes = [len(m) for m in mips]
    struct.pack_into('<II', header, 0, *offsets)
    struct.pack_into('<HH', header, 22, LOGO_W, LOGO_H)
    header[30], header[31] = len(mips), 0
    header[40:56] = chunk_guid
    struct.pack_into('<15I', header, 56, *(sizes + [0] * (15 - len(sizes))))
    struct.pack_into('<I', header, 116, sum(sizes))
    struct.pack_into('<Q', header, 120, djb(name, 64))
    return bytes(header)


# ------------------------------------------------------------------ preset values
def preset_values(dump, param_names):
    """{shader parameter name: (offset in the document, value bytes)} of a preset's boxed values."""
    imports = {int(i): f for i, f in re.findall(r'^import #(\d+) file=(\S+) inst=', dump, re.M)}
    out = {}
    for block in re.split(r'\n\s+\[\d+\] AppearanceExpressionParamInfo', dump)[1:]:
        p = re.search(r'Parameter = import\(#(\d+)', block)
        v = re.search(r'Value = boxed\(\d+ at=(\d+) bytes=([0-9a-f]*)\)', block)
        if p and v:
            out[param_names.get(imports[int(p.group(1))], '?')] = (int(v.group(1)), bytes.fromhex(v.group(2)))
    return out


# ------------------------------------------------------------------ thumbnails
def decode_texture(g, toc, header, work, label, bundle_chunks=()):
    """Mip 0 of a BC7 texture as RGBA (its chunk from the TOC, else whole in its bundle)."""
    w, h = struct.unpack_from('<HH', header, 22)
    size0 = struct.unpack_from('<I', header, 56)[0]
    c = toc.get(header[40:56])
    if c is not None:
        data = fb.decode_cas(g.read(fb.FileInfo(c.patch, c.install_chunk, c.archive, c.offset, c.size)), g.root)
    else:
        hit = [(a, f) for a, f in bundle_chunks if a.guid == header[40:56] and a.logical_offset == 0]
        if not hit:
            raise SystemExit(f'{label}: texture data not found')
        data = g.payload(hit[0][1])
    src, dst = os.path.join(work, label + '.bc7'), os.path.join(work, label + '.rgba')
    with open(src, 'wb') as f:
        f.write(data[:size0])
    ebxtool('bc7dec', src, w, h, dst)
    return np.frombuffer(read(dst), np.uint8).reshape(h, w, 4).astype(np.float32)


def print_map(src_wh, src, dst_wh, dst):
    """Pixel map from one print texture to another through the shirt's print UVs.
    src/dst = (offset horizontal, offset vertical, scale, aspect)."""
    def to_uv(wh, p):
        offh, offv, s, a = p
        return np.array([[s / wh[0], 0, -0.5 * s + 0.5 + offh], [0, s / a / wh[1], -0.5 * s / a + 0.5 - offv], [0, 0, 1]])
    return np.linalg.inv(to_uv(dst_wh, dst)) @ to_uv(src_wh, src)


def thumbnail(base_thumb, base_print, base_p, canvas, logo_p):
    """The game's render of the base shirt with its print painted out and the logo drawn through
    the same projection (fitted to the print in the render), lit like the old print."""
    T = base_thumb
    S = T.shape[0]
    lum = T[..., :3].mean(2)
    roi = np.zeros(lum.shape, bool)
    roi[int(0.30 * S):int(0.68 * S), int(0.22 * S):int(0.83 * S)] = True
    bright = ((lum > 120) & roi).astype(np.float32)
    tmpl = base_print[..., 3] / 255
    ys, xs = np.nonzero(bright)
    iy, ix = np.nonzero(tmpl > 0.3)
    src = np.float32([[ix.min(), iy.min()], [ix.max(), iy.min()], [ix.max(), iy.max()], [ix.min(), iy.max()]])
    dst = np.float32([[xs.min(), ys.min()], [xs.max(), ys.min()], [xs.max(), ys.max()], [xs.min(), ys.max()]])
    H = cv2.getPerspectiveTransform(src, dst).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 500, 1e-7)
    fit, H = cv2.findTransformECC(cv2.GaussianBlur(tmpl.astype(np.float32), (0, 0), 1.0),
                                  cv2.GaussianBlur(bright, (0, 0), 1.5), H, cv2.MOTION_HOMOGRAPHY, crit, None, 5)
    if fit < 0.85:
        raise SystemExit(f'Could not find the print in the base thumbnail (fit {fit:.2f}).')
    warped = cv2.warpPerspective(tmpl.astype(np.float32), H, (S, S))
    near = cv2.dilate((warped > 0.02).astype(np.uint8), np.ones((9, 9), np.uint8))
    hole = cv2.dilate(((near > 0) & ((warped > 0.02) | (lum > 70))).astype(np.uint8), np.ones((3, 3), np.uint8))
    rgb = np.ascontiguousarray(T[..., 2::-1].astype(np.uint8))
    clean = cv2.inpaint(rgb, hole * 255, 6, cv2.INPAINT_TELEA)[..., ::-1].astype(np.float32)
    # the light on the print: the render over the print's own colour, where it was solid
    ink = warped > 0.95
    print_rgb = cv2.warpPerspective(np.ascontiguousarray(base_print[..., :3]), H, (S, S))
    ratio = np.where(ink, lum / np.maximum(print_rgb.mean(2), 1), 0).astype(np.float32)
    weight = ink.astype(np.float32)
    light = cv2.GaussianBlur(ratio, (0, 0), 25) / np.maximum(cv2.GaussianBlur(weight, (0, 0), 25), 1e-4)
    far = cv2.GaussianBlur(ratio, (0, 0), 90) / np.maximum(cv2.GaussianBlur(weight, (0, 0), 90), 1e-4)
    light = np.where(cv2.GaussianBlur(weight, (0, 0), 25) > 0.02, light, far)
    # the logo, drawn at twice the size and brought down for clean edges
    C = np.asarray(canvas, np.float32) / 255
    pm = np.dstack([C[..., :3] * C[..., 3:], C[..., 3:]])
    A = print_map((canvas.width, canvas.height), logo_p, (base_print.shape[1], base_print.shape[0]), base_p)
    big = cv2.warpPerspective(pm, np.diag([2, 2, 1]) @ H.astype(np.float64) @ A, (2 * S, 2 * S), flags=cv2.INTER_LINEAR)
    small = cv2.resize(big, (S, S), interpolation=cv2.INTER_AREA)
    a = small[..., 3:]
    colour = np.clip(small[..., :3] / np.maximum(a, 1e-4) * 255 * light[..., None], 0, 255)
    out = clean * (1 - a) + colour * a
    return Image.fromarray(np.dstack([np.clip(out, 0, 255), T[..., 3]]).astype(np.uint8), 'RGBA')


# ------------------------------------------------------------------ build
def build(args):
    game = args.game
    key_words = slugify(args.name)
    key = 'Own_' + key_words
    ident = 'own_' + key_words.lower()
    author = re.sub(r'[^A-Za-z0-9_]', '_', args.author).strip('_') or 'socioculture'
    package = re.sub(r'[^A-Za-z0-9_]', '_', args.package or key_words)
    folder_name = f'{author}-{package}'
    out_root = os.path.abspath(args.out)
    mod_dir = os.path.join(out_root, folder_name)
    rng = random.SystemRandom()

    print(f'Crop top "{args.name}"  key={key}  mod={folder_name}')
    g = fb.GameData(game)
    patch_chunk = next(k for k, v in g.chunk_dirs.items() if v == PATCH_DIRECTORY)
    _, base_bundles, base_toc_chunks = g.toc('Win32/items.toc')
    by_name = {b.name: b for b in base_bundles}
    toc_base = {c.guid: c for c in base_toc_chunks}
    if COLORWAY_BUNDLE not in by_name:
        raise SystemExit('The black Vans crop crewneck colorway is not in this skate. build.')
    others = bundles_of_other_mods(game, folder_name).get(COLORWAY_BUNDLE)
    if others:
        raise SystemExit(f'Another installed mod ships the crop top colorway bundle ({", ".join(others)}); '
                         'ReSkate would merge the two and drop the logo texture.')

    work = tempfile.mkdtemp(prefix='topmod_')
    try:
        def wpath(f):
            return os.path.join(work, f)

        # ---- the base colorway: its preset, print texture and the names of its parameters
        files, _ = fb.read_bundle_region(by_name[COLORWAY_BUNDLE].region)
        b_ebx, b_res, b_chunks, b_meta = g.manifest(files)
        b_ebx_files = files[1:1 + len(b_ebx)]
        b_res_files = files[1 + len(b_ebx):1 + len(b_ebx) + len(b_res)]
        b_chunk_files = files[1 + len(b_ebx) + len(b_res):]
        param_names, tmpl = {}, {}
        for a, f in zip(b_ebx, b_ebx_files):
            data = g.payload(f)
            param_names[ebx_file_guid(data)] = a.name.split('/')[-1]
            if a.name.endswith('/' + BASE + '_ap'):
                tmpl['preset'] = (a.name, data)
        preset_path = wpath('preset_tmpl.ebx')
        with open(preset_path, 'wb') as f:
            f.write(tmpl['preset'][1])
        pinfo = ebx_info(preset_path)
        values = preset_values(pinfo['dump'], param_names)
        need = ('param_usegraphic', 'param_graphic0_co', 'param_graphic0_scale', 'param_graphic0_aspect',
                'param_graphic0_offsethorizontal', 'param_graphic0_offsetvertical')
        if any(n not in values for n in need):
            raise SystemExit(f'{BASE}_ap: no chest print in this build ({sorted(values)})')
        f32 = {n: struct.unpack_from('<f', values[n][1])[0] for n in need[2:]}
        print_import = struct.unpack_from('<Q', values['param_graphic0_co'][1])[0] >> 1
        print_file, print_inst = pinfo['imports'][print_import]
        for a, f in zip(b_ebx, b_ebx_files):
            if param_names.get(print_file) == a.name.split('/')[-1] and ebx_file_guid(g.payload(f)) == print_file:
                tmpl['texture'] = (a.name, g.payload(f))
        for a, f in zip(b_res, b_res_files):
            if a.name == tmpl['texture'][0] and a.res_type == TEXTURE_RES:
                tmpl['texture_res'] = (a, g.payload(f))
        base_print = decode_texture(g, toc_base, tmpl['texture_res'][1], work, 'base_print', zip(b_chunks, b_chunk_files))

        # ---- placement: centred on the body, the base print's height, the chosen width
        front = shirt_front(g)
        offh = front[2] - 0.5
        offv, aspect = f32['param_graphic0_offsetvertical'], f32['param_graphic0_aspect']
        if abs(aspect - LOGO_W / LOGO_H) > 1e-3:
            raise SystemExit(f'The base print is {aspect}:1, expected 2:1.')
        canvas = load_logo(args.image)
        alpha = np.asarray(canvas)[..., 3].astype(np.float32) / 255
        off, facing = print_coverage(front, alpha, offh, offv, args.size, aspect)
        print(f'  logo: {args.size:.3f} wide (the Vans print is {f32["param_graphic0_scale"]:.3f}), '
              f'centre line u={front[2]:.4f}; off the front {off:.1%}, faces the front >= {facing:.2f}')
        if off > 0 or facing < MIN_FACING:
            raise SystemExit('At this size the logo would wrap round the sides of the shirt; use a smaller --size.')

        # ---- names
        stem = tmpl['preset'][0].rsplit('/', 1)[0]
        taken = {a.name for b in (b_ebx,) for a in b}
        sb_files, _ = fb.read_bundle_region(by_name[SHARED_BUNDLE].region)
        s_ebx, s_res, s_chunks, s_meta = g.manifest(sb_files)
        ne, nr = len(s_ebx), len(s_res)
        br = next(i for i, a in enumerate(s_res) if a.name == BUNDLE_REF_TABLE)
        table = bundleref.Table(g.payload(sb_files[1 + ne + br]), s_res[br].res_meta)
        number = next(f'{n:05d}' for n in iter(lambda: rng.randrange(70000, 99999), None)
                      if f'{stem}/{BASE[:-5]}{n:05d}_ap' not in table.presets and f'{stem}/{BASE[:-5]}{n:05d}_ap' not in taken)
        names = {'preset': f'{stem}/{BASE[:-5]}{number}_ap',
                 'texture': f'characters/materials/logo/custom/{ident}_co',
                 'item': f'items/cust_tops/{ident}',
                 'thumb_tool': f'thumbnail/tool/{ident}', 'thumb_tool_lrg': f'thumbnail/tool/{ident}_lrg',
                 'thumb_cdn': f'thumbnail/cdn/img_{ident}', 'thumb_cdn_lrg': f'thumbnail/cdn/img_{ident}_lrg'}
        if any(n in {a.name for a in s_ebx} for n in (names['item'], names['thumb_tool'])):
            raise SystemExit(f'The game already has an item named {key}; pick another name.')
        print(f'  colorway bundle: {BASE}_ap (the new preset is number {number})')

        # ---- art: the print texture (whole in the bundle) and thumbnails
        mips = logo_mips(canvas, work)
        total = sum(len(m) for m in mips)
        th_res = [f for a, f in zip(s_res, sb_files[1 + ne:1 + ne + nr]) if a.name == BASE_THUMB]
        if not th_res:
            raise SystemExit(f'{BASE_THUMB} is not in the shared bundle')
        base_thumb = decode_texture(g, toc_base, g.payload(th_res[0]), work, 'base_thumb',
                                    zip(s_chunks, sb_files[1 + ne + nr:]))
        large = thumbnail(base_thumb, base_print,
                          (f32['param_graphic0_offsethorizontal'], offv, f32['param_graphic0_scale'], aspect),
                          canvas, (offh, offv, args.size, aspect))
        thumbs = {'thumb_tool_lrg': large, 'thumb_tool': large.resize((THUMB, THUMB), Image.LANCZOS)}
        thumb_bc7 = {lab: bc7(im, work, lab) for lab, im in thumbs.items()}

        ids = {lab: {'file': new_guid(), 'inst': new_guid()} for lab in
               ('preset', 'texture', 'thumb_tool', 'thumb_tool_lrg', 'thumb_cdn', 'thumb_cdn_lrg')}
        res_ids = {lab: rng.getrandbits(64) | 1 for lab in ('texture', 'thumb_tool', 'thumb_tool_lrg')}
        chunk_ids = {lab: uuid.uuid4().bytes for lab in ('texture', 'thumb_tool', 'thumb_tool_lrg')}
        cas = CasWriter(patch_chunk, game, compress=not args.no_compress)
        ebx, res = {}, {}

        tex_tmpl = wpath('texture.ebx')
        with open(tex_tmpl, 'wb') as f:
            f.write(tmpl['texture'][1])
        ebx['texture'] = edit_ebx(tex_tmpl, wpath('texture_new.ebx'), '--file-guid', ids['texture']['file'],
                                  '--inst-guid', 0, ids['texture']['inst'],
                                  '--set', '0:Name', 'str:Characters/Materials/Logo/Custom/' + key + '_CO',
                                  '--set', '0:Resource', f'res:{res_ids["texture"]}',
                                  '--set', '0:CropInfo.Z', f'u64:{LOGO_W}', '--set', '0:CropInfo.W', f'u64:{LOGO_H}')
        parts = [cas.encode(m) for m in mips]
        stream = b''.join(parts)
        res['texture'] = logo_header(tmpl['texture_res'][1], names['texture'], chunk_ids['texture'], mips,
                                     (len(parts[0]), len(parts[0]) + len(parts[1])))

        # ---- the preset, byte for byte (ReSkate's writer drops boxed values): new ids, the
        #      logo in place of the print, its size and centre, renumbered
        data = bytearray(tmpl['preset'][1])
        for old, new in ((pinfo['file'], ids['preset']['file']), (pinfo['instances'][0], ids['preset']['inst']),
                         (print_file, ids['texture']['file']), (print_inst, ids['texture']['inst'])):
            assert data.count(guid_bytes(old)) == 1, f'preset: {old} should appear once'
            data = data.replace(guid_bytes(old), guid_bytes(new))
        for name, value in (('param_graphic0_scale', args.size), ('param_graphic0_offsethorizontal', offh)):
            struct.pack_into('<f', data, values[name][0], value)
        old_text = ('CropCrewNeck_' + BASE[-5:]).encode()
        assert data.count(old_text) == 1, 'preset: its name should hold the colorway number once'
        ebx['preset'] = bytes(data.replace(old_text, ('CropCrewNeck_' + number).encode()))

        # ---- thumbnails and the item (shared bundle)
        def tpath(f):
            return os.path.join(TEMPLATE, f)
        for lab in ('thumb_tool', 'thumb_tool_lrg'):
            ebx[lab] = edit_ebx(tpath(lab + '.ebx'), wpath(lab + '.ebx'), '--file-guid', ids[lab]['file'],
                                '--inst-guid', 0, ids[lab]['inst'], '--set', '0:Name', 'str:' + names[lab],
                                '--set', '0:Resource', f'res:{res_ids[lab]}')
            res[lab] = texture_header(read(tpath(lab + '.res')), names[lab], chunk_ids[lab], len(thumb_bc7[lab]))
        for lab, tool in (('thumb_cdn', 'thumb_tool'), ('thumb_cdn_lrg', 'thumb_tool_lrg')):
            ebx[lab] = edit_ebx(tpath(lab + '.ebx'), wpath(lab + '.ebx'), '--file-guid', ids[lab]['file'],
                                '--inst-guid', 0, ids[lab]['inst'], '--import', 0, ids[tool]['file'], ids[tool]['inst'],
                                '--set', '0:Name', 'str:' + names[lab], '--set', '0:NameHash', f'u64:{djb(names[lab])}',
                                '--set', '0:ContentHash', 'sha1:' + hashlib.sha1(thumb_bc7[tool]).hexdigest())
        sb_ebx_files, sb_res_files, sb_chunk_files = sb_files[1:1 + ne], sb_files[1 + ne:1 + ne + nr], sb_files[1 + ne + nr:]
        ii = next((i for i, a in enumerate(s_ebx) if a.name == BASE_ITEM), None)
        if ii is None:
            raise SystemExit(f'{BASE_ITEM} is not in the shared bundle')
        with open(wpath('item_tmpl.ebx'), 'wb') as f:
            f.write(g.payload(sb_ebx_files[ii]))
        iinfo = ebx_info(wpath('item_tmpl.ebx'))
        idump = iinfo['dump']
        thumb_slot = int(re.search(r'\n\s+Thumbnail = import\(#(\d+)', idump).group(1))
        large_slot = int(re.search(r'\n\s+ThumbnailLarge = import\(#(\d+)', idump).group(1))
        paths = re.findall(r'\[(\d+)\] DingoAssetPathInfo \{\s+AssetTypeId = (\d+)\s+AssetName = "([^"]+)"', idump)
        slot = [int(k) for k, t, n in paths if t == '3']
        if len(slot) != 1:
            raise SystemExit(f'{BASE_ITEM}: expected one appearance preset path ({paths})')
        preset_leaf = re.search(r'Name = "([^"]+)"', ebx_info_bytes(ebx['preset'], work)).group(1).rsplit('/', 1)[1]
        item_ops = ['--file-guid', new_guid()]
        for k in range(len(iinfo['instances'])):
            item_ops += ['--inst-guid', k, new_guid()]
        ebx['item'] = edit_ebx(wpath('item_tmpl.ebx'), wpath('item.ebx'), *item_ops,
                               '--set', '0:Name', 'str:' + names['item'], '--set', '0:Key', 'str:' + key,
                               '--set', '0:HashedAssetKey', f'u64:{djb(key)}',
                               '--import', thumb_slot, ids['thumb_cdn']['file'], ids['thumb_cdn']['inst'],
                               '--import', large_slot, ids['thumb_cdn_lrg']['file'], ids['thumb_cdn_lrg']['inst'],
                               '--set', f'0:ItemData.AssetPaths[{slot[0]}].AssetName', 'str:' + preset_leaf)
        item_info = ebx_info(wpath('item.ebx'))

        ic = next(i for i, a in enumerate(s_ebx) if a.name == ITEM_COLLECTION)
        with open(wpath('collection.ebx'), 'wb') as f:
            f.write(g.payload(sb_ebx_files[ic]))
        ebxtool('additem', wpath('collection.ebx'), wpath('collection_new.ebx'), item_info['file'], item_info['instances'][0])
        collection = read(wpath('collection_new.ebx'))
        fi, sha = cas.add(collection)
        s_ebx[ic].sha1, s_ebx[ic].original_size = sha, len(collection)
        sb_ebx_files[ic] = fi
        table.insert(names['preset'], table.bundle_index(bundle_ref(COLORWAY_BUNDLE)))
        fi, sha = cas.add(bytes(table.data))
        s_res[br].sha1, s_res[br].original_size, s_res[br].res_meta = sha, len(table.data), bytes(table.meta)
        sb_res_files[br] = fi

        def add_ebx(labels):
            assets, fis = [], []
            for lab in labels:
                fi, sha = cas.add(ebx[lab])
                assets.append(fb.Asset(kind='ebx', name=names[lab], sha1=sha, original_size=len(ebx[lab])))
                fis.append(fi)
            return assets, fis

        def add_res(labels, meta):
            assets, fis = [], []
            for lab in labels:
                fi, sha = cas.add(res[lab])
                assets.append(fb.Asset(kind='res', name=names[lab], sha1=sha, original_size=len(res[lab]),
                                       res_type=TEXTURE_RES, res_meta=meta, res_id=res_ids[lab]))
                fis.append(fi)
            return assets, fis
        e_new, e_files = add_ebx(['item', 'thumb_tool', 'thumb_tool_lrg', 'thumb_cdn', 'thumb_cdn_lrg'])
        r_new, r_files = add_res(['thumb_tool', 'thumb_tool_lrg'], bytes.fromhex('0c000000090000000000000000000000'))
        toc_chunks = []
        for lab in ('thumb_tool', 'thumb_tool_lrg'):
            fi, _ = cas.add(thumb_bc7[lab])
            toc_chunks.append(fb.TocChunk(chunk_ids[lab], True, patch_chunk, 1, fi.offset, fi.size))
        manifest = fb.write_binary_bundle(s_ebx + e_new, s_res + r_new, s_chunks, s_meta)
        shared = fb.TocBundle(SHARED_BUNDLE, fb.write_bundle_region(
            [cas.add_raw(manifest)] + sb_ebx_files + e_files + sb_res_files + r_files + sb_chunk_files))

        # ---- the colorway bundle: + preset, logo texture (asset, resource, whole chunk)
        ce, ce_files = add_ebx(['preset', 'texture'])
        cr, cr_files = add_res(['texture'], tmpl['texture_res'][0].res_meta)
        fi = cas.add_raw(stream)
        toc_chunks.append(fb.TocChunk(chunk_ids['texture'], True, patch_chunk, 1, fi.offset, fi.size))
        cc = [fb.Asset(kind='chunk', name=fb.guid_str(chunk_ids['texture']), guid=chunk_ids['texture'],
                       sha1=hashlib.sha1(stream).digest(), logical_offset=0, logical_size=total)]
        tree, _ = fb.read_db(b_meta, 0)
        tree['children'].append({'type': 2, 'name': None, 'terminated': True, 'children': [
            {'type': 9, 'name': 'h64', 'value': struct.pack('<Q', djb(names['texture'], 64))},
            {'type': 2, 'name': 'meta', 'terminated': True, 'children': [
                {'type': 8, 'name': 'firstMip', 'value': struct.pack('<i', 0)}]}]})
        manifest = fb.write_binary_bundle(b_ebx + ce, b_res + cr, b_chunks + cc, fb.write_db(tree))
        colorway = fb.TocBundle(COLORWAY_BUNDLE, fb.write_bundle_region(
            [cas.add_raw(manifest)] + b_ebx_files + ce_files + b_res_files + cr_files + b_chunk_files + [fi]))
        toc = fb.write_patch_toc([shared, colorway], toc_chunks, 3)

        # ---- mod folder
        if os.path.exists(mod_dir):
            shutil.rmtree(mod_dir)
        cas_dir = os.path.join(mod_dir, 'Win32', *PATCH_DIRECTORY.split('/'))
        os.makedirs(cas_dir)
        with open(os.path.join(cas_dir, 'cas_01.cas'), 'wb') as f:
            f.write(cas.data)
        with open(os.path.join(mod_dir, 'Win32', 'items.toc'), 'wb') as f:
            f.write(toc)
        layout = read(os.path.join(game, 'Data', 'layout.toc'))
        with open(os.path.join(mod_dir, 'layout.toc'), 'wb') as f:
            f.write(layout[:4] + bytes(fb.TOC_ENVELOPE - 4) + layout[fb.TOC_ENVELOPE:])
        shutil.copyfile(os.path.join(game, 'Data', 'initfs_Win32'), os.path.join(mod_dir, 'initfs_win32'))
        skate_sha = hashlib.sha256(read(os.path.join(game, 'Skate.exe'))).hexdigest()
        with open(os.path.join(mod_dir, '.reskate-studio-patch'), 'w', newline='\n') as f:
            f.write(f'ReSkate Studio native Patch v1\nskate_sha256={skate_sha}\n')
        with open(os.path.join(mod_dir, 'manifest.json'), 'w') as f:
            json.dump({'author': author, 'dependencies': [], 'description': f'{args.name}: a black crop top with a chest logo',
                       'name': package, 'version_number': args.version, 'website_url': ''}, f, indent=4)
        with open(os.path.join(mod_dir, 'README.md'), 'w', encoding='utf-8') as f:
            f.write(f'# {args.name}\n\nAdds "{args.name}", a black crop top with a chest logo, to skate. (ReSkate). '
                    'Find it under tops.\n')
        thumbs['thumb_tool'].save(os.path.join(mod_dir, 'icon.png'))
        canvas.save(os.path.join(out_root, f'{folder_name}-logo.png'))
        thumbs['thumb_tool_lrg'].save(os.path.join(out_root, f'{folder_name}-thumbnail.png'))
        zip_path = os.path.join(out_root, f'{folder_name}.zip')
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
            for dirpath, _, fs in os.walk(mod_dir):
                for n in fs:
                    full = os.path.join(dirpath, n)
                    z.write(full, os.path.relpath(full, mod_dir))
        print(f'  wrote {mod_dir}\n  wrote {zip_path}')

        if not args.no_verify:
            r = subprocess.run([sys.executable, os.path.join(HERE, 'verify_top_mod.py'), mod_dir, '--game', game],
                               capture_output=True, text=True)
            print(r.stdout[-3000:])
            if r.returncode:
                print(r.stderr)
                raise SystemExit('Verification failed; the mod was not installed.')
        if args.install:
            running = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq Skate.exe'], capture_output=True, text=True).stdout
            if 'skate.exe' in running.lower():
                print('  skate. is running: close it, then copy the mod folder into Mods (or run this again).')
            else:
                for old in earlier_builds(game, folder_name):
                    shutil.rmtree(os.path.join(game, 'Mods', old))
                    print(f'  removed the earlier build {old}')
                target = os.path.join(game, 'Mods', folder_name)
                if os.path.exists(target):
                    shutil.rmtree(target)
                shutil.copytree(mod_dir, target)
                print(f'  installed to {target}')
        return mod_dir
    finally:
        shutil.rmtree(work, ignore_errors=True)


def ebx_info_bytes(data, work):
    path = os.path.join(work, 'probe.ebx')
    with open(path, 'wb') as f:
        f.write(data)
    return ebx_info(path)['dump']


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('image', help='logo image (PNG with transparency works best)')
    p.add_argument('--name', required=True, help='top name shown in game')
    p.add_argument('--size', type=float, default=DEFAULT_SIZE,
                   help=f'logo texture width across the chest (default {DEFAULT_SIZE}; the game\'s Vans print is 0.373)')
    p.add_argument('--author', default='socioculture', help='mod author (folder prefix)')
    p.add_argument('--package')
    p.add_argument('--version', default='1.0.0')
    p.add_argument('--game', default=DEFAULT_GAME)
    p.add_argument('--out', default=os.path.join(HERE, '..', 'mods'))
    p.add_argument('--install', action='store_true')
    p.add_argument('--no-compress', action='store_true')
    p.add_argument('--no-verify', action='store_true')
    build(p.parse_args())


if __name__ == '__main__':
    main()
