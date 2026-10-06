"""Recolours the game's skinny jeans: a ReSkate mod that adds them to skate. as a new pair.

    python make_jeans_mod.py --name "Black Skinny Jeans" [--color black] [--install]

The game's only skinny jeans (gen_pants_jeansskinny_00004, light grey acid wash) carry their
colour in the colorway's own colour texture (a two-slice texture array: legs and fly / back and
pockets), not in tint values. The new pair gets a copy of that texture with the denim recoloured:
the wash, whiskers, creases and seams kept as shading around the new colour, the button and
pocket lining left alone (the region mask tells them apart). "black" is black denim (#222327),
never flat black. The small Extravert logo on the back is switched off.

Like the other builders: the new appearance preset and texture go into that colorway's bundle,
which no other installed mod may ship (ReSkate would merge it and drop the texture's metadata),
with the texture whole in the bundle (first mip 0); the item, thumbnails, item-list entry and
bundle-reference row go into the shared bundle. The thumbnail is the game's own render of the
jeans, recoloured inside a GrabCut mask of the jeans.
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
from PIL import Image, ImageColor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fb  # noqa: E402
import bundleref  # noqa: E402
from make_deck_mod import (DEFAULT_GAME, SHARED_BUNDLE, BUNDLE_REF_TABLE, PATCH_DIRECTORY, TEXTURE_RES,  # noqa: E402
                           ITEM_COLLECTION, TEMPLATE, djb, guid_bytes, new_guid, ebxtool, ebx_info, edit_ebx, bc7,
                           texture_header, CasWriter, slugify)
from make_shoe_mod import bundle_ref, bundles_of_other_mods, earlier_builds, ebx_file_guid  # noqa: E402
from make_top_mod import preset_values, decode_texture  # noqa: E402

BASE = 'gen_pants_jeansskinny_00004'
COLORWAY_BUNDLE = ('win32/characters/maincharacters/generic/cas/clothing/unlicensed/generic/apparel/bottom/pants/'
                   f'jeansskinny/2026/colorways/{BASE}_ap_cas_main_bundlereftable')
BASE_ITEM = 'items/cust_bottoms/own_bottompants_gen_jeansskinny_00004'
BASE_THUMB = 'thumbnail/tool/own_bottompants_gen_jeansskinny_00004_lrg'
BLACK_DENIM = (34, 35, 39)       # what "black" means for denim
MIN_LUMA = 30                    # darker than this, denim reads as a hole
DETAIL = 0.5                     # how much of the wash, whiskers and creases survive (relative)
LIT = 1.45                       # the thumbnail's light: rendered denim / its colour texture
THUMB, THUMB_LARGE = 256, 768


def read(path):
    with open(path, 'rb') as f:
        return f.read()


# ------------------------------------------------------------------ colour
def denim_colour(name):
    if name.strip().lower() == 'black':
        return np.array(BLACK_DENIM, np.float32)
    c = np.array(ImageColor.getrgb(name)[:3], np.float32)
    luma = c.mean()
    if luma < MIN_LUMA:
        c = c + (MIN_LUMA - luma) if luma < 1 else c * (MIN_LUMA / luma)
    return c


# ------------------------------------------------------------------ the texture array
def array_slices(g, toc, header, work):
    """Mip 0 of every slice of a BC7 texture array (mips stored mip by mip, slices inside)."""
    w, h = struct.unpack_from('<HH', header, 22)
    slices = struct.unpack_from('<H', header, 28)[0]
    size0 = struct.unpack_from('<I', header, 56)[0]
    c = toc[header[40:56]]
    data = fb.decode_cas(g.read(fb.FileInfo(c.patch, c.install_chunk, c.archive, c.offset, c.size)), g.root)
    out = []
    for s in range(slices):
        src, dst = os.path.join(work, f'slice{s}.bc7'), os.path.join(work, f'slice{s}.rgba')
        with open(src, 'wb') as f:
            f.write(data[s * size0:(s + 1) * size0])
        ebxtool('bc7dec', src, w, h, dst)
        out.append(np.frombuffer(read(dst), np.uint8).reshape(h, w, 4).astype(np.float32))
    return out


def mask_slices(g, toc, header, work):
    """Mip 0 of every slice of the BC1 region mask, as RGB."""
    w, h = struct.unpack_from('<HH', header, 22)
    slices = struct.unpack_from('<H', header, 28)[0]
    size0 = struct.unpack_from('<I', header, 56)[0]
    c = toc[header[40:56]]
    data = fb.decode_cas(g.read(fb.FileInfo(c.patch, c.install_chunk, c.archive, c.offset, c.size)), g.root)
    out = []
    for s in range(slices):
        dds = bytearray(128)
        dds[0:4] = b'DDS '
        struct.pack_into('<IIIIIII', dds, 4, 124, 0x1007, h, w, size0, 0, 1)
        struct.pack_into('<II4s', dds, 76, 32, 4, b'DXT1')
        path = os.path.join(work, f'mask{s}.dds')
        with open(path, 'wb') as f:
            f.write(bytes(dds) + data[s * size0:(s + 1) * size0])
        out.append(np.asarray(Image.open(path).convert('RGB'), np.float32))
    return out


def denim_weight(mask):
    """1 on denim (the mask's blue regions), 0 on the button (green) and pocket lining (red)."""
    r, b = mask[..., 0], mask[..., 2]
    return (np.clip((b - 48) / 48, 0, 1) * (1 - np.clip((r - 48) / 48, 0, 1)))[..., None]


def recolour(slices, masks, colour):
    weights = [denim_weight(m) for m in masks]
    lum = [s[..., :3].mean(2, keepdims=True) for s in slices]
    mean = sum((l * w).sum() for l, w in zip(lum, weights)) / sum(w.sum() for w in weights)
    out = []
    for s, l, w in zip(slices, lum, weights):
        new = colour * np.clip(1 + DETAIL * (l - mean) / mean, 0, None)
        rgb = s[..., :3] * (1 - w) + new * w
        out.append(np.dstack([np.clip(rgb, 0, 255), s[..., 3:]]))
    return out, float(mean)


def array_mips(slices, work):
    """BC7 mips of every slice, largest first: [[mip0 slice0, mip0 slice1], [mip1 ...], ...]."""
    mips, level = [], 0
    cur = [Image.fromarray(s.astype(np.uint8), 'RGBA') for s in slices]
    while True:
        mips.append([bc7(im, work, f'jeans_m{level}_s{k}') for k, im in enumerate(cur)])
        if cur[0].width == 1 and cur[0].height == 1:
            return mips
        cur = [im.resize((max(1, im.width // 2), max(1, im.height // 2)), Image.BOX) for im in cur]
        level += 1


# ------------------------------------------------------------------ thumbnail
def jeans_mask(T):
    """GrabCut the jeans out of the game's pants thumbnail (square RGBA): seeded on the middle of
    each thigh and shin; the bare torso, the feet and the arms and hands beside the hips (bright
    and smooth, where the jeans' wash is mottled) marked as not jeans. Then grown into the
    mottled pixels along its edge that GrabCut left out (highlights on the hip, the hem)."""
    bgr = np.ascontiguousarray(T[..., 2::-1].astype(np.uint8))
    a = T[..., 3]
    L = T[..., :3].mean(2).astype(np.float32)
    mu = cv2.blur(L, (5, 5))
    sd = np.sqrt(np.maximum(cv2.blur(L * L, (5, 5)) - mu * mu, 0))
    skin = (L > 180) & (sd < 4)
    S = T.shape[0]
    yy, xx = np.mgrid[0:S, 0:S] / S
    mask = np.full(a.shape, cv2.GC_PR_BGD, np.uint8)
    mask[a < 128] = cv2.GC_BGD
    mask[yy < 0.023] = cv2.GC_BGD
    mask[yy > 0.86] = cv2.GC_BGD
    arms = (yy < 0.36) & ((xx < 0.39) | (xx > 0.65))           # beside the hips: arms and hands
    mask[(L > 172) & (sd < 6) & (a > 200) & arms] = cv2.GC_BGD
    for y0, y1, x0, x1 in ((0.08, 0.16, 0.39, 0.55), (0.23, 0.43, 0.39, 0.49), (0.23, 0.43, 0.55, 0.65),
                           (0.49, 0.73, 0.34, 0.43), (0.49, 0.73, 0.56, 0.65)):
        mask[int(y0 * S):int(y1 * S), int(x0 * S):int(x1 * S)] = cv2.GC_FGD
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(bgr, mask, None, bgd, fgd, 8, cv2.GC_INIT_WITH_MASK)
    m = np.isin(mask, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    m = (lab == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    grow = (L < 172) & ~skin & (a > 200) & (yy < 0.86)
    hem = (yy > 0.80) & (yy < 0.86) & (sd > 5.5) & (L > 140) & (L < 205) & (a > 200)   # the frayed hem edge
    for _ in range(4):
        m = np.maximum(m, (cv2.dilate(m, np.ones((3, 3), np.uint8)) > 0) & (grow | hem)).astype(np.uint8)
    m = (cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.5) > 0.5).astype(np.uint8)   # a smooth outline
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(m)
    cv2.drawContours(filled, contours, -1, 1, cv2.FILLED)
    return filled


def thumbnail(base_thumb, colour):
    """The game's jeans render in the new colour: its light and shading kept, the wash detail
    toned down the way the texture's is."""
    T = base_thumb
    m = jeans_mask(T)
    soft = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 0.8)[..., None]
    L = T[..., :3].mean(2)
    inside = m > 0
    shade = cv2.GaussianBlur(L * m, (0, 0), 10) / np.maximum(cv2.GaussianBlur(m.astype(np.float32), (0, 0), 10), 1e-3)
    detail = np.where(inside, L / np.maximum(shade, 1), 1)
    shade = shade / L[inside].mean()
    new = colour * LIT * (shade ** 0.9 * (1 + DETAIL * (detail - 1)))[..., None]
    rgb = T[..., :3] * (1 - soft) + np.clip(new, 0, 255) * soft
    return Image.fromarray(np.dstack([rgb, T[..., 3:]]).astype(np.uint8), 'RGBA'), int(m.sum())


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
    colour = denim_colour(args.color)

    print(f'Skinny jeans "{args.name}"  key={key}  mod={folder_name}  colour {args.color} -> {tuple(int(v) for v in colour)}')
    g = fb.GameData(game)
    patch_chunk = next(k for k, v in g.chunk_dirs.items() if v == PATCH_DIRECTORY)
    _, base_bundles, base_toc_chunks = g.toc('Win32/items.toc')
    by_name = {b.name: b for b in base_bundles}
    toc_base = {c.guid: c for c in base_toc_chunks}
    if COLORWAY_BUNDLE not in by_name:
        raise SystemExit('The skinny jeans colorway is not in this skate. build.')
    others = bundles_of_other_mods(game, folder_name).get(COLORWAY_BUNDLE)
    if others:
        raise SystemExit(f'Another installed mod ships the skinny jeans colorway bundle ({", ".join(others)}); '
                         'ReSkate would merge the two and drop the new texture.')

    work = tempfile.mkdtemp(prefix='jeansmod_')
    try:
        def wpath(f):
            return os.path.join(work, f)

        # ---- the base colorway: preset, colour texture array, region mask, parameter names
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
            elif a.name.endswith('/' + BASE + '_c_ta'):
                tmpl['texture'] = (a.name, data)
        for a, f in zip(b_res, b_res_files):
            if a.res_type != TEXTURE_RES:
                continue
            if a.name.endswith('/' + BASE + '_c_ta'):
                tmpl['texture_res'] = (a, g.payload(f))
            elif a.name.endswith('_msk_ta'):
                tmpl['mask_res'] = (a, g.payload(f))
        if len(tmpl) != 4:
            raise SystemExit(f'{BASE}_ap: unexpected contents ({sorted(tmpl)})')
        preset_path = wpath('preset_tmpl.ebx')
        with open(preset_path, 'wb') as f:
            f.write(tmpl['preset'][1])
        pinfo = ebx_info(preset_path)
        values = preset_values(pinfo['dump'], param_names)
        if 'param_usegraphic' not in values or 'param_base_c_ta' not in values:
            raise SystemExit(f'{BASE}_ap: no colour texture / graphic switch in this build ({sorted(values)})')
        tinfo_path = wpath('texture_tmpl.ebx')
        with open(tinfo_path, 'wb') as f:
            f.write(tmpl['texture'][1])
        tinfo = ebx_info(tinfo_path)
        header = tmpl['texture_res'][1]
        if struct.unpack_from('<I', header, 12)[0] != 67 or struct.unpack_from('<HHHH', header, 22) != (1024, 1024, 2, 2):
            raise SystemExit('The skinny jeans colour texture is not the 1024x1024 two-slice BC7 array this tool expects.')

        # ---- recolour
        slices = array_slices(g, toc_base, header, work)
        masks = mask_slices(g, toc_base, tmpl['mask_res'][1], work)
        new_slices, base_mean = recolour(slices, masks, colour)
        mips = array_mips(new_slices, work)
        parts_raw = [b''.join(m) for m in mips]
        total = sum(len(p) for p in parts_raw)
        per_slice = sum(len(m[0]) for m in mips)
        print(f'  colour texture: 2 slices, {len(mips)} mips, denim recoloured (it was {base_mean:.0f} grey)')

        # ---- names
        stem = tmpl['preset'][0].rsplit('/', 1)[0]
        sb_files, _ = fb.read_bundle_region(by_name[SHARED_BUNDLE].region)
        s_ebx, s_res, s_chunks, s_meta = g.manifest(sb_files)
        ne, nr = len(s_ebx), len(s_res)
        br = next(i for i, a in enumerate(s_res) if a.name == BUNDLE_REF_TABLE)
        table = bundleref.Table(g.payload(sb_files[1 + ne + br]), s_res[br].res_meta)
        taken = {a.name for a in b_ebx}
        number = next(f'{n:05d}' for n in iter(lambda: rng.randrange(70000, 99999), None)
                      if f'{stem}/{BASE[:-5]}{n:05d}_ap' not in table.presets and f'{stem}/{BASE[:-5]}{n:05d}_ap' not in taken)
        names = {'preset': f'{stem}/{BASE[:-5]}{number}_ap', 'texture': f'{stem}/{BASE[:-5]}{number}_c_ta',
                 'item': f'items/cust_bottoms/{ident}',
                 'thumb_tool': f'thumbnail/tool/{ident}', 'thumb_tool_lrg': f'thumbnail/tool/{ident}_lrg',
                 'thumb_cdn': f'thumbnail/cdn/img_{ident}', 'thumb_cdn_lrg': f'thumbnail/cdn/img_{ident}_lrg'}
        if any(n in {a.name for a in s_ebx} for n in (names['item'], names['thumb_tool'])):
            raise SystemExit(f'The game already has an item named {key}; pick another name.')
        print(f'  colorway bundle: {BASE}_ap (the new colorway is number {number})')

        # ---- thumbnails
        th_res = [f for a, f in zip(s_res, sb_files[1 + ne:1 + ne + nr]) if a.name == BASE_THUMB]
        if not th_res:
            raise SystemExit(f'{BASE_THUMB} is not in the shared bundle')
        base_thumb = decode_texture(g, toc_base, g.payload(th_res[0]), work, 'base_thumb', zip(s_chunks, sb_files[1 + ne + nr:]))
        large, covered = thumbnail(base_thumb, colour)
        if covered < 0.15 * base_thumb.shape[0] ** 2:
            raise SystemExit(f'Could not find the jeans in the base thumbnail ({covered} pixels).')
        thumbs = {'thumb_tool_lrg': large, 'thumb_tool': large.resize((THUMB, THUMB), Image.LANCZOS)}
        thumb_bc7 = {lab: bc7(im, work, lab) for lab, im in thumbs.items()}

        ids = {lab: {'file': new_guid(), 'inst': new_guid()} for lab in
               ('preset', 'texture', 'thumb_tool', 'thumb_tool_lrg', 'thumb_cdn', 'thumb_cdn_lrg')}
        res_ids = {lab: rng.getrandbits(64) | 1 for lab in ('texture', 'thumb_tool', 'thumb_tool_lrg')}
        chunk_ids = {lab: uuid.uuid4().bytes for lab in ('texture', 'thumb_tool', 'thumb_tool_lrg')}
        cas = CasWriter(patch_chunk, game, compress=not args.no_compress)
        ebx, res = {}, {}

        # ---- the colour texture: whole in the bundle, every mip (both slices) its own blocks
        texture_name = tinfo['dump'].split('Name = "')[1].split('"')[0].replace(f'_{BASE[-5:]}_', f'_{number}_')
        ebx['texture'] = edit_ebx(tinfo_path, wpath('texture_new.ebx'), '--file-guid', ids['texture']['file'],
                                  '--inst-guid', 0, ids['texture']['inst'], '--set', '0:Name', 'str:' + texture_name,
                                  '--set', '0:Resource', f'res:{res_ids["texture"]}')
        parts = [cas.encode(p) for p in parts_raw]
        stream = b''.join(parts)
        h = bytearray(header)
        struct.pack_into('<II', h, 0, len(parts[0]), len(parts[0]) + len(parts[1]))
        h[31] = 0                                    # the bundle carries every mip
        h[40:56] = chunk_ids['texture']
        assert struct.unpack_from('<I', h, 116)[0] == per_slice, 'texture size differs from the game\'s'
        struct.pack_into('<Q', h, 120, djb(names['texture'], 64))
        res['texture'] = bytes(h)

        # ---- the preset, byte for byte: new ids, the new colour texture, no logo, renumbered
        data = bytearray(tmpl['preset'][1])
        for old, new in ((pinfo['file'], ids['preset']['file']), (pinfo['instances'][0], ids['preset']['inst']),
                         (tinfo['file'], ids['texture']['file']), (tinfo['instances'][0], ids['texture']['inst'])):
            assert data.count(guid_bytes(old)) == 1, f'preset: {old} should appear once'
            data = data.replace(guid_bytes(old), guid_bytes(new))
        at, old_value = values['param_usegraphic']
        assert old_value[:1] == b'\x01', 'the base colorway should have its logo on'
        data[at] = 0
        old_text = ('JeansSkinny_' + BASE[-5:]).encode()
        assert data.count(old_text) == 1, 'preset: its name should hold the colorway number once'
        ebx['preset'] = bytes(data.replace(old_text, ('JeansSkinny_' + number).encode()))
        preset_leaf = 'Gen_Pants_JeansSkinny_' + number + '_AP'

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
        swatch = ((colour / 255) ** 2.2).round(6)     # linear, like the game's BaseColor values
        item_ops = ['--file-guid', new_guid()]
        for k in range(len(iinfo['instances'])):
            item_ops += ['--inst-guid', k, new_guid()]
        ebx['item'] = edit_ebx(wpath('item_tmpl.ebx'), wpath('item.ebx'), *item_ops,
                               '--set', '0:Name', 'str:' + names['item'], '--set', '0:Key', 'str:' + key,
                               '--set', '0:HashedAssetKey', f'u64:{djb(key)}',
                               '--import', thumb_slot, ids['thumb_cdn']['file'], ids['thumb_cdn']['inst'],
                               '--import', large_slot, ids['thumb_cdn_lrg']['file'], ids['thumb_cdn_lrg']['inst'],
                               '--set', f'0:ItemData.AssetPaths[{slot[0]}].AssetName', 'str:' + preset_leaf,
                               '--set', '0:ItemData.BaseColor.x', f'f64:{swatch[0]}',
                               '--set', '0:ItemData.BaseColor.y', f'f64:{swatch[1]}',
                               '--set', '0:ItemData.BaseColor.z', f'f64:{swatch[2]}')
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

        # ---- the colorway bundle: + preset, colour texture (asset, resource, whole chunk)
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
            json.dump({'author': author, 'dependencies': [], 'description': f'{args.name}: skinny jeans in {args.color}',
                       'name': package, 'version_number': args.version, 'website_url': ''}, f, indent=4)
        with open(os.path.join(mod_dir, 'README.md'), 'w', encoding='utf-8') as f:
            f.write(f'# {args.name}\n\nAdds "{args.name}", the game\'s skinny jeans in {args.color}, to skate. (ReSkate). '
                    'Find them under bottoms.\n')
        thumbs['thumb_tool'].save(os.path.join(mod_dir, 'icon.png'))
        thumbs['thumb_tool_lrg'].save(os.path.join(out_root, f'{folder_name}-thumbnail.png'))
        Image.fromarray(np.hstack([s[..., :3] for s in new_slices]).astype(np.uint8)).resize((1024, 512), Image.LANCZOS) \
            .save(os.path.join(out_root, f'{folder_name}-texture.png'))
        zip_path = os.path.join(out_root, f'{folder_name}.zip')
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
            for dirpath, _, fs in os.walk(mod_dir):
                for n in fs:
                    full = os.path.join(dirpath, n)
                    z.write(full, os.path.relpath(full, mod_dir))
        print(f'  wrote {mod_dir}\n  wrote {zip_path}')

        if not args.no_verify:
            r = subprocess.run([sys.executable, os.path.join(HERE, 'verify_jeans_mod.py'), mod_dir, '--game', game],
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


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--name', required=True, help='name shown in game')
    p.add_argument('--color', default='black', help='denim colour: black, a colour name, or #rrggbb (default black)')
    p.add_argument('--author', default='socioculture', help='mod author (folder prefix)')
    p.add_argument('--package')
    p.add_argument('--version', default='1.0.0')
    p.add_argument('--game', default=DEFAULT_GAME)
    p.add_argument('--out', default=os.path.join(HERE, '..', 'mods'))
    p.add_argument('--install', action='store_true')
    p.add_argument('--no-compress', action='store_true')
    p.add_argument('--no-verify', action='store_true')
    a = p.parse_args()
    if a.color.strip().lower() != 'black':
        try:
            ImageColor.getrgb(a.color)
        except ValueError:
            p.error(f'unknown colour "{a.color}" (use #rrggbb or a colour name like navy, white, olive)')
    build(a)


if __name__ == '__main__':
    main()
