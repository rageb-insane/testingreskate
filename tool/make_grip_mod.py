"""Turns a grip tape design (an image painted on grip-template.png, or just a colour) into a
ReSkate mod that adds it to skate. as a new grip tape.

    python make_grip_mod.py my_grip.png --name "My Grip" [--install]
    python make_grip_mod.py --color "#d01818" --name "Red Grip" [--install]
    python make_grip_mod.py --template grip-template.png        (writes the painting template)

The grip art is the game's 512x2048 grip texture (authored at 1024x4096): the top of the image
is the nose, seen from above; only the deck's outline shows, the rest is bleed. The new grip
is a copy of one of the game's own grip colours (popsicle deck): its appearance preset, grip
texture and material preset, renumbered, in that grip's bundle, which no other installed mod
may ship (ReSkate would merge it and drop the texture's metadata); plus the item, thumbnails,
item-list entry and bundle-reference row in the shared bundle. The texture sits whole in the
bundle (first mip 0), the layout the community mods use, which streams nothing.
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

import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFilter, ImageFont

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

GRIP_W, GRIP_H = 512, 2048            # the game's grip texture
ART_W, ART_H = 1024, 4096             # its authored size (TextureAsset.AuthoredWidth/Height)
GRIP_DIR = 'win32/characters/skateboard/unlicensed/deck/common/generic/gripcolor/presets/'
GRIP_BUNDLE = re.compile(r'deck_gen_popsiclegripcolor_(\d{5})(?:/\d{4})?/deck_gen_popsiclegripcolor_\1_ap_cas_main_bundlereftable$')
DECK_MESH = 'characters/skateboard/unlicensed/deck/generic/popsicle/2022/deck_gen_popsicle_mesh'
# Truck bolt holes of the popsicle deck in grip-texture space (u, v), found on the deck's own
# wood texture and carried over through its mesh.
BOLTS = [(0.4063, 0.2468), (0.5901, 0.2468), (0.4065, 0.3061), (0.5907, 0.3064),
         (0.5899, 0.7104), (0.4052, 0.7107), (0.4060, 0.7696), (0.5896, 0.7697)]
THUMB, THUMB_LARGE = 256, 768


def read(path):
    with open(path, 'rb') as f:
        return f.read()


# ------------------------------------------------------------------ the deck's grip layout
def grip_layout(g):
    """The grip surface of the game's popsicle deck in grip-texture space (0..1): its outline
    as one closed polygon, and where the nose and tail kicks start (v)."""
    _, bundles, toc_chunks = g.toc('Win32/items.toc')
    toc = {c.guid: c for c in toc_chunks}
    for b in bundles:
        if not b.name.endswith('deck_gen_popsicle_cas_main_bundlereftable'):
            continue
        files, _ = fb.read_bundle_region(b.region)
        ebx, res, chunks, _ = g.manifest(files)
        for a, f in zip(res, files[1 + len(ebx):]):
            if a.res_type != MESHSET_RES or a.name != DECK_MESH:
                continue
            ms = meshset.MeshSet(g.payload(f))
            lod = ms.lods[0]
            c = toc.get(lod.chunk)
            data = fb.decode_cas(g.read(fb.FileInfo(c.patch, c.install_chunk, c.archive, c.offset, c.size)), g.root) if c \
                else g.payload(dict(zip([x.guid for x in chunks], files[1 + len(ebx) + len(res):]))[lod.chunk])
            top = [s for s in lod.sections if s.material == 'DeckTop_mat'][0]
            v = meshset.decode_section(lod, top, data)
            uv = v['uv1'] - [1, 0]                 # the grip graphic samples UDIM tile 2
            tri, pos = v['triangles'], v['pos'][:, :3]
            count = {}
            for t in tri:
                for e in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
                    k = (min(e), max(e))
                    count[k] = count.get(k, 0) + 1
            nxt = {}
            for a_, b_ in (k for k, n in count.items() if n == 1):
                nxt.setdefault(a_, []).append(b_)
                nxt.setdefault(b_, []).append(a_)
            start = min(nxt, key=lambda i: uv[i, 1])
            loop, prev, cur = [start], None, start
            while True:
                step = [x for x in nxt[cur] if x != prev][0]
                if step == start:
                    break
                loop.append(step)
                prev, cur = cur, step
            # the kicks, along the board's centre line (the concave lifts the rails everywhere)
            centre = np.abs(pos[:, 0]) < 0.25 * np.abs(pos[:, 0]).max()
            flat = np.median(pos[centre & (np.abs(pos[:, 2]) < 0.15), 1])
            up = centre & (pos[:, 1] > flat + 0.004)
            nose = uv[up & (pos[:, 2] > 0), 1].max(initial=0.2)
            tail = uv[up & (pos[:, 2] < 0), 1].min(initial=0.8)
            return [tuple(uv[i]) for i in loop], float(nose), float(tail)
    raise SystemExit('The popsicle deck mesh was not found in this skate. build.')


def write_template(g, path):
    """grip-template.png: the authored 1024x4096 grip with the deck outline, kicks and bolts."""
    outline, nose, tail = grip_layout(g)
    s = 2
    W, H = ART_W * s, ART_H * s
    im = Image.new('RGB', (W, H), (205, 205, 210))
    d = ImageDraw.Draw(im)
    for k in range(-H, W + H, 60 * s):          # hatch the bleed
        d.line([(k, 0), (k + H, H)], fill=(190, 190, 196), width=6 * s)
    poly = [(u * W, v * H) for u, v in outline]
    d.polygon(poly, fill=(255, 255, 255))
    d.line(poly + [poly[0]], fill=(30, 30, 30), width=5 * s)
    xs = [p[0] for p in poly]
    for v in (nose, tail):                       # where the board bends up into the nose/tail
        y = v * H
        for x in range(int(min(xs)), int(max(xs)), 40 * s):
            d.line([(x, y), (x + 20 * s, y)], fill=(80, 140, 230), width=4 * s)
    for u, v in BOLTS:
        r = 20 * s
        d.ellipse([u * W - r, v * H - r, u * W + r, v * H + r], outline=(220, 60, 60), width=5 * s)
    try:
        font = ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf', 64 * s)
        small = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 34 * s)
    except OSError:
        font = small = ImageFont.load_default()
    def label(text, y, f, fill=(30, 30, 30)):
        w = d.textlength(text, font=f)
        d.text(((W - w) / 2, y), text, font=f, fill=fill)
    label('NOSE', 0.11 * H, font)
    label('TAIL', 0.86 * H, font)
    notes = [('Paint your grip here, nose at the top.', (30, 30, 30)),
             ('Blue dashes: where the nose and tail bend up.', (80, 140, 230)),
             ('Red circles: truck bolts.', (220, 60, 60)),
             ('Hatched area is off the board:', (120, 120, 125)),
             ('fill it with your art too (bleed).', (120, 120, 125))]
    for i, (text, colour) in enumerate(notes):
        label(text, (0.43 + 0.022 * i) * H, small, colour)
    im.resize((ART_W, ART_H), Image.LANCZOS).save(path)
    print('wrote', path)


# ------------------------------------------------------------------ the art
def load_art(path):
    """The grip image at 512x2048, nose at the top. Images of another shape are cropped to fit
    (centred), landscape ones turned upright; transparent parts repeat their nearest colour."""
    im = Image.open(path)
    im.load()
    im = im.convert('RGBA')
    if im.width > im.height:
        im = im.rotate(90, expand=True)
    aspect = GRIP_W / GRIP_H
    if abs(im.width / im.height - aspect) > 0.02:
        if im.width / im.height > aspect:
            w = round(im.height * aspect)
            im = im.crop(((im.width - w) // 2, 0, (im.width - w) // 2 + w, im.height))
        else:
            h = round(im.width / aspect)
            im = im.crop((0, (im.height - h) // 2, im.width, (im.height - h) // 2 + h))
    im = im.resize((GRIP_W, GRIP_H), Image.LANCZOS)
    a = np.asarray(im, np.float32)
    alpha = a[..., 3:] / 255
    if alpha.min() < 0.99:                       # fill transparency from the surroundings
        rgb, w = a[..., :3] * alpha, alpha.copy()
        filled = rgb.copy()
        for radius in (2, 4, 8, 16, 32, 64, 128, 256):
            br = np.asarray(Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).filter(ImageFilter.BoxBlur(radius)), np.float32)
            bw = np.asarray(Image.fromarray((w[..., 0] * 255).astype(np.uint8)).filter(ImageFilter.BoxBlur(radius)), np.float32)[..., None] / 255
            est = np.where(bw > 1e-3, br / np.maximum(bw, 1e-3), filled)
            filled = np.where(w > 0.99, filled, np.where(bw > 0.05, est, filled))
        a[..., :3] = rgb + filled * (1 - alpha)
    return Image.fromarray(np.clip(a[..., :3], 0, 255).astype(np.uint8))


def colour_art(colour, seed=1):
    """A plain grip in one colour, with the fine grain and slight mottling of real grip tape
    (the game adds its own grit on top from the grip's shared normal/mask maps)."""
    r, g_, b = ImageColor.getrgb(colour)[:3]
    rng = np.random.default_rng(seed)
    grain = rng.normal(0, 1, (GRIP_H, GRIP_W)).astype(np.float32)
    mottle = np.asarray(Image.fromarray(((rng.random((GRIP_H // 32, GRIP_W // 32)) * 255)).astype(np.uint8))
                        .resize((GRIP_W, GRIP_H), Image.BICUBIC), np.float32) / 255 - 0.5
    shade = 1 + 0.045 * grain + 0.06 * mottle
    base = np.array([r, g_, b], np.float32)
    return Image.fromarray(np.clip(base * shade[..., None] + 6 * grain[..., None], 0, 255).astype(np.uint8))


def thumbnail(art, outline, size):
    """The grip on the deck's outline, standing upright and centred on transparency, like the
    game's own grip thumbnails."""
    big = art.convert('RGBA')
    mask = Image.new('L', big.size, 0)
    ImageDraw.Draw(mask).polygon([(u * big.width, v * big.height) for u, v in outline], fill=255)
    big.putalpha(mask)
    shape = big.crop(mask.getbbox())
    h = round(size * 0.93)
    w = max(1, round(shape.width * h / shape.height))
    shape = shape.resize((w, h), Image.LANCZOS)
    out = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    out.alpha_composite(shape, ((size - w) // 2, (size - h) // 2))
    return out


def grip_mips(art, work):
    mips, level = [], 0
    while True:
        w, h = max(1, GRIP_W >> level), max(1, GRIP_H >> level)
        mips.append(bc7(art if level == 0 else art.resize((w, h), Image.LANCZOS), work, f'grip_mip{level}'))
        if w == 1 and h == 1:
            return mips
        level += 1


# ------------------------------------------------------------------ build
def pick_grip_bundle(by_name, taken):
    """A grip colour bundle no other installed mod ships, and its five-digit grip number."""
    free = []
    for name in sorted(by_name):
        m = GRIP_BUNDLE.search(name)
        if m and name.startswith(GRIP_DIR) and name not in taken:
            free.append((name, m.group(1)))
    if not free:
        raise SystemExit('Every grip colour bundle is already shipped by another installed mod.')
    return free[-1]                              # the newest grip


def build(args):
    game = args.game
    key_words = slugify(args.name)
    key = 'Own_' + key_words + '_Grip'
    ident = 'own_' + key_words.lower() + '_grip'
    author = re.sub(r'[^A-Za-z0-9_]', '_', args.author).strip('_') or 'socioculture'
    package = re.sub(r'[^A-Za-z0-9_]', '_', args.package or (key_words + '_Grip'))
    folder_name = f'{author}-{package}'
    out_root = os.path.abspath(args.out)
    mod_dir = os.path.join(out_root, folder_name)
    rng = random.SystemRandom()

    print(f'Grip "{args.name}"  key={key}  mod={folder_name}')
    g = fb.GameData(game)
    patch_chunk = next(k for k, v in g.chunk_dirs.items() if v == PATCH_DIRECTORY)
    _, base_bundles, _ = g.toc('Win32/items.toc')
    by_name = {b.name: b for b in base_bundles}
    others = bundles_of_other_mods(game, folder_name)
    bundle, number = pick_grip_bundle(by_name, others)
    used = {GRIP_BUNDLE.search(n).group(1) for n in by_name if GRIP_BUNDLE.search(n)}
    new_number = next(f'{n:05d}' for n in iter(lambda: rng.randrange(70000, 99999), None) if f'{n:05d}' not in used)
    print(f'  grip bundle: deck_gen_popsiclegripcolor_{number}_ap (the new grip is number {new_number})')
    outline, _, _ = grip_layout(g)

    work = tempfile.mkdtemp(prefix='gripmod_')
    try:
        def wpath(f):
            return os.path.join(work, f)

        # ---- the grip's own assets, from its bundle
        files, _ = fb.read_bundle_region(by_name[bundle].region)
        b_ebx, b_res, b_chunks, b_meta = g.manifest(files)
        b_ebx_files = files[1:1 + len(b_ebx)]
        b_res_files = files[1 + len(b_ebx):1 + len(b_ebx) + len(b_res)]
        b_chunk_files = files[1 + len(b_ebx) + len(b_res):]
        stem = f'deck_gen_popsiclegripcolor_{number}'
        tmpl = {}
        for a, f in zip(b_ebx, b_ebx_files):
            for lab, suffix in (('preset', '_ap'), ('material', '_m'), ('texture', '_c')):
                if a.name.endswith(stem + suffix):
                    tmpl[lab] = (a.name, g.payload(f))
        for a, f in zip(b_res, b_res_files):
            if a.name.endswith(stem + '_c') and a.res_type == TEXTURE_RES:
                tmpl['texture_res'] = (a, g.payload(f))
        if len(tmpl) != 4:
            raise SystemExit(f'{bundle}: unexpected contents ({sorted(tmpl)})')
        names = {lab: tmpl[lab][0].replace(f'_{number}', f'_{new_number}') for lab in ('preset', 'material', 'texture')}
        names.update({'item': f'items/boardtapecolors0/{ident}',
                      'thumb_tool': f'thumbnail/tool/{ident}', 'thumb_tool_lrg': f'thumbnail/tool/{ident}_lrg',
                      'thumb_cdn': f'thumbnail/cdn/img_{ident}', 'thumb_cdn_lrg': f'thumbnail/cdn/img_{ident}_lrg'})

        # ---- art
        if args.image:
            art = load_art(args.image)
            print(f'  art from {os.path.basename(args.image)}')
        else:
            art = colour_art(args.color)
            print(f'  plain {args.color} grip')
        mips = grip_mips(art, work)
        total = sum(len(m) for m in mips)
        thumbs = {lab: thumbnail(art, outline, size) for lab, size in (('thumb_tool', THUMB), ('thumb_tool_lrg', THUMB_LARGE))}
        thumb_bc7 = {lab: bc7(im, work, lab) for lab, im in thumbs.items()}

        ids = {lab: {'file': new_guid(), 'inst': new_guid()} for lab in
               ('preset', 'material', 'texture', 'thumb_tool', 'thumb_tool_lrg', 'thumb_cdn', 'thumb_cdn_lrg')}
        ids['item'] = {'file': new_guid(), 'inst': new_guid(), 'data': new_guid()}
        res_ids = {lab: rng.getrandbits(64) | 1 for lab in ('texture', 'thumb_tool', 'thumb_tool_lrg')}
        chunk_ids = {lab: uuid.uuid4().bytes for lab in ('texture', 'thumb_tool', 'thumb_tool_lrg')}
        cas = CasWriter(patch_chunk, game, compress=not args.no_compress)
        ebx, res = {}, {}

        # ---- grip texture: whole in the bundle, every mip starting its own blocks
        tex_tmpl = os.path.join(work, 'texture.ebx')
        with open(tex_tmpl, 'wb') as f:
            f.write(tmpl['texture'][1])
        tinfo = ebx_info(tex_tmpl)
        ebx['texture'] = edit_ebx(tex_tmpl, wpath('texture_new.ebx'), '--file-guid', ids['texture']['file'],
                                  '--inst-guid', 0, ids['texture']['inst'],
                                  '--set', '0:Name', 'str:' + tinfo['dump'].split('Name = "')[1].split('"')[0].replace(f'_{number}', f'_{new_number}'),
                                  '--set', '0:Resource', f'res:{res_ids["texture"]}')
        parts = [cas.encode(m) for m in mips]
        stream = b''.join(parts)
        header = bytearray(tmpl['texture_res'][1])
        header[31] = 0                              # the bundle carries every mip
        res['texture'] = texture_header(bytes(header), names['texture'], chunk_ids['texture'], total,
                                        (len(parts[0]), len(parts[0]) + len(parts[1])))

        # ---- presets: renumbered and re-pointed byte for byte (ReSkate's writer drops boxed values)
        def patch(lab, swaps):
            data = bytearray(tmpl[lab][1])
            info_path = wpath(lab + '_tmpl.ebx')
            with open(info_path, 'wb') as f:
                f.write(data)
            info = ebx_info(info_path)
            swaps = [(info['file'], ids[lab]['file']), (info['instances'][0], ids[lab]['inst'])] + swaps
            for old, new in swaps:
                assert data.count(guid_bytes(old)) >= 1, f'{lab}: {old} not found'
                data = data.replace(guid_bytes(old), guid_bytes(new))
            old_text = f'PopsicleGripColor_{number}'.encode()
            assert data.count(old_text) == 2, f'{lab}: its name should hold the grip number twice'
            return bytes(data.replace(old_text, f'PopsicleGripColor_{new_number}'.encode()))
        ebx['preset'] = patch('preset', [(tinfo['file'], ids['texture']['file']),
                                         (tinfo['instances'][0], ids['texture']['inst'])])
        ebx['material'] = patch('material', [])

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
        sb_files, _ = fb.read_bundle_region(by_name[SHARED_BUNDLE].region)
        s_ebx, s_res, s_chunks, s_meta = g.manifest(sb_files)
        ne, nr = len(s_ebx), len(s_res)
        sb_ebx_files, sb_res_files, sb_chunk_files = sb_files[1:1 + ne], sb_files[1 + ne:1 + ne + nr], sb_files[1 + ne + nr:]
        base_item = f'items/boardtapecolors0/own_deckgripcolor_gen_popsicle_{number}'
        ii = next((i for i, a in enumerate(s_ebx) if a.name == base_item), None)
        if ii is None:
            raise SystemExit(f'{base_item} is not in the shared bundle')
        with open(wpath('item_tmpl.ebx'), 'wb') as f:
            f.write(g.payload(sb_ebx_files[ii]))
        idump = ebx_info(wpath('item_tmpl.ebx'))['dump']
        thumb_slot = int(re.search(r'\n\s+Thumbnail = import\(#(\d+)', idump).group(1))
        large_slot = int(re.search(r'\n\s+ThumbnailLarge = import\(#(\d+)', idump).group(1))
        avg = np.asarray(art.resize((64, 256), Image.BOX), np.float32).reshape(-1, 3).mean(0) / 255
        swatch = (avg ** 2.2).round(6)               # linear, like the game's BaseColor values
        ebx['item'] = edit_ebx(wpath('item_tmpl.ebx'), wpath('item.ebx'), '--file-guid', ids['item']['file'],
                               '--inst-guid', 0, ids['item']['inst'], '--inst-guid', 1, ids['item']['data'],
                               '--set', '0:Name', 'str:' + names['item'], '--set', '0:Key', 'str:' + key,
                               '--set', '0:HashedAssetKey', f'u64:{djb(key)}',
                               '--import', thumb_slot, ids['thumb_cdn']['file'], ids['thumb_cdn']['inst'],
                               '--import', large_slot, ids['thumb_cdn_lrg']['file'], ids['thumb_cdn_lrg']['inst'],
                               '--set', '0:ItemData.AssetPaths[0].AssetName', 'str:' + names['preset'].rsplit('/', 1)[1],
                               '--set', '0:ItemData.BaseColor.x', f'f64:{swatch[0]}',
                               '--set', '0:ItemData.BaseColor.y', f'f64:{swatch[1]}',
                               '--set', '0:ItemData.BaseColor.z', f'f64:{swatch[2]}')

        ic = next(i for i, a in enumerate(s_ebx) if a.name == ITEM_COLLECTION)
        with open(wpath('collection.ebx'), 'wb') as f:
            f.write(g.payload(sb_ebx_files[ic]))
        ebxtool('additem', wpath('collection.ebx'), wpath('collection_new.ebx'), ids['item']['file'], ids['item']['inst'])
        collection = read(wpath('collection_new.ebx'))
        fi, sha = cas.add(collection)
        s_ebx[ic].sha1, s_ebx[ic].original_size = sha, len(collection)
        sb_ebx_files[ic] = fi
        br = next(i for i, a in enumerate(s_res) if a.name == BUNDLE_REF_TABLE)
        table = bundleref.Table(g.payload(sb_res_files[br]), s_res[br].res_meta)
        table.insert(names['preset'], table.bundle_index(bundle_ref(bundle)))
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

        # ---- the grip bundle: + preset, material preset, texture (asset, resource, whole chunk)
        ge, ge_files = add_ebx(['preset', 'material', 'texture'])
        gr, gr_files = add_res(['texture'], tmpl['texture_res'][0].res_meta)
        fi = cas.add_raw(stream)
        toc_chunks.append(fb.TocChunk(chunk_ids['texture'], True, patch_chunk, 1, fi.offset, fi.size))
        gc = [fb.Asset(kind='chunk', name=fb.guid_str(chunk_ids['texture']), guid=chunk_ids['texture'],
                       sha1=hashlib.sha1(stream).digest(), logical_offset=0, logical_size=total)]
        tree, _ = fb.read_db(b_meta, 0)
        tree['children'].append({'type': 2, 'name': None, 'terminated': True, 'children': [
            {'type': 9, 'name': 'h64', 'value': struct.pack('<Q', djb(names['texture'], 64))},
            {'type': 2, 'name': 'meta', 'terminated': True, 'children': [
                {'type': 8, 'name': 'firstMip', 'value': struct.pack('<i', 0)}]}]})
        manifest = fb.write_binary_bundle(b_ebx + ge, b_res + gr, b_chunks + gc, fb.write_db(tree))
        grip = fb.TocBundle(bundle, fb.write_bundle_region(
            [cas.add_raw(manifest)] + b_ebx_files + ge_files + b_res_files + gr_files + b_chunk_files + [fi]))
        toc = fb.write_patch_toc([shared, grip], toc_chunks, 3)

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
            json.dump({'author': author, 'dependencies': [], 'description': f'{args.name} grip tape',
                       'name': package, 'version_number': args.version, 'website_url': ''}, f, indent=4)
        with open(os.path.join(mod_dir, 'README.md'), 'w', encoding='utf-8') as f:
            f.write(f'# {args.name}\n\nAdds the "{args.name}" grip tape to skate. (ReSkate). Find it under grip tape.\n')
        thumbs['thumb_tool'].save(os.path.join(mod_dir, 'icon.png'))
        art.save(os.path.join(out_root, f'{folder_name}-texture.png'))
        thumbs['thumb_tool_lrg'].save(os.path.join(out_root, f'{folder_name}-thumbnail.png'))
        zip_path = os.path.join(out_root, f'{folder_name}.zip')
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
            for dirpath, _, fs in os.walk(mod_dir):
                for n in fs:
                    full = os.path.join(dirpath, n)
                    z.write(full, os.path.relpath(full, mod_dir))
        print(f'  wrote {mod_dir}\n  wrote {zip_path}')

        if not args.no_verify:
            r = subprocess.run([sys.executable, os.path.join(HERE, 'verify_grip_mod.py'), mod_dir, '--game', game],
                               capture_output=True, text=True)
            print(r.stdout[-2500:])
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
    p.add_argument('image', nargs='?', help='grip image (PNG/JPG), nose at the top, ideally painted on grip-template.png')
    p.add_argument('--color', help='make a plain grip in this colour instead (#rrggbb or a colour name)')
    p.add_argument('--name', help='grip name shown in game')
    p.add_argument('--template', metavar='PNG', help='write the painting template to this file and stop')
    p.add_argument('--author', default='socioculture', help='mod author (folder prefix)')
    p.add_argument('--package')
    p.add_argument('--version', default='1.0.0')
    p.add_argument('--game', default=DEFAULT_GAME)
    p.add_argument('--out', default=os.path.join(HERE, '..', 'mods'))
    p.add_argument('--install', action='store_true')
    p.add_argument('--no-compress', action='store_true')
    p.add_argument('--no-verify', action='store_true')
    a = p.parse_args()
    if a.template:
        write_template(fb.GameData(a.game), a.template)
        return
    if bool(a.image) == bool(a.color):
        p.error('give a grip image or --color (one of them)')
    if a.color:
        try:
            ImageColor.getrgb(a.color)
        except ValueError:
            p.error(f'unknown colour "{a.color}" (use #rrggbb or a colour name like red, navy, hotpink)')
    if not a.name:
        p.error('--name is required')
    build(a)


if __name__ == '__main__':
    main()
