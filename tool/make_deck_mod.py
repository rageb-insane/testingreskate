"""Turns a deck image into a ReSkate mod that adds it to skate. as a new deck graphic.

    python make_deck_mod.py my_deck.png --name "My Deck" [--author socioculture] [--install]

The image is the bottom of the board, nose at the top. A transparent background
around the deck shape (like blank-deck-template.png) is cropped away; a plain
rectangle is used as-is. Output: a mod folder plus a .zip under ../mods/, which the
ReSkate launcher installs by drag-and-drop (or pass --install to copy it into Mods/).

How the mod is put together (mirrors what ReSkate Studio deck mods ship):
  * win32/characters/customization/configs/cas_main_sharedbundle gains the item
    (items/board_bottomart/own_<id>), its two thumbnails, and edits to the season
    item collection and the character bundle-reference table.
  * The Baker deck_baker_graphic_00003 bundle gains an appearance preset and the
    512x2048 BC7 texture it points at.
Templates for every asset are in template/ (one Studio-made deck); ebxtool.exe
does the EBX edits with ReSkate's own reader/writer.
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

from PIL import Image, ImageChops, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fb  # noqa: E402
import bundleref  # noqa: E402

EBXTOOL = os.path.join(HERE, 'ebxtool', 'ebxtool.exe')
TEMPLATE = os.path.join(HERE, 'template')
DEFAULT_GAME = r'E:\Steam\steamapps\common\Skate'

SHARED_BUNDLE = 'win32/characters/customization/configs/cas_main_sharedbundle'
DECK_BUNDLE = ('win32/characters/skateboard/licensed/deck/common/baker/graphic/deck_baker_graphic_00003/2026/'
               'deck_baker_graphic_00003_ap_cas_main_bundlereftable')
DECK_BUNDLE_REF = 'characters/skateboard/licensed/deck/common/baker/graphic/deck_baker_graphic_00003/2026/deck_baker_graphic_00003_ap'
ITEM_COLLECTION = 'items/_seasons/0.32.0/ownables/0.32.0_itemcollection'
BUNDLE_REF_TABLE = 'characters/customization/configs/cas_main_bundlereftable'
PATCH_DIRECTORY = 'configurations/layout/initialinstallpackage'
TEXTURE_RES = 0x6BDE20BA
TEXTURE_PARAM = '2259d419-387f-b1ba-30fc-c5140100b012'  # the deck graphic's shader parameter key
TEMPLATE_EDIT = 'f440c076-6457-4866-b8f7-a100b30086e3'   # textureedit id inside the template names
DECK_W, DECK_H, DECK_MIPS = 512, 2048, 12
THUMB, THUMB_LARGE = 256, 768


# ------------------------------------------------------------------ helpers
def djb(text, bits=32):
    mask = (1 << bits) - 1
    h = 5381
    for c in text.encode():
        h = ((h * 33) ^ c) & mask
    return h


def guid_bytes(text):
    """Canonical GUID text -> the 16 bytes EBX/manifests store (first three fields swapped)."""
    c = bytes.fromhex(text.replace('-', ''))
    return bytes([c[3], c[2], c[1], c[0], c[5], c[4], c[7], c[6]]) + c[8:]


def new_guid():
    return str(uuid.uuid4())


def ebxtool(*args):
    result = subprocess.run([EBXTOOL, *map(str, args)], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f'ebxtool {args[0]} failed: {result.stderr.strip() or result.stdout.strip()}')
    return result.stdout


def ebx_info(path):
    """File guid, instance guids and imports, read from `ebxtool dump`."""
    text = ebxtool('dump', path)
    info = {'file': re.search(r'^file (\S+)', text, re.M).group(1),
            'instances': re.findall(r'^instance #\d+ guid=(\S+)', text, re.M),
            'imports': re.findall(r'^import #\d+ file=(\S+) inst=(\S+)', text, re.M),
            'dump': text}
    return info


def edit_ebx(src, dst, *ops):
    ebxtool('edit', src, dst, *ops)
    with open(dst, 'rb') as f:
        return f.read()


# ------------------------------------------------------------------ image work
def deck_outline_mask(width, height):
    """Rounded deck outline (fitted to a real popsicle shape) for art that has no alpha."""
    scale = 4
    w, h = width * scale, height * scale
    mask = Image.new('L', (w, h), 0)
    half = w / 2
    cap = min(h / 2, half * 175 / 131)  # nose/tail length relative to half-width
    n = 2.26
    pts = []
    steps = 200
    for i in range(steps + 1):
        t = i / steps
        pts.append((half + half * (1 - (1 - t) ** n) ** (1 / n), cap * t))
    right = pts + [(x, h - y) for x, y in reversed(pts)]
    left = [(w - x, y) for x, y in reversed(right)]
    ImageDraw.Draw(mask).polygon(right + left, fill=255)
    return mask.resize((width, height), Image.LANCZOS)


def load_deck(path):
    """Returns (art RGBA cropped to the deck, had_alpha)."""
    im = Image.open(path)
    im.load()
    had_alpha = im.mode in ('RGBA', 'LA', 'PA') or (im.mode == 'P' and 'transparency' in im.info)
    im = im.convert('RGBA')
    alpha = im.getchannel('A')
    if had_alpha and alpha.getextrema()[0] < 250:
        box = alpha.point(lambda a: 255 if a > 16 else 0).getbbox()
    else:
        had_alpha = False
        # Crop a uniform border (e.g. a white background around the board).
        rgb = im.convert('RGB')
        diff = ImageChops.difference(rgb, Image.new('RGB', rgb.size, rgb.getpixel((0, 0))))
        box = diff.convert('L').point(lambda v: 255 if v > 24 else 0).getbbox()
    if box:
        im = im.crop(box)
    if im.width > im.height:
        im = im.rotate(90, expand=True)
    return im, had_alpha


def flatten(art):
    """Opaque RGB copy where pixels outside the deck outline repeat the nearest edge
    colour of their row (rows past the nose/tail repeat the nearest row), so nothing
    dark shows at the rails or bleeds into the smaller mips."""
    w, h = art.size
    rgb = art.convert('RGB')
    solid = art.getchannel('A').point(lambda a: 255 if a > 128 else 0).tobytes()
    out = rgb.copy()
    spans = []
    for y in range(h):
        row = solid[y * w:(y + 1) * w]
        left, right = row.find(b'\xff'), row.rfind(b'\xff')
        spans.append((left, right) if left >= 0 else None)
        if left < 0:
            continue
        if left > 0:
            out.paste(rgb.getpixel((left, y)), (0, y, left, y + 1))
        if right < w - 1:
            out.paste(rgb.getpixel((right, y)), (right + 1, y, w, y + 1))
    filled = [y for y, s in enumerate(spans) if s]
    if not filled:
        return rgb
    first, last = filled[0], filled[-1]
    for y in range(first):
        out.paste(out.crop((0, first, w, first + 1)), (0, y))
    for y in range(last + 1, h):
        out.paste(out.crop((0, last, w, last + 1)), (0, y))
    for y in range(first, last + 1):  # gaps inside the deck (fully transparent rows)
        if not spans[y]:
            out.paste(out.crop((0, y - 1, w, y)), (0, y))
    return out


def bc7(image, work, label):
    """Encodes one RGBA image to BC7 with ebxtool."""
    raw = os.path.join(work, label + '.rgba')
    out = os.path.join(work, label + '.bc7')
    with open(raw, 'wb') as f:
        f.write(image.convert('RGBA').tobytes())
    ebxtool('bc7enc', raw, image.width, image.height, out)
    with open(out, 'rb') as f:
        return f.read()


def deck_texture(art, work):
    """BC7 mips of the deck graphic, largest first."""
    base = flatten(art).resize((DECK_W, DECK_H), Image.LANCZOS)
    mips = []
    for level in range(DECK_MIPS):
        w, h = max(1, DECK_W >> level), max(1, DECK_H >> level)
        mip = base if level == 0 else base.resize((w, h), Image.LANCZOS)
        mips.append(bc7(mip, work, f'deck_mip{level}'))
    return mips, base


def thumbnail(art, had_alpha, size):
    """The deck standing upright and centred, on transparency, like the game's own thumbnails."""
    shaped = art.copy()
    if not had_alpha:
        shaped.putalpha(deck_outline_mask(art.width, art.height))
    target_h = round(size * 658 / 768)
    target_w = max(1, round(shaped.width * target_h / shaped.height))
    if target_w > size * 0.9:
        target_w = round(size * 0.9)
        target_h = round(shaped.height * target_w / shaped.width)
    shaped = shaped.resize((target_w, target_h), Image.LANCZOS)
    canvas = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    canvas.alpha_composite(shaped, ((size - target_w) // 2, (size - target_h) // 2))
    return canvas


def texture_header(template, name, chunk_guid, chunk_size, stream_offsets=(0, 0)):
    """A texture resource made from a template header.

    180-byte v12 layout: 0 compressed offsets of mips 1 and 2 inside the streamed chunk
    (the streamer seeks to them), 8 type, 12 format, 20 flags, 22 width/height/depth/slices,
    30 mip count, 31 first mip the bundle carries, 40 chunk GUID, 56 mip sizes,
    116 chunk size, 120 djb64 of the resource's own name, 128 texture group."""
    header = bytearray(template)
    struct.pack_into('<II', header, 0, *stream_offsets)
    header[40:56] = chunk_guid
    assert struct.unpack_from('<I', header, 116)[0] == chunk_size, 'texture size differs from the template'
    struct.pack_into('<Q', header, 120, djb(name, 64))
    return bytes(header)


# ------------------------------------------------------------------ mod assembly
class CasWriter:
    def __init__(self, install_chunk, game_root, compress=True):
        self.data = bytearray()
        self.chunk = install_chunk
        self.game_root = game_root
        self.compress = compress

    def add_raw(self, payload):
        f = fb.FileInfo(True, self.chunk, 1, len(self.data), len(payload))
        self.data += payload
        return f

    def encode(self, decoded):
        encoded = fb.encode_cas(decoded, self.game_root, self.compress)
        assert fb.decode_cas(encoded, self.game_root) == decoded
        return encoded

    def add(self, decoded):
        """CAS-encodes a payload; returns (file info, sha1 of the stored bytes)."""
        return self.add_encoded(self.encode(decoded))

    def add_encoded(self, encoded):
        return self.add_raw(encoded), hashlib.sha1(encoded).digest()


DECK_FIRST_MIP = 3  # like the game's own decks: the bundle carries mips 3+, mips 0-2 stream


def pack_deck_chunk(cas, mips):
    """Encodes the deck texture the way the game stores its own: every mip the
    streamer loads separately (0..first-1) starts a block, and the mip tail it
    loads with the bundle is one block. Returns (streamed chunk, tail, offsets of
    mips 1 and 2 inside the streamed chunk)."""
    parts = [cas.encode(m) for m in mips[:DECK_FIRST_MIP]]
    tail = b''.join(mips[DECK_FIRST_MIP:])
    assert len(tail) <= 0x10000, 'mip tail must fit one CAS block'
    tail_encoded = cas.encode(tail)
    offsets = (len(parts[0]), len(parts[0]) + len(parts[1]))
    return b''.join(parts) + tail_encoded, tail_encoded, offsets


def slugify(name):
    words = re.findall(r'[A-Za-z0-9]+', name)
    if not words:
        raise SystemExit('The deck name needs at least one letter or digit.')
    return '_'.join(w[:1].upper() + w[1:] for w in words)


def build(args):
    game = args.game
    if not os.path.isfile(os.path.join(game, 'Skate.exe')):
        raise SystemExit(f'skate. not found at {game} (use --game)')
    for need in (EBXTOOL,):
        if not os.path.isfile(need):
            raise SystemExit(f'missing {need}: run ebxtool\\build.bat first')

    key_words = slugify(args.name)              # e.g. Blank_Baker
    key = 'Own_' + key_words                     # item key; ReSkate shows it as "Blank Baker"
    ident = 'own_' + key_words.lower()           # asset leaf, e.g. own_blank_baker
    author = re.sub(r'[^A-Za-z0-9_]', '_', args.author).strip('_') or 'socioculture'
    package = re.sub(r'[^A-Za-z0-9_]', '_', args.package or (key_words + '_Deck'))
    folder_name = f'{author}-{package}'
    out_root = os.path.abspath(args.out)
    mod_dir = os.path.join(out_root, folder_name)

    rng = random.SystemRandom()
    edit_id = new_guid()
    names = {
        'item': f'items/board_bottomart/{ident}',
        'thumb_tool': f'thumbnail/tool/{ident}',
        'thumb_tool_lrg': f'thumbnail/tool/{ident}_lrg',
        'thumb_cdn': f'thumbnail/cdn/img_{ident}',
        'thumb_cdn_lrg': f'thumbnail/cdn/img_{ident}_lrg',
        'preset': f'characters/customization/reskate/textureedits/{edit_id}_ap',
        'tex_asset': f'characters/customization/reskate/textureedits/{edit_id}_ap_{TEXTURE_PARAM}',
    }

    print(f'Deck "{args.name}"  key={key}  mod={folder_name}')
    g = fb.GameData(game)
    patch_chunk = next(k for k, v in g.chunk_dirs.items() if v == PATCH_DIRECTORY)
    _, base_bundles, _ = g.toc('Win32/items.toc')
    by_name = {b.name: b for b in base_bundles}
    if SHARED_BUNDLE not in by_name or DECK_BUNDLE not in by_name:
        raise SystemExit('This skate. build does not have the bundles this tool expects.')

    work = tempfile.mkdtemp(prefix='deckmod_')
    try:
        # ---- textures
        art, had_alpha = load_deck(args.image)
        print(f'  art {art.width}x{art.height} (transparent outline: {had_alpha})')
        deck_mips, preview = deck_texture(art, work)
        deck_size = sum(len(m) for m in deck_mips)
        thumbs = {}
        for label, size in (('thumb_tool', THUMB), ('thumb_tool_lrg', THUMB_LARGE)):
            image = thumbnail(art, had_alpha, size)
            thumbs[label] = (image, bc7(image, work, label))

        def tpath(name):
            return os.path.join(TEMPLATE, name)

        def wpath(name):
            return os.path.join(work, name)

        def read(path):
            with open(path, 'rb') as f:
                return f.read()

        # ---- new identities
        ids = {k: {'file': new_guid(), 'inst': new_guid()} for k in
               ('thumb_tool', 'thumb_tool_lrg', 'thumb_cdn', 'thumb_cdn_lrg', 'preset', 'tex_asset')}
        ids['item'] = {'file': new_guid(), 'inst': new_guid(), 'data': new_guid()}
        chunk_ids = {k: uuid.uuid4().bytes for k in ('thumb_tool', 'thumb_tool_lrg', 'tex_asset')}
        # The engine tags resolved resource pointers in the low bits, so an unresolved
        # resource id must have bit 0 set (every id the game ships ends in binary 01 or 11);
        # one ending in 10 is taken for a pointer and crashes the level load.
        res_ids = {k: rng.getrandbits(64) | 1 for k in ('thumb_tool', 'thumb_tool_lrg', 'tex_asset')}

        ebx = {}
        # thumbnails (TextureAsset) + their resources
        res = {}
        for k in ('thumb_tool', 'thumb_tool_lrg'):
            ebx[k] = edit_ebx(tpath(k + '.ebx'), wpath(k + '.ebx'),
                              '--file-guid', ids[k]['file'], '--inst-guid', 0, ids[k]['inst'],
                              '--set', '0:Name', 'str:' + names[k], '--set', '0:Resource', f'res:{res_ids[k]}')
            res[k] = texture_header(read(tpath(k + '.res')), names[k], chunk_ids[k], len(thumbs[k][1]))
        # CDN thumbnail records that the item points at
        for k, tool in (('thumb_cdn', 'thumb_tool'), ('thumb_cdn_lrg', 'thumb_tool_lrg')):
            ebx[k] = edit_ebx(tpath(k + '.ebx'), wpath(k + '.ebx'),
                              '--file-guid', ids[k]['file'], '--inst-guid', 0, ids[k]['inst'],
                              '--import', 0, ids[tool]['file'], ids[tool]['inst'],
                              '--set', '0:Name', 'str:' + names[k],
                              '--set', '0:NameHash', f'u64:{djb(names[k])}',
                              '--set', '0:ContentHash', 'sha1:' + hashlib.sha1(thumbs[tool][1]).hexdigest())
        # deck TextureAsset + resource
        ebx['tex_asset'] = edit_ebx(tpath('tex_asset.ebx'), wpath('tex_asset.ebx'),
                                    '--file-guid', ids['tex_asset']['file'], '--inst-guid', 0, ids['tex_asset']['inst'],
                                    '--set', '0:Name', 'str:' + names['tex_asset'],
                                    '--set', '0:Resource', f'res:{res_ids["tex_asset"]}')
        cas = CasWriter(patch_chunk, game, compress=not args.no_compress)
        deck_stream, deck_tail, stream_offsets = pack_deck_chunk(cas, deck_mips)
        # Header of the game's own deck_baker_graphic_00003_c (BC7 512x2048, first mip 3).
        res['tex_asset'] = texture_header(read(tpath('base_deck_c.res')), names['tex_asset'], chunk_ids['tex_asset'],
                                          deck_size, stream_offsets)
        assert res['tex_asset'][31] == DECK_FIRST_MIP
        # appearance preset: ReSkate's writer drops its boxed values, so patch bytes in place
        tmpl = ebx_info(tpath('preset.ebx'))
        tex_tmpl = ebx_info(tpath('tex_asset.ebx'))
        preset = bytearray(read(tpath('preset.ebx')))
        swaps = [(tmpl['file'], ids['preset']['file']), (tmpl['instances'][0], ids['preset']['inst']),
                 (tex_tmpl['file'], ids['tex_asset']['file']), (tex_tmpl['instances'][0], ids['tex_asset']['inst'])]
        for old, new in swaps:
            ob, nb = guid_bytes(old), guid_bytes(new)
            assert preset.count(ob) >= 1, f'{old} not found in preset template'
            preset = preset.replace(ob, nb)
        old_text, new_text = TEMPLATE_EDIT.encode(), edit_id.encode()
        assert len(old_text) == len(new_text) and preset.count(old_text) == 1
        preset = preset.replace(old_text, new_text)
        ebx['preset'] = bytes(preset)
        with open(wpath('preset.ebx'), 'wb') as f:
            f.write(ebx['preset'])
        # the item itself
        item_tmpl = ebx_info(tpath('item.ebx'))
        imports = item_tmpl['imports']
        thumb_slot = [i for i, (f_, _) in enumerate(imports) if f_ == ebx_info(tpath('thumb_cdn.ebx'))['file']][0]
        large_slot = [i for i, (f_, _) in enumerate(imports) if f_ == ebx_info(tpath('thumb_cdn_lrg.ebx'))['file']][0]
        ebx['item'] = edit_ebx(tpath('item.ebx'), wpath('item.ebx'),
                               '--file-guid', ids['item']['file'],
                               '--inst-guid', 0, ids['item']['inst'], '--inst-guid', 1, ids['item']['data'],
                               '--set', '0:Name', 'str:' + names['item'],
                               '--set', '0:Key', 'str:' + key,
                               '--set', '0:HashedAssetKey', f'u64:{djb(key)}',
                               '--import', thumb_slot, ids['thumb_cdn']['file'], ids['thumb_cdn']['inst'],
                               '--import', large_slot, ids['thumb_cdn_lrg']['file'], ids['thumb_cdn_lrg']['inst'],
                               '--set', '0:ItemData.AssetPaths[0].AssetName', 'str:' + names['preset'].rsplit('/', 1)[1])

        # ---- shared bundle: base + item, thumbnails, item collection and bundle-reference edits
        sb_files, _ = fb.read_bundle_region(by_name[SHARED_BUNDLE].region)
        s_ebx, s_res, s_chunks, s_meta = g.manifest(sb_files)
        ne, nr = len(s_ebx), len(s_res)
        sb_ebx_files = sb_files[1:1 + ne]
        sb_res_files = sb_files[1 + ne:1 + ne + nr]
        sb_chunk_files = sb_files[1 + ne + nr:]

        ic_index = next(i for i, a in enumerate(s_ebx) if a.name == ITEM_COLLECTION)
        with open(wpath('collection.ebx'), 'wb') as f:
            f.write(g.payload(sb_ebx_files[ic_index]))
        ebxtool('additem', wpath('collection.ebx'), wpath('collection_new.ebx'), ids['item']['file'], ids['item']['inst'])
        collection = read(wpath('collection_new.ebx'))
        fi, sha = cas.add(collection)
        s_ebx[ic_index].sha1, s_ebx[ic_index].original_size = sha, len(collection)
        sb_ebx_files[ic_index] = fi

        br_index = next(i for i, a in enumerate(s_res) if a.name == BUNDLE_REF_TABLE)
        table = bundleref.Table(g.payload(sb_res_files[br_index]), s_res[br_index].res_meta)
        table.insert(names['preset'], table.bundle_index(DECK_BUNDLE_REF))
        fi, sha = cas.add(bytes(table.data))
        s_res[br_index].sha1, s_res[br_index].original_size = sha, len(table.data)
        s_res[br_index].res_meta = bytes(table.meta)
        sb_res_files[br_index] = fi

        new_ebx, new_ebx_files = [], []
        for k in ('item', 'thumb_tool', 'thumb_tool_lrg', 'thumb_cdn', 'thumb_cdn_lrg'):
            fi, sha = cas.add(ebx[k])
            new_ebx.append(fb.Asset(kind='ebx', name=names[k], sha1=sha, original_size=len(ebx[k])))
            new_ebx_files.append(fi)
        new_res, new_res_files = [], []
        for k in ('thumb_tool', 'thumb_tool_lrg'):
            fi, sha = cas.add(res[k])
            new_res.append(fb.Asset(kind='res', name=names[k], sha1=sha, original_size=len(res[k]), res_type=TEXTURE_RES,
                                    res_meta=bytes.fromhex('0c000000090000000000000000000000'), res_id=res_ids[k]))
            new_res_files.append(fi)
        toc_chunks = []
        for k in ('thumb_tool', 'thumb_tool_lrg'):
            fi, _ = cas.add(thumbs[k][1])
            toc_chunks.append(fb.TocChunk(chunk_ids[k], True, patch_chunk, 1, fi.offset, fi.size))

        manifest = fb.write_binary_bundle(s_ebx + new_ebx, s_res + new_res, s_chunks, s_meta)
        sb_region = fb.write_bundle_region([cas.add_raw(manifest)] + sb_ebx_files + new_ebx_files + sb_res_files +
                                           new_res_files + sb_chunk_files)

        # ---- deck bundle: base + preset, texture asset, texture resource and chunk
        db_files, _ = fb.read_bundle_region(by_name[DECK_BUNDLE].region)
        d_ebx, d_res, d_chunks, d_meta = g.manifest(db_files)
        ne, nr = len(d_ebx), len(d_res)
        db_ebx_files, db_res_files, db_chunk_files = db_files[1:1 + ne], db_files[1 + ne:1 + ne + nr], db_files[1 + ne + nr:]
        add_ebx, add_ebx_files = [], []
        for k in ('preset', 'tex_asset'):
            fi, sha = cas.add(ebx[k])
            add_ebx.append(fb.Asset(kind='ebx', name=names[k], sha1=sha, original_size=len(ebx[k])))
            add_ebx_files.append(fi)
        fi, sha = cas.add(res['tex_asset'])
        add_res = [fb.Asset(kind='res', name=names['tex_asset'], sha1=sha, original_size=len(res['tex_asset']),
                            res_type=TEXTURE_RES, res_meta=bytes.fromhex('0c000000010000000000000000000000'),
                            res_id=res_ids['tex_asset'])]
        add_res_files = [fi]
        # The full texture is a TOC chunk the streamer reads mip by mip; the bundle
        # carries its own copy of the mip tail (logical range firstMip..end).
        fi = cas.add_raw(deck_stream)
        toc_chunks.append(fb.TocChunk(chunk_ids['tex_asset'], True, patch_chunk, 1, fi.offset, fi.size))
        tail_offset = sum(len(m) for m in deck_mips[:DECK_FIRST_MIP])
        fi, sha = cas.add_encoded(deck_tail)
        add_chunks = [fb.Asset(kind='chunk', name=fb.guid_str(chunk_ids['tex_asset']), guid=chunk_ids['tex_asset'],
                               sha1=sha, logical_offset=tail_offset, logical_size=deck_size - tail_offset)]
        add_chunk_files = [fi]
        meta_tree, _ = fb.read_db(d_meta, 0)
        meta_tree['children'].append({'type': 2, 'name': None, 'terminated': True, 'children': [
            {'type': 9, 'name': 'h64', 'value': struct.pack('<Q', djb(names['tex_asset'], 64))},
            {'type': 2, 'name': 'meta', 'terminated': True, 'children': [
                {'type': 8, 'name': 'firstMip', 'value': struct.pack('<i', DECK_FIRST_MIP)}]}]})
        manifest = fb.write_binary_bundle(d_ebx + add_ebx, d_res + add_res, d_chunks + add_chunks, fb.write_db(meta_tree))
        db_region = fb.write_bundle_region([cas.add_raw(manifest)] + db_ebx_files + add_ebx_files + db_res_files +
                                           add_res_files + db_chunk_files + add_chunk_files)

        toc = fb.write_patch_toc([fb.TocBundle(SHARED_BUNDLE, sb_region), fb.TocBundle(DECK_BUNDLE, db_region)],
                                 toc_chunks, 3)

        # ---- write the mod folder
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
        with open(os.path.join(game, 'Skate.exe'), 'rb') as f:
            skate_sha = hashlib.sha256(f.read()).hexdigest()
        with open(os.path.join(mod_dir, '.reskate-studio-patch'), 'w', newline='\n') as f:
            f.write(f'ReSkate Studio native Patch v1\nskate_sha256={skate_sha}\n')
        with open(os.path.join(mod_dir, 'manifest.json'), 'w') as f:
            json.dump({'author': author, 'dependencies': [], 'description': f'{args.name} deck graphic',
                       'name': package, 'version_number': args.version, 'website_url': ''}, f, indent=4)
        with open(os.path.join(mod_dir, 'README.md'), 'w') as f:
            f.write(f'# {args.name}\n\nAdds the "{args.name}" deck graphic to skate. (ReSkate). '
                    f'Find it under board graphics.\n')
        thumbs['thumb_tool'][0].save(os.path.join(mod_dir, 'icon.png'))
        preview.save(os.path.join(out_root, f'{folder_name}-texture-preview.png'))

        zip_path = os.path.join(out_root, f'{folder_name}.zip')
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
            for dirpath, _, files in os.walk(mod_dir):
                for name in files:
                    full = os.path.join(dirpath, name)
                    z.write(full, os.path.relpath(full, mod_dir))
        print(f'  wrote {mod_dir}')
        print(f'  wrote {zip_path}')

        if not args.no_verify:
            check = subprocess.run([sys.executable, os.path.join(HERE, 'verify_mod.py'), mod_dir, '--game', game],
                                   capture_output=True, text=True)
            if check.returncode:
                print(check.stdout, check.stderr)
                raise SystemExit('Verification failed; the mod was not installed.')
            print('  verified: every reference in the mod resolves')

        if args.install:
            target = os.path.join(game, 'Mods', folder_name)
            if os.path.exists(target):
                shutil.rmtree(target)
            shutil.copytree(mod_dir, target)
            print(f'  installed to {target}')
        return mod_dir
    finally:
        if args.keep_work:
            print('  work files kept in', work)
        else:
            shutil.rmtree(work, ignore_errors=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('image', help='deck image (PNG/JPG), nose at the top')
    p.add_argument('--name', required=True, help='deck name shown in game, e.g. "Blank Baker"')
    p.add_argument('--author', default='socioculture', help='mod author (folder prefix)')
    p.add_argument('--package', help='mod package name (default: <Name>_Deck)')
    p.add_argument('--version', default='1.0.0')
    p.add_argument('--game', default=DEFAULT_GAME, help='skate. install folder')
    p.add_argument('--out', default=os.path.join(HERE, '..', 'mods'), help='where to write the mod')
    p.add_argument('--install', action='store_true', help="also copy the mod into the game's Mods folder")
    p.add_argument('--no-compress', action='store_true', help='store CAS data uncompressed')
    p.add_argument('--keep-work', action='store_true')
    p.add_argument('--no-verify', action='store_true', help='skip the post-build checks')
    build(p.parse_args())


if __name__ == '__main__':
    main()
