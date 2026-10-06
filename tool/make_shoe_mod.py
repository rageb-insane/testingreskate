"""Turns a 3D shoe model (.gltf or .glb, a pair or one shoe) into a ReSkate mod that adds it
to skate. as a new shoe.

    python make_shoe_mod.py model/scene.gltf --name "Dior Jordan 1" [--credit "..."] [--install]

Display props, posing and orientation in the model don't matter: shoe_prep stands each shoe
up, works out which foot it is, and leaves out backdrops and floors.

The mesh is fitted onto the game's own high-top sneaker (gen_sneaker_vertclassic):
same feet, toe-out and length, skinned by copying that sneaker's bone weights. It is
added beside that sneaker the way working custom-shoe mods do (template_shoe/):
  * mesh bundle: <id>_mesh (SkinnedMeshAsset), <id>_geometry, <id>_variation_2,
    a MeshSet patched for the new geometry, and one chunk per detail level;
  * colorway bundle: appearance presets and five textures (color from the model,
    neutral normal/mask/stitch maps), in a sneaker colorway no other installed mod
    ships (see pick_colorway);
  * shared bundle: the item, its thumbnails, the season item collection and the
    character bundle-reference table.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import uuid
import zipfile
import random

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fb  # noqa: E402
import atlas  # noqa: E402
import bundleref  # noqa: E402
import gltf  # noqa: E402
import objfile  # noqa: E402
import meshreduce  # noqa: E402
import meshset  # noqa: E402
import shoe_mesh  # noqa: E402
import shoe_prep  # noqa: E402
import tangentspace  # noqa: E402
import thumbs as icon_renderer  # noqa: E402
from make_deck_mod import (EBXTOOL, DEFAULT_GAME, SHARED_BUNDLE, BUNDLE_REF_TABLE, PATCH_DIRECTORY, TEXTURE_RES,  # noqa: E402
                           djb, guid_bytes, new_guid, ebxtool, ebx_info, edit_ebx, bc7, texture_header, CasWriter,
                           slugify)

TEMPLATE = os.path.join(HERE, 'template_shoe')
MESH_BUNDLE = shoe_mesh.BASE_MESH_BUNDLE
COLORWAY_DIR = ('win32/characters/maincharacters/generic/cas/clothing/unlicensed/generic/footwear/shoe/sneaker/'
                'vertclassic/2025/colorways/')
COLORWAY_BUNDLE = COLORWAY_DIR + 'gen_sneaker_vertclassic_00001_ap_cas_main_bundlereftable'   # the template's
MESH_BUNDLE_REF = 'characters/maincharacters/generic/cas/clothing/unlicensed/generic/footwear/shoe/sneaker/vertclassic/2025/gen_sneaker_vertclassic'
ITEM_COLLECTION = 'items/_seasons/0.27.5/ownables/0.27.5_itemcollection'
MAX_VERTICES = 64000                      # under the 16-bit index limit (65535), with room to spare
FOOT_REGIONS = (0x7C7EB057, 0xB17618AF)   # the body's feet, which the template (VertClassic) item hides
FOOT_REGION_TOP = 0.133                   # how high up the ankle those regions reach (metres)
PUFFY90S_BUNDLE = ('win32/characters/maincharacters/generic/cas/clothing/unlicensed/generic/footwear/shoe/sneaker/'
                   'puffy90s/2025/gen_sneaker_puffy90s_cas_main_bundlereftable')
BASE_NAMES = {'vertclassic': 'VertClassic sneaker', 'puffy90s': 'Puffy 90s sneaker', 'dunkhigh': 'Nike SB Dunk High'}
TEXTURE_KEYS = {  # shader parameter -> what goes in it
    '0bb23445': 'color', '3f4cd4ca': 'normal', '00c854f4': 'mask', '89aaa101': 'stitches'}
NEUTRAL = {'normal': (128, 128, 255, 255), 'mask': (0, 87, 0, 255), 'stitches': (131, 0, 114, 255)}


def read(path):
    with open(path, 'rb') as f:
        return f.read()


# ------------------------------------------------------------------ bundles
def bundle_ref(bundle_name):
    """A bundle's name as the bundle-reference table spells it."""
    return bundle_name[len('win32/'):].replace('_cas_main_bundlereftable', '')


def ebx_file_guid(data):
    """The file guid of an EBX document: the start of its EFIX chunk."""
    at = 12
    while at + 8 <= len(data):
        size = struct.unpack_from('<I', data, at + 4)[0]
        if data[at:at + 4] == b'EFIX':
            return fb.guid_str(data[at + 8:at + 24])
        at += 8 + size + (size & 1)
    raise ValueError('EBX document without an EFIX chunk')


def earlier_builds(game, own_folder):
    """Installed folders holding this same package under another author name (an earlier
    build of this shoe): not a conflict, and replaced when this one is installed."""
    package = own_folder.split('-', 1)[-1]
    mods = os.path.join(game, 'Mods')
    return [m for m in sorted(os.listdir(mods)) if m != own_folder and m.endswith('-' + package)] \
        if os.path.isdir(mods) else []


def bundles_of_other_mods(game, own_folder):
    """{bundle name: [mod folders]} for every installed mod but this one, enabled or not."""
    shipped = {}
    mods = os.path.join(game, 'Mods')
    skip = {own_folder, *earlier_builds(game, own_folder)}
    for mod in sorted(os.listdir(mods)) if os.path.isdir(mods) else []:
        if mod.startswith('.') or mod in skip:
            continue
        for dirpath, _, files in os.walk(os.path.join(mods, mod, 'Win32')):
            for name in files:
                if not name.endswith('.toc'):
                    continue
                try:
                    _, bundles, _ = fb.read_toc(read(os.path.join(dirpath, name)))
                except Exception:
                    continue
                for b in bundles:
                    shipped.setdefault(b.name, []).append(mod)
    return shipped


def pick_colorway(g, by_name, imports, taken):
    """The vertclassic colorway bundle the presets and textures go in.

    When two mods ship the same bundle, ReSkate merges them but keeps only the game's
    chunk metadata, so the merged bundle loses each added texture's firstMip entry. The
    game then streams the texture's top mips without its bundled mip tail and copies past
    the end of the buffer (a crash as soon as the shoe is previewed). So the textures go
    in a colorway no other installed mod ships. `imports` are the presets' (file, instance)
    imports; the colorway must hold every one, except:
      * a colorway texture, which is swapped for the chosen colorway's own (the template's
        presets list one they never use);
      * a shader parameter or other non-texture asset the template's colorway holds: it is
        carried into the chosen bundle (every bundle keeps its own copy of such shared
        assets), with whatever it imports in turn.
    Returns (bundle name, [(old guid, new guid)], [(asset name, payload) to add])."""
    def contents(name):
        files, _ = fb.read_bundle_region(by_name[name].region)
        assets = g.manifest(files)[0]
        return {ebx_file_guid(data): (a.name, data) for a, data in
                ((a, g.payload(files[1 + i])) for i, a in enumerate(assets))}

    def info_of(data):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.ebx') as tmp:
            tmp.write(data)
        try:
            return ebx_info(tmp.name)
        finally:
            os.unlink(tmp.name)
    template = contents(COLORWAY_BUNDLE)
    candidates = sorted(n for n in by_name if n.startswith(COLORWAY_DIR) and n.endswith('_ap_cas_main_bundlereftable'))
    candidates.sort(key=lambda n: n != COLORWAY_BUNDLE)
    choices = []
    for name in candidates:
        if name in taken:
            continue
        have = template if name == COLORWAY_BUNDLE else contents(name)
        swaps, carry = [], {}
        pending = list(imports)
        while pending and swaps is not None:
            file_guid, inst_guid = pending.pop()
            if file_guid in have or file_guid in carry:
                continue
            old_name, old_data = template.get(file_guid, ('?', b''))
            if old_data and 'TextureAsset' not in info_of(old_data)['dump'].splitlines()[0]:
                carry[file_guid] = (old_name, old_data)      # a shared non-texture asset: bring it along
                pending += info_of(old_data)['imports']
                continue
            suffix = old_name.rsplit('_', 1)[-1]
            # the colorway's own texture of that kind, else the sneaker's shared one
            same = sorted((('/colorways/' not in n), n, f, d) for f, (n, d) in have.items()
                          if suffix in ('c', 'ny', 'rgb', 'msk') and '/gen_sneaker_vertclassic_' in old_name
                          and '/gen_sneaker_vertclassic_' in n and n.endswith('_' + suffix))
            if not same:
                swaps = None
                break
            _, n, f, d = same[0]
            swaps += [(file_guid, f), (inst_guid, info_of(d)['instances'][0])]
        if swaps is not None:
            choices.append((len(swaps) + len(carry), name, swaps, list(carry.values())))
            if not swaps and not carry:
                break
    if not choices:
        raise SystemExit('Every sneaker colorway bundle is already shipped by another installed mod (one per custom\n'
                         'shoe); ReSkate would merge them and drop this mod\'s texture metadata (the game crashes on that).\n'
                         'Remove a custom shoe mod you no longer use to free one.')
    _, name, swaps, carry = min(choices, key=lambda c: c[:2])
    return name, swaps, carry


# ------------------------------------------------------------------ textures
def mip_chain(image, levels):
    out = [image]
    for i in range(1, levels):
        w, h = max(1, image.width >> i), max(1, image.height >> i)
        out.append(image.resize((w, h), Image.LANCZOS))
    return out


def bc1_constant(width, height, rgb):
    r, g, b = rgb[:3]
    c = ((r * 31 + 127) // 255) << 11 | ((g * 63 + 127) // 255) << 5 | ((b * 31 + 127) // 255)
    block = struct.pack('<HHI', c, c, 0)
    return block * (max(1, (width + 3) // 4) * max(1, (height + 3) // 4))


def texture_mips(kind, size, color_image, work, label, normal_image=None):
    """The mip chain of one shoe texture, each mip already block-compressed."""
    levels = int(np.log2(size)) + 1
    if kind == 'color':
        return [bc7(m, work, f'{label}_{i}') for i, m in enumerate(mip_chain(color_image.resize((size, size), Image.LANCZOS), levels))]
    if kind == 'normal' and normal_image is not None:
        return [bc7(m, work, f'{label}_{i}') for i, m in enumerate(normal_mips(normal_image, size, levels))]
    if kind == 'normal':
        return [bc7(Image.new('RGBA', (max(1, size >> i),) * 2, NEUTRAL['normal']), work, f'{label}_{i}') for i in range(levels)]
    return [bc1_constant(max(1, size >> i), max(1, size >> i), NEUTRAL[kind]) for i in range(levels)]


def normal_mips(image, size, levels):
    """Mips of a tangent-space normal map: each level averaged from the full map as vectors and
    renormalised (averaging the colours would shorten them and flatten the detail)."""
    import cv2
    n = np.asarray(image.convert('RGB'), np.float32) / 127.5 - 1
    out = []
    for i in range(levels):
        s = max(1, size >> i)
        v = cv2.resize(n, (s, s), interpolation=cv2.INTER_AREA).reshape(s, s, 3)
        v /= np.maximum(np.linalg.norm(v, axis=2, keepdims=True), 1e-6)
        rgb = ((v + 1) * 127.5).round().clip(0, 255).astype(np.uint8)
        out.append(Image.fromarray(np.dstack([rgb, np.full((s, s), 255, np.uint8)]), 'RGBA'))
    return out


def green_points_up(image):
    """Guesses whether a tangent-space normal map is OpenGL-style (green toward the top of the
    image) or DirectX-style (toward the bottom). A normal map is the slope of a surface, so its
    two slope channels only agree (curl-free) when green is read the right way round. Measured at
    full size: shrunk, fine grain blurs into noise and the answer can flip (it did on the
    CGTrader Jordan 1 pack, a DirectX map that reads OpenGL at 1024). Baked maps follow the low
    poly's facets as well as the surface, so the agreement can be weak: returns (up, sure), where
    sure is False when the two readings differ by under 25%; pass --normal-style then."""
    n = np.asarray(image.convert('RGB'), np.float32)
    nx, ny = (n[..., 0] - 128) / 127, (n[..., 1] - 128) / 127
    a, b = np.diff(nx, axis=0)[:, :-1], np.diff(ny, axis=1)[:-1, :]
    ok = (np.abs(a) < 0.3) & (np.abs(b) < 0.3)            # not across island seams
    down, up = np.mean((a[ok] - b[ok]) ** 2), np.mean((a[ok] + b[ok]) ** 2)
    return bool(up < down), bool(max(up, down) > 1.25 * min(up, down))


def fill_holes(image, shade):
    """A colour texture's see-through texels (alpha under half: a netting's holes, which the model
    shows the lining through) filled from the opaque texels around them, times `shade` (the
    lining is in the netting's shadow): the game's shoe shader draws every texel opaque, so
    left alone they show whatever colour the model kept under them (black, on the Jordan 4).
    Returns (RGBA image, texels filled)."""
    a = np.asarray(image.convert('RGBA'))
    holes = a[..., 3] < 128
    filled = np.asarray(atlas.pad(Image.fromarray(a[..., :3]), ~holes), np.float32)
    rgb = a[..., :3].astype(np.float32)
    rgb[holes] = filled[holes] * shade
    out = np.dstack([np.clip(rgb, 0, 255).astype(np.uint8), np.full(holes.shape, 255, np.uint8)])
    return Image.fromarray(out, 'RGBA'), int(holes.sum())


METAL_TINT = np.array([0.98, 0.99, 1.02], np.float32)     # silver reads a touch cool


def paint_metal(image, metallic, shade):
    """The parts a metallic map marks (eyelets, logos) painted as a flat silver: their colour
    times `shade` and a slight cool tint, blended in by the map. A PBR model leaves metal
    white in its colour texture (its reflections make it silver); the game's shoe shader has
    no metal input, so it would draw them as white leather. Returns (image, share painted)."""
    a = np.asarray(image.convert('RGBA')).astype(np.float32)
    m = np.asarray(metallic.convert('L').resize(image.size, Image.BILINEAR), np.float32)[..., None] / 255
    a[..., :3] = a[..., :3] * (1 - m) + a[..., :3] * shade * METAL_TINT * m
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), 'RGBA'), float((m > 0.5).mean())


# what a chrome surface shows by how much it faces up (-1 down .. 1 up): bright and dark bands,
# as chrome picks up sky, horizon, buildings and ground, so it reads as metal at any angle
CHROME_UP = [-1.0, -0.6, -0.3, -0.1, 0.05, 0.15, 0.35, 0.55, 0.75, 1.0]
CHROME_SKY = [165, 120, 205, 85, 60, 200, 248, 140, 238, 252]


def chrome_metal(image, metallic, shoe, normal_image=None, green_up=False):
    """The parts a metallic map marks painted as chrome: the game's shoe shader has no metal input
    and draws them as plain colour, so a flat grey looks like grey plastic. Each metal texel gets
    a baked reflection of a simple outdoor scene by the way its surface faces (bright sky
    above, a dark band at the horizon, lighter ground below), from the shoe's own normals
    and normal map: curved and bevelled parts (an eyelet's edges and holes) catch bright and
    dark bands as chrome does. Both feet share the texture; facing up or down is the same on
    each. The texture's own shading is kept as detail. Returns (image, share of the texture)."""
    a = np.asarray(image.convert('RGBA')).astype(np.float32)
    h, w = a.shape[:2]
    m = np.asarray(metallic.convert('L').resize((w, h), Image.BILINEAR), np.float32) / 255
    metal = m > 0.02
    nmap = None
    if normal_image is not None:
        nmap = np.asarray(normal_image.convert('RGB').resize((w, h), Image.BILINEAR), np.float32) / 127.5 - 1
        if green_up:
            nmap[..., 1] *= -1
    up = np.full((h, w), np.nan, np.float32)
    pos, uv, tri = shoe.positions, shoe.uv, shoe.triangles
    tri = tri[(pos[tri][:, :, 0] > 0).all(1)]                   # one foot is enough: the other mirrors it
    for t in tri:
        xs, ys = uv[t, 0] * w, uv[t, 1] * h
        x0, x1 = max(int(xs.min()), 0), min(int(xs.max()) + 1, w - 1)
        y0, y1 = max(int(ys.min()), 0), min(int(ys.max()) + 1, h - 1)
        if x0 > x1 or y0 > y1 or not metal[y0:y1 + 1, x0:x1 + 1].any():
            continue
        d = (xs[1] - xs[0]) * (ys[2] - ys[0]) - (xs[2] - xs[0]) * (ys[1] - ys[0])
        if abs(d) < 1e-12:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        w1 = ((gx - xs[0]) * (ys[2] - ys[0]) - (xs[2] - xs[0]) * (gy - ys[0])) / d
        w2 = ((xs[1] - xs[0]) * (gy - ys[0]) - (gx - xs[0]) * (ys[1] - ys[0])) / d
        w0 = 1 - w1 - w2
        ins = (w0 >= -1e-3) & (w1 >= -1e-3) & (w2 >= -1e-3) & metal[y0:y1 + 1, x0:x1 + 1]
        if not ins.any():
            continue
        W = np.stack([w0[ins], w1[ins], w2[ins]], 1)
        n = W @ shoe.normals[t]
        if nmap is not None:
            ts = nmap[y0:y1 + 1, x0:x1 + 1][ins]
            n = ts[:, :1] * (W @ shoe.tangents[t]) + ts[:, 1:2] * (W @ shoe.bitangents[t]) + ts[:, 2:] * n
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
        up[y0:y1 + 1, x0:x1 + 1][ins] = n[:, 1]
    seen = metal & np.isfinite(up)
    sky = np.interp(np.nan_to_num(up), CHROME_UP, CHROME_SKY).astype(np.float32)
    lum = a[..., :3].mean(2)
    detail = np.clip(lum / max(float(np.median(lum[seen])) if seen.any() else 1.0, 1.0), 0.6, 1.1)
    chrome = (sky * detail)[..., None] * METAL_TINT
    k = (m * seen)[..., None]
    a[..., :3] = a[..., :3] * (1 - k) + chrome * k
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), 'RGBA'), float((m > 0.5).mean())


def gold_parts(shoe, image, palette_uv, name, colour):
    """A palette model's parts whose material name contains `name` (the aglets) made shiny gold:
    the free bottom of the palette texture becomes a little environment (bright sky above, a dark
    band at the horizon, lighter ground below, two soft lights to the sides) in `colour` (the
    game's gold deck, deck_gen_graphic_00041: #fcce71), and each of those points looks it up by
    the way it faces (up/down, and around), so the colour runs in bands over the part as on
    polished metal. The game's shoe shader has no metal input, so the shine is painted in."""
    uvs = [uv for n, uv in palette_uv.items() if name.lower() in n.lower()]
    if not uvs:
        raise SystemExit(f'No material named like "{name}" in the model.')
    mask = np.zeros(len(shoe.uv), bool)
    for uv in uvs:
        mask |= np.abs(shoe.uv - np.array(uv)).max(1) < 1e-6
    a = np.asarray(image.convert('RGBA')).astype(np.float32)
    h, w = a.shape[:2]
    y0, y1 = int(h * 0.66), int(h * 0.98)                  # under the colour squares (they fill the top 5/8)
    gx, gy = np.meshgrid((np.arange(w) + 0.5) / w, (np.arange(y0, y1) + 0.5 - y0) / (y1 - y0))
    up = 1 - 2 * gy
    around = gx * np.pi
    sky = np.interp(up, CHROME_UP, CHROME_SKY) / 255
    lights = 0.55 * np.exp(-((around - 0.9) ** 2) / 0.04) + 0.35 * np.exp(-((around - 2.3) ** 2) / 0.06)
    value = np.clip(0.2 + 0.85 * sky + lights * np.clip(1 - np.abs(up) * 1.4, 0, 1), 0.18, 1.25)
    gold = np.array([int(colour.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4)], np.float32)
    rgb = gold * np.minimum(value, 1)[..., None] + (255 - gold) * np.clip(value - 1, 0, 1)[..., None] * 2
    a[y0:y1, :, :3] = np.clip(rgb, 0, 255)
    n = shoe.normals[mask]
    shoe.uv = shoe.uv.copy()
    shoe.uv[mask, 0] = 0.01 + 0.98 * np.abs(np.arctan2(n[:, 0], n[:, 2])) / np.pi   # |angle|: both feet alike, no seam
    shoe.uv[mask, 1] = (y0 + (y1 - y0) * (1 - np.clip(n[:, 1], -1, 1)) / 2) / h
    shoe.tangents, shoe.bitangents = shoe_mesh.tangents(shoe.positions, shoe.normals, shoe.uv, shoe.triangles)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), 'RGBA'), int(mask.sum())


def sized_header(header, size, mips):
    """A texture header from the template's, for a texture of another size carried whole in the
    bundle (first mip 0): width and height, mip count, each mip's size and the chunk size."""
    header = bytearray(header)
    sizes = [len(m) for m in mips]
    struct.pack_into('<HH', header, 22, size, size)
    header[30], header[31] = len(mips), 0
    struct.pack_into('<15I', header, 56, *(sizes + [0] * (15 - len(sizes))))
    struct.pack_into('<I', header, 116, sum(sizes))
    return bytes(header)


def pack_texture(cas, mips, first):
    """CAS-encodes a texture with every mip starting its own blocks (the header's offsets of
    mips 1 and 2 point at them). first > 0 is the game's own layout: mips first.. are the tail
    the bundle carries. first == 0 is the community mods' layout: the bundle carries the whole
    texture, so nothing has to stream (this is the default; see build()).
    Returns (chunk, the bundle's copy, (offset of mip 1, offset of mip 2))."""
    if first == 0:
        parts = [cas.encode(m) for m in mips]
        stream = b''.join(parts)
        return stream, stream, (len(parts[0]), len(parts[0]) + len(parts[1]))
    parts = [cas.encode(m) for m in mips[:first]]
    tail = b''.join(mips[first:])
    assert len(tail) <= 0x10000, 'mip tail must fit one CAS block'
    tail_encoded = cas.encode(tail)
    parts.append(tail_encoded)
    return b''.join(parts), tail_encoded, (len(parts[0]), len(parts[0]) + len(parts[1]))


# ------------------------------------------------------------------ mesh
def section_vertices(shoe, section):
    """Encodes the shoe's vertices with this section's layout (stream-major)."""
    n = len(shoe.positions)
    eight = any(u == 3 for u, *_ in section.elements)
    w = shoe.weights.astype(np.int64)
    b = shoe.bones.astype(np.uint16)
    if not eight:   # keep the 4 strongest influences, renormalised to 255
        order = np.argsort(-w, axis=1)[:, :4]
        w = np.take_along_axis(w, order, 1)
        b = np.take_along_axis(b, order, 1)
        total = w.sum(1, keepdims=True)
        w = np.where(total > 0, np.floor(w * 255 / np.maximum(total, 1)), 0).astype(np.int64)
        w[:, 0] += 255 - w.sum(1)
    pos = np.c_[shoe.positions, np.ones(n)].astype(np.float16)
    uv = shoe.uv.astype(np.float16)
    frame = tangentspace.encode(shoe.normals, shoe.tangents, shoe.bitangents)
    by_usage = {1: pos.tobytes(), 2: b[:, :4].astype('<u2').tobytes(), 4: w[:, :4].astype(np.uint8).tobytes(),
                0x21: uv.tobytes(), 0x22: uv.tobytes(), 0x34: frame.tobytes()}
    if eight:
        by_usage[3] = b[:, 4:8].astype('<u2').tobytes()
        by_usage[5] = w[:, 4:8].astype(np.uint8).tobytes()
    out = bytearray()
    for stream, stride in enumerate(section.strides):
        usage = [u for u, f, o, s in section.elements if s == stream]
        assert len(usage) == 1, 'one element per stream expected'
        data = by_usage[usage[0]]
        assert len(data) == stride * n, f'stream {stream} size'
        out += data
    return bytes(out)


def build_meshset(template, shoe, mesh_id, template_mesh_id, chunk_ids):
    """Patches the template MeshSet for the new geometry; returns (resource, [chunk bytes per LOD])."""
    data = bytearray(template.replace(template_mesh_id.encode(), mesh_id.encode()))
    ms = meshset.MeshSet(bytes(data))
    lo, hi = shoe.positions.min(0), shoe.positions.max(0)
    struct.pack_into('<3f', data, 0x10, *lo)
    struct.pack_into('<3f', data, 0x20, *hi)
    struct.pack_into('<I', data, 0x78, djb(ms.name))
    tris = shoe.triangles.astype('<u2').tobytes()
    chunks = []
    for lod, chunk_id in zip(ms.lods, chunk_ids):
        vertex = bytearray()
        start_index = 0
        for s in lod.sections:
            at = s.at
            struct.pack_into('<4I', data, at + 0x20, len(shoe.triangles), start_index, len(vertex), len(shoe.positions))
            if start_index or len(vertex):
                struct.pack_into('<I', data, at + 0x54, len(vertex))
                struct.pack_into('<I', data, at + 0x140, start_index)
            struct.pack_into('<3f', data, at + 0x150, *lo)
            struct.pack_into('<3f', data, at + 0x160, *hi)
            vertex += section_vertices(shoe, s)
            start_index += len(shoe.triangles) * 3
        index = tris * len(lod.sections)
        struct.pack_into('<II', data, lod.at + 0x58, len(index), len(vertex))
        data[lod.at + 0x74:lod.at + 0x84] = chunk_id
        struct.pack_into('<I', data, lod.at + 0xA4, djb(lod.name[len('Mesh:'):]))
        chunk = bytes(vertex) + index
        chunks.append(chunk + bytes(-len(chunk) % 16))
    return bytes(data), chunks


# ------------------------------------------------------------------ thumbnails
def render_shoe(shoe, color_image, size, side=+1):
    """A simple shaded side view of one shoe on transparency, for the item thumbnails."""
    sel = shoe.positions[:, 0] * side > 0
    keep = sel[shoe.triangles].all(1)
    tri = shoe.triangles[keep]
    p = shoe.positions
    tex = np.asarray(color_image.convert('RGB').resize((256, 256)), dtype=np.float64)
    uvc = shoe.uv[tri].mean(1)
    tx = np.clip((uvc[:, 0] % 1) * 255, 0, 255).astype(int)
    ty = np.clip((uvc[:, 1] % 1) * 255, 0, 255).astype(int)
    base = tex[ty, tx]
    v0, v1, v2 = p[tri[:, 0]], p[tri[:, 1]], p[tri[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
    view = np.array([side * 0.9, 0.35, 0.25])    # outer side, slightly from above
    view /= np.linalg.norm(view)
    facing = fn @ view > 0
    tri, base, fn = tri[facing], base[facing], fn[facing]
    light = np.array([0.5, 0.8, 0.3])
    light /= np.linalg.norm(light)
    shade = 0.55 + 0.45 * np.clip(fn @ light, 0, 1)
    # project: x axis = along the shoe (z), y = up, looking along -view
    right = np.cross([0, 1, 0], view)
    right /= np.linalg.norm(right)
    up = np.cross(view, right)
    q = np.stack([p @ right, p @ up], 1)
    qs = q[tri.reshape(-1)].reshape(-1, 3, 2)
    mn, mx = q[sel].min(0), q[sel].max(0)
    scale = size * 0.86 / (mx - mn).max()
    off = (size - (mx - mn) * scale) / 2
    qs = (qs - mn) * scale + off
    qs[..., 1] = size - qs[..., 1]
    depth = (p[tri] @ view).mean(1)
    big = size * 2
    img = Image.new('RGBA', (big, big), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for i in np.argsort(depth):
        c = tuple(int(x) for x in np.clip(base[i] * shade[i], 0, 255)) + (255,)
        d.polygon([tuple(v * 2) for v in qs[i]], fill=c)
    return img.resize((size, size), Image.LANCZOS)


# ------------------------------------------------------------------ build
def build(args):
    game = args.game
    for need in (EBXTOOL, os.path.join(TEMPLATE, 'template.json')):
        if not os.path.isfile(need):
            raise SystemExit(f'missing {need}')
    tmpl = json.load(open(os.path.join(TEMPLATE, 'template.json')))
    key_words = slugify(args.name)
    key = 'Own_' + key_words
    ident = 'own_' + key_words.lower()
    author = re.sub(r'[^A-Za-z0-9_]', '_', args.author).strip('_') or 'socioculture'
    package = re.sub(r'[^A-Za-z0-9_]', '_', args.package or (key_words + '_Shoes'))
    folder_name = f'{author}-{package}'
    out_root = os.path.abspath(args.out)
    mod_dir = os.path.join(out_root, folder_name)
    rng = random.SystemRandom()

    print(f'Shoe "{args.name}"  key={key}  mod={folder_name}')
    g = fb.GameData(game)
    patch_chunk = next(k for k, v in g.chunk_dirs.items() if v == PATCH_DIRECTORY)
    _, base_bundles, _ = g.toc('Win32/items.toc')
    by_name = {b.name: b for b in base_bundles}
    for b in (SHARED_BUNDLE, MESH_BUNDLE, COLORWAY_BUNDLE):
        if b not in by_name:
            raise SystemExit('This skate. build does not have the bundles this tool expects: ' + b)

    # ---- geometry
    reference = shoe_mesh.Reference(g)
    prims, doc = (objfile.load if args.model.lower().endswith('.obj') else gltf.load)(args.model)
    palette = None
    textured = bool(args.texture) or any(doc.get('obj_textures', {}).values()) or any(
        gltf.base_colour_image(doc, m) is not None for m in {p.material for p in prims} if m is not None)
    if not textured:
        # painted by material colours alone: one flat square per colour, each part pointed at its own
        colours = None
        if args.colours:
            with open(args.colours) as f:
                colours = list(json.load(f).items())
        palette, palette_uv = gltf.colour_palette(prims, doc, colours=colours)
        print(f'  no textures: {len({tuple(c) for c in np.asarray(palette).reshape(-1, 3)})} material colours '
              'made into a palette texture')
    up = None if args.up == 'auto' else np.array({'x': (1, 0, 0), 'y': (0, 1, 0), 'z': (0, 0, 1)}[args.up[-1]], float) \
        * (-1 if args.up[0] == '-' else 1)
    prims, material = shoe_prep.prepare(prims, up=up)
    # the game indexes a shoe's mesh with 16 bits: both feet together at most 65535 vertices
    per_shoe = MAX_VERTICES // max(len(prims), 1)
    reduced = False
    twin = None
    if len(prims) == 2 and len(prims[0].positions) == len(prims[1].positions):
        a, b = prims                                   # one model shoe mirrored for the other foot?
        ca, cb = a.positions - a.positions.mean(0), b.positions - b.positions.mean(0)
        if np.allclose(ca * [-1, 1, 1], cb, atol=1e-6 * np.ptp(ca)):
            twin = (a, b)
    for p in prims:
        if len(p.positions) > per_shoe and (twin is None or p is twin[0]):
            reduced = True
            p.positions, p.normals, p.uv, p.indices = meshreduce.reduce(
                p.positions, p.uv, p.indices, per_shoe, label=p.name, nrm=p.normals)
    if reduced and twin is not None:
        # the other foot is the reduced one mirrored, so the two match exactly
        a, b = twin
        b.positions = a.positions * [-1, 1, 1]           # shoe_prep sets the pair symmetric about x = 0
        b.normals, b.uv, b.indices = a.normals * [-1, 1, 1], a.uv.copy(), a.indices[:, ::-1].copy()
        print(f'  {b.name}: the reduced {a.name} mirrored, so both feet match')
    if args.unmirror and twin is not None and palette is not None:
        # the mirrored foot's lettering reads backwards: flip each named part back along its own
        # reading direction, in place (it stays where it was, on the same surface)
        b = twin[1]
        for name in args.unmirror:
            for mat, uv in palette_uv.items():
                if name.lower() not in mat.lower():
                    continue
                on = np.abs(b.uv - np.array(uv)).max(1) < 1e-6
                if on.sum() < 4:
                    continue
                pts = b.positions[on]
                centre = pts.mean(0)
                w, v = np.linalg.eigh(np.cov((pts - centre).T))
                along = v[:, 2]                          # the longest extent: the line of text
                b.positions = b.positions.copy(); b.normals = b.normals.copy()
                b.positions[on] -= 2 * ((pts - centre) @ along)[:, None] * along
                b.normals[on] -= 2 * (b.normals[on] @ along)[:, None] * along
                flip = on[b.indices].all(1)
                b.indices = b.indices.copy()
                b.indices[flip] = b.indices[flip][:, ::-1]
                print(f'  {b.name}: lettering "{mat}" flipped back so it reads the right way')
    garments = shoe_mesh.lower_leg_garments(g, reference.positions[:, 1].max() + 0.02)
    print(f'  fitting the ankle around {len(garments)} pieces of the game\'s legwear')
    shape = shoe_mesh.Reference(g, shoe_mesh.DUNK_HIGH_BUNDLE, None) if args.fit in ('auto', 'model', 'scale') else None
    leg = shoe_mesh.Reference(g, shoe_mesh.BODY_BUNDLE, None) if args.fit == 'model' else None
    base_ref = None
    if args.base != 'auto' and args.fit in ('model', 'scale'):
        base_ref = {'vertclassic': lambda: reference, 'dunkhigh': lambda: shape,
                    'puffy90s': lambda: shoe_mesh.Reference(g, PUFFY90S_BUNDLE, None)}[args.base]()
    shoe = shoe_mesh.build(prims, reference, garments, collar=args.collar, shape=shape,
                           as_modelled=args.fit in ('model', 'scale'), leg=leg, adjust=args.fit != 'scale',
                           base_shoe=base_ref, base_name=BASE_NAMES.get(args.base))
    if reduced:
        # a reduced model keeps slivers too small for the game's 16-bit positions: drop the ones
        # that would collapse or turn over there (invisible either way)
        bad = meshreduce.unstable_triangles(shoe.positions, shoe.triangles, shoe.normals)
        if bad.any():
            shoe.triangles = shoe.triangles[~bad]
            shoe.tangents, shoe.bitangents = shoe_mesh.tangents(shoe.positions, shoe.normals, shoe.uv, shoe.triangles)
            print(f'  dropped {int(bad.sum())} sub-millimetre triangles that 16-bit positions would turn over')
    print(f'  mesh: {len(shoe.positions)} vertices, {len(shoe.triangles)} triangles')
    if args.texture:
        color_image = Image.open(args.texture).convert('RGBA')
    elif palette is not None:                     # flat colours: kept flat when scaled up
        size = struct.unpack_from('<H', read(os.path.join(TEMPLATE, 'tex_set_0bb23445.res')), 22)[0]
        color_image = palette.resize((size, size), Image.NEAREST).convert('RGBA')
        if args.gold:
            color_image, n = gold_parts(shoe, color_image, palette_uv, args.gold, args.gold_colour)
            print(f'  {n} points of the parts named "{args.gold}" made shiny gold ({args.gold_colour}, '
                  'reflections baked by which way each faces)')
    elif 'obj_textures' in doc:                   # an OBJ: its .mtl's map_Kd
        found = doc['obj_textures'].get(material)
        if not found:
            raise SystemExit("The shoes' main material has no colour texture in the .mtl; pass one with --texture.")
        color_image = Image.open(found).convert('RGBA')
    else:
        color_image = gltf.base_colour_image(doc, material)
        if color_image is None:
            raise SystemExit("The shoes' material has no colour texture; pass one with --texture.")
        color_image = color_image.convert('RGBA')
    if args.fill_holes is not None:
        color_image, n = fill_holes(color_image, args.fill_holes)
        print(f'  {n} see-through texels (netting holes) filled from around them at x{args.fill_holes:.2f} '
              "(the game's shoe shader is opaque)")
    if args.metallic and args.metal_style == 'flat':
        color_image, share = paint_metal(color_image, Image.open(args.metallic), args.metal_shade)
        print(f"  metal parts ({share:.1%} of the texture, from the metallic map) painted silver at x{args.metal_shade:.2f}")
    elif args.metallic:
        color_image, share = chrome_metal(color_image, Image.open(args.metallic), shoe,
                                          Image.open(args.normal) if args.normal else None, args.normal_style == 'opengl')
        print(f"  metal parts ({share:.1%} of the texture, from the metallic map) given a baked chrome reflection")
    # per-foot outsoles: left_sole.png / right_sole.png beside the model replace that foot's
    # sole art (the left foot is x>0, on the skeleton's LeftFoot). A foot without one keeps the
    # model's own sole untouched: flipping it to un-mirror lettering drags the upper's texels
    # (just outside the asymmetric sole outline) onto the sole rim.
    sole_art = {}
    for side, fname in ((1, 'left_sole.png'), (-1, 'right_sole.png')):
        path = os.path.join(os.path.dirname(os.path.abspath(args.model)), fname)
        if os.path.isfile(path):
            sole_art[side] = Image.open(path)
    if sole_art:
        color_image, kept = atlas.own_soles(shoe, color_image, sole_art)
        color_image = color_image.convert('RGBA')
        print(f'  own outsole art for the {" and ".join("left" if s > 0 else "right" for s in sorted(sole_art, reverse=True))} '
              f'foot; rest of the texture kept at {kept:.0%} scale')

    normal_image = None
    if args.normal:
        normal_image = Image.open(args.normal).convert('RGB')
        style, how = args.normal_style, 'as given'
        if style == 'auto':
            up, sure = green_points_up(normal_image)
            style, how = ('opengl' if up else 'directx'), ('detected' if sure else 'a weak guess; set --normal-style if '
                                                                         'its seams and stitching look sunken in game')
        if style == 'opengl':
            # the shoe's tangent frames run with the texture's rows (DirectX-style maps)
            r, g_, b = normal_image.split()
            normal_image = Image.merge('RGB', (r, g_.point(lambda x: 255 - x), b))
            print(f'  normal map: OpenGL-style ({how}), green flipped for the game')
        else:
            print(f'  normal map: DirectX-style ({how}), used as is')
    if args.pad_seams:
        mask = atlas.coverage(shoe.uv, shoe.triangles, color_image.width)
        color_image = atlas.pad(color_image.convert('RGB'), mask).convert('RGBA')
        if normal_image is not None:
            if normal_image.size != color_image.size:
                mask = atlas.coverage(shoe.uv, shoe.triangles, normal_image.width)
            normal_image = atlas.pad(normal_image, mask)
        print(f'  texture seams padded ({mask.mean():.0%} of the texture is painted on the shoe)')

    # a high-top placed as modelled: its own collar decides where pants are hidden (make_tuck_mod
    # reads <mod>-tuck.json and gives every pair of pants a region for it, which the item culls)
    own_tuck = None
    lip = args.tuck_rim == 'lip'
    if (args.fit == 'model' or (args.fit == 'scale' and args.tuck)) and shoe.high_top:
        region_name = shoe_mesh.OWN_TUCK_PREFIX + slugify(args.name)
        rims = shoe_mesh.own_collar_rims(shoe.positions, base_ref or shape, angle_step=5, lip=lip,
                                         margin=args.tuck_margin / 1000)
        own_tuck = (djb(region_name), region_name, {str(side): {'axis': [float(x) for x in axis], 'rim': [float(x) for x in rim]}
                                                    for side, (axis, rim) in rims.items()})
        how = ("the lip's real top per 5 deg sector" if lip else 'each 5 deg sector lowered to its lowest neighbour') + \
              (f', plus up to {args.tuck_margin:g} mm at the heel' if args.tuck_margin else '')
        print(f'  pants hidden under its own collar ({rims[1][1].min() * 100:.1f}-{rims[1][1].max() * 100:.1f} cm; {how}): '
              f'region {region_name} (rebuild the High_Top_Pants_Tuck mod after this)')
        if lip:
            # what the default would have read, so the difference the lip makes is on record
            safe = shoe_mesh.own_collar_rims(shoe.positions, base_ref or shape, angle_step=5)
            rim_l = rims[1][1] - args.tuck_margin / 1000 * (1 - np.cos(np.radians(np.arange(72) * 5 + 2.5))) / 2
            gap = rim_l - safe[1][1]
            k = int(np.argmax(gap))
            print(f'  left foot: the lip reads up to {gap[k] * 1000:.1f} mm above the sector-min rim (at {k * 5 + 2.5:.0f} deg '
                  f'from the toe); lip at the heel, 150-210 deg: ' + ' '.join(f'{h * 100:.1f}' for h in rim_l[30:43]) + ' cm')
        if args.fit == 'scale':
            # the collar the pants tuck into follows the leg as they do (fit 'model' skins the part
            # above the Dunk High's collar the same way)
            body = shoe_mesh.Reference(g, shoe_mesh.BODY_BUNDLE, None)
            shoe.bones, shoe.weights, moved = shoe_mesh.skin_collar_to_leg(shoe.positions, shoe.bones, shoe.weights, rims, body)
            print(f'  {moved} points of the top of the collar follow the leg like the pants over them (skinned like the body)')
    # the bare feet: the template (the VertClassic) hides the body's feet up to FOOT_REGION_TOP; a shoe
    # whose collar dips lower would show a gap there (no ankle between shorts and shoe), so like most
    # of the game's own shoes it hides nothing and the foot stays inside it
    hide_feet = args.hide_feet == 'yes'
    if args.hide_feet == 'auto':
        hide_feet = True
        if args.fit in ('model', 'scale'):
            low = min(r.min() for _, r in shoe_mesh.own_collar_rims(shoe.positions, base_ref or shape or reference,
                                                                     angle_step=5, lip=lip).values())
            hide_feet = bool(low >= FOOT_REGION_TOP)
    if not hide_feet:
        print("  the body's feet stay drawn (this collar dips under the top of the foot regions the game sneaker hides)")

    # ---- identities
    mesh_id, set_id, item_preset_id = new_guid(), new_guid(), new_guid()
    T_MESH, T_SET, T_ITEM = tmpl['mesh_id'], tmpl['set_id'], tmpl['item_preset_id']
    names = {
        'item': f'items/cust_shoes/{ident}',
        'thumb_tool': f'thumbnail/tool/{ident}', 'thumb_tool_lrg': f'thumbnail/tool/{ident}_lrg',
        'thumb_cdn': f'thumbnail/cdn/img_{ident}', 'thumb_cdn_lrg': f'thumbnail/cdn/img_{ident}_lrg',
        'geometry': f'characters/maincharacters/reskate/{mesh_id}_geometry',
        'mesh': f'characters/maincharacters/reskate/{mesh_id}_mesh',
        'variation': f'characters/maincharacters/reskate/{mesh_id}_variation_2',
        'preset_set': f'characters/customization/reskate/textureedits/{set_id}_ap',
        'preset_item': f'characters/customization/reskate/textureedits/{item_preset_id}_ap',
    }
    tex_labels = [k[:-4] for k in tmpl['assets'] if k.startswith('tex_') and k.endswith('.ebx')]
    for lab in tex_labels:
        owner = set_id if lab.startswith('tex_set_') else item_preset_id
        full = tmpl['assets'][lab + '.ebx']['name']
        names[lab] = f'characters/customization/reskate/textureedits/{owner}_ap_' + full.split('_ap_')[1]
    infos = {lab: ebx_info(os.path.join(TEMPLATE, lab + '.ebx')) for lab in
             ['item', 'thumb_tool', 'thumb_tool_lrg', 'thumb_cdn', 'thumb_cdn_lrg', 'geometry', 'mesh', 'variation',
              'preset_set', 'preset_item'] + tex_labels}
    ids = {lab: {'file': new_guid(), 'inst': [new_guid() for _ in info['instances']]} for lab, info in infos.items()}
    ids['item']['file'] = mesh_id          # as the template does: the item's file guid names the mesh

    # ---- bundles no other installed mod ships, so ReSkate passes ours through unmerged
    others = bundles_of_other_mods(game, folder_name)
    own_textures = {infos[lab]['file'] for lab in tex_labels}
    preset_imports = sorted({imp for lab in ('preset_set', 'preset_item') for imp in infos[lab]['imports']
                             if imp[0] not in own_textures})
    colorway, import_swaps, carried = pick_colorway(g, by_name, preset_imports, others)
    if carried:
        print(f'  carried {len(carried)} shared shader setting(s) the colorway lacks: '
              + ', '.join(n.split('/')[-1] for n, _ in carried))
    print(f'  colorway bundle: {colorway.split("/")[-1].replace("_cas_main_bundlereftable", "")}'
          + (f' ({COLORWAY_BUNDLE.split("/")[-1][:31]} is shipped by {", ".join(others[COLORWAY_BUNDLE])})'
             if COLORWAY_BUNDLE in others else ''))
    if MESH_BUNDLE in others:
        print(f'  note: the sneaker mesh bundle is also shipped by {", ".join(others[MESH_BUNDLE])}; '
              'ReSkate merges it (its mesh chunks need no firstMip, so this is safe)')
    res_ids = {lab: rng.getrandbits(64) | 1 for lab in ['mesh', 'thumb_tool', 'thumb_tool_lrg'] + tex_labels}
    chunk_ids = {lab: uuid.uuid4().bytes for lab in ['thumb_tool', 'thumb_tool_lrg'] + tex_labels}
    lod_chunk_ids = [uuid.uuid4().bytes for _ in range(6)]

    work = tempfile.mkdtemp(prefix='shoemod_')
    try:
        def tpath(f): return os.path.join(TEMPLATE, f)
        def wpath(f): return os.path.join(work, f)

        def guid_ops(lab, skip_zero=True):
            ops = ['--file-guid', ids[lab]['file']]
            for i, old in enumerate(infos[lab]['instances']):
                if skip_zero and old == '00000000-0000-0000-0000-000000000000':
                    continue
                ops += ['--inst-guid', i, ids[lab]['inst'][i]]
            return ops

        def inst_of(lab, i=0):
            return ids[lab]['inst'][i]

        ebx, res = {}, {}
        cas = CasWriter(patch_chunk, game, compress=not args.no_compress)
        toc_chunks = []

        # ---- textures
        tex_chunks = {}
        for lab in tex_labels:
            kind = TEXTURE_KEYS[lab.split('_')[-1]]
            header = read(tpath(lab + '.res'))
            size = struct.unpack_from('<H', header, 22)[0]
            if args.texture_size and kind in ('color', 'normal') and args.texture_size != size:
                size = args.texture_size
                mips = texture_mips(kind, size, color_image, work, lab, normal_image)
                header = sized_header(header, size, mips)
            else:
                mips = texture_mips(kind, size, color_image, work, lab, normal_image)
            # The bundle carries the whole texture (first mip 0), as the community shoe mods do.
            # The game's own layout (bundle tail from byte 32's mip, the rest streamed) left the
            # textures unfilled (noise) or crashed in some builds, depending on their random ids.
            first = header[32] if args.streamed_textures and not args.texture_size else 0
            header = header[:31] + bytes([first]) + header[32:]
            stream, tail, offsets = pack_texture(cas, mips, first)
            total = sum(len(m) for m in mips)
            res[lab] = texture_header(header, names[lab], chunk_ids[lab], total, offsets)
            tex_chunks[lab] = (stream, tail, total, first, sum(len(m) for m in mips[:first]))
            ebx[lab] = edit_ebx(tpath(lab + '.ebx'), wpath(lab + '.ebx'), *guid_ops(lab),
                                '--set', '0:Name', 'str:' + names[lab], '--set', '0:Resource', f'res:{res_ids[lab]}')
        print(f'  textures: {len(tex_labels)} ({", ".join(TEXTURE_KEYS[l.split("_")[-1]] for l in tex_labels)})'
              + (f', colour and normal at {args.texture_size}' if args.texture_size else ''))

        # ---- thumbnails
        thumbs = {}
        icon = icon_renderer.render(shoe, color_image, 768, 2, legs=icon_renderer.body_legs(g))
        for lab, size in (('thumb_tool', 256), ('thumb_tool_lrg', 768)):
            image = icon if size == 768 else icon.resize((size, size), Image.LANCZOS)
            thumbs[lab] = (image, bc7(image, work, lab))
            res[lab] = texture_header(read(tpath(lab + '.res')), names[lab], chunk_ids[lab], len(thumbs[lab][1]))
            ebx[lab] = edit_ebx(tpath(lab + '.ebx'), wpath(lab + '.ebx'), *guid_ops(lab),
                                '--set', '0:Name', 'str:' + names[lab], '--set', '0:Resource', f'res:{res_ids[lab]}')
        for lab, tool in (('thumb_cdn', 'thumb_tool'), ('thumb_cdn_lrg', 'thumb_tool_lrg')):
            ebx[lab] = edit_ebx(tpath(lab + '.ebx'), wpath(lab + '.ebx'), *guid_ops(lab),
                                '--import', 0, ids[tool]['file'], inst_of(tool),
                                '--set', '0:Name', 'str:' + names[lab], '--set', '0:NameHash', f'u64:{djb(names[lab])}',
                                '--set', '0:ContentHash', 'sha1:' + hashlib.sha1(thumbs[tool][1]).hexdigest())

        # ---- mesh assets
        meshset_res, lod_chunks = build_meshset(read(tpath('mesh.res')), shoe, mesh_id, T_MESH, lod_chunk_ids)
        res['mesh'] = meshset_res
        # patched in place, so the payload and relocation table sizes in its metadata still hold
        assert len(meshset_res) == len(read(tpath('mesh.res')))
        ebx['mesh'] = edit_ebx(tpath('mesh.ebx'), wpath('mesh.ebx'), *guid_ops('mesh'),
                               '--set', '0:Name', 'str:' + names['mesh'], '--set', '0:NameHash', f'u64:{djb(names["mesh"])}',
                               '--set', '0:MeshSetResource', f'res:{res_ids["mesh"]}')
        ebx['geometry'] = edit_ebx(tpath('geometry.ebx'), wpath('geometry.ebx'), *guid_ops('geometry'),
                                   '--import', 0, ids['mesh']['file'], inst_of('mesh', 0),
                                   '--set', '0:Name', 'str:' + names['geometry'])
        mesh_insts = infos['mesh']['instances']
        var_imports = infos['variation']['imports']
        var_ops = []
        for i, (f_, inst) in enumerate(var_imports):
            if f_ == infos['mesh']['file']:
                var_ops += ['--import', i, ids['mesh']['file'], ids['mesh']['inst'][mesh_insts.index(inst)]]
        ebx['variation'] = edit_ebx(tpath('variation.ebx'), wpath('variation.ebx'), *guid_ops('variation'), *var_ops,
                                    '--set', '0:Name', 'str:' + names['variation'])

        # ---- presets: ReSkate's writer drops their boxed values, so patch bytes in place
        def patch_preset(lab, uuid_swaps):
            data = bytearray(read(tpath(lab + '.ebx')))
            info = infos[lab]
            swaps = [(info['file'], ids[lab]['file'])] + list(zip(info['instances'], ids[lab]['inst']))
            for f_, inst in info['imports']:
                for other, oinfo in infos.items():
                    if other.startswith('tex_') and oinfo['file'] == f_:
                        swaps += [(f_, ids[other]['file']), (inst, ids[other]['inst'][oinfo['instances'].index(inst)])]
            for old, new in swaps:
                if old == '00000000-0000-0000-0000-000000000000':
                    continue
                ob, nb = guid_bytes(old), guid_bytes(new)
                assert data.count(ob) >= 1, f'{lab}: {old} not found'
                data = data.replace(ob, nb)
            for old, new in import_swaps:      # imports the chosen colorway holds under its own guids
                data = data.replace(guid_bytes(old), guid_bytes(new))
            for old, new in uuid_swaps:
                data = data.replace(old.encode(), new.encode())
            return bytes(data)
        ebx['preset_set'] = patch_preset('preset_set', [(T_SET, set_id)])
        ebx['preset_item'] = patch_preset('preset_item', [(T_ITEM, item_preset_id)])

        # The shader multiplies the colour texture by twice the per-region tint (the game's own
        # sneakers pair a mid-grey colour texture with ~0.6 for white), so 0.5 shows the texture
        # as painted; at 1.0 the whites clip and only the baked-in wrinkle shading is left. The
        # template's tints are a black colourway's. Set every colour parameter (boxed type 18).
        def tint_preset(lab):
            path = wpath(lab + '_tint.ebx')
            with open(path, 'wb') as f:
                f.write(ebx[lab])
            data = bytearray(ebx[lab])
            spots = [int(at) for at in re.findall(r'boxed\(18 at=(\d+) bytes=', ebxtool('dump', path))]
            for at in spots:
                struct.pack_into('<3f', data, at, *args.tint)
            ebx[lab] = bytes(data)
            return len(spots)
        tinted = tint_preset('preset_set') + tint_preset('preset_item')
        print(f'  colour tints set to {tuple(args.tint)} ({tinted} parameters)')

        # ---- item
        item_imports = infos['item']['imports']
        thumb_slot = [i for i, (f_, _) in enumerate(item_imports) if f_ == infos['thumb_cdn']['file']][0]
        large_slot = [i for i, (f_, _) in enumerate(item_imports) if f_ == infos['thumb_cdn_lrg']['file']][0]
        dunk_tuck = shoe.high_top and args.fit not in ('model', 'scale')    # Dunk-shaped: hides the Dunk High set
        culled = ((list(FOOT_REGIONS) if hide_feet else []) + ([shoe_mesh.TUCK_REGION] if dunk_tuck else [])
                  + ([own_tuck[0]] if own_tuck else [])) or [0]
        culled += [culled[0]] * max(0, 2 - len(culled))           # the template lists two; extras repeat
        region_ops = [op for _ in range(len(culled) - 2) for op in ('--append', '0:ItemData.CulledRegions')]
        region_ops += [op for n, rid in enumerate(culled) for op in ('--set', f'0:ItemData.CulledRegions[{n}].RegionId', f'u64:{rid}')]
        ebx['item'] = edit_ebx(tpath('item.ebx'), wpath('item.ebx'), *guid_ops('item'),
                               '--set', '0:Name', 'str:' + names['item'], '--set', '0:Key', 'str:' + key,
                               '--set', '0:HashedAssetKey', f'u64:{djb(key)}',
                               '--import', thumb_slot, ids['thumb_cdn']['file'], inst_of('thumb_cdn'),
                               '--import', large_slot, ids['thumb_cdn_lrg']['file'], inst_of('thumb_cdn_lrg'),
                               '--set', '0:ItemData.AssetPaths[0].AssetName', 'str:' + names['geometry'].rsplit('/', 1)[1],
                               '--set', '0:ItemData.AssetPaths[1].AssetName', 'str:' + names['preset_item'].rsplit('/', 1)[1],
                               *region_ops)

        def res_meta(lab):
            return bytes.fromhex(tmpl['assets'][lab + '.res']['res_meta'])

        def chunk_meta(name, first_mip=None):
            meta = {'type': 2, 'name': 'meta', 'terminated': True, 'children': []}
            if first_mip is not None:
                meta['children'].append({'type': 8, 'name': 'firstMip', 'value': struct.pack('<i', first_mip)})
            return {'type': 2, 'name': None, 'terminated': True, 'children': [
                {'type': 9, 'name': 'h64', 'value': struct.pack('<Q', djb(name, 64))}, meta]}

        def patch_bundle(bundle_name, new_ebx, new_res, new_chunks, edit=None):
            files, _ = fb.read_bundle_region(by_name[bundle_name].region)
            b_ebx, b_res, b_ch, b_meta = g.manifest(files)
            ne, nr = len(b_ebx), len(b_res)
            f_ebx, f_res, f_ch = files[1:1 + ne], files[1 + ne:1 + ne + nr], files[1 + ne + nr:]
            if edit:
                edit(b_ebx, b_res, f_ebx, f_res)
            add_e, add_ef = [], []
            for lab in new_ebx:
                fi, sha = cas.add(ebx[lab])
                add_e.append(fb.Asset(kind='ebx', name=names[lab], sha1=sha, original_size=len(ebx[lab])))
                add_ef.append(fi)
            add_r, add_rf = [], []
            for lab in new_res:
                fi, sha = cas.add(res[lab])
                add_r.append(fb.Asset(kind='res', name=names[lab], sha1=sha, original_size=len(res[lab]),
                                      res_type=TEXTURE_RES if lab != 'mesh' else shoe_mesh.MESHSET_RES,
                                      res_meta=res_meta(lab), res_id=res_ids[lab]))
                add_rf.append(fi)
            add_c, add_cf = [], []
            tree, _ = fb.read_db(b_meta, 0)
            for guid, fi, sha, offset, size, meta in new_chunks:
                add_c.append(fb.Asset(kind='chunk', name=fb.guid_str(guid), guid=guid, sha1=sha,
                                      logical_offset=offset, logical_size=size))
                add_cf.append(fi)
                tree['children'].append(meta)
            manifest = fb.write_binary_bundle(b_ebx + add_e, b_res + add_r, b_ch + add_c, fb.write_db(tree))
            return fb.TocBundle(bundle_name, fb.write_bundle_region(
                [cas.add_raw(manifest)] + f_ebx + add_ef + f_res + add_rf + f_ch + add_cf))

        # chunks: every texture and LOD in items.toc; textures and LODs 1-5 also in their bundle
        color_chunks = []
        for lab in tex_labels:
            stream, tail, total, first, tail_at = tex_chunks[lab]
            fi = cas.add_raw(stream)
            toc_chunks.append(fb.TocChunk(chunk_ids[lab], True, patch_chunk, 1, fi.offset, fi.size))
            if tail is stream:              # the bundle's copy is the whole chunk: share its bytes
                sha = hashlib.sha1(stream).digest()
            else:
                fi, sha = cas.add_encoded(tail)
            color_chunks.append((chunk_ids[lab], fi, sha, tail_at, total - tail_at, chunk_meta(names[lab], first)))
        mesh_chunk_entries = []
        for i, (cid, data) in enumerate(zip(lod_chunk_ids, lod_chunks)):
            fi, sha = cas.add(data)
            toc_chunks.append(fb.TocChunk(cid, True, patch_chunk, 1, fi.offset, fi.size))
            if i > 0:
                mesh_chunk_entries.append((cid, fi, sha, 0, len(data), chunk_meta(names['mesh'])))
        for lab in ('thumb_tool', 'thumb_tool_lrg'):
            fi, _ = cas.add(thumbs[lab][1])
            toc_chunks.append(fb.TocChunk(chunk_ids[lab], True, patch_chunk, 1, fi.offset, fi.size))

        def edit_shared(b_ebx, b_res, f_ebx, f_res):
            ic = next(i for i, a in enumerate(b_ebx) if a.name == ITEM_COLLECTION)
            with open(wpath('collection.ebx'), 'wb') as f:
                f.write(g.payload(f_ebx[ic]))
            ebxtool('additem', wpath('collection.ebx'), wpath('collection_new.ebx'), mesh_id, inst_of('item'))
            data = read(wpath('collection_new.ebx'))
            fi, sha = cas.add(data)
            b_ebx[ic].sha1, b_ebx[ic].original_size, f_ebx[ic] = sha, len(data), fi
            br = next(i for i, a in enumerate(b_res) if a.name == BUNDLE_REF_TABLE)
            table = bundleref.Table(g.payload(f_res[br]), b_res[br].res_meta)
            for preset in ('preset_set', 'preset_item'):
                table.insert(names[preset], table.bundle_index(bundle_ref(colorway)))
            table.insert(names['geometry'], table.bundle_index(MESH_BUNDLE_REF))
            fi, sha = cas.add(bytes(table.data))
            b_res[br].sha1, b_res[br].original_size, b_res[br].res_meta, f_res[br] = sha, len(table.data), bytes(table.meta), fi

        carry_labels = []                      # shared shader settings the chosen colorway lacks
        for k, (asset_name, data) in enumerate(carried):
            names[f'carried{k}'], ebx[f'carried{k}'] = asset_name, data
            carry_labels.append(f'carried{k}')
        bundles = [
            patch_bundle(SHARED_BUNDLE, ['item', 'thumb_tool', 'thumb_tool_lrg', 'thumb_cdn', 'thumb_cdn_lrg'],
                         ['thumb_tool', 'thumb_tool_lrg'], [], edit_shared),
            patch_bundle(colorway, ['preset_set', 'preset_item'] + tex_labels + carry_labels, tex_labels, color_chunks),
            patch_bundle(MESH_BUNDLE, ['geometry', 'mesh', 'variation'], ['mesh'], mesh_chunk_entries),
        ]
        toc = fb.write_patch_toc(bundles, toc_chunks, 3)

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
        with open(os.path.join(game, 'Skate.exe'), 'rb') as f:
            skate_sha = hashlib.sha256(f.read()).hexdigest()
        with open(os.path.join(mod_dir, '.reskate-studio-patch'), 'w', newline='\n') as f:
            f.write(f'ReSkate Studio native Patch v1\nskate_sha256={skate_sha}\n')
        with open(os.path.join(mod_dir, 'manifest.json'), 'w') as f:
            json.dump({'author': author, 'dependencies': [], 'description': f'{args.name} shoes',
                       'name': package, 'version_number': args.version, 'website_url': ''}, f, indent=4)
        with open(os.path.join(mod_dir, 'README.md'), 'w', encoding='utf-8') as f:
            f.write(f'# {args.name}\n\nAdds the "{args.name}" shoes to skate. (ReSkate).\n')
            if args.credit:
                f.write(f'\n## Credits\n\n{args.credit}\n')
        thumbs['thumb_tool'][0].save(os.path.join(mod_dir, 'icon.png'))
        thumbs['thumb_tool_lrg'][0].save(os.path.join(out_root, f'{folder_name}-thumbnail.png'))
        zip_path = os.path.join(out_root, f'{folder_name}.zip')
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
            for dirpath, _, files in os.walk(mod_dir):
                for name in files:
                    full = os.path.join(dirpath, name)
                    z.write(full, os.path.relpath(full, mod_dir))
        print(f'  wrote {mod_dir}\n  wrote {zip_path}')
        # the tuck registry make_tuck_mod reads: this shoe's own rim, or the Dunk High set it culls
        tuck_json = os.path.join(out_root, f'{folder_name}-tuck.json')
        tucks = bool(own_tuck) or (shoe.high_top and args.fit not in ('model', 'scale'))
        if own_tuck:
            with open(tuck_json, 'w') as f:
                json.dump({'region': own_tuck[0], 'name': own_tuck[1], 'rims': own_tuck[2], 'hide_feet': hide_feet}, f, indent=1)
            print(f'  wrote {tuck_json}')
        elif tucks:
            with open(tuck_json, 'w') as f:
                json.dump({'set': 'dunk', 'hide_feet': hide_feet}, f)
        elif os.path.exists(tuck_json):
            os.remove(tuck_json)

        if not args.no_verify:
            check = subprocess.run([sys.executable, os.path.join(HERE, 'verify_shoe_mod.py'), mod_dir, '--game', game]
                                   + (['--as-modelled'] if args.fit in ('model', 'scale') else []), capture_output=True, text=True)
            print(check.stdout[-3000:])
            if check.returncode:
                print(check.stderr)
                raise SystemExit('Verification failed; the mod was not installed.')
        # a tucking shoe changes how pants are split: rebuild the tuck mod, which also updates the
        # culled regions of every tucking shoe in out_root (this one included)
        also = []
        if tucks:
            import make_tuck_mod
            print('\nRebuilding the pants tuck for this shoe:')
            tuck_dir, changed = make_tuck_mod.build(argparse.Namespace(
                author=args.author, version='1.0.0', game=game, out=out_root, install=False, no_compress=args.no_compress))
            also = [os.path.basename(tuck_dir)] + [c for c in changed if c != folder_name]
        if args.install:
            running = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq Skate.exe'], capture_output=True, text=True).stdout
            if 'skate.exe' in running.lower():
                print('  skate. is running: close it, then copy these into Mods (or run this again): '
                      + ', '.join([folder_name] + also))
                return mod_dir
            for old in earlier_builds(game, folder_name):
                shutil.rmtree(os.path.join(game, 'Mods', old))
                print(f'  removed the earlier build {old}')
            for folder in [folder_name] + also:
                target = os.path.join(game, 'Mods', folder)
                if os.path.exists(target):
                    shutil.rmtree(target)
                shutil.copytree(os.path.join(out_root, folder), target)
                print(f'  installed to {target}')
        elif also:
            print('  copy into Mods together: ' + ', '.join([folder_name] + also))
        return mod_dir
    finally:
        if args.keep_work:
            print('  work files kept in', work)
        else:
            shutil.rmtree(work, ignore_errors=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('model', help='shoe model: .gltf (with its .bin/textures), .glb or .obj (with its .mtl)')
    p.add_argument('--name', required=True, help='shoe name shown in game')
    p.add_argument('--texture', help='color texture (default: the model\'s base color texture)')
    p.add_argument('--credit', help='credit line for the README (required by CC-BY models)')
    p.add_argument('--tint', type=lambda s: [float(x) for x in s.split(',')], default=[0.5, 0.5, 0.5],
                   help='linear RGB region tint; the shader doubles it (default 0.5,0.5,0.5 = as painted, '
                        'higher brightens, lower darkens)')
    p.add_argument('--fit', choices=('auto', 'sneaker', 'model', 'scale'), default='auto',
                   help="auto: high-tops are shaped and skinned like the game's Nike SB Dunk High (slim, real "
                        'proportions, the same pants fit the game ships for it), other shoes are fitted to the '
                        'game sneaker; sneaker: every shoe fitted to the game sneaker (wider collars); model: the '
                        "shoe exactly as modelled, only turned, scaled to the game's foot and set level (heel and "
                        'ball on the floor); nothing is reshaped, so pants may show where it is slimmer; scale: a first '
                        "look, only turned, scaled to the game's foot and set on the floor (no tipping, no pants culling "
                        'beyond the bare foot, skinned by the nearest game-shoe point)')
    p.add_argument('--up', choices=('auto', 'x', 'y', 'z', '-x', '-y', '-z'), default='auto',
                   help='the axis the model stands up along when it already stands on its sole (auto: found from the '
                        "sole, the default; the Yeezy 2 OBJ's flat side panel fools that, it needs y)")
    p.add_argument('--collar', choices=('classic', 'smooth'), default='classic',
                   help='ankle collar fit: classic pulls single points in (the default), smooth draws whole rings in '
                        '(for bulky collars and tall tongues that classic crumples)')
    p.add_argument('--normal', help="tangent-space normal map for the model's UVs (OpenGL- or DirectX-style, "
                                    'told apart automatically); default: flat')
    p.add_argument('--normal-style', choices=('auto', 'directx', 'opengl'), default='auto',
                   help="the normal map's green channel: directx (down the image: 3ds Max, Unreal, most "
                        'CGTrader packs), opengl (up: glTF, Blender, Substance default) or auto (guessed)')
    p.add_argument('--texture-size', type=int, choices=(1024, 2048), help='colour and normal texture size (default: '
                   "the game sneaker's 1024)")
    p.add_argument('--base', choices=('auto', 'vertclassic', 'puffy90s', 'dunkhigh'), default='auto',
                   help='with --fit model/scale: the game shoe to place on and skin from (auto: the Dunk High for '
                        'high-tops, else the VertClassic)')
    p.add_argument('--hide-feet', choices=('auto', 'yes', 'no'), default='auto',
                   help="hide the body's bare feet like the template sneaker (auto: unless the collar dips under the "
                        'top of those foot regions, 13.3 cm, which would leave a see-through gap at the ankle)')
    p.add_argument('--tuck', action='store_true',
                   help='with --fit scale: hide pants under this shoe\'s own collar (--fit model always does)')
    p.add_argument('--tuck-rim', choices=('sectors', 'lip'), default='sectors',
                   help='how the collar rim the pants tuck under is measured (with --tuck or --fit model; it also decides '
                        '--hide-feet auto): sectors lowers each 5-degree sector to the lowest of itself and its neighbours '
                        '(the default; under a lip that rises quickly, as the Yeezy 2 does at the back, it reads up to 8 mm '
                        "low, and pants kept down to there drape over the lip); lip takes each sector's own highest point, "
                        "the lip's real top")
    p.add_argument('--tuck-margin', type=float, default=0.0, metavar='MM',
                   help='hide pants up to this many millimetres above the collar lip at the heel, easing to nothing at the '
                        'toe (tight jeans sit outside the lip at the back and slant across it); default 0')
    p.add_argument('--colours', metavar='JSON', help='a colourway for a model painted by material colours: '
                   '{"part of a material name": "#rrggbb", ..., "*": "#rrggbb"} (tried in order)')
    p.add_argument('--unmirror', nargs='+', metavar='NAME', help='on a foot mirrored from the other, flip the parts '
                   'whose material name contains NAME (lettering) back so they read the right way')
    p.add_argument('--gold', metavar='NAME', help='make the parts whose material name contains NAME shiny gold')
    p.add_argument('--gold-colour', default='#fcce71', help="the gold (default: the game's gold deck, #fcce71)")
    p.add_argument('--fill-holes', type=float, nargs='?', const=0.86, metavar='SHADE',
                   help='fill see-through texels (netting holes) from around them, darkened by SHADE (default 0.86)')
    p.add_argument('--metallic', metavar='MAP', help="the model's metallic map: paint its metal parts silver")
    p.add_argument('--metal-style', choices=('chrome', 'flat'), default='chrome',
                   help='chrome: a reflection baked from which way each metal part faces (default); flat: plain silver grey')
    p.add_argument('--metal-shade', type=float, default=0.78, help='with --metal-style flat: how much darker the silver is')
    p.add_argument('--pad-seams', action='store_true',
                   help='fill the texture around each UV island from its edges, so no line shows where pieces meet')
    p.add_argument('--author', default='socioculture', help='mod author (folder prefix)')
    p.add_argument('--package')
    p.add_argument('--version', default='1.0.0')
    p.add_argument('--game', default=DEFAULT_GAME)
    p.add_argument('--out', default=os.path.join(HERE, '..', 'mods'))
    p.add_argument('--install', action='store_true')
    p.add_argument('--no-compress', action='store_true')
    p.add_argument('--no-verify', action='store_true')
    p.add_argument('--streamed-textures', action='store_true',
                   help="the game's own texture layout (mip tail in the bundle, the rest streamed) instead of whole textures in the bundle")
    p.add_argument('--keep-work', action='store_true')
    build(p.parse_args())


if __name__ == '__main__':
    main()
