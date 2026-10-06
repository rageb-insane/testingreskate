"""Checks a crop top mod built by make_top_mod.py the way ReSkate and the game will read it.

    python verify_top_mod.py <mod folder> [--game E:\\...\\Skate]
"""
import argparse
import os
import re
import struct
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fb  # noqa: E402
import bundleref  # noqa: E402
from make_deck_mod import EBXTOOL, DEFAULT_GAME, SHARED_BUNDLE, BUNDLE_REF_TABLE, ITEM_COLLECTION, TEXTURE_RES, djb  # noqa: E402
from make_shoe_mod import bundle_ref, bundles_of_other_mods, ebx_file_guid  # noqa: E402
from make_top_mod import (COLORWAY_BUNDLE, LOGO_W, LOGO_H, MIN_FACING, shirt_front, print_coverage,  # noqa: E402
                          preset_values)

problems = []


def check(ok, message):
    print(('  ok   ' if ok else '  FAIL ') + message)
    if not ok:
        problems.append(message)


def dump(data):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.ebx') as f:
        f.write(data)
    try:
        return subprocess.run([EBXTOOL, 'dump', f.name], capture_output=True, text=True).stdout
    finally:
        os.unlink(f.name)


def bc7_alpha(data, w, h):
    with tempfile.TemporaryDirectory() as d:
        src, dst = os.path.join(d, 'm.bc7'), os.path.join(d, 'm.rgba')
        with open(src, 'wb') as f:
            f.write(data)
        subprocess.run([EBXTOOL, 'bc7dec', src, str(w), str(h), dst], check=True, capture_output=True)
        with open(dst, 'rb') as f:
            return np.frombuffer(f.read(), np.uint8).reshape(h, w, 4)[..., 3].astype(np.float32) / 255


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mod')
    p.add_argument('--game', default=DEFAULT_GAME)
    a = p.parse_args()
    g = fb.GameData(a.game)
    flags, bundles, chunks = fb.read_toc(open(os.path.join(a.mod, 'Win32', 'items.toc'), 'rb').read())
    check(flags == 3 and len(bundles) == 2 and len(chunks) == 3, f'items.toc: {len(bundles)} bundles, {len(chunks)} chunks')
    toc = {c.guid: c for c in chunks}
    cas_path = os.path.join(a.mod, 'Win32', 'configurations', 'layout', 'initialinstallpackage', 'cas_01.cas')
    cas = open(cas_path, 'rb').read()
    check(all(c.offset + c.size <= len(cas) for c in chunks), 'items.toc chunks lie inside cas_01.cas')
    _, base_bundles, _ = g.toc('Win32/items.toc')
    base = {b.name: b for b in base_bundles}

    def payload(f):
        return fb.decode_cas(cas[f.offset:f.offset + f.size], g.root) if f.patch else g.payload(f)

    content = {}
    for b in bundles:
        print(b.name.split('/')[-1])
        files, _ = fb.read_bundle_region(b.region)
        ebx, res, ch, meta = g.manifest(files, a.mod)
        bf, _ = fb.read_bundle_region(base[b.name].region)
        be, br, bc, _ = g.manifest(bf)
        check(len(files) == 1 + len(ebx) + len(res) + len(ch), 'region files match the manifest')
        check([x.name for x in ebx[:len(be)]] == [x.name for x in be] and [x.name for x in res[:len(br)]] == [x.name for x in br]
              and [x.guid for x in ch[:len(bc)]] == [x.guid for x in bc], 'base assets kept, additions appended')
        tree, _ = fb.read_db(meta, 0)
        check(len(tree['children']) == len(ch), 'chunk metadata: one entry per chunk')
        metas = {struct.unpack('<Q', fb.db_field(e, 'h64')['value'])[0]: e for e in tree['children'] if fb.db_field(e, 'h64')}
        bad = 0
        for x, f in zip(ebx + res + ch, files[1:]):
            if f.patch:
                raw = cas[f.offset:f.offset + f.size]
                size = len(fb.decode_cas(raw, g.root))
                bad += fb.sha1(raw) != x.sha1 or size != (x.logical_size if x.kind == 'chunk' else x.original_size)
        check(not bad, 'every stored file matches its sha1 and size')
        content[b.name] = dict(files=files, ebx=ebx, res=res, chunks=ch, metas=metas,
                               guids={ebx_file_guid(payload(f)): x.name for x, f in zip(ebx, files[1:1 + len(ebx)])})

    shared, colorway = content.get(SHARED_BUNDLE), content.get(COLORWAY_BUNDLE)
    if not shared or not colorway:
        check(False, 'the shared bundle and the crop top colorway bundle')
        return 1
    print('item')
    items = [(x, f) for x, f in zip(shared['ebx'], shared['files'][1:]) if x.name.startswith('items/cust_tops/own_') and f.patch]
    check(len(items) == 1, 'one new top')
    item = dump(payload(items[0][1]))
    key = re.search(r'Key = "([^"]+)"', item).group(1)
    check(re.search(r'HashedAssetKey = (\d+)', item).group(1) == str(djb(key)), f'HashedAssetKey of {key}')
    paths = re.findall(r'AssetTypeId = (\d+)\s+AssetName = "([^"]+)"', item)
    check(('1', 'Gen_Shirt_CropCrewNeck_complex_dmPreset') in paths, '  the crop crewneck mesh')
    preset_leaf = [n for t, n in paths if t == '3'][0].lower()
    presets = [(x, f) for x, f in zip(colorway['ebx'], colorway['files'][1:]) if x.name.endswith('/' + preset_leaf)]
    check(len(presets) == 1 and presets[0][1].patch, f'  its preset {preset_leaf} is new, in the colorway bundle')
    thumbs = {x.name: (x, f) for x, f in zip(shared['ebx'], shared['files'][1:]) if x.name.startswith('thumbnail/')}
    for slot in ('Thumbnail', 'ThumbnailLarge'):
        fg = re.search(rf'\n\s+{slot} = import\(#\d+ file=(\S+)', item).group(1)
        hit = [n for n, (x, f) in thumbs.items() if f.patch and ebx_file_guid(payload(f)) == fg]
        check(len(hit) == 1, f'  {slot} -> {hit[0] if hit else "nothing"}')
    for n, (x, f) in thumbs.items():
        if not f.patch or not n.startswith('thumbnail/tool/'):
            continue
        rid = int(re.search(r'res\(0x([0-9a-f]+)\)', dump(payload(f))).group(1), 16)
        r = [(rx, rf) for rx, rf in zip(shared['res'], shared['files'][1 + len(shared['ebx']):]) if rx.name == n]
        check(r and r[0][0].res_id == rid and payload(r[0][1])[40:56] in toc, f'  {n.split("/")[-1]}: resource and chunk')

    print('appearance')
    preset = dump(payload(presets[0][1]))
    imports = re.findall(r'^import #\d+ file=(\S+) inst=(\S+)', preset, re.M)
    missing = [f_ for f_, _ in imports if f_ not in colorway['guids']]
    check(not missing, 'every preset import is in the colorway bundle' + (f' (missing {missing})' if missing else ''))
    names = {guid: name.split('/')[-1] for guid, name in colorway['guids'].items()}
    values = preset_values(preset, names)
    f32 = {n: struct.unpack_from('<f', v[1])[0] for n, v in values.items() if n.startswith('param_graphic0_') and n != 'param_graphic0_co'}
    check(values.get('param_usegraphic', (0, b''))[1][:1] == b'\x01', '  the graphic slot is on')
    co = values.get('param_graphic0_co')
    logo_file = imports[struct.unpack_from('<Q', co[1])[0] >> 1][0] if co else None
    tex = [(x, f) for x, f in zip(colorway['ebx'], colorway['files'][1:]) if f.patch and logo_file and ebx_file_guid(payload(f)) == logo_file]
    check(len(tex) == 1, f'  the print is the new logo texture {tex[0][0].name if tex else ""}')
    if not tex:
        return 1
    tdump = dump(payload(tex[0][1]))
    rid = int(re.search(r'res\(0x([0-9a-f]+)\)', tdump).group(1), 16)
    check(re.search(r'Z = (\d+)\s+W = (\d+)', tdump).groups() == (str(LOGO_W), str(LOGO_H)), f'  crop {LOGO_W}x{LOGO_H}')
    rr = [(x, f) for x, f in zip(colorway['res'], colorway['files'][1 + len(colorway['ebx']):]) if x.name == tex[0][0].name]
    check(len(rr) == 1 and rr[0][0].res_id == rid and rr[0][0].res_type == TEXTURE_RES, '  its resource')
    h = payload(rr[0][1])
    mips, first = h[30], h[31]
    sizes = struct.unpack_from(f'<{mips}I', h, 56)
    total = struct.unpack_from('<I', h, 116)[0]
    check(struct.unpack_from('<HH', h, 22) == (LOGO_W, LOGO_H) and mips == 11 and sum(sizes) == total and first == 0,
          f'  {LOGO_W}x{LOGO_H}, {mips} mips, first mip 0 (the whole texture in the bundle)')
    check(struct.unpack_from('<Q', h, 120)[0] == djb(tex[0][0].name, 64), '  name hash')
    c = toc.get(h[40:56])
    check(c is not None, '  chunk in items.toc')
    raw = cas[c.offset:c.offset + c.size]
    at, logical, starts = 0, 0, {}
    while at < len(raw):
        starts[logical] = at
        comp = struct.unpack_from('<H', raw, at + 4)[0]
        logical += fb.be32(raw, at) & 0xFFFFFF
        at += 8 + struct.unpack_from('>H', raw, at + 6)[0] + (((comp >> 8) & 0x0F) << 16)
    check(logical == total, f'  chunk is {logical} bytes')
    check(struct.unpack_from('<II', h, 0) == (starts.get(sizes[0], -1), starts.get(sizes[0] + sizes[1], -1)),
          '  header offsets of mips 1 and 2')
    whole = fb.decode_cas(raw, g.root)
    bundled = [(x, f) for x, f in zip(colorway['chunks'], colorway['files'][1 + len(colorway['ebx']) + len(colorway['res']):]) if x.guid == h[40:56]]
    check(len(bundled) == 1 and bundled[0][0].logical_offset == 0 and bundled[0][0].logical_size == total and
          payload(bundled[0][1]) == whole, '  the bundle carries the whole texture')
    m = colorway['metas'].get(djb(tex[0][0].name, 64))
    fm = fb.db_field(fb.db_field(m, 'meta'), 'firstMip') if m else None
    check(fm is not None and struct.unpack('<i', fm['value'])[0] == 0, '  chunk metadata: h64 and firstMip 0')

    print('placement')
    front = shirt_front(g)
    alpha = bc7_alpha(whole[:sizes[0]], LOGO_W, LOGO_H)
    check(alpha.max() > 0.9 and alpha[0].max() < 0.05 and alpha[-1].max() < 0.05 and alpha[:, 0].max() < 0.05
          and alpha[:, -1].max() < 0.05, '  the logo is inside a clear border')
    offh, offv = f32.get('param_graphic0_offsethorizontal', 9), f32.get('param_graphic0_offsetvertical', 9)
    scale, aspect = f32.get('param_graphic0_scale', 9), f32.get('param_graphic0_aspect', 9)
    check(abs(offh + 0.5 - front[2]) < 0.002, f'  centred on the body (u {offh + 0.5:.4f}, centre line {front[2]:.4f})')
    check(abs(aspect - LOGO_W / LOGO_H) < 1e-3, f'  aspect {aspect:.2f} matches the texture')
    off, facing = print_coverage(front, alpha, offh, offv, scale, aspect)
    check(off == 0 and facing >= MIN_FACING, f'  whole logo on the front of the shirt (width {scale:.3f}, '
                                             f'{off:.1%} off it, faces front >= {facing:.2f})')

    print('shared lists')
    sr = [(x, f) for x, f in zip(shared['res'], shared['files'][1 + len(shared['ebx']):]) if x.name == BUNDLE_REF_TABLE][0]
    t = bundleref.Table(payload(sr[1]), sr[0].res_meta)
    check(t.presets.get(presets[0][0].name) is not None and t.bundle_name_token(t.presets[presets[0][0].name])[0] == bundle_ref(COLORWAY_BUNDLE),
          'bundle-reference table loads the preset from the colorway bundle')
    ic = [(x, f) for x, f in zip(shared['ebx'], shared['files'][1:]) if x.name == ITEM_COLLECTION][0]
    check(ebx_file_guid(payload(items[0][1])) in dump(payload(ic[1])), f'{ITEM_COLLECTION.split("/")[-1]} lists the item')
    others = bundles_of_other_mods(a.game, os.path.basename(os.path.normpath(a.mod))).get(COLORWAY_BUNDLE, [])
    check(not others, 'no other installed mod ships the colorway bundle' + (f': {others}' if others else ''))
    print('\nALL CHECKS PASSED' if not problems else f'\n{len(problems)} PROBLEM(S)')
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
