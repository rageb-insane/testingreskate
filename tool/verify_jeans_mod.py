"""Checks a skinny jeans mod built by make_jeans_mod.py the way ReSkate and the game will read it.

    python verify_jeans_mod.py <mod folder> [--game E:\\...\\Skate]
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
from make_top_mod import preset_values  # noqa: E402
from make_jeans_mod import COLORWAY_BUNDLE, BASE  # noqa: E402

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


def bc7_rgb(data, w, h):
    with tempfile.TemporaryDirectory() as d:
        src, dst = os.path.join(d, 'm.bc7'), os.path.join(d, 'm.rgba')
        with open(src, 'wb') as f:
            f.write(data)
        subprocess.run([EBXTOOL, 'bc7dec', src, str(w), str(h), dst], check=True, capture_output=True)
        with open(dst, 'rb') as f:
            return np.frombuffer(f.read(), np.uint8).reshape(h, w, 4)[..., :3].astype(np.float32)


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
    _, base_bundles, base_chunks = g.toc('Win32/items.toc')
    base = {b.name: b for b in base_bundles}
    base_toc = {c.guid: c for c in base_chunks}

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
        check(False, 'the shared bundle and the skinny jeans colorway bundle')
        return 1
    print('item')
    items = [(x, f) for x, f in zip(shared['ebx'], shared['files'][1:]) if x.name.startswith('items/cust_bottoms/own_') and f.patch]
    check(len(items) == 1, 'one new pair of jeans')
    item = dump(payload(items[0][1]))
    key = re.search(r'Key = "([^"]+)"', item).group(1)
    check(re.search(r'HashedAssetKey = (\d+)', item).group(1) == str(djb(key)), f'HashedAssetKey of {key}')
    paths = re.findall(r'AssetTypeId = (\d+)\s+AssetName = "([^"]+)"', item)
    check(('1', 'Gen_Pants_JeansSkinny_complex_dmPreset') in paths, '  the skinny jeans mesh')
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
    check(values.get('param_usegraphic', (0, b'?'))[1][:1] == b'\x00', '  the back logo is off')
    ct = values.get('param_base_c_ta')
    c_file = imports[struct.unpack_from('<Q', ct[1])[0] >> 1][0] if ct else None
    tex = [(x, f) for x, f in zip(colorway['ebx'], colorway['files'][1:]) if f.patch and c_file and ebx_file_guid(payload(f)) == c_file]
    check(len(tex) == 1, f'  the colour texture is the new one {tex[0][0].name.split("/")[-1] if tex else ""}')
    if not tex:
        return 1
    tdump = dump(payload(tex[0][1]))
    check('TextureArrayAsset' in tdump, '  a texture array, like the game\'s')
    rid = int(re.search(r'res\(0x([0-9a-f]+)\)', tdump).group(1), 16)
    rr = [(x, f) for x, f in zip(colorway['res'], colorway['files'][1 + len(colorway['ebx']):]) if x.name == tex[0][0].name]
    check(len(rr) == 1 and rr[0][0].res_id == rid and rr[0][0].res_type == TEXTURE_RES, '  its resource')
    h = payload(rr[0][1])
    mips, first = h[30], h[31]
    sizes = struct.unpack_from(f'<{mips}I', h, 56)
    per_slice = struct.unpack_from('<I', h, 116)[0]
    slices = struct.unpack_from('<H', h, 28)[0]
    check(struct.unpack_from('<HHHH', h, 22) == (1024, 1024, 2, 2) and mips == 11 and sum(sizes) == per_slice and first == 0,
          f'  1024x1024, {slices} slices, {mips} mips, first mip 0 (the whole texture in the bundle)')
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
    check(logical == per_slice * slices, f'  chunk is {logical} bytes ({slices} slices)')
    check(struct.unpack_from('<II', h, 0) == (starts.get(sizes[0] * slices, -1), starts.get((sizes[0] + sizes[1]) * slices, -1)),
          '  header offsets of mips 1 and 2 (stored mip by mip)')
    whole = fb.decode_cas(raw, g.root)
    bundled = [(x, f) for x, f in zip(colorway['chunks'], colorway['files'][1 + len(colorway['ebx']) + len(colorway['res']):]) if x.guid == h[40:56]]
    check(len(bundled) == 1 and bundled[0][0].logical_offset == 0 and bundled[0][0].logical_size == len(whole) and
          payload(bundled[0][1]) == whole, '  the bundle carries the whole texture')
    m = colorway['metas'].get(djb(tex[0][0].name, 64))
    fm = fb.db_field(fb.db_field(m, 'meta'), 'firstMip') if m else None
    check(fm is not None and struct.unpack('<i', fm['value'])[0] == 0, '  chunk metadata: h64 and firstMip 0')

    print('colour')
    base_files, _ = fb.read_bundle_region(base[COLORWAY_BUNDLE].region)
    be, br_, bc_, _ = g.manifest(base_files)
    bh = [g.payload(f) for x, f in zip(br_, base_files[1 + len(be):]) if x.name.endswith('/' + BASE + '_c_ta')][0]
    bt = base_toc[bh[40:56]]
    base_whole = fb.decode_cas(g.read(fb.FileInfo(bt.patch, bt.install_chunk, bt.archive, bt.offset, bt.size)), g.root)
    for s in range(slices):
        new = bc7_rgb(whole[s * sizes[0]:(s + 1) * sizes[0]], 1024, 1024)
        old = bc7_rgb(base_whole[s * sizes[0]:(s + 1) * sizes[0]], 1024, 1024)
        changed = np.abs(new - old).max(2) > 12
        check(changed.mean() > 0.5, f'  slice {s}: {changed.mean():.0%} recoloured, now averaging {tuple(int(v) for v in new[changed].mean(0))}')
    top = bc7_rgb(whole[sizes[0] * slices:sizes[0] * slices + sizes[1]], 512, 512)
    check(abs(top.mean() - bc7_rgb(whole[:sizes[0]], 1024, 1024).mean()) < 12, '  mip 1 matches mip 0')

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
