"""Checks a grip mod built by make_grip_mod.py the way ReSkate and the game will read it.

    python verify_grip_mod.py <mod folder> [--game E:\\...\\Skate]
"""
import argparse
import os
import re
import struct
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fb  # noqa: E402
import bundleref  # noqa: E402
from make_deck_mod import EBXTOOL, DEFAULT_GAME, SHARED_BUNDLE, BUNDLE_REF_TABLE, ITEM_COLLECTION, TEXTURE_RES, djb  # noqa: E402
from make_shoe_mod import bundle_ref, bundles_of_other_mods, ebx_file_guid  # noqa: E402

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
                               guids={ebx_file_guid(payload(f)) for f in files[1:1 + len(ebx)]})

    shared = content.get(SHARED_BUNDLE)
    grip_name = next((n for n in content if n != SHARED_BUNDLE), None)
    grip = content.get(grip_name)
    if not shared or not grip:
        check(False, 'the shared bundle and a grip bundle')
        return 1
    print('item')
    items = [(x, f) for x, f in zip(shared['ebx'], shared['files'][1:]) if x.name.startswith('items/boardtapecolors0/own_') and f.patch]
    check(len(items) == 1, 'one new grip item')
    item = dump(payload(items[0][1]))
    key = re.search(r'Key = "([^"]+)"', item).group(1)
    check(re.search(r'HashedAssetKey = (\d+)', item).group(1) == str(djb(key)), f'HashedAssetKey of {key}')
    preset_leaf = re.search(r'AssetTypeId = 3\s+AssetName = "([^"]+)"', item).group(1).lower()
    presets = [(x, f) for x, f in zip(grip['ebx'], grip['files'][1:]) if x.name.endswith('/' + preset_leaf)]
    check(len(presets) == 1, f'its preset {preset_leaf} is in the grip bundle')
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
    missing = [f_ for f_, _ in re.findall(r'^import #\d+ file=(\S+) inst=(\S+)', preset, re.M) if f_ not in grip['guids']]
    check(not missing, 'every preset import is in the grip bundle' + (f' (missing {missing})' if missing else ''))
    tex = [(x, f) for x, f in zip(grip['ebx'], grip['files'][1:]) if f.patch and x.name.endswith('_c')]
    check(len(tex) == 1, 'one new grip texture')
    tdump = dump(payload(tex[0][1]))
    check(ebx_file_guid(payload(tex[0][1])) in {f_ for f_, _ in re.findall(r'^import #\d+ file=(\S+) inst=(\S+)', preset, re.M)},
          '  the preset imports it')
    rid = int(re.search(r'res\(0x([0-9a-f]+)\)', tdump).group(1), 16)
    rr = [(x, f) for x, f in zip(grip['res'], grip['files'][1 + len(grip['ebx']):]) if x.name == tex[0][0].name]
    check(len(rr) == 1 and rr[0][0].res_id == rid and rr[0][0].res_type == TEXTURE_RES, '  its resource')
    h = payload(rr[0][1])
    mips, first = h[30], h[31]
    sizes = struct.unpack_from(f'<{mips}I', h, 56)
    total = struct.unpack_from('<I', h, 116)[0]
    check(struct.unpack_from('<HH', h, 22) == (512, 2048) and sum(sizes) == total and first == 0,
          f'  512x2048, {mips} mips, first mip 0 (the whole texture in the bundle)')
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
    bundled = [(x, f) for x, f in zip(grip['chunks'], grip['files'][1 + len(grip['ebx']) + len(grip['res']):]) if x.guid == h[40:56]]
    check(len(bundled) == 1 and bundled[0][0].logical_offset == 0 and bundled[0][0].logical_size == total and
          payload(bundled[0][1]) == fb.decode_cas(raw, g.root), '  the bundle carries the whole texture')
    m = grip['metas'].get(djb(tex[0][0].name, 64))
    fm = fb.db_field(fb.db_field(m, 'meta'), 'firstMip') if m else None
    check(fm is not None and struct.unpack('<i', fm['value'])[0] == 0, '  chunk metadata: h64 and firstMip 0')

    print('shared lists')
    sr = [(x, f) for x, f in zip(shared['res'], shared['files'][1 + len(shared['ebx']):]) if x.name == BUNDLE_REF_TABLE][0]
    t = bundleref.Table(payload(sr[1]), sr[0].res_meta)
    check(t.presets.get(presets[0][0].name) is not None and t.bundle_name_token(t.presets[presets[0][0].name])[0] == bundle_ref(grip_name),
          'bundle-reference table loads the preset from the grip bundle')
    ic = [(x, f) for x, f in zip(shared['ebx'], shared['files'][1:]) if x.name == ITEM_COLLECTION][0]
    item_guid = ebx_file_guid(payload(items[0][1]))
    check(item_guid in dump(payload(ic[1])), f'{ITEM_COLLECTION.split("/")[-1]} lists the item')
    others = bundles_of_other_mods(a.game, os.path.basename(os.path.normpath(a.mod))).get(grip_name, [])
    check(not others, 'no other installed mod ships the grip bundle' + (f': {others}' if others else ''))
    print('\nALL CHECKS PASSED' if not problems else f'\n{len(problems)} PROBLEM(S)')
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
